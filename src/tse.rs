#![deny(unsafe_code)]

//! Acoustic Directional Target Sound Extraction (TSE) with 3D Spatial Conditioning.
//!
//! Features:
//! - Steered Minimum Variance Distortionless Response (MVDR / Capon) adaptive beamformer.
//! - Closed-form stack-allocated complex linear system solver with partial pivoting for dynamic spatial covariance inversion.
//! - 3D GPS ground station operator line-of-sight bearing & elevation conditioning.
//! - Non-linear spatial mask neural/linear gating isolating desired voice in multi-speaker cocktail party scenarios.
//! - Constant Overlap-Add (COLA) time-frequency resynthesis with zero amplitude ripple.
//! - MAVLink telemetry packet emission (`MavlinkNamedValueFloat`).

use crate::beamforming::{ArrayGeometry, Point3D, SPEED_OF_SOUND};
use crate::health::MavlinkNamedValueFloat;
use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// WGS-84 Global Positioning System coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpsCoordinate {
    /// Latitude in degrees [-90.0, +90.0].
    pub latitude_deg: f64,
    /// Longitude in degrees [-180.0, +180.0].
    pub longitude_deg: f64,
    /// Altitude above mean sea level in meters.
    pub altitude_m: f32,
}

impl GpsCoordinate {
    /// Construct a new GPS coordinate.
    pub fn new(latitude_deg: f64, longitude_deg: f64, altitude_m: f32) -> Self {
        Self {
            latitude_deg,
            longitude_deg,
            altitude_m,
        }
    }

    /// Approximate 3D Euclidean distance in meters to another GPS coordinate.
    pub fn distance_to(&self, other: &GpsCoordinate) -> f32 {
        const EARTH_RADIUS: f64 = 6378137.0;
        let d_lat = (other.latitude_deg - self.latitude_deg).to_radians();
        let d_lon = (other.longitude_deg - self.longitude_deg).to_radians();
        let mean_lat = ((self.latitude_deg + other.latitude_deg) * 0.5).to_radians();

        let north_m = (EARTH_RADIUS * d_lat) as f32;
        let east_m = (EARTH_RADIUS * d_lon * mean_lat.cos()) as f32;
        let down_m = self.altitude_m - other.altitude_m;

        (north_m * north_m + east_m * east_m + down_m * down_m).sqrt()
    }
}

/// Computes target relative azimuth and elevation in drone body frame from GPS positions and drone heading.
///
/// In aerospace body-fixed coordinate conventions:
/// - +X is Forward (nose), +Y is Right (starboard wing), +Z is Down.
/// - Azimuth: 0 rad is straight ahead (+X), +pi/2 is right (+Y), -pi/2 is left (-Y).
/// - Elevation: 0 rad is horizontal, +pi/2 is straight up, -pi/2 is straight down.
pub fn calculate_bearing_from_gps(
    drone_gps: &GpsCoordinate,
    operator_gps: &GpsCoordinate,
    drone_yaw_rad: f32,
) -> (f32, f32) {
    const EARTH_RADIUS: f64 = 6378137.0;
    let d_lat = (operator_gps.latitude_deg - drone_gps.latitude_deg).to_radians();
    let d_lon = (operator_gps.longitude_deg - drone_gps.longitude_deg).to_radians();
    let mean_lat = ((drone_gps.latitude_deg + operator_gps.latitude_deg) * 0.5).to_radians();

    let north_m = (EARTH_RADIUS * d_lat) as f32;
    let east_m = (EARTH_RADIUS * d_lon * mean_lat.cos()) as f32;
    let down_m = drone_gps.altitude_m - operator_gps.altitude_m;

    // Rotate NED by -drone_yaw to convert into drone body frame
    let cos_yaw = drone_yaw_rad.cos();
    let sin_yaw = drone_yaw_rad.sin();
    let body_x = north_m * cos_yaw + east_m * sin_yaw;
    let body_y = -north_m * sin_yaw + east_m * cos_yaw;
    let body_z = down_m;

    let horizontal_dist = (body_x * body_x + body_y * body_y).sqrt();
    let azimuth_rad = body_y.atan2(body_x);
    let elevation_rad = (-body_z).atan2(horizontal_dist.max(1e-3));

    (azimuth_rad, elevation_rad)
}

/// Conditioning target mode for spatial extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialConditioningTarget {
    /// Fixed direct angles in vehicle body frame (radians).
    BodyFrame {
        azimuth_rad: f32,
        elevation_rad: f32,
    },
    /// Live GPS operator tracking with dynamic drone heading.
    GpsTracked {
        drone_gps: GpsCoordinate,
        operator_gps: GpsCoordinate,
        drone_yaw_rad: f32,
    },
}

impl SpatialConditioningTarget {
    /// Returns current (azimuth_rad, elevation_rad) in body frame.
    pub fn angles(&self) -> (f32, f32) {
        match self {
            SpatialConditioningTarget::BodyFrame {
                azimuth_rad,
                elevation_rad,
            } => (*azimuth_rad, *elevation_rad),
            SpatialConditioningTarget::GpsTracked {
                drone_gps,
                operator_gps,
                drone_yaw_rad,
            } => calculate_bearing_from_gps(drone_gps, operator_gps, *drone_yaw_rad),
        }
    }
}

/// Lightweight single-precision complex number helper for numerical beamforming.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> f32 {
        self.norm_sqr().sqrt()
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    pub fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sqr().max(1e-12);
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }

    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// In-place solver for an M x M complex linear system A * x = b via Gauss-Jordan elimination with partial pivoting.
/// For M <= 8, utilizes a stack-allocated buffer (72 elements) with zero heap allocations.
pub fn solve_complex_linear_system_in_place(
    m: usize,
    a_mat: &[Complex32],
    b_vec: &[Complex32],
    solution: &mut [Complex32],
) -> bool {
    if m == 0 || a_mat.len() != m * m || b_vec.len() != m || solution.len() < m {
        return false;
    }

    if m <= 8 {
        let mut aug = [Complex32::zero(); 72];
        let stride = m + 1;
        for r in 0..m {
            for c in 0..m {
                aug[r * stride + c] = a_mat[r * m + c];
            }
            aug[r * stride + m] = b_vec[r];
        }

        for col in 0..m {
            let mut max_norm = aug[col * stride + col].norm_sqr();
            let mut pivot_row = col;
            for r in (col + 1)..m {
                let n = aug[r * stride + col].norm_sqr();
                if n > max_norm {
                    max_norm = n;
                    pivot_row = r;
                }
            }

            if max_norm < 1e-12 {
                return false;
            }

            if pivot_row != col {
                for c in 0..=m {
                    aug.swap(col * stride + c, pivot_row * stride + c);
                }
            }

            let pivot_val = aug[col * stride + col];
            for c in col..=m {
                aug[col * stride + c] = aug[col * stride + c].div(pivot_val);
            }

            for r in 0..m {
                if r != col {
                    let factor = aug[r * stride + col];
                    if factor.norm_sqr() > 1e-12 {
                        for c in col..=m {
                            let sub_term = factor.mul(aug[col * stride + c]);
                            aug[r * stride + c] = aug[r * stride + c].sub(sub_term);
                        }
                    }
                }
            }
        }

        for r in 0..m {
            solution[r] = aug[r * stride + m];
        }
        true
    } else {
        let stride = m + 1;
        let mut aug = vec![Complex32::zero(); m * stride];
        for r in 0..m {
            for c in 0..m {
                aug[r * stride + c] = a_mat[r * m + c];
            }
            aug[r * stride + m] = b_vec[r];
        }

        for col in 0..m {
            let mut max_norm = aug[col * stride + col].norm_sqr();
            let mut pivot_row = col;
            for r in (col + 1)..m {
                let n = aug[r * stride + col].norm_sqr();
                if n > max_norm {
                    max_norm = n;
                    pivot_row = r;
                }
            }

            if max_norm < 1e-12 {
                return false;
            }

            if pivot_row != col {
                for c in 0..=m {
                    aug.swap(col * stride + c, pivot_row * stride + c);
                }
            }

            let pivot_val = aug[col * stride + col];
            for c in col..=m {
                aug[col * stride + c] = aug[col * stride + c].div(pivot_val);
            }

            for r in 0..m {
                if r != col {
                    let factor = aug[r * stride + col];
                    if factor.norm_sqr() > 1e-12 {
                        for c in col..=m {
                            let sub_term = factor.mul(aug[col * stride + c]);
                            aug[r * stride + c] = aug[r * stride + c].sub(sub_term);
                        }
                    }
                }
            }
        }

        for r in 0..m {
            solution[r] = aug[r * stride + m];
        }
        true
    }
}

/// Solves an M x M complex linear system A * x = b via Gauss-Jordan elimination with partial pivoting.
pub fn solve_complex_linear_system(
    m: usize,
    a_mat: &[Complex32],
    b_vec: &[Complex32],
) -> Option<Vec<Complex32>> {
    let mut solution = vec![Complex32::zero(); m];
    if solve_complex_linear_system_in_place(m, a_mat, b_vec, &mut solution) {
        Some(solution)
    } else {
        None
    }
}

/// Target Sound Extraction (TSE) configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TseConfig {
    /// FFT frame size (must be power of two, e.g. 512).
    pub fft_size: usize,
    /// Hop size for streaming execution (typically fft_size / 2, e.g. 256).
    pub hop_size: usize,
    /// Recursive covariance matrix exponential forgetting factor (e.g. 0.90 to 0.98).
    pub covariance_alpha: f32,
    /// Diagonal loading regularization factor added to trace (e.g. 0.01 to 0.05).
    pub diagonal_loading: f32,
    /// Spatial similarity threshold xi for soft mask gating (e.g. 0.50).
    pub spatial_mask_threshold: f32,
    /// Spatial mask sigmoid steepness parameter gamma (e.g. 8.0).
    pub spatial_mask_gamma: f32,
    /// Minimum attenuation floor for spatial mask (e.g. 0.03 = -30 dB).
    pub attenuation_floor: f32,
    /// Minimum frequency for spatial processing in Hz (e.g. 150.0).
    pub min_freq_hz: f32,
    /// Maximum frequency for spatial processing in Hz (e.g. 4000.0).
    pub max_freq_hz: f32,
}

impl Default for TseConfig {
    fn default() -> Self {
        Self {
            fft_size: 512,
            hop_size: 256,
            covariance_alpha: 0.92,
            diagonal_loading: 0.02,
            spatial_mask_threshold: 0.50,
            spatial_mask_gamma: 8.0,
            attenuation_floor: 0.03,
            min_freq_hz: 150.0,
            max_freq_hz: 4000.0,
        }
    }
}

/// Diagnostic report evaluated for each processed target extraction block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TseReport {
    /// Active steered azimuth in radians.
    pub steered_azimuth_rad: f32,
    /// Active steered elevation in radians.
    pub steered_elevation_rad: f32,
    /// Target RMS energy of the extracted stream.
    pub target_rms: f32,
    /// Estimated interference suppression relative to omnidirectional input in dB.
    pub interference_suppression_db: f32,
    /// Mean spatial mask coefficient across voice frequencies [0.0, 1.0].
    pub mean_spatial_mask: f32,
    /// Target acoustic activity flag.
    pub is_target_present: bool,
    /// Evaluation timestamp in seconds.
    pub timestamp_sec: f64,
}

impl TseReport {
    /// Convert TSE diagnostic telemetry into standard MAVLink named value float packets.
    pub fn to_mavlink_packets(&self, time_boot_ms: u32) -> Vec<MavlinkNamedValueFloat> {
        vec![
            MavlinkNamedValueFloat::new(
                time_boot_ms,
                "TSE_AZIM",
                self.steered_azimuth_rad.to_degrees(),
            ),
            MavlinkNamedValueFloat::new(
                time_boot_ms,
                "TSE_ELEV",
                self.steered_elevation_rad.to_degrees(),
            ),
            MavlinkNamedValueFloat::new(time_boot_ms, "TSE_MASK", self.mean_spatial_mask),
            MavlinkNamedValueFloat::new(
                time_boot_ms,
                "TSE_SUPP",
                self.interference_suppression_db,
            ),
        ]
    }
}

/// Steered MVDR Adaptive Beamformer with 3D Spatial Mask Conditioning.
#[derive(Debug, Clone)]
pub struct TargetSoundExtractor {
    geometry: ArrayGeometry,
    positions: Vec<Point3D>,
    sample_rate: f32,
    config: TseConfig,
    num_mics: usize,
    target: SpatialConditioningTarget,
    fft: FftProcessor,
    analysis_window: Window,
    covariances: Vec<Vec<Complex32>>,
    overlap_buffer: Vec<f32>,
    input_buffers: Vec<Vec<f32>>,
    latest_report: Option<TseReport>,

    // Preallocated scratch buffers for zero-allocation streaming execution
    scratch_stft_real: Vec<f32>,
    scratch_stft_imag: Vec<f32>,
    scratch_stft_frames: Vec<Vec<Complex32>>,
    scratch_reg_cov: Vec<Complex32>,
    scratch_mvdr_sol: Vec<Complex32>,
    scratch_mvdr_weights: Vec<Complex32>,
    scratch_a_k: Vec<Complex32>,
    scratch_target_spectrum: Vec<Complex32>,
    scratch_time_real: Vec<f32>,
    scratch_time_imag: Vec<f32>,
}

impl TargetSoundExtractor {
    /// Construct a new Target Sound Extractor for the given array geometry and sampling rate.
    pub fn new(geometry: ArrayGeometry, sample_rate: f32, config: TseConfig) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(
            config.fft_size > 0 && (config.fft_size & (config.fft_size - 1)) == 0,
            "FFT size must be power of two"
        );
        assert!(
            config.hop_size > 0 && config.hop_size <= config.fft_size,
            "Hop size must be <= FFT size"
        );

        let positions = geometry.positions();
        let num_mics = positions.len();
        assert!(num_mics >= 2, "Array must contain at least 2 microphones");

        let num_bins = config.fft_size / 2 + 1;
        let mut covariances = Vec::with_capacity(num_bins);
        for _ in 0..num_bins {
            let mut cov = vec![Complex32::zero(); num_mics * num_mics];
            for m in 0..num_mics {
                cov[m * num_mics + m] = Complex32::new(1e-4, 0.0);
            }
            covariances.push(cov);
        }

        let input_buffers = vec![Vec::with_capacity(config.fft_size * 2); num_mics];
        let overlap_buffer = vec![0.0f32; config.fft_size];

        let scratch_stft_real = vec![0.0f32; config.fft_size];
        let scratch_stft_imag = vec![0.0f32; config.fft_size];
        let scratch_stft_frames = vec![vec![Complex32::zero(); num_bins]; num_mics];
        let scratch_reg_cov = vec![Complex32::zero(); num_mics * num_mics];
        let scratch_mvdr_sol = vec![Complex32::zero(); num_mics];
        let scratch_mvdr_weights = vec![Complex32::zero(); num_mics];
        let scratch_a_k = vec![Complex32::zero(); num_mics];
        let scratch_target_spectrum = vec![Complex32::zero(); num_bins];
        let scratch_time_real = vec![0.0f32; config.fft_size];
        let scratch_time_imag = vec![0.0f32; config.fft_size];

        Self {
            geometry,
            positions,
            sample_rate,
            config: config.clone(),
            num_mics,
            target: SpatialConditioningTarget::BodyFrame {
                azimuth_rad: 0.0,
                elevation_rad: 0.0,
            },
            fft: FftProcessor::new(config.fft_size),
            analysis_window: Window::new(WindowType::Hann, config.fft_size),
            covariances,
            overlap_buffer,
            input_buffers,
            latest_report: None,
            scratch_stft_real,
            scratch_stft_imag,
            scratch_stft_frames,
            scratch_reg_cov,
            scratch_mvdr_sol,
            scratch_mvdr_weights,
            scratch_a_k,
            scratch_target_spectrum,
            scratch_time_real,
            scratch_time_imag,
        }
    }

    /// Dynamically update steered target angles in body frame (radians).
    pub fn set_target_bearing(&mut self, azimuth_rad: f32, elevation_rad: f32) {
        self.target = SpatialConditioningTarget::BodyFrame {
            azimuth_rad,
            elevation_rad,
        };
    }

    /// Dynamically update steered target using GPS drone + operator coordinates and drone yaw.
    pub fn set_target_gps(
        &mut self,
        drone_gps: GpsCoordinate,
        operator_gps: GpsCoordinate,
        drone_yaw_rad: f32,
    ) {
        self.target = SpatialConditioningTarget::GpsTracked {
            drone_gps,
            operator_gps,
            drone_yaw_rad,
        };
    }

    /// Get current active steered (azimuth_rad, elevation_rad).
    pub fn steered_bearing(&self) -> (f32, f32) {
        self.target.angles()
    }

    /// Compute acoustic steering vector a(k) for given frequency and angles.
    pub fn compute_steering_vector(
        &self,
        freq_hz: f32,
        azimuth_rad: f32,
        elevation_rad: f32,
    ) -> Vec<Complex32> {
        let mut a = vec![Complex32::zero(); self.num_mics];
        self.compute_steering_vector_in_place(freq_hz, azimuth_rad, elevation_rad, &mut a);
        a
    }

    /// In-place calculation of acoustic steering vector into destination slice.
    pub fn compute_steering_vector_in_place(
        &self,
        freq_hz: f32,
        azimuth_rad: f32,
        elevation_rad: f32,
        dest: &mut [Complex32],
    ) {
        compute_steering_vector_static(&self.positions, freq_hz, azimuth_rad, elevation_rad, dest);
    }
}

/// Static helper for in-place steering vector computation from array positions.
pub fn compute_steering_vector_static(
    positions: &[Point3D],
    freq_hz: f32,
    azimuth_rad: f32,
    elevation_rad: f32,
    dest: &mut [Complex32],
) {
    let u_x = elevation_rad.cos() * azimuth_rad.cos();
    let u_y = elevation_rad.cos() * azimuth_rad.sin();
    let u_z = elevation_rad.sin();

    for (m, p) in positions.iter().enumerate().take(dest.len()) {
        let proj_m = p.x * u_x + p.y * u_y + p.z * u_z;
        let phase = 2.0 * PI * freq_hz * proj_m / SPEED_OF_SOUND;
        dest[m] = Complex32::from_polar(1.0, phase);
    }
}

impl TargetSoundExtractor {

    /// Process a multi-channel block of length `hop_size`, returning extracted target audio.
    /// Runs with strictly zero heap allocations in steady state.
    pub fn process_block(
        &mut self,
        multi_channel_inputs: &[&[f32]],
        output: &mut [f32],
        timestamp_sec: f64,
    ) -> TseReport {
        let hop = self.config.hop_size;
        let n = self.config.fft_size;
        assert_eq!(output.len(), hop, "Output length must match hop size");
        assert!(
            multi_channel_inputs.len() >= self.num_mics,
            "Input channels must match microphone count"
        );

        for (m, ch) in multi_channel_inputs.iter().take(self.num_mics).enumerate() {
            self.input_buffers[m].extend_from_slice(&ch[..hop]);
        }

        // If buffer does not yet contain a full FFT frame, fill output with zeros
        if self.input_buffers[0].len() < n {
            output.fill(0.0);
            return TseReport {
                steered_azimuth_rad: 0.0,
                steered_elevation_rad: 0.0,
                target_rms: 0.0,
                interference_suppression_db: 0.0,
                mean_spatial_mask: 0.0,
                is_target_present: false,
                timestamp_sec,
            };
        }

        let (azimuth_rad, elevation_rad) = self.target.angles();
        let num_bins = n / 2 + 1;
        let freq_bin_hz = self.sample_rate / (n as f32);

        // 1. Forward STFT for all M channels into scratch_stft_frames
        for m in 0..self.num_mics {
            self.scratch_stft_real.copy_from_slice(&self.input_buffers[m][..n]);
            self.scratch_stft_imag.fill(0.0);
            self.analysis_window.apply(&mut self.scratch_stft_real);
            self.fft
                .fft_in_place(&mut self.scratch_stft_real, &mut self.scratch_stft_imag);

            for k in 0..num_bins {
                self.scratch_stft_frames[m][k] =
                    Complex32::new(self.scratch_stft_real[k], self.scratch_stft_imag[k]);
            }
        }

        // 2. Spatial Covariance Matrix Update & MVDR + Spatial Mask Processing
        let alpha = self.config.covariance_alpha;
        let one_minus_alpha = 1.0 - alpha;
        let mut mask_sum = 0.0f32;
        let mut mask_count = 0;
        let num_pairs = (self.num_mics * (self.num_mics - 1)) / 2;

        let mut input_power_sum = 0.0f32;
        let mut target_power_sum = 0.0f32;

        for k in 0..num_bins {
            let freq_hz = (k as f32) * freq_bin_hz;

            // Compute input channel power
            let ch0_pwr = self.scratch_stft_frames[0][k].norm_sqr();
            input_power_sum += ch0_pwr;

            // Recursive covariance update R(k) = alpha * R(k) + (1 - alpha) * X(k) * X(k)^H
            let cov = &mut self.covariances[k];
            let mut trace = 0.0f32;
            for r in 0..self.num_mics {
                let xr = self.scratch_stft_frames[r][k];
                for c in 0..self.num_mics {
                    let xc = self.scratch_stft_frames[c][k];
                    let outer = xr.mul(xc.conj());
                    let old_val = cov[r * self.num_mics + c];
                    let updated = old_val.scale(alpha).add(outer.scale(one_minus_alpha));
                    cov[r * self.num_mics + c] = updated;
                }
                trace += cov[r * self.num_mics + r].re;
            }

            // Regularized covariance R_reg = R + delta * trace * I
            let reg_delta = (trace * self.config.diagonal_loading).max(1e-5);
            self.scratch_reg_cov.copy_from_slice(cov);
            for m in 0..self.num_mics {
                let idx = m * self.num_mics + m;
                self.scratch_reg_cov[idx].re += reg_delta;
            }

            // Steering vector a(k)
            compute_steering_vector_static(
                &self.positions,
                freq_hz,
                azimuth_rad,
                elevation_rad,
                &mut self.scratch_a_k,
            );

            // Solve R_reg * v = a in-place
            let solved = solve_complex_linear_system_in_place(
                self.num_mics,
                &self.scratch_reg_cov,
                &self.scratch_a_k,
                &mut self.scratch_mvdr_sol,
            );

            if solved {
                // Denominator: a^H * v = sum(a_m^* * v_m)
                let mut denom = Complex32::zero();
                for m in 0..self.num_mics {
                    denom = denom.add(self.scratch_a_k[m].conj().mul(self.scratch_mvdr_sol[m]));
                }
                if denom.re > 1e-9 {
                    for m in 0..self.num_mics {
                        self.scratch_mvdr_weights[m] = self.scratch_mvdr_sol[m].div(denom);
                    }
                } else {
                    let inv_m = 1.0 / (self.num_mics as f32);
                    for m in 0..self.num_mics {
                        self.scratch_mvdr_weights[m] = self.scratch_a_k[m].scale(inv_m);
                    }
                }
            } else {
                let inv_m = 1.0 / (self.num_mics as f32);
                for m in 0..self.num_mics {
                    self.scratch_mvdr_weights[m] = self.scratch_a_k[m].scale(inv_m);
                }
            }

            // MVDR Beamformed Output Y(k) = w^H * X(k) = sum(w_m^* * X_m)
            let mut y_k = Complex32::zero();
            for m in 0..self.num_mics {
                let xm = self.scratch_stft_frames[m][k];
                y_k = y_k.add(self.scratch_mvdr_weights[m].conj().mul(xm));
            }

            // Spatial Conditioning Mask Computation
            let spatial_mask = if freq_hz >= self.config.min_freq_hz
                && freq_hz <= self.config.max_freq_hz
                && num_pairs > 0
            {
                // Inter-channel Phase Alignment Similarity
                let mut sim_sum = 0.0f32;
                for i in 0..self.num_mics {
                    let xi = self.scratch_stft_frames[i][k];
                    let ai = self.scratch_a_k[i];
                    for j in (i + 1)..self.num_mics {
                        let xj = self.scratch_stft_frames[j][k];
                        let aj = self.scratch_a_k[j];
                        // Z = (X_i * X_j^*) * (a_i^* * a_j)
                        let x_cross = xi.mul(xj.conj());
                        let a_cross = ai.conj().mul(aj);
                        let z = x_cross.mul(a_cross);
                        let norm_term = xi.norm() * xj.norm() + 1e-9;
                        let cos_phase_err = (z.re / norm_term).clamp(-1.0, 1.0);
                        sim_sum += cos_phase_err;
                    }
                }
                let mean_sim = sim_sum / (num_pairs as f32);

                // Sigmoidal Spatial Gating Mask
                let arg = self.config.spatial_mask_gamma
                    * (mean_sim - self.config.spatial_mask_threshold);
                let sigmoid = 1.0 / (1.0 + (-arg).exp());
                let mask = self.config.attenuation_floor
                    + (1.0 - self.config.attenuation_floor) * sigmoid;

                mask_sum += mask;
                mask_count += 1;
                mask
            } else if freq_hz < self.config.min_freq_hz {
                // Attenuate sub-speech low frequency rotor rumble
                self.config.attenuation_floor
            } else {
                1.0
            };

            // Gated Target Spectrum S(k) = Y(k) * M(k)
            let s_k = y_k.scale(spatial_mask);
            self.scratch_target_spectrum[k] = s_k;
            target_power_sum += s_k.norm_sqr();
        }

        // 3. Inverse STFT Reconstruction via Complex Conjugate FFT
        self.scratch_time_real.fill(0.0);
        self.scratch_time_imag.fill(0.0);

        self.scratch_time_real[0] = self.scratch_target_spectrum[0].re;
        self.scratch_time_imag[0] = 0.0;

        for k in 1..(n / 2) {
            self.scratch_time_real[k] = self.scratch_target_spectrum[k].re;
            self.scratch_time_imag[k] = -self.scratch_target_spectrum[k].im; // Conjugate for IFFT trick

            self.scratch_time_real[n - k] = self.scratch_target_spectrum[k].re;
            self.scratch_time_imag[n - k] = self.scratch_target_spectrum[k].im;
        }

        self.scratch_time_real[n / 2] = self.scratch_target_spectrum[n / 2].re;
        self.scratch_time_imag[n / 2] = 0.0;

        self.fft
            .fft_in_place(&mut self.scratch_time_real, &mut self.scratch_time_imag);

        let inv_n = 1.0 / (n as f32);
        for re in &mut self.scratch_time_real {
            *re *= inv_n;
        }

        // 4. Overlap-Add Resynthesis (50% Overlap Hann Analysis yields exact unity COLA)
        for i in 0..n {
            self.overlap_buffer[i] += self.scratch_time_real[i];
        }

        // Drain hop_size samples to output
        output.copy_from_slice(&self.overlap_buffer[..hop]);

        // Shift overlap buffer forward by hop_size
        self.overlap_buffer.copy_within(hop..n, 0);
        self.overlap_buffer[(n - hop)..n].fill(0.0);

        // Slide input history buffers forward by hop_size
        for buf in &mut self.input_buffers {
            buf.drain(..hop);
        }

        // 5. Compute Diagnostic Telemetry Report
        let mut out_sq_sum = 0.0f32;
        for &s in output.iter() {
            out_sq_sum += s * s;
        }
        let target_rms = (out_sq_sum / (hop as f32)).sqrt();

        let mean_mask = if mask_count > 0 {
            mask_sum / (mask_count as f32)
        } else {
            1.0
        };

        let in_pwr = input_power_sum.max(1e-9);
        let out_pwr = target_power_sum.max(1e-9);
        let suppression_ratio = in_pwr / out_pwr;
        let interference_suppression_db = (10.0 * suppression_ratio.log10()).max(0.0);

        let is_target_present = target_rms > 0.015 && mean_mask > 0.40;

        let report = TseReport {
            steered_azimuth_rad: azimuth_rad,
            steered_elevation_rad: elevation_rad,
            target_rms,
            interference_suppression_db,
            mean_spatial_mask: mean_mask,
            is_target_present,
            timestamp_sec,
        };

        self.latest_report = Some(report.clone());
        report
    }

    /// Access reference to active array geometry.
    pub fn geometry(&self) -> &ArrayGeometry {
        &self.geometry
    }

    /// Access configuration parameters.
    pub fn config(&self) -> &TseConfig {
        &self.config
    }

    /// Number of microphones in the array.
    pub fn num_mics(&self) -> usize {
        self.num_mics
    }

    /// Sampling rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Access reference to latest evaluated diagnostic report.
    pub fn latest_report(&self) -> Option<&TseReport> {
        self.latest_report.as_ref()
    }

    /// Reset all internal state and buffers.
    pub fn reset(&mut self) {
        for cov in &mut self.covariances {
            cov.fill(Complex32::zero());
            for m in 0..self.num_mics {
                cov[m * self.num_mics + m] = Complex32::new(1e-4, 0.0);
            }
        }
        for buf in &mut self.input_buffers {
            buf.clear();
        }
        self.overlap_buffer.fill(0.0);
        self.latest_report = None;
    }
}
