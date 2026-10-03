#![deny(unsafe_code)]

use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Speed of sound in dry air at 20 degrees Celsius in meters/second.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// 3D Spatial Cartesian Coordinate in meters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Microphone Array Geometric Topologies for Drones and Robotics.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayGeometry {
    /// Uniform Linear Array (ULA) along the Y-axis (lateral wing mount, broadside is forward +X).
    Linear { spacing: f32, num_mics: usize },
    /// Uniform Circular Array (UCA) on the XY plane with given radius.
    Circular { radius: f32, num_mics: usize },
    /// 3D Regular Tetrahedral Array centered at origin (4 microphones).
    Tetrahedral { radius: f32 },
    /// Custom arbitrary 3D microphone coordinates.
    Custom(Vec<Point3D>),
}

impl ArrayGeometry {
    /// Generates microphone coordinate points for the specified geometry.
    ///
    /// In aerospace body-fixed coordinates:
    /// +X is Forward (nose), +Y is Right (starboard wing), +Z is Down.
    /// A linear array mounted along the wings (Y-axis) has its broadside looking forward (+X).
    pub fn positions(&self) -> Vec<Point3D> {
        match self {
            ArrayGeometry::Linear { spacing, num_mics } => {
                let count = (*num_mics).max(2);
                let mut pts = Vec::with_capacity(count);
                let half_span = (count as f32 - 1.0) * spacing * 0.5;
                for i in 0..count {
                    // Mounted laterally along Y-axis so forward +X is broadside
                    pts.push(Point3D::new(0.0, (i as f32) * spacing - half_span, 0.0));
                }
                pts
            }
            ArrayGeometry::Circular { radius, num_mics } => {
                let count = (*num_mics).max(3);
                let mut pts = Vec::with_capacity(count);
                for i in 0..count {
                    let angle = 2.0 * PI * (i as f32) / (count as f32);
                    pts.push(Point3D::new(radius * angle.cos(), radius * angle.sin(), 0.0));
                }
                pts
            }
            ArrayGeometry::Tetrahedral { radius } => {
                let r = *radius;
                vec![
                    Point3D::new(0.0, 0.0, -r),
                    Point3D::new(r * (8.0f32 / 9.0).sqrt(), 0.0, r / 3.0),
                    Point3D::new(
                        -r * (2.0f32 / 9.0).sqrt(),
                        r * (2.0f32 / 3.0).sqrt(),
                        r / 3.0,
                    ),
                    Point3D::new(
                        -r * (2.0f32 / 9.0).sqrt(),
                        -r * (2.0f32 / 3.0).sqrt(),
                        r / 3.0,
                    ),
                ]
            }
            ArrayGeometry::Custom(pts) => pts.clone(),
        }
    }

    /// Number of microphones in the array.
    pub fn num_mics(&self) -> usize {
        match self {
            ArrayGeometry::Linear { num_mics, .. } => *num_mics,
            ArrayGeometry::Circular { num_mics, .. } => *num_mics,
            ArrayGeometry::Tetrahedral { .. } => 4,
            ArrayGeometry::Custom(pts) => pts.len(),
        }
    }
}

/// Generalized Cross-Correlation with Phase Transform (GCC-PHAT) Engine.
///
/// Computes sub-millisecond Time Difference of Arrival (TDOA) between pairs
/// of microphone signals using windowed cross-spectral phase normalization:
///
/// Psi_ij[k] = (X_j[k] * X_i^*[k]) / (|X_j[k] * X_i^*[k]| + epsilon)
#[derive(Debug, Clone)]
pub struct GccPhatEstimator {
    fft_size: usize,
    sample_rate: f32,
    fft: FftProcessor,
    window: Window,
}

impl GccPhatEstimator {
    /// Constructs a new GCC-PHAT TDOA estimator.
    ///
    /// # Arguments
    /// * `fft_size` - FFT size (power of 2, typically 512 or 1024).
    /// * `sample_rate` - Audio sampling frequency in Hz (e.g. 16000.0).
    pub fn new(fft_size: usize, sample_rate: f32) -> Self {
        assert!(fft_size > 0 && (fft_size & (fft_size - 1)) == 0);
        let window_len = fft_size / 2;
        Self {
            fft_size,
            sample_rate,
            fft: FftProcessor::new(fft_size),
            window: Window::new(WindowType::Hann, window_len),
        }
    }

    /// Estimates Time Difference of Arrival (TDOA) in seconds between two channel frames.
    ///
    /// Positive delay indicates signal arrived at channel 1 before channel 2 (ch2 is delayed relative to ch1).
    /// Returns (delay_seconds, correlation_peak_value).
    pub fn estimate_tdoa(&self, ch1: &[f32], ch2: &[f32], max_delay_sec: f32) -> (f32, f32) {
        let n = self.fft_size;
        let win_len = self.window.len();

        let mut x1_re = vec![0.0f32; n];
        let mut x1_im = vec![0.0f32; n];
        let mut x2_re = vec![0.0f32; n];
        let mut x2_im = vec![0.0f32; n];

        let l1 = ch1.len().min(win_len);
        let l2 = ch2.len().min(win_len);
        x1_re[..l1].copy_from_slice(&ch1[..l1]);
        x2_re[..l2].copy_from_slice(&ch2[..l2]);

        // Apply Hann window to eliminate truncation sinc leakage
        self.window.apply(&mut x1_re[..win_len]);
        self.window.apply(&mut x2_re[..win_len]);

        // Forward FFT on both channels
        self.fft.fft_in_place(&mut x1_re, &mut x1_im);
        self.fft.fft_in_place(&mut x2_re, &mut x2_im);

        // Cross-spectrum G = X2 * conj(X1)
        let mut cross_re = vec![0.0f32; n];
        let mut cross_im = vec![0.0f32; n];

        let mut sum_mag = 0.0f32;
        for k in 0..n {
            let re = x2_re[k] * x1_re[k] + x2_im[k] * x1_im[k];
            let im = x2_im[k] * x1_re[k] - x2_re[k] * x1_im[k];
            let mag = (re * re + im * im).sqrt();
            sum_mag += mag;
            cross_re[k] = re;
            cross_im[k] = im;
        }

        // Regularized PHAT weighting
        let eps = (sum_mag / (n as f32) * 0.05).max(1e-6);
        for k in 0..n {
            let mag = (cross_re[k] * cross_re[k] + cross_im[k] * cross_im[k]).sqrt();
            let denom = mag + eps;
            cross_re[k] /= denom;
            cross_im[k] /= denom;
        }

        // IFFT via conjugate forward FFT: IFFT(Z) = conj(FFT(conj(Z))) / N
        for im in cross_im.iter_mut() {
            *im = -*im;
        }
        self.fft.fft_in_place(&mut cross_re, &mut cross_im);

        let inv_n = 1.0 / (n as f32);
        for re in cross_re.iter_mut() {
            *re *= inv_n;
        }

        // Search within [-max_lag_samples, +max_lag_samples]
        let max_lag_samples = ((max_delay_sec * self.sample_rate).ceil() as isize).min((n / 4) as isize);
        let mut best_lag = 0isize;
        let mut best_peak = -1e9f32;

        for lag in -max_lag_samples..=max_lag_samples {
            let idx = if lag >= 0 {
                lag as usize
            } else {
                (n as isize + lag) as usize
            };
            let val = cross_re[idx];
            if val > best_peak {
                best_peak = val;
                best_lag = lag;
            }
        }

        // Parabolic interpolation for sub-sample accuracy
        let peak_idx = if best_lag >= 0 {
            best_lag as usize
        } else {
            (n as isize + best_lag) as usize
        };
        let prev_idx = if peak_idx == 0 { n - 1 } else { peak_idx - 1 };
        let next_idx = (peak_idx + 1) % n;

        let y_prev = cross_re[prev_idx];
        let y_peak = cross_re[peak_idx];
        let y_next = cross_re[next_idx];

        let denom = 2.0 * (2.0 * y_peak - y_prev - y_next);
        let delta = if denom.abs() > 1e-9 {
            (y_next - y_prev) / denom
        } else {
            0.0
        };

        let refined_lag = (best_lag as f32 + delta.clamp(-0.5, 0.5)) / self.sample_rate;
        (refined_lag, best_peak)
    }
}

/// 2D / 3D Direction of Arrival (DoA) Triangulator.
#[derive(Debug, Clone)]
pub struct DoaEstimator {
    geometry: ArrayGeometry,
    sample_rate: f32,
    gcc_phat: GccPhatEstimator,
}

impl DoaEstimator {
    /// Constructs a new DoA Estimator for the given array geometry.
    pub fn new(geometry: ArrayGeometry, sample_rate: f32, fft_size: usize) -> Self {
        Self {
            geometry,
            sample_rate,
            gcc_phat: GccPhatEstimator::new(fft_size, sample_rate),
        }
    }

    /// Estimates 2D azimuth angle in radians [-PI, PI].
    ///
    /// For a lateral wing-mounted Linear Array along Y-axis:
    /// Azimuth = 0 is straight ahead (broadside +X).
    /// Azimuth > 0 is starboard (right +Y), Azimuth < 0 is port (left -Y).
    pub fn estimate_azimuth(&self, multi_channel_frames: &[&[f32]]) -> f32 {
        let positions = self.geometry.positions();
        let num_mics = positions.len().min(multi_channel_frames.len());
        if num_mics < 2 {
            return 0.0;
        }

        match &self.geometry {
            ArrayGeometry::Linear { spacing, .. } => {
                let max_delay = *spacing / SPEED_OF_SOUND;
                let (tdoa, _) = self.gcc_phat.estimate_tdoa(
                    multi_channel_frames[0],
                    multi_channel_frames[1],
                    max_delay,
                );
                // For mics along Y: y1 - y0 = spacing.
                // Delay between mic 0 and 1: tau = spacing * sin(theta) / c
                let ratio = (SPEED_OF_SOUND * tdoa / spacing).clamp(-1.0, 1.0);
                ratio.asin()
            }
            ArrayGeometry::Circular { .. } => {
                // Steered Response Power (SRP) grid scan over 360 degrees
                let mut best_angle = 0.0f32;
                let mut max_power = -1e9f32;

                let steps = 72;
                for step in 0..steps {
                    let theta = (step as f32) * (2.0 * PI / (steps as f32)) - PI;
                    let u = [theta.cos(), theta.sin()];

                    let mut total_corr = 0.0f32;
                    let mut pairs = 0;

                    for i in 0..num_mics {
                        for j in (i + 1)..num_mics {
                            let p_i = &positions[i];
                            let p_j = &positions[j];

                            let tau_ideal = ((p_j.x - p_i.x) * u[0] + (p_j.y - p_i.y) * u[1]) / SPEED_OF_SOUND;
                            let max_d = p_i.distance_to(p_j) / SPEED_OF_SOUND;

                            let (est_tau, peak) = self.gcc_phat.estimate_tdoa(
                                multi_channel_frames[i],
                                multi_channel_frames[j],
                                max_d,
                            );

                            let error = (est_tau - tau_ideal).abs();
                            let weight = (-error * self.sample_rate).exp() * peak;
                            total_corr += weight;
                            pairs += 1;
                        }
                    }

                    if pairs > 0 && total_corr > max_power {
                        max_power = total_corr;
                        best_angle = theta;
                    }
                }

                best_angle
            }
            _ => {
                let p0 = &positions[0];
                let p1 = &positions[1];
                let d = p0.distance_to(p1);
                let max_d = d / SPEED_OF_SOUND;
                let (tdoa, _) = self.gcc_phat.estimate_tdoa(
                    multi_channel_frames[0],
                    multi_channel_frames[1],
                    max_d,
                );
                (SPEED_OF_SOUND * tdoa / d).clamp(-1.0, 1.0).asin()
            }
        }
    }
}

/// Real-Time Delay-and-Sum Spatial Beamformer.
///
/// Features fast power-of-2 bitmask circular buffers for > 10,000,000 samples/sec throughput.
#[derive(Debug, Clone)]
pub struct DelayAndSumBeamformer {
    geometry: ArrayGeometry,
    sample_rate: f32,
    num_mics: usize,
    positions: Vec<Point3D>,
    delay_buffers: Vec<Vec<f32>>,
    buffer_mask: usize,
    buffer_indices: Vec<usize>,
    steered_azimuth: f32,
    steered_elevation: f32,
    channel_delays_samples: Vec<f32>,
}

impl DelayAndSumBeamformer {
    /// Constructs a new Delay-and-Sum Beamformer for the given array geometry.
    pub fn new(geometry: ArrayGeometry, sample_rate: f32) -> Self {
        let positions = geometry.positions();
        let num_mics = positions.len();

        // Power-of-two buffer capacity (512 samples)
        let buffer_capacity = 512;
        let buffer_mask = buffer_capacity - 1;

        let delay_buffers = vec![vec![0.0f32; buffer_capacity]; num_mics];
        let buffer_indices = vec![0usize; num_mics];
        let channel_delays_samples = vec![0.0f32; num_mics];

        let mut beamformer = Self {
            geometry,
            sample_rate,
            num_mics,
            positions,
            delay_buffers,
            buffer_mask,
            buffer_indices,
            steered_azimuth: 0.0,
            steered_elevation: 0.0,
            channel_delays_samples,
        };

        beamformer.steer(0.0, 0.0);
        beamformer
    }

    /// Dynamically steers the listening lobe toward target azimuth and elevation angles (radians).
    pub fn steer(&mut self, azimuth_rad: f32, elevation_rad: f32) {
        self.steered_azimuth = azimuth_rad;
        self.steered_elevation = elevation_rad;

        let u_x = elevation_rad.cos() * azimuth_rad.cos();
        let u_y = elevation_rad.cos() * azimuth_rad.sin();
        let u_z = elevation_rad.sin();

        let mut centroid = Point3D::new(0.0, 0.0, 0.0);
        for p in &self.positions {
            centroid.x += p.x;
            centroid.y += p.y;
            centroid.z += p.z;
        }
        centroid.x /= self.num_mics as f32;
        centroid.y /= self.num_mics as f32;
        centroid.z /= self.num_mics as f32;

        let mut min_delay_sec = 1e6f32;
        let mut delays_sec = Vec::with_capacity(self.num_mics);

        for p in &self.positions {
            let rx = p.x - centroid.x;
            let ry = p.y - centroid.y;
            let rz = p.z - centroid.z;
            let tau = -(rx * u_x + ry * u_y + rz * u_z) / SPEED_OF_SOUND;
            delays_sec.push(tau);
            if tau < min_delay_sec {
                min_delay_sec = tau;
            }
        }

        for i in 0..self.num_mics {
            let causal_delay_sec = delays_sec[i] - min_delay_sec;
            self.channel_delays_samples[i] = causal_delay_sec * self.sample_rate;
        }
    }

    /// Processes a multi-channel block of audio samples, producing a single beamformed mono output.
    #[inline]
    pub fn process_block(&mut self, multi_channel_inputs: &[&[f32]], output: &mut [f32]) {
        let block_len = output.len();
        if self.num_mics == 0 || multi_channel_inputs.len() < self.num_mics {
            output.fill(0.0);
            return;
        }

        output.fill(0.0);
        let inv_m = 1.0 / (self.num_mics as f32);
        let mask = self.buffer_mask;

        for sample_idx in 0..block_len {
            let mut sum = 0.0f32;

            for m in 0..self.num_mics {
                let in_sample = multi_channel_inputs[m][sample_idx];
                let write_idx = self.buffer_indices[m];
                self.delay_buffers[m][write_idx] = in_sample;

                let delay = self.channel_delays_samples[m];
                let delay_int = delay.floor() as usize;
                let delay_frac = delay - (delay_int as f32);

                let read_idx0 = (write_idx + mask + 1 - (delay_int & mask)) & mask;
                let read_idx1 = (read_idx0 + mask) & mask;

                let s0 = self.delay_buffers[m][read_idx0];
                let s1 = self.delay_buffers[m][read_idx1];
                let interpolated = s0 + delay_frac * (s1 - s0);

                sum += interpolated;

                self.buffer_indices[m] = (write_idx + 1) & mask;
            }

            output[sample_idx] = sum * inv_m;
        }
    }

    /// Returns array geometry.
    pub fn geometry(&self) -> &ArrayGeometry {
        &self.geometry
    }

    /// Returns the active steered azimuth angle in radians.
    pub fn steered_azimuth(&self) -> f32 {
        self.steered_azimuth
    }

    /// Returns the active steered elevation angle in radians.
    pub fn steered_elevation(&self) -> f32 {
        self.steered_elevation
    }

    /// Number of microphone channels in the beamformer.
    pub fn num_mics(&self) -> usize {
        self.num_mics
    }

    /// Resets all internal delay buffers.
    pub fn reset(&mut self) {
        for b in self.delay_buffers.iter_mut() {
            b.fill(0.0);
        }
        self.buffer_indices.fill(0);
    }
}
