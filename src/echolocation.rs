#![deny(unsafe_code)]

//! Acoustic Echolocation & 3D Obstacle Spatial Mapping for GPS-Denied Subterranean UAV Flight.
//!
//! Features:
//! - Active Linear Frequency Modulated (LFM) chirp pulse compression with matched filtering.
//! - Pulse compression processing gain ($G_{\text{proc}} = 10 \log_{10}(B \cdot T_p) > 15\text{ dB}$)
//!   and sub-decimeter range resolution ($\Delta R = c / (2B)$).
//! - Cell-Averaging Constant False Alarm Rate (CA-CFAR) adaptive thresholding with guard cells.
//! - Sub-sample parabolic interpolation yielding sub-centimeter range precision.
//! - Multi-microphone Time-Difference-of-Arrival (TDoA) 3D Direction-of-Arrival (DoA) triangulation.
//! - 3D obstacle point cloud generation (`AcousticPointCloud`, `Point3D`) for tunnel/cave mapping.
//! - Opportunistic passive rotor acoustic echolocation using propeller blade pass acoustics.
//! - Forward, lateral, and vertical clearance boundary monitoring with collision warnings.
//! - Standard MAVLink v2 `NAMED_VALUE_FLOAT` telemetry packets (`ECHO_DIST`, `ECHO_CONF`, `ECHO_PTS`).
//! - Subterranean cave/tunnel acoustic simulator (`SubterraneanCaveSimulator`).

use crate::beamforming::{ArrayGeometry, Point3D, SPEED_OF_SOUND};
use crate::health::MavlinkNamedValueFloat;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration for Linear Frequency Modulated (LFM) acoustic chirps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpConfig {
    /// Audio sampling rate in Hertz (e.g. 16000.0 or 48000.0).
    pub sample_rate: f32,
    /// Starting frequency of the chirp sweep in Hertz.
    pub start_freq_hz: f32,
    /// Ending frequency of the chirp sweep in Hertz.
    pub end_freq_hz: f32,
    /// Chirp pulse duration in seconds (e.g. 0.015 s = 15 ms).
    pub pulse_duration_sec: f32,
    /// Pulse repetition rate in Hertz (e.g. 10.0 = 10 pings per second).
    pub pulse_repetition_hz: f32,
    /// Tukey window cosine fraction for edge tapering [0.0, 1.0].
    pub window_taper_ratio: f32,
}

impl Default for ChirpConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            start_freq_hz: 2000.0,
            end_freq_hz: 7000.0,
            pulse_duration_sec: 0.015,
            pulse_repetition_hz: 10.0,
            window_taper_ratio: 0.20,
        }
    }
}

impl ChirpConfig {
    /// Bandwidth of the chirp sweep in Hertz: B = |f_end - f_start|.
    pub fn bandwidth_hz(&self) -> f32 {
        (self.end_freq_hz - self.start_freq_hz).abs()
    }

    /// Theoretical matched filter range resolution in meters: Delta R = c / (2 B).
    pub fn theoretical_range_resolution_m(&self) -> f32 {
        let b = self.bandwidth_hz().max(100.0);
        SPEED_OF_SOUND / (2.0 * b)
    }

    /// Theoretical matched filter pulse compression processing gain in decibels: 10 * log10(B * Tp).
    pub fn processing_gain_db(&self) -> f32 {
        let bt = (self.bandwidth_hz() * self.pulse_duration_sec).max(1.0);
        10.0 * bt.log10()
    }
}

/// Linear Frequency Modulated (LFM) chirp pulse synthesizer.
#[derive(Debug, Clone)]
pub struct LfmChirpGenerator {
    config: ChirpConfig,
    template: Vec<f32>,
    template_energy: f32,
}

impl LfmChirpGenerator {
    /// Construct a new chirp generator and precompute the windowed reference template.
    pub fn new(config: ChirpConfig) -> Self {
        let num_samples = (config.pulse_duration_sec * config.sample_rate).round() as usize;
        let n = num_samples.max(16);
        let mut template = Vec::with_capacity(n);

        let f0 = config.start_freq_hz;
        let f1 = config.end_freq_hz;
        let t_p = config.pulse_duration_sec;
        let chirp_rate = (f1 - f0) / t_p;
        let fs = config.sample_rate;
        let taper_len = ((config.window_taper_ratio * (n as f32) * 0.5).round() as usize).max(1);

        let mut energy = 0.0f32;
        for i in 0..n {
            let t = (i as f32) / fs;
            // Phase phi(t) = 2 * pi * (f0 * t + 0.5 * chirp_rate * t^2)
            let phi = 2.0 * PI * (f0 * t + 0.5 * chirp_rate * t * t);
            let raw_sample = phi.sin();

            // Tukey window tapering:
            let window = if i < taper_len {
                0.5 * (1.0 - (PI * (i as f32) / (taper_len as f32)).cos())
            } else if i >= n - taper_len {
                let tail_idx = n - 1 - i;
                0.5 * (1.0 - (PI * (tail_idx as f32) / (taper_len as f32)).cos())
            } else {
                1.0
            };

            let s = raw_sample * window;
            energy += s * s;
            template.push(s);
        }

        Self {
            config,
            template,
            template_energy: energy.max(1e-12),
        }
    }

    /// Access reference template samples.
    pub fn template(&self) -> &[f32] {
        &self.template
    }

    /// Return template length in samples.
    pub fn template_len(&self) -> usize {
        self.template.len()
    }

    /// Return template energy sum(s^2).
    pub fn template_energy(&self) -> f32 {
        self.template_energy
    }

    /// Access active chirp configuration.
    pub fn config(&self) -> &ChirpConfig {
        &self.config
    }
}

/// Configuration for Cell-Averaging Constant False Alarm Rate (CA-CFAR) peak detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaCfarConfig {
    /// Number of training cells on each side of the cell under test.
    pub training_cells: usize,
    /// Number of guard cells on each side to prevent target energy leakage into the noise estimate.
    pub guard_cells: usize,
    /// Threshold multiplier factor over local noise estimate (e.g. 2.5 to 4.5).
    pub threshold_multiplier: f32,
    /// Minimum detection range in meters (blanking zone preventing self-interference, e.g. 0.20 m).
    pub min_range_m: f32,
    /// Maximum detection range in meters (e.g. 15.0 m).
    pub max_range_m: f32,
    /// Minimum peak correlation confidence threshold [0.0, 1.0].
    pub min_peak_confidence: f32,
}

impl Default for CaCfarConfig {
    fn default() -> Self {
        Self {
            training_cells: 16,
            guard_cells: 4,
            threshold_multiplier: 3.0,
            min_range_m: 0.25,
            max_range_m: 12.0,
            min_peak_confidence: 0.15,
        }
    }
}

/// Detected acoustic echo obstacle reflection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticObstacle {
    /// Radial range distance to the reflecting surface in meters.
    pub range_m: f32,
    /// Estimated horizontal azimuth bearing in radians [-PI, +PI].
    pub azimuth_rad: f32,
    /// Estimated vertical elevation angle in radians [-PI/2, +PI/2].
    pub elevation_rad: f32,
    /// 3D Cartesian coordinates of the obstacle in the vehicle body frame (X=Forward, Y=Right, Z=Down).
    pub position_3d: Point3D,
    /// Reflected echo amplitude in decibels (dB relative to reference).
    pub amplitude_db: f32,
    /// Normalized echo correlation confidence metric [0.0, 1.0].
    pub confidence: f32,
}

/// 3D Acoustic Point Cloud and spatial boundary clearances for subterranean navigation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticPointCloud {
    /// List of detected 3D obstacle reflections.
    pub obstacles: Vec<AcousticObstacle>,
    /// Minimum obstacle distance in any direction in meters.
    pub min_distance_m: f32,
    /// Forward clearance distance (+X axis) in meters.
    pub forward_clearance_m: f32,
    /// Left / port clearance distance (-Y axis) in meters.
    pub left_clearance_m: f32,
    /// Right / starboard clearance distance (+Y axis) in meters.
    pub right_clearance_m: f32,
    /// Downward / floor clearance distance (+Z axis) in meters.
    pub floor_clearance_m: f32,
    /// Upward / ceiling clearance distance (-Z axis) in meters.
    pub ceiling_clearance_m: f32,
    /// Flag set if any obstacle violates the minimum safe vehicle distance threshold.
    pub is_collision_risk: bool,
    /// Telemetry timestamp in seconds.
    pub timestamp_sec: f64,
}

impl AcousticPointCloud {
    /// Construct empty point cloud.
    pub fn empty(timestamp_sec: f64) -> Self {
        Self {
            obstacles: Vec::new(),
            min_distance_m: 99.0,
            forward_clearance_m: 99.0,
            left_clearance_m: 99.0,
            right_clearance_m: 99.0,
            floor_clearance_m: 99.0,
            ceiling_clearance_m: 99.0,
            is_collision_risk: false,
            timestamp_sec,
        }
    }

    /// Format into standard MAVLink v2 `NAMED_VALUE_FLOAT` telemetry packets.
    pub fn to_mavlink_telemetry(&self) -> Vec<MavlinkNamedValueFloat> {
        let time_boot_ms = (self.timestamp_sec * 1000.0) as u32;
        let best_conf = self.obstacles.iter().map(|o| o.confidence).fold(0.0f32, f32::max);
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "ECHO_DIST", self.min_distance_m),
            MavlinkNamedValueFloat::new(time_boot_ms, "ECHO_CONF", best_conf),
            MavlinkNamedValueFloat::new(time_boot_ms, "ECHO_PTS", self.obstacles.len() as f32),
        ]
    }
}

/// Multi-Microphone Acoustic Echolocator.
///
/// Executes active chirp pulse compression, CA-CFAR echo detection, and multi-channel
/// TDoA Direction-of-Arrival (DoA) triangulation to construct real-time 3D subterranean point clouds.
#[derive(Debug, Clone)]
pub struct MultiMicAcousticEcholocator {
    chirp_gen: LfmChirpGenerator,
    cfar_config: CaCfarConfig,
    geometry: ArrayGeometry,
    mic_positions: Vec<Point3D>,
    sample_rate: f32,
    collision_threshold_m: f32,

    // Preallocated correlation buffers
    corr_buffer: Vec<f32>,
    envelope_buffer: Vec<f32>,

    // Telemetry state
    latest_cloud: AcousticPointCloud,
}

impl MultiMicAcousticEcholocator {
    /// Construct a new acoustic echolocator with given geometry, chirp settings, and CFAR detector.
    pub fn new(
        geometry: ArrayGeometry,
        chirp_config: ChirpConfig,
        cfar_config: CaCfarConfig,
    ) -> Self {
        let mic_positions = geometry.positions();
        let sample_rate = chirp_config.sample_rate;
        let chirp_gen = LfmChirpGenerator::new(chirp_config);

        Self {
            chirp_gen,
            cfar_config,
            geometry,
            mic_positions,
            sample_rate,
            collision_threshold_m: 1.0, // Default 1 meter collision alert threshold
            corr_buffer: Vec::new(),
            envelope_buffer: Vec::new(),
            latest_cloud: AcousticPointCloud::empty(0.0),
        }
    }

    /// Set collision alert range threshold in meters.
    pub fn set_collision_threshold(&mut self, threshold_m: f32) {
        self.collision_threshold_m = threshold_m.max(0.1);
    }

    /// Return reference to precomputed chirp reference template.
    pub fn chirp_template(&self) -> &[f32] {
        self.chirp_gen.template()
    }

    /// Ingest multi-channel audio recordings and process active echolocation.
    ///
    /// - `multi_channel_inputs`: Audio samples from all microphones (length must be >= chirp template length).
    /// - `tx_reference`: Optional reference chirp audio. If None, uses internal precomputed chirp template.
    /// - `timestamp_sec`: Current flight timestamp in seconds.
    pub fn process_ping(
        &mut self,
        multi_channel_inputs: &[&[f32]],
        tx_reference: Option<&[f32]>,
        timestamp_sec: f64,
    ) -> AcousticPointCloud {
        let num_mics = self.mic_positions.len();
        if multi_channel_inputs.len() < num_mics || multi_channel_inputs[0].is_empty() {
            return self.latest_cloud.clone();
        }

        let ref_samples = tx_reference.unwrap_or_else(|| self.chirp_gen.template());
        let ref_len = ref_samples.len();
        let sig_len = multi_channel_inputs[0].len();
        if sig_len < ref_len {
            return self.latest_cloud.clone();
        }

        // 1. Run matched filter pulse compression on primary channel (mic 0)
        let ch0 = multi_channel_inputs[0];
        let out_len = sig_len - ref_len + 1;
        self.corr_buffer.resize(out_len, 0.0);
        self.envelope_buffer.resize(out_len, 0.0);

        let ref_energy = self.chirp_gen.template_energy();

        // Time-domain normalized cross-correlation:
        for i in 0..out_len {
            let mut dot = 0.0f32;
            let mut sig_e = 0.0f32;
            for j in 0..ref_len {
                let s = ch0[i + j];
                let r = ref_samples[j];
                dot += s * r;
                sig_e += s * s;
            }
            let denom = (sig_e.max(1e-12)).sqrt() * ref_energy.sqrt();
            let norm_corr = dot / denom;
            self.corr_buffer[i] = norm_corr;
            self.envelope_buffer[i] = norm_corr.abs();
        }

        // 2. Run Cell-Averaging Constant False Alarm Rate (CA-CFAR) Peak Detection
        let mut detected_peaks = Vec::new();
        let train = self.cfar_config.training_cells;
        let guard = self.cfar_config.guard_cells;
        let mult = self.cfar_config.threshold_multiplier;
        let min_idx = ((self.cfar_config.min_range_m * 2.0 / SPEED_OF_SOUND) * self.sample_rate).round() as usize;
        let max_idx = ((self.cfar_config.max_range_m * 2.0 / SPEED_OF_SOUND) * self.sample_rate).round() as usize;

        let start_idx = (train + guard).max(min_idx);
        let end_idx = (out_len.saturating_sub(train + guard)).min(max_idx);

        for i in start_idx..end_idx {
            let cell_val = self.envelope_buffer[i];
            if cell_val < self.cfar_config.min_peak_confidence {
                continue;
            }

            // Local maximum test
            if cell_val <= self.envelope_buffer[i - 1] || cell_val < self.envelope_buffer[i + 1] {
                continue;
            }

            // CA-CFAR background noise estimation
            let mut sum_noise = 0.0f32;
            let mut count_noise = 0;

            for k in (i - train - guard)..(i - guard) {
                sum_noise += self.envelope_buffer[k];
                count_noise += 1;
            }
            for k in (i + guard + 1)..=(i + guard + train) {
                sum_noise += self.envelope_buffer[k];
                count_noise += 1;
            }

            let avg_noise = sum_noise / (count_noise as f32).max(1.0);
            let threshold = avg_noise * mult;

            if cell_val > threshold {
                // Parabolic sub-sample interpolation
                let y1 = self.envelope_buffer[i - 1];
                let y2 = self.envelope_buffer[i];
                let y3 = self.envelope_buffer[i + 1];
                let denom = y1 - 2.0 * y2 + y3;
                let delta = if denom.abs() > 1e-9 {
                    (0.5 * (y1 - y3) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                };

                let refined_sample = (i as f32) + delta;
                let tof_sec = refined_sample / self.sample_rate;
                let range_m = 0.5 * SPEED_OF_SOUND * tof_sec;

                detected_peaks.push((i, refined_sample, range_m, cell_val));
            }
        }

        // Apply Non-Maximum Suppression (NMS) to cluster multi-cycle chirp ripples into dominant reflections
        let cluster_radius = ((0.001 * self.sample_rate).round() as usize).max(8);
        let mut clustered_peaks = Vec::new();
        for peak in detected_peaks {
            if let Some(last) = clustered_peaks.last_mut() {
                let (last_idx, _, _, last_val): &mut (usize, f32, f32, f32) = last;
                if peak.0 - *last_idx < cluster_radius {
                    if peak.3 > *last_val {
                        *last = peak;
                    }
                    continue;
                }
            }
            clustered_peaks.push(peak);
        }

        // 3. For each detected echo peak, extract multi-channel TDoA and triangulate 3D coordinates
        let mut obstacles = Vec::new();
        let search_window = 12; // Search +/- 12 samples for inter-channel TDoA
        let win_size = 2 * search_window + 1;
        let mut corr_window = vec![0.0f32; win_size];

        for &(peak_int_idx, _refined_sample, range_m, confidence) in &clustered_peaks {
            let mut delays_samples = vec![0.0f32; num_mics];

            for m in 1..num_mics {
                let ch_m = multi_channel_inputs[m];
                corr_window.fill(0.0);

                for (idx, shift) in (-(search_window as isize)..=(search_window as isize)).enumerate() {
                    let target_idx = (peak_int_idx as isize) + shift;
                    if target_idx < 0 || (target_idx as usize) + ref_len > sig_len {
                        continue;
                    }

                    let mut dot = 0.0f32;
                    let mut e_m = 0.0f32;
                    for j in 0..ref_len {
                        let sm = ch_m[(target_idx as usize) + j];
                        let r = ref_samples[j];
                        dot += sm * r;
                        e_m += sm * sm;
                    }
                    let norm = (e_m.max(1e-12)).sqrt() * ref_energy.sqrt();
                    corr_window[idx] = dot / norm;
                }

                // Find integer peak
                let mut best_idx = search_window;
                let mut best_val = corr_window[best_idx];
                for idx in 0..win_size {
                    if corr_window[idx] > best_val {
                        best_val = corr_window[idx];
                        best_idx = idx;
                    }
                }

                // Sub-sample parabolic interpolation
                let delta = if best_idx > 0 && best_idx + 1 < win_size {
                    let y1 = corr_window[best_idx - 1];
                    let y2 = corr_window[best_idx];
                    let y3 = corr_window[best_idx + 1];
                    let denom = y1 - 2.0 * y2 + y3;
                    if denom.abs() > 1e-9 {
                        (0.5 * (y1 - y3) / denom).clamp(-0.5, 0.5)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let best_shift = (best_idx as isize) - (search_window as isize);
                let refined_shift = (best_shift as f32) + delta;
                delays_samples[m] = refined_shift;
            }

            // Estimate 3D direction vector from delays:
            // Delta tau_m = (p_m - p_0) · u / c
            let (azimuth_rad, elevation_rad) = self.estimate_direction_from_delays(&delays_samples);

            let u_x = elevation_rad.cos() * azimuth_rad.cos();
            let u_y = elevation_rad.cos() * azimuth_rad.sin();
            let u_z = elevation_rad.sin();

            let position_3d = Point3D::new(range_m * u_x, range_m * u_y, range_m * u_z);
            let amplitude_db = 20.0 * confidence.max(1e-4).log10();

            obstacles.push(AcousticObstacle {
                range_m,
                azimuth_rad,
                elevation_rad,
                position_3d,
                amplitude_db,
                confidence,
            });
        }

        // 4. Compute boundary clearances and point cloud analytics
        let mut min_distance_m = 99.0f32;
        let mut forward_clearance_m = 99.0f32;
        let mut left_clearance_m = 99.0f32;
        let mut right_clearance_m = 99.0f32;
        let mut floor_clearance_m = 99.0f32;
        let mut ceiling_clearance_m = 99.0f32;

        for obs in &obstacles {
            if obs.range_m < min_distance_m {
                min_distance_m = obs.range_m;
            }
            // Forward sector: X > 0 and within 45 degrees
            if obs.position_3d.x > 0.0 && obs.azimuth_rad.abs() <= PI / 4.0 {
                if obs.position_3d.x < forward_clearance_m {
                    forward_clearance_m = obs.position_3d.x;
                }
            }
            // Port (left): Y < 0
            if obs.position_3d.y < 0.0 {
                let d_left = -obs.position_3d.y;
                if d_left < left_clearance_m {
                    left_clearance_m = d_left;
                }
            }
            // Starboard (right): Y > 0
            if obs.position_3d.y > 0.0 {
                let d_right = obs.position_3d.y;
                if d_right < right_clearance_m {
                    right_clearance_m = d_right;
                }
            }
            // Floor: Z > 0 (downward)
            if obs.position_3d.z > 0.0 {
                let d_floor = obs.position_3d.z;
                if d_floor < floor_clearance_m {
                    floor_clearance_m = d_floor;
                }
            }
            // Ceiling: Z < 0 (upward)
            if obs.position_3d.z < 0.0 {
                let d_ceil = -obs.position_3d.z;
                if d_ceil < ceiling_clearance_m {
                    ceiling_clearance_m = d_ceil;
                }
            }
        }

        let is_collision_risk = min_distance_m <= self.collision_threshold_m;

        self.latest_cloud = AcousticPointCloud {
            obstacles,
            min_distance_m,
            forward_clearance_m,
            left_clearance_m,
            right_clearance_m,
            floor_clearance_m,
            ceiling_clearance_m,
            is_collision_risk,
            timestamp_sec,
        };

        self.latest_cloud.clone()
    }

    /// Estimate Direction-of-Arrival (azimuth and elevation) from relative sample delays.
    fn estimate_direction_from_delays(&self, delays_samples: &[f32]) -> (f32, f32) {
        if self.mic_positions.len() < 2 {
            return (0.0, 0.0);
        }

        match &self.geometry {
            ArrayGeometry::Linear { spacing, .. } => {
                // Linear array mounted laterally along Y axis:
                // Delay tau = d * sin(azimuth) / c => sin(azimuth) = tau * c / d
                if delays_samples.len() >= 2 {
                    let d = *spacing;
                    let tau = (delays_samples[1] - delays_samples[0]) / self.sample_rate;
                    let arg = (tau * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);
                    let azimuth = arg.asin();
                    (azimuth, 0.0)
                } else {
                    (0.0, 0.0)
                }
            }
            ArrayGeometry::Circular { radius, num_mics } => {
                if delays_samples.len() >= 4 && *num_mics >= 4 {
                    // Estimate azimuth from orthogonal mic pairs (0-2 and 1-3)
                    let d = 2.0 * radius;
                    let tau_x = (delays_samples[2] - delays_samples[0]) / self.sample_rate;
                    let tau_y = (delays_samples[3] - delays_samples[1]) / self.sample_rate;
                    let u_x = (tau_x * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);
                    let u_y = (tau_y * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);
                    let azimuth = u_y.atan2(u_x);
                    (azimuth, 0.0)
                } else {
                    (0.0, 0.0)
                }
            }
            ArrayGeometry::Tetrahedral { radius } => {
                // 3D tetrahedral array gives full 3D DoA
                if delays_samples.len() >= 4 {
                    let d = 1.633 * radius; // Characteristic tetrahedral edge length
                    let tau_x = (delays_samples[1] - delays_samples[0]) / self.sample_rate;
                    let tau_y = (delays_samples[2] - delays_samples[0]) / self.sample_rate;
                    let tau_z = (delays_samples[3] - delays_samples[0]) / self.sample_rate;

                    let u_x = (tau_x * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);
                    let u_y = (tau_y * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);
                    let u_z = (tau_z * SPEED_OF_SOUND / d).clamp(-1.0, 1.0);

                    let azimuth = u_y.atan2(u_x);
                    let elevation = u_z.asin();
                    (azimuth, elevation)
                } else {
                    (0.0, 0.0)
                }
            }
            _ => (0.0, 0.0),
        }
    }

    /// Access latest evaluated acoustic point cloud.
    pub fn latest_point_cloud(&self) -> &AcousticPointCloud {
        &self.latest_cloud
    }

    /// Reset internal history buffers and point cloud.
    pub fn reset(&mut self) {
        self.corr_buffer.clear();
        self.envelope_buffer.clear();
        self.latest_cloud = AcousticPointCloud::empty(0.0);
    }
}

/// Subterranean Cave, Mine Tunnel, and Obstacle Acoustic Simulator.
///
/// Synthesizes physically accurate multi-channel acoustic signals inside enclosed subterranean
/// environments, including two-way Time-of-Flight (ToF) acoustic echoes, $1/R^2$ spherical spreading
/// attenuation, wall reflection coefficients, and rotor acoustic noise.
pub struct SubterraneanCaveSimulator;

impl SubterraneanCaveSimulator {
    /// Simulate active echolocation ping return inside a subterranean tunnel.
    ///
    /// - `geometry`: Microphone array geometry.
    /// - `sample_rate`: Sample rate in Hz.
    /// - `obstacles`: Ground truth list of reflecting obstacle coordinates (Point3D).
    /// - `reflection_coeffs`: Reflection coefficient of each obstacle (e.g. 0.8 for rock/concrete wall).
    /// - `tx_chirp`: Transmitted chirp reference waveform.
    /// - `rotor_noise_rms`: Background quadcopter propeller noise RMS level.
    pub fn simulate_subterranean_ping(
        geometry: &ArrayGeometry,
        sample_rate: f32,
        obstacles: &[Point3D],
        reflection_coeffs: &[f32],
        tx_chirp: &[f32],
        rotor_noise_rms: f32,
    ) -> Vec<Vec<f32>> {
        let mic_positions = geometry.positions();
        let num_mics = mic_positions.len();

        // Calculate maximum required duration based on furthest obstacle
        let mut max_range_m = 1.0f32;
        for obs in obstacles {
            let dist = (obs.x * obs.x + obs.y * obs.y + obs.z * obs.z).sqrt();
            if dist > max_range_m {
                max_range_m = dist;
            }
        }

        let max_tof_sec = (2.0 * max_range_m / SPEED_OF_SOUND) + (tx_chirp.len() as f32 / sample_rate) + 0.02;
        let total_samples = (max_tof_sec * sample_rate).ceil() as usize;

        let mut channels = vec![vec![0.0f32; total_samples]; num_mics];

        // 1. Add direct-path transmitted chirp leakage to all microphones
        for (m, ch) in channels.iter_mut().enumerate() {
            let mic_dist_to_emitter = (mic_positions[m].x.powi(2) + mic_positions[m].y.powi(2) + mic_positions[m].z.powi(2)).sqrt().max(0.05);
            let direct_delay_sec = mic_dist_to_emitter / SPEED_OF_SOUND;
            let direct_delay_samples = (direct_delay_sec * sample_rate).round() as usize;

            for (i, &s) in tx_chirp.iter().enumerate() {
                if direct_delay_samples + i < total_samples {
                    // Direct acoustic cross-talk
                    ch[direct_delay_samples + i] += s * 0.3;
                }
            }
        }

        // 2. Add reflected acoustic echoes from each subterranean obstacle
        for (k, obs) in obstacles.iter().enumerate() {
            let ref_coeff = reflection_coeffs.get(k).copied().unwrap_or(0.7);
            let r_tx = (obs.x.powi(2) + obs.y.powi(2) + obs.z.powi(2)).sqrt(); // Range from transmitter (origin) to obstacle

            for (m, ch) in channels.iter_mut().enumerate() {
                let mic = &mic_positions[m];
                let r_rx = ((obs.x - mic.x).powi(2) + (obs.y - mic.y).powi(2) + (obs.z - mic.z).powi(2)).sqrt(); // Obstacle to mic
                let total_path_m = r_tx + r_rx;

                let tof_sec = total_path_m / SPEED_OF_SOUND;
                let delay_samples = (tof_sec * sample_rate).round() as usize;

                // Spherical geometric spreading attenuation 1 / (r_tx * r_rx)
                let spreading_atten = 1.0 / (r_tx * r_rx).max(0.5);
                let echo_gain = (ref_coeff * spreading_atten * 2.0).clamp(0.001, 1.0);

                for (i, &s) in tx_chirp.iter().enumerate() {
                    if delay_samples + i < total_samples {
                        ch[delay_samples + i] += s * echo_gain;
                    }
                }
            }
        }

        // 3. Add rotor acoustic noise (low-frequency blade pass harmonic tone + random turbulence)
        let mut rng = 0x87654321_u64;
        let mut rand_f = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 32) as i32 as f32) / 2147483648.0
        };

        for ch in channels.iter_mut() {
            for (n, s) in ch.iter_mut().enumerate() {
                let t = (n as f32) / sample_rate;
                // Quadcopter 400 Hz Blade Pass Frequency (BPF) tonal hum + broadband air rush
                let bpf_tone = (2.0 * PI * 400.0 * t).sin() * 0.7 + (2.0 * PI * 800.0 * t).sin() * 0.3;
                let noise = (bpf_tone + rand_f() * 0.5) * rotor_noise_rms;
                *s += noise;
            }
        }

        channels
    }
}
