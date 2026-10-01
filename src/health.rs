//! Drone acoustic health monitoring, propeller blade damage detection,
//! motor bearing friction diagnostics, and autonomous MAVLink telemetry emission.

#![deny(unsafe_code)]

use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};

/// Qualitative anomaly severity rating for airframe components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Nominal acoustic operation; no acoustic degradation detected.
    Normal = 0,
    /// Subtle acoustic divergence; scheduled maintenance or visual check advised.
    Advisory = 1,
    /// Pronounced acoustic defect (e.g. cracked blade, initial bearing spalling).
    Warning = 2,
    /// Critical acoustic defect (severe blade damage, imminent failure). Emergency landing advised.
    Critical = 3,
}

impl AnomalySeverity {
    /// Returns true if component requires operator attention.
    pub fn is_anomalous(&self) -> bool {
        *self != AnomalySeverity::Normal
    }

    /// Numerical telemetry code (0.0 = Normal, 1.0 = Advisory, 2.0 = Warning, 3.0 = Critical).
    pub fn as_code(&self) -> f32 {
        *self as u8 as f32
    }
}

/// Configuration parameters for rotor damage and bearing acoustic analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorHealthConfig {
    /// Number of blades per propeller (typically 2 for standard multirotors).
    pub num_blades: usize,
    /// Frequency bandwidth (Hz) searched around nominal target frequency bins.
    pub search_bandwidth_hz: f32,
    /// Subharmonic-to-BPF energy ratio threshold triggering Advisory status.
    pub imbalance_advisory_ratio: f32,
    /// Subharmonic-to-BPF energy ratio threshold triggering Warning status.
    pub imbalance_warning_ratio: f32,
    /// Subharmonic-to-BPF energy ratio threshold triggering Critical status.
    pub imbalance_critical_ratio: f32,
    /// High-frequency friction ratio threshold triggering Warning status.
    pub bearing_friction_warn_ratio: f32,
    /// High-frequency friction ratio threshold triggering Critical status.
    pub bearing_friction_crit_ratio: f32,
    /// Spectral kurtosis threshold triggering Warning status.
    pub kurtosis_warn_threshold: f32,
    /// Spectral kurtosis threshold triggering Critical status.
    pub kurtosis_crit_threshold: f32,
}

impl Default for MotorHealthConfig {
    fn default() -> Self {
        Self {
            num_blades: 2,
            search_bandwidth_hz: 6.0,
            imbalance_advisory_ratio: 0.20,
            imbalance_warning_ratio: 0.45,
            imbalance_critical_ratio: 0.75,
            bearing_friction_warn_ratio: 0.35,
            bearing_friction_crit_ratio: 0.65,
            kurtosis_warn_threshold: 8.0,
            kurtosis_crit_threshold: 15.0,
        }
    }
}

/// Individual motor and propeller acoustic diagnostic assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorHealthReport {
    /// Zero-indexed motor identifier (0 = Motor 1, 1 = Motor 2, etc.).
    pub motor_id: usize,
    /// Telemetry RPM of this motor.
    pub rpm: f32,
    /// Primary Blade Pass Frequency (BPF = blades * RPM / 60) in Hz.
    pub bpf_hz: f32,
    /// Shaft rotational frequency (RPM / 60) in Hz.
    pub rot_hz: f32,
    /// Integrated acoustic energy around the Blade Pass Frequency.
    pub bpf_power: f32,
    /// Integrated acoustic energy around the shaft rotational frequency (asymmetry subharmonic).
    pub subharmonic_power: f32,
    /// Ratio of rotational subharmonic power to BPF power.
    pub imbalance_ratio: f32,
    /// Ratio of ultrasonic/high-frequency friction energy to mid-frequency energy.
    pub bearing_friction_ratio: f32,
    /// Spectral kurtosis across high-frequency friction band.
    pub spectral_kurtosis: f32,
    /// Assessed severity rating for this motor.
    pub severity: AnomalySeverity,
}

/// Aggregate airframe acoustic diagnostic state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirframeHealthSnapshot {
    /// Unified airframe health rating (1.0 = pristine, 0.0 = catastrophic failure).
    pub overall_health_score: f32,
    /// Worst anomaly severity across all monitored motors.
    pub worst_severity: AnomalySeverity,
    /// Detailed diagnostic report per motor.
    pub motor_reports: Vec<MotorHealthReport>,
    /// Stream timestamp in seconds when snapshot was evaluated.
    pub timestamp_sec: f64,
}

/// MAVLink standard NAMED_VALUE_FLOAT packet structure (Message ID 251).
///
/// Dispatched directly to ground stations or autopilot flight executives
/// over serial/UDP MAVLink streams.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MavlinkNamedValueFloat {
    /// Timestamp in milliseconds since flight controller boot.
    pub time_boot_ms: u32,
    /// Fixed 10-character null-terminated ASCII identifier.
    pub name: [u8; 10],
    /// Telemetry scalar value.
    pub value: f32,
}

impl MavlinkNamedValueFloat {
    /// Creates a new named value float message from string identifier.
    pub fn new(time_boot_ms: u32, name_str: &str, value: f32) -> Self {
        let mut name = [0u8; 10];
        let bytes = name_str.as_bytes();
        let len = bytes.len().min(10);
        name[..len].copy_from_slice(&bytes[..len]);
        Self {
            time_boot_ms,
            name,
            value,
        }
    }

    /// Returns string slice of the parameter name up to null terminator or 10 bytes.
    pub fn name_as_str(&self) -> &str {
        let len = self.name.iter().position(|&c| c == 0).unwrap_or(10);
        std::str::from_utf8(&self.name[..len]).unwrap_or("")
    }
}

/// Real-time acoustic health monitoring and anomaly detection engine.
pub struct AcousticHealthMonitor {
    sample_rate: f32,
    fft_size: usize,
    fft: FftProcessor,
    window: Window,
    config: MotorHealthConfig,
    motor_rpms: Vec<f32>,
    real_scratch: Vec<f32>,
    imag_scratch: Vec<f32>,
    windowed_scratch: Vec<f32>,
    power_scratch: Vec<f32>,
}

impl AcousticHealthMonitor {
    /// Construct a new acoustic health monitor.
    pub fn new(sample_rate: f32, fft_size: usize, config: MotorHealthConfig) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(
            fft_size > 0 && (fft_size & (fft_size - 1)) == 0,
            "FFT size must be power of two"
        );

        let fft = FftProcessor::new(fft_size);
        let window = Window::new(WindowType::Blackman, fft_size);
        let num_bins = fft_size / 2 + 1;

        Self {
            sample_rate,
            fft_size,
            fft,
            window,
            config,
            motor_rpms: Vec::new(),
            real_scratch: vec![0.0; fft_size],
            imag_scratch: vec![0.0; fft_size],
            windowed_scratch: vec![0.0; fft_size],
            power_scratch: vec![0.0; num_bins],
        }
    }

    /// Update RPM telemetry for all motors.
    pub fn update_motor_rpms(&mut self, rpms: &[f32]) {
        self.motor_rpms.clear();
        self.motor_rpms.extend_from_slice(rpms);
    }

    /// Update RPM telemetry for a single motor index.
    pub fn update_motor_rpm(&mut self, motor_id: usize, rpm: f32) {
        if motor_id >= self.motor_rpms.len() {
            self.motor_rpms.resize(motor_id + 1, 0.0);
        }
        self.motor_rpms[motor_id] = rpm;
    }

    /// Computes band power between start_hz and end_hz from power spectrum bins.
    pub fn compute_band_power(&self, power_spectrum: &[f32], start_hz: f32, end_hz: f32) -> f32 {
        let bin_hz = self.sample_rate / (self.fft_size as f32);
        let min_bin = ((start_hz.max(0.0) / bin_hz).round() as usize).min(power_spectrum.len() - 1);
        let max_bin = ((end_hz.max(0.0) / bin_hz).round() as usize).min(power_spectrum.len() - 1);

        if min_bin > max_bin {
            return power_spectrum[min_bin];
        }

        let mut sum = 0.0f32;
        for bin in min_bin..=max_bin {
            sum += power_spectrum[bin];
        }
        sum
    }

    /// Computes spectral kurtosis across the specified frequency range.
    ///
    /// Spectral kurtosis measures the presence of impulsive, non-Gaussian spikes in the spectrum
    /// typical of bearing ball impacts and cracked raceways.
    pub fn compute_spectral_kurtosis(
        &self,
        power_spectrum: &[f32],
        start_hz: f32,
        end_hz: f32,
    ) -> f32 {
        let bin_hz = self.sample_rate / (self.fft_size as f32);
        let min_bin = ((start_hz.max(0.0) / bin_hz).floor() as usize).min(power_spectrum.len() - 1);
        let max_bin = ((end_hz.max(0.0) / bin_hz).ceil() as usize).min(power_spectrum.len() - 1);

        let count = if max_bin >= min_bin {
            max_bin - min_bin + 1
        } else {
            0
        };

        if count < 4 {
            return 3.0; // Nominal Gaussian kurtosis
        }

        // Calculate mean
        let mut sum = 0.0f32;
        for bin in min_bin..=max_bin {
            sum += power_spectrum[bin];
        }
        let mean = sum / (count as f32);

        // Calculate second and fourth central moments
        let mut m2 = 0.0f32;
        let mut m4 = 0.0f32;
        for bin in min_bin..=max_bin {
            let diff = power_spectrum[bin] - mean;
            let diff2 = diff * diff;
            m2 += diff2;
            m4 += diff2 * diff2;
        }
        let var = m2 / (count as f32);

        if var < 1e-12 {
            return 3.0;
        }

        (m4 / (count as f32)) / (var * var)
    }

    /// Analyzes an incoming time-domain audio slice (length must match FFT size)
    /// and evaluates current motor health against active telemetry RPM.
    pub fn analyze_frame(
        &mut self,
        audio_frame: &[f32],
        timestamp_sec: f64,
    ) -> AirframeHealthSnapshot {
        assert_eq!(
            audio_frame.len(),
            self.fft_size,
            "Audio frame length must equal FFT size"
        );

        // Apply Blackman window to eliminate side-lobes
        self.windowed_scratch.copy_from_slice(audio_frame);
        self.window.apply(&mut self.windowed_scratch);

        // Compute FFT in-place
        self.real_scratch.copy_from_slice(&self.windowed_scratch);
        self.imag_scratch.fill(0.0);
        self.fft.fft_in_place(&mut self.real_scratch, &mut self.imag_scratch);

        // Compute one-sided power spectrum
        let norm = 1.0 / (self.fft_size as f32);
        for k in 0..self.power_scratch.len() {
            let r = self.real_scratch[k];
            let i = self.imag_scratch[k];
            self.power_scratch[k] = (r * r + i * i) * norm;
        }

        self.evaluate_power_spectrum(&self.power_scratch.clone(), timestamp_sec)
    }

    /// Analyzes a precomputed power spectrum slice directly.
    pub fn analyze_spectrum(
        &self,
        power_spectrum: &[f32],
        timestamp_sec: f64,
    ) -> AirframeHealthSnapshot {
        self.evaluate_power_spectrum(power_spectrum, timestamp_sec)
    }

    /// Internal evaluation logic mapping spectral features to motor health diagnostics.
    fn evaluate_power_spectrum(
        &self,
        power_spectrum: &[f32],
        timestamp_sec: f64,
    ) -> AirframeHealthSnapshot {
        let mut motor_reports = Vec::with_capacity(self.motor_rpms.len());
        let mut worst_severity = AnomalySeverity::Normal;
        let mut max_penalty = 0.0f32;

        let nyquist = self.sample_rate / 2.0;
        // High-frequency friction band: 3500 Hz to Nyquist
        let friction_start_hz = 3500.0f32.min(nyquist * 0.5);
        let friction_end_hz = 7500.0f32.min(nyquist);

        let mid_start_hz = 200.0f32;
        let mid_end_hz = friction_start_hz;

        let high_power = self.compute_band_power(power_spectrum, friction_start_hz, friction_end_hz);
        let mid_power = self.compute_band_power(power_spectrum, mid_start_hz, mid_end_hz);
        let global_bearing_friction_ratio = high_power / (mid_power + 1e-9);
        let global_kurtosis =
            self.compute_spectral_kurtosis(power_spectrum, friction_start_hz, friction_end_hz);

        for (motor_id, &rpm) in self.motor_rpms.iter().enumerate() {
            if rpm < 100.0 {
                // Motor is stationary or telemetry absent
                motor_reports.push(MotorHealthReport {
                    motor_id,
                    rpm,
                    bpf_hz: 0.0,
                    rot_hz: 0.0,
                    bpf_power: 0.0,
                    subharmonic_power: 0.0,
                    imbalance_ratio: 0.0,
                    bearing_friction_ratio: 0.0,
                    spectral_kurtosis: 3.0,
                    severity: AnomalySeverity::Normal,
                });
                continue;
            }

            let rot_hz = rpm / 60.0;
            let bpf_hz = (self.config.num_blades as f32) * rot_hz;

            let bw = self.config.search_bandwidth_hz;
            let bpf_power = self.compute_band_power(power_spectrum, (bpf_hz - bw).max(0.0), bpf_hz + bw);
            let rot_power = self.compute_band_power(power_spectrum, (rot_hz - bw).max(0.0), rot_hz + bw);

            let imbalance_ratio = rot_power / (bpf_power + 1e-9);

            // Determine motor anomaly severity
            let mut severity = AnomalySeverity::Normal;

            // Check blade imbalance
            if imbalance_ratio >= self.config.imbalance_critical_ratio {
                severity = severity.max(AnomalySeverity::Critical);
            } else if imbalance_ratio >= self.config.imbalance_warning_ratio {
                severity = severity.max(AnomalySeverity::Warning);
            } else if imbalance_ratio >= self.config.imbalance_advisory_ratio {
                severity = severity.max(AnomalySeverity::Advisory);
            }

            // Check bearing friction and kurtosis (ISO 10816 / ISO 13373-1 condition monitoring:
            // kurtosis indicates impact/friction only when high-frequency energy is elevated)
            let bearing_crit = global_bearing_friction_ratio >= self.config.bearing_friction_crit_ratio
                || (global_bearing_friction_ratio >= self.config.bearing_friction_warn_ratio
                    && global_kurtosis >= self.config.kurtosis_crit_threshold);

            let bearing_warn = global_bearing_friction_ratio >= self.config.bearing_friction_warn_ratio
                || (global_bearing_friction_ratio >= (self.config.bearing_friction_warn_ratio * 0.5)
                    && global_kurtosis >= self.config.kurtosis_warn_threshold);

            if bearing_crit {
                severity = severity.max(AnomalySeverity::Critical);
            } else if bearing_warn {
                severity = severity.max(AnomalySeverity::Warning);
            }

            worst_severity = worst_severity.max(severity);

            // Compute health penalty for this motor
            let imb_penalty = (imbalance_ratio / self.config.imbalance_critical_ratio).min(1.0);
            let brg_penalty = (global_bearing_friction_ratio
                / self.config.bearing_friction_crit_ratio)
                .min(1.0);
            let motor_penalty = 0.65 * imb_penalty + 0.35 * brg_penalty;
            if motor_penalty > max_penalty {
                max_penalty = motor_penalty;
            }

            motor_reports.push(MotorHealthReport {
                motor_id,
                rpm,
                bpf_hz,
                rot_hz,
                bpf_power,
                subharmonic_power: rot_power,
                imbalance_ratio,
                bearing_friction_ratio: global_bearing_friction_ratio,
                spectral_kurtosis: global_kurtosis,
                severity,
            });
        }

        let overall_health_score = (1.0 - max_penalty).clamp(0.0, 1.0);

        AirframeHealthSnapshot {
            overall_health_score,
            worst_severity,
            motor_reports,
            timestamp_sec,
        }
    }

    /// Generates standard MAVLink `NAMED_VALUE_FLOAT` telemetry packets for flight controllers
    /// (e.g. Kestrel, ArduPilot, PX4).
    pub fn generate_mavlink_telemetry(
        &self,
        snapshot: &AirframeHealthSnapshot,
        time_boot_ms: u32,
    ) -> Vec<MavlinkNamedValueFloat> {
        let mut packets = Vec::with_capacity(snapshot.motor_reports.len() * 2 + 2);

        // 1. Overall health score
        packets.push(MavlinkNamedValueFloat::new(
            time_boot_ms,
            "SONON_HLTH",
            snapshot.overall_health_score,
        ));

        // 2. Worst severity code
        packets.push(MavlinkNamedValueFloat::new(
            time_boot_ms,
            "SONON_STAT",
            snapshot.worst_severity.as_code(),
        ));

        // 3. Per-motor imbalance and bearing metrics
        for report in &snapshot.motor_reports {
            let id = report.motor_id + 1; // 1-indexed for pilot convention
            let imb_name = format!("SON_IMB{id}");
            let brg_name = format!("SON_BRG{id}");

            packets.push(MavlinkNamedValueFloat::new(
                time_boot_ms,
                &imb_name,
                report.imbalance_ratio,
            ));

            packets.push(MavlinkNamedValueFloat::new(
                time_boot_ms,
                &brg_name,
                report.bearing_friction_ratio,
            ));
        }

        packets
    }

    /// Returns sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}
