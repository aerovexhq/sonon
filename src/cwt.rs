//! Continuous Wavelet Transform (CWT) non-stationary rotor micro-damage profiler,
//! multi-resolution time-frequency vibration decomposition, and autonomous MAVLink telemetry emission.

#![deny(unsafe_code)]

use crate::health::{AnomalySeverity, MavlinkNamedValueFloat};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Wavelet mother kernel function family for Continuous Wavelet Transform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WaveletType {
    /// Complex Morlet wavelet with central dimensionless angular frequency parameter $\omega_0$.
    /// Provides joint analytic amplitude envelope and phase decomposition.
    ComplexMorlet { omega_0: f32 },
    /// Mexican Hat (Ricker) wavelet, second derivative of a Gaussian.
    /// Purely real with zero DC bias; optimal for impulsive mechanical impact shocks and sharp discontinuities.
    MexicanHat,
}

impl Default for WaveletType {
    fn default() -> Self {
        WaveletType::ComplexMorlet { omega_0: 6.0 }
    }
}

/// Continuous Wavelet Transform 2D scalogram representation.
#[derive(Debug, Clone)]
pub struct CwtScalogram {
    /// Analyzed wavelet dilation scales $a$.
    pub scales: Vec<f32>,
    /// Pseudo-frequencies corresponding to each scale in Hz.
    pub frequencies: Vec<f32>,
    /// Wavelet coefficient magnitude matrix (outer dimension = scales, inner dimension = time samples).
    pub magnitudes: Vec<Vec<f32>>,
}

impl CwtScalogram {
    /// Total integrated scalogram energy across all scales and time samples.
    pub fn total_energy(&self) -> f32 {
        let mut sum = 0.0f32;
        for scale_mags in &self.magnitudes {
            for &mag in scale_mags {
                sum += mag * mag;
            }
        }
        sum
    }

    /// Integrated acoustic energy at a specific scale index.
    pub fn scale_energy(&self, scale_idx: usize) -> f32 {
        if scale_idx >= self.magnitudes.len() {
            return 0.0;
        }
        self.magnitudes[scale_idx].iter().map(|&m| m * m).sum()
    }

    /// Evaluates statistical kurtosis along the time axis for a specific scale.
    ///
    /// Kurtosis measures non-Gaussian peakedness:
    /// - Nominal stationary Gaussian noise produces kurtosis $\approx 3.0$.
    /// - Micro-crack impacts, blade chips, or bearing spall clicks produce impulsive spikes ($> 4.5$).
    pub fn scale_kurtosis(&self, scale_idx: usize) -> f32 {
        if scale_idx >= self.magnitudes.len() {
            return 3.0;
        }

        let slice = &self.magnitudes[scale_idx];
        let n_total = slice.len();
        if n_total < 16 {
            return 3.0;
        }

        // Exclude boundary samples (cone of influence) to avoid edge truncation distortions
        let margin = (n_total / 8).clamp(4, 32);
        if margin * 2 >= n_total {
            return 3.0;
        }
        let interior = &slice[margin..n_total - margin];

        let n = interior.len() as f32;
        let mean = interior.iter().sum::<f32>() / n;

        let var = interior.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        if var < 1e-8 {
            return 3.0;
        }

        let m4 = interior
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d * d * d
            })
            .sum::<f32>()
            / n;

        (m4 / (var * var)).clamp(1.0, 50.0)
    }

    /// Returns (scale_idx, peak_kurtosis, freq_hz) for the scale exhibiting maximum impulsive peakedness.
    pub fn peak_kurtosis(&self) -> (usize, f32, f32) {
        let mut max_k = 3.0f32;
        let mut best_idx = 0;
        let mut best_freq = self.frequencies.first().copied().unwrap_or(0.0);
        let tot_energy = self.total_energy();

        for i in 0..self.scales.len() {
            let scale_eng = self.scale_energy(i);
            // Require scale to possess meaningful acoustic energy (> 0.02 of total energy)
            if tot_energy > 1e-8 && scale_eng < 0.02 * tot_energy {
                continue;
            }

            let k = self.scale_kurtosis(i);
            if k > max_k {
                max_k = k;
                best_idx = i;
                best_freq = self.frequencies.get(i).copied().unwrap_or(0.0);
            }
        }

        (best_idx, max_k, best_freq)
    }

    /// Returns scale index with largest total integrated energy.
    pub fn dominant_scale(&self) -> (usize, f32) {
        let mut max_eng = 0.0f32;
        let mut best_idx = 0;

        for (i, _) in self.scales.iter().enumerate() {
            let eng = self.scale_energy(i);
            if eng > max_eng {
                max_eng = eng;
                best_idx = i;
            }
        }

        (best_idx, max_eng)
    }
}

/// Continuous Wavelet Transform filterbank for multi-resolution acoustic analysis.
#[derive(Debug, Clone)]
pub struct ContinuousWaveletFilterbank {
    wavelet_type: WaveletType,
    scales: Vec<f32>,
    frequencies: Vec<f32>,
    real_kernels: Vec<Vec<f32>>,
    imag_kernels: Vec<Vec<f32>>,
    sample_rate: f32,
}

impl ContinuousWaveletFilterbank {
    /// Construct a logarithmically-spaced continuous wavelet filterbank across specified frequency bounds.
    pub fn new(
        wavelet_type: WaveletType,
        num_scales: usize,
        min_freq_hz: f32,
        max_freq_hz: f32,
        sample_rate: f32,
    ) -> Self {
        assert!(num_scales > 0, "Number of scales must be positive");
        assert!(min_freq_hz > 0.0 && max_freq_hz > min_freq_hz, "Invalid frequency bounds");
        assert!(max_freq_hz <= sample_rate / 2.0, "Max frequency exceeds Nyquist");

        // Central frequency f_c of the mother wavelet at a = 1.0
        let fc = match wavelet_type {
            WaveletType::ComplexMorlet { omega_0 } => omega_0 / (2.0 * PI),
            WaveletType::MexicanHat => (2.5f32).sqrt() / (2.0 * PI), // ~0.2516 Hz
        };

        // Pseudo-frequency formula: f(a) = (fc * sample_rate) / a
        // => scale a = (fc * sample_rate) / f
        let min_scale = (fc * sample_rate) / max_freq_hz;
        let max_scale = (fc * sample_rate) / min_freq_hz;

        let mut scales = Vec::with_capacity(num_scales);
        let mut frequencies = Vec::with_capacity(num_scales);
        let mut real_kernels = Vec::with_capacity(num_scales);
        let mut imag_kernels = Vec::with_capacity(num_scales);

        let log_min = min_scale.ln();
        let log_max = max_scale.ln();
        let step = if num_scales > 1 {
            (log_max - log_min) / ((num_scales - 1) as f32)
        } else {
            0.0
        };

        let pi_fourth_inv = 1.0 / PI.powf(0.25);
        let mexican_c = 2.0 / ((3.0f32).sqrt() * PI.powf(0.25));

        for s in 0..num_scales {
            // Traverse from max_scale (min_freq) down to min_scale (max_freq) so frequencies increase
            let a = (log_max - (s as f32) * step).exp();
            scales.push(a);

            let freq = (fc * sample_rate) / a;
            frequencies.push(freq);

            // Compute discrete convolution kernel for scale a
            // Truncate kernel support to [-4a, +4a]
            let half_len = (4.0 * a).ceil() as isize;
            let half_len = half_len.clamp(2, 256);
            let kernel_len = (2 * half_len + 1) as usize;

            let mut real_k = Vec::with_capacity(kernel_len);
            let mut imag_k = Vec::with_capacity(kernel_len);
            let norm = 1.0 / a.sqrt();

            match wavelet_type {
                WaveletType::ComplexMorlet { omega_0 } => {
                    for n in -half_len..=half_len {
                        let t = (n as f32) / a;
                        let gauss = (-0.5 * t * t).exp();
                        let r = norm * pi_fourth_inv * gauss * (omega_0 * t).cos();
                        let i = norm * pi_fourth_inv * gauss * (omega_0 * t).sin();
                        real_k.push(r);
                        imag_k.push(i);
                    }
                }
                WaveletType::MexicanHat => {
                    for n in -half_len..=half_len {
                        let t = (n as f32) / a;
                        let gauss = (-0.5 * t * t).exp();
                        let r = norm * mexican_c * (1.0 - t * t) * gauss;
                        real_k.push(r);
                    }
                    // Subtract discrete kernel mean to guarantee exact zero DC bias
                    let mean = real_k.iter().sum::<f32>() / (real_k.len() as f32);
                    for val in &mut real_k {
                        *val -= mean;
                    }
                }
            }

            real_kernels.push(real_k);
            imag_kernels.push(imag_k);
        }

        Self {
            wavelet_type,
            scales,
            frequencies,
            real_kernels,
            imag_kernels,
            sample_rate,
        }
    }

    /// Compute Continuous Wavelet Transform on an input audio slice.
    ///
    /// Computes discrete 1D convolution of `signal` with scaled wavelet kernels:
    /// $$W(a, b) = \sum_n x[n] \psi^*_a[n - b]$$
    pub fn transform(&self, signal: &[f32]) -> CwtScalogram {
        let n_samples = signal.len();
        let num_scales = self.scales.len();
        let mut magnitudes = Vec::with_capacity(num_scales);

        for (s, real_k) in self.real_kernels.iter().enumerate() {
            let k_len = real_k.len();
            let half_k = k_len / 2;
            let mut scale_mags = vec![0.0f32; n_samples];

            let has_imag = !self.imag_kernels[s].is_empty();
            let imag_k = if has_imag {
                &self.imag_kernels[s][..]
            } else {
                &[]
            };

            let interior_start = half_k;
            let interior_end = n_samples.saturating_sub(k_len - half_k);

            // Left boundary
            for i in 0..interior_start.min(n_samples) {
                let mut sum_r = 0.0f32;
                let mut sum_i = 0.0f32;
                for k in 0..k_len {
                    let sig_idx = (i + k) as isize - (half_k as isize);
                    if sig_idx >= 0 && (sig_idx as usize) < n_samples {
                        let x = signal[sig_idx as usize];
                        sum_r += x * real_k[k];
                        if has_imag {
                            sum_i += x * imag_k[k];
                        }
                    }
                }
                scale_mags[i] = if has_imag {
                    (sum_r * sum_r + sum_i * sum_i).sqrt()
                } else {
                    sum_r.abs()
                };
            }

            // Central interior (no bounds checks, vectorizable)
            if interior_start < interior_end {
                for i in interior_start..interior_end {
                    let sig_start = i - half_k;
                    let sig_slice = &signal[sig_start..sig_start + k_len];

                    let mut sum_r = 0.0f32;
                    let mut sum_i = 0.0f32;

                    if has_imag {
                        for k in 0..k_len {
                            let x = sig_slice[k];
                            sum_r += x * real_k[k];
                            sum_i += x * imag_k[k];
                        }
                        scale_mags[i] = (sum_r * sum_r + sum_i * sum_i).sqrt();
                    } else {
                        for k in 0..k_len {
                            sum_r += sig_slice[k] * real_k[k];
                        }
                        scale_mags[i] = sum_r.abs();
                    }
                }
            }

            // Right boundary
            for i in interior_end.max(interior_start)..n_samples {
                let mut sum_r = 0.0f32;
                let mut sum_i = 0.0f32;
                for k in 0..k_len {
                    let sig_idx = (i + k) as isize - (half_k as isize);
                    if sig_idx >= 0 && (sig_idx as usize) < n_samples {
                        let x = signal[sig_idx as usize];
                        sum_r += x * real_k[k];
                        if has_imag {
                            sum_i += x * imag_k[k];
                        }
                    }
                }
                scale_mags[i] = if has_imag {
                    (sum_r * sum_r + sum_i * sum_i).sqrt()
                } else {
                    sum_r.abs()
                };
            }

            magnitudes.push(scale_mags);
        }

        CwtScalogram {
            scales: self.scales.clone(),
            frequencies: self.frequencies.clone(),
            magnitudes,
        }
    }

    /// Return wavelet family type.
    pub fn wavelet_type(&self) -> WaveletType {
        self.wavelet_type
    }

    /// Return reference to center frequencies in Hz for all scales.
    pub fn frequencies(&self) -> &[f32] {
        &self.frequencies
    }

    /// Return reference to scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Return sample rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

/// Configuration parameters for CWT rotor micro-damage and structural vibration analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CwtProfilerConfig {
    /// Wavelet kernel type.
    pub wavelet_type: WaveletType,
    /// Number of logarithmically spaced scales.
    pub num_scales: usize,
    /// Lower pseudo-frequency limit in Hz (typically ~100 Hz).
    pub min_freq_hz: f32,
    /// Upper pseudo-frequency limit in Hz (typically 7500 Hz).
    pub max_freq_hz: f32,
    /// Statistical kurtosis threshold triggering Advisory warning.
    pub kurtosis_advisory_threshold: f32,
    /// Statistical kurtosis threshold triggering Critical failure alert.
    pub kurtosis_critical_threshold: f32,
    /// Rhythmic blade flutter modulation index threshold triggering Warning.
    pub flutter_warning_threshold: f32,
    /// High-frequency micro-crack energy ratio threshold triggering Warning.
    pub crack_energy_warning_ratio: f32,
}

impl Default for CwtProfilerConfig {
    fn default() -> Self {
        Self {
            wavelet_type: WaveletType::ComplexMorlet { omega_0: 6.0 },
            num_scales: 16,
            min_freq_hz: 100.0,
            max_freq_hz: 7500.0,
            kurtosis_advisory_threshold: 6.0,
            kurtosis_critical_threshold: 10.0,
            flutter_warning_threshold: 0.35,
            crack_energy_warning_ratio: 0.30,
        }
    }
}

/// Diagnostic evaluation report generated by the CWT rotor micro-damage profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorDamageReport {
    /// Normalized composite airframe fatigue index (0.0 = pristine, 1.0 = imminent mechanical failure).
    pub airframe_fatigue_index: f32,
    /// Peak statistical kurtosis observed across all wavelet scales.
    pub max_kurtosis: f32,
    /// Frequency in Hz at which peak kurtosis was observed.
    pub kurtosis_peak_freq_hz: f32,
    /// Dynamic blade flutter modulation index.
    pub flutter_index: f32,
    /// Ratio of high-frequency (>3000 Hz) micro-crack acoustic energy to total acoustic energy.
    pub high_freq_crack_energy_ratio: f32,
    /// Overall structural severity assessment.
    pub severity: AnomalySeverity,
    /// Tracked motor shaft RPM during evaluation.
    pub rpm: f32,
    /// Evaluation timestamp in seconds.
    pub timestamp_sec: f64,
}

impl RotorDamageReport {
    /// Generate standardized MAVLink `NAMED_VALUE_FLOAT` telemetry packets.
    pub fn generate_mavlink_telemetry(&self, time_boot_ms: u32) -> Vec<MavlinkNamedValueFloat> {
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "FATIGUE", self.airframe_fatigue_index),
            MavlinkNamedValueFloat::new(time_boot_ms, "FLUTTER", self.flutter_index),
            MavlinkNamedValueFloat::new(time_boot_ms, "CWT_KURT", self.max_kurtosis),
            MavlinkNamedValueFloat::new(time_boot_ms, "CRACK_ENG", self.high_freq_crack_energy_ratio),
        ]
    }
}

/// Continuous Wavelet Transform non-stationary rotor micro-damage profiler.
#[derive(Debug, Clone)]
pub struct RotorDamageProfiler {
    filterbank: ContinuousWaveletFilterbank,
    config: CwtProfilerConfig,
    active_rpm: f32,
    num_blades: usize,
}

impl RotorDamageProfiler {
    /// Construct a new CWT rotor micro-damage profiler.
    pub fn new(sample_rate: f32, config: CwtProfilerConfig, num_blades: usize) -> Self {
        let filterbank = ContinuousWaveletFilterbank::new(
            config.wavelet_type,
            config.num_scales,
            config.min_freq_hz,
            config.max_freq_hz,
            sample_rate,
        );

        Self {
            filterbank,
            config,
            active_rpm: 0.0,
            num_blades: num_blades.max(1),
        }
    }

    /// Update active rotor RPM from autopilot or ESC telemetry.
    pub fn update_rpm(&mut self, rpm: f32) {
        self.active_rpm = rpm.max(0.0);
    }

    /// Analyze an audio frame and evaluate non-stationary structural damage signatures.
    pub fn analyze_frame(&self, frame: &[f32], timestamp_sec: f64) -> RotorDamageReport {
        let scalogram = self.filterbank.transform(frame);
        let total_energy = scalogram.total_energy().max(1e-9);

        // 1. Peak Kurtosis across all wavelet scales
        let (_best_idx, max_kurtosis, kurtosis_freq) = scalogram.peak_kurtosis();

        // 2. High-Frequency Micro-Crack Energy Ratio (> 3000 Hz)
        let mut crack_energy = 0.0f32;
        for (i, &freq) in scalogram.frequencies.iter().enumerate() {
            if freq >= 3000.0 {
                crack_energy += scalogram.scale_energy(i);
            }
        }
        let high_freq_crack_ratio = (crack_energy / total_energy).clamp(0.0, 1.0);

        // 3. Dynamic Blade Flutter Index
        // If RPM is known, measure amplitude modulation depth around the blade rotation period
        let flutter_index = if self.active_rpm > 500.0 && frame.len() > 32 {
            let bpf_hz = (self.num_blades as f32) * self.active_rpm / 60.0;
            // Find closest wavelet scale to BPF
            let bpf_scale_idx = scalogram
                .frequencies
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    (a.1 - bpf_hz)
                        .abs()
                        .partial_cmp(&(b.1 - bpf_hz).abs())
                        .unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let bpf_mags = &scalogram.magnitudes[bpf_scale_idx];
            let n_total = bpf_mags.len();
            let margin = (n_total / 8).clamp(4, 32);
            if margin * 2 < n_total {
                let interior = &bpf_mags[margin..n_total - margin];
                let peak_mag = interior.iter().fold(0.0f32, |acc, &x| acc.max(x));
                let min_mag = interior.iter().fold(f32::MAX, |acc, &x| acc.min(x));
                if peak_mag + min_mag > 1e-6 {
                    ((peak_mag - min_mag) / (peak_mag + min_mag)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 4. Composite Airframe Fatigue Index
        // Normalized metric in [0.0, 1.0]
        let kurt_score = ((max_kurtosis - 3.0) / (self.config.kurtosis_critical_threshold - 3.0)).clamp(0.0, 1.0);
        let crack_score = (high_freq_crack_ratio / self.config.crack_energy_warning_ratio).clamp(0.0, 1.0);
        let flutter_score = (flutter_index / self.config.flutter_warning_threshold).clamp(0.0, 1.0);

        let fatigue_index = (0.45 * kurt_score + 0.35 * crack_score + 0.20 * flutter_score).clamp(0.0, 1.0);

        // 5. Severity Rating
        let severity = if max_kurtosis >= self.config.kurtosis_critical_threshold || fatigue_index >= 0.75 {
            AnomalySeverity::Critical
        } else if max_kurtosis >= self.config.kurtosis_advisory_threshold || fatigue_index >= 0.45 {
            AnomalySeverity::Warning
        } else if fatigue_index >= 0.25 {
            AnomalySeverity::Advisory
        } else {
            AnomalySeverity::Normal
        };

        RotorDamageReport {
            airframe_fatigue_index: fatigue_index,
            max_kurtosis,
            kurtosis_peak_freq_hz: kurtosis_freq,
            flutter_index,
            high_freq_crack_energy_ratio: high_freq_crack_ratio,
            severity,
            rpm: self.active_rpm,
            timestamp_sec,
        }
    }

    /// Access reference to underlying continuous wavelet filterbank.
    pub fn filterbank(&self) -> &ContinuousWaveletFilterbank {
        &self.filterbank
    }

    /// Access configuration parameters.
    pub fn config(&self) -> &CwtProfilerConfig {
        &self.config
    }
}
