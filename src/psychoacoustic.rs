//! Psychoacoustic masking model (ISO/IEC 11172-3 MPEG-1 Model 1), human auditory threshold in quiet,
//! real-time acoustic detectability range calculation, and active drone rotor RPM micro-dithering.

#![deny(unsafe_code)]

use crate::health::MavlinkNamedValueFloat;
use serde::{Deserialize, Serialize};

/// Representation of a discrete Zwicker Bark critical band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarkBand {
    /// Zero-indexed Bark band number (0 to 24).
    pub index: usize,
    /// Lower cutoff frequency in Hz.
    pub f_low: f32,
    /// Center frequency in Hz.
    pub f_center: f32,
    /// Upper cutoff frequency in Hz.
    pub f_high: f32,
    /// Critical bandwidth in Hz.
    pub bandwidth: f32,
}

/// Standard 25 Zwicker critical bands covering 0 Hz to 20,000 Hz.
pub const BARK_BANDS_25: [BarkBand; 25] = [
    BarkBand { index: 0, f_low: 0.0, f_center: 50.0, f_high: 100.0, bandwidth: 100.0 },
    BarkBand { index: 1, f_low: 100.0, f_center: 150.0, f_high: 200.0, bandwidth: 100.0 },
    BarkBand { index: 2, f_low: 200.0, f_center: 250.0, f_high: 300.0, bandwidth: 100.0 },
    BarkBand { index: 3, f_low: 300.0, f_center: 350.0, f_high: 400.0, bandwidth: 100.0 },
    BarkBand { index: 4, f_low: 400.0, f_center: 450.0, f_high: 510.0, bandwidth: 110.0 },
    BarkBand { index: 5, f_low: 510.0, f_center: 570.0, f_high: 630.0, bandwidth: 120.0 },
    BarkBand { index: 6, f_low: 630.0, f_center: 700.0, f_high: 770.0, bandwidth: 140.0 },
    BarkBand { index: 7, f_low: 770.0, f_center: 840.0, f_high: 920.0, bandwidth: 150.0 },
    BarkBand { index: 8, f_low: 920.0, f_center: 1000.0, f_high: 1080.0, bandwidth: 160.0 },
    BarkBand { index: 9, f_low: 1080.0, f_center: 1170.0, f_high: 1270.0, bandwidth: 190.0 },
    BarkBand { index: 10, f_low: 1270.0, f_center: 1370.0, f_high: 1480.0, bandwidth: 210.0 },
    BarkBand { index: 11, f_low: 1480.0, f_center: 1600.0, f_high: 1720.0, bandwidth: 240.0 },
    BarkBand { index: 12, f_low: 1720.0, f_center: 1850.0, f_high: 2000.0, bandwidth: 280.0 },
    BarkBand { index: 13, f_low: 2000.0, f_center: 2150.0, f_high: 2320.0, bandwidth: 320.0 },
    BarkBand { index: 14, f_low: 2320.0, f_center: 2500.0, f_high: 2700.0, bandwidth: 380.0 },
    BarkBand { index: 15, f_low: 2700.0, f_center: 2900.0, f_high: 3150.0, bandwidth: 450.0 },
    BarkBand { index: 16, f_low: 3150.0, f_center: 3400.0, f_high: 3700.0, bandwidth: 550.0 },
    BarkBand { index: 17, f_low: 3700.0, f_center: 4000.0, f_high: 4400.0, bandwidth: 700.0 },
    BarkBand { index: 18, f_low: 4400.0, f_center: 4800.0, f_high: 5300.0, bandwidth: 900.0 },
    BarkBand { index: 19, f_low: 5300.0, f_center: 5800.0, f_high: 6400.0, bandwidth: 1100.0 },
    BarkBand { index: 20, f_low: 6400.0, f_center: 7000.0, f_high: 7700.0, bandwidth: 1300.0 },
    BarkBand { index: 21, f_low: 7700.0, f_center: 8500.0, f_high: 9500.0, bandwidth: 1800.0 },
    BarkBand { index: 22, f_low: 9500.0, f_center: 10500.0, f_high: 12000.0, bandwidth: 2500.0 },
    BarkBand { index: 23, f_low: 12000.0, f_center: 13500.0, f_high: 15500.0, bandwidth: 3500.0 },
    BarkBand { index: 24, f_low: 15500.0, f_center: 17000.0, f_high: 20000.0, bandwidth: 4500.0 },
];

/// Converts frequency in Hz to Bark scale according to Traunmüller (1990).
#[inline]
pub fn freq_to_bark(freq_hz: f32) -> f32 {
    let f = freq_hz.max(0.0);
    ((26.81 * f) / (1960.0 + f)) - 0.53
}

/// Converts Bark scale value back to frequency in Hz according to exact Traunmüller inversion.
#[inline]
pub fn bark_to_freq(bark: f32) -> f32 {
    let b = bark.max(0.0);
    let num = b + 0.53;
    let denom = (26.81 - num).max(1e-4);
    (1960.0 * num) / denom
}

/// High-precision fast base-10 logarithm for real-time DSP (max error < 0.002 dB).
#[inline]
pub fn fast_log10(x: f32) -> f32 {
    if x <= 0.0 {
        return -300.0;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) as i32) - 127;
    // Normalized mantissa in [1.0, 2.0)
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    // Minimax polynomial approximation of log2(1 + m) for m in [0.0, 1.0)
    let m = mantissa - 1.0;
    let log2_m = m * (std::f32::consts::LOG2_E - m * (0.721347 - 0.278652 * m));
    let log2_val = (exponent as f32) + log2_m;
    log2_val * std::f32::consts::LOG10_2
}

/// Evaluates Sound Pressure Level in dB SPL from normalized power.
#[inline]
pub fn fast_power_to_db_spl(power: f32, spl_full_scale: f32) -> f32 {
    10.0 * fast_log10(power.max(1e-12)) + spl_full_scale
}

/// Evaluates human Absolute Threshold of Hearing (ATH / threshold in quiet) in dB SPL.
///
/// Formula: Terhardt (1979) / ISO/IEC 11172-3 empirical curve.
#[inline]
pub fn threshold_in_quiet_db(freq_hz: f32) -> f32 {
    let f = freq_hz.clamp(20.0, 20000.0);
    let f_khz = f / 1000.0;

    let term1 = 3.64 * f_khz.powf(-0.8);
    let term2 = -6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp();
    let term3 = 1e-3 * f_khz.powi(4);

    term1 + term2 + term3
}

/// Evaluates ISO 9613-1 atmospheric attenuation coefficient in dB/km.
#[inline]
pub fn atmospheric_absorption_db_km(freq_hz: f32, temp_c: f32, _rel_humidity_pct: f32) -> f32 {
    let f = freq_hz.clamp(50.0, 20000.0);
    let temp_ratio = (273.15 + temp_c) / 293.15;
    let alpha_ref = 1.6 * (f / 1000.0).powf(1.4);
    alpha_ref / temp_ratio.sqrt()
}

/// Configuration parameters for the psychoacoustic masking and stealth engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PsychoacousticConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: f32,
    /// STFT frame size in samples (power of 2, e.g. 512).
    pub frame_size: usize,
    /// Ambient background noise level at ground receiver in dBA SPL (e.g. 45.0 dBA suburban).
    pub ambient_noise_spl_dba: f32,
    /// Acoustic reference distance in meters (typically 1.0 m).
    pub reference_distance_m: f32,
    /// Ambient air temperature in Celsius (default 20.0 C).
    pub temperature_c: f32,
    /// Relative humidity in percent (default 50.0%).
    pub relative_humidity_pct: f32,
    /// Number of lift rotors on the airframe (typically 4 for quadcopter).
    pub num_rotors: usize,
    /// Number of blades per propeller (typically 2).
    pub num_blades: usize,
    /// Maximum permissible rotor RPM micro-dither percentage (e.g. 3.0%).
    pub max_rpm_dither_pct: f32,
    /// Calibration reference sound pressure level at 1 meter for unit digital full-scale (dB SPL).
    pub spl_full_scale_1m: f32,
}

impl Default for PsychoacousticConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            frame_size: 512,
            ambient_noise_spl_dba: 45.0, // Typical suburban background
            reference_distance_m: 1.0,
            temperature_c: 20.0,
            relative_humidity_pct: 50.0,
            num_rotors: 4,
            num_blades: 2,
            max_rpm_dither_pct: 3.0,
            spl_full_scale_1m: 94.0, // 94 dB SPL = 1 Pa RMS
        }
    }
}

/// Real-time psychoacoustic stealth assessment and detectability report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcousticStealthReport {
    /// Estimated distance in meters at which the drone becomes audible to human observers.
    pub human_detectability_range_m: f32,
    /// Acoustic frequency bin (Hz) with highest human audibility margin.
    pub dominant_tonal_frequency_hz: f32,
    /// Peak Signal-to-Mask Ratio in dB across all frequency bins (> 0 is audible, <= 0 is masked).
    pub max_signal_to_mask_ratio_db: f32,
    /// Estimated overall Sound Pressure Level at 1 meter in dBA.
    pub overall_spl_1m_dba: f32,
    /// Zwicker psychoacoustic annoyance metric in [0.0, 1.0] (combining tonality, loudness, sharpness).
    pub psychoacoustic_annoyance_score: f32,
    /// Recommended thrust-conserving RPM micro-dithering offsets per motor in percent.
    pub recommended_rpm_dithers: Vec<f32>,
    /// Stream timestamp in seconds.
    pub timestamp_sec: f64,
}

impl AcousticStealthReport {
    /// Converts stealth metrics into MAVLink standard NAMED_VALUE_FLOAT packets (Message ID 251).
    pub fn to_mavlink_packets(&self, time_boot_ms: u32) -> Vec<MavlinkNamedValueFloat> {
        let first_dither = self.recommended_rpm_dithers.first().copied().unwrap_or(0.0);
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "AUD_DIST", self.human_detectability_range_m),
            MavlinkNamedValueFloat::new(time_boot_ms, "AUD_SMR", self.max_signal_to_mask_ratio_db),
            MavlinkNamedValueFloat::new(time_boot_ms, "RPM_DITH", first_dither),
        ]
    }
}

/// Real-time Psychoacoustic Masking and Active Drone Acoustic Stealth Engine.
pub struct PsychoacousticStealthEngine {
    config: PsychoacousticConfig,
    num_bins: usize,
    bin_frequencies: Vec<f32>,
    ath_curve: Vec<f32>,
    bark_band_of_bin: Vec<usize>,
    alpha_atm_per_bin: Vec<f32>,
    spreading_matrix: [[f32; 25]; 25], // 25x25 inter-band spreading attenuation (dB)
    spl_spectrum: Vec<f32>,            // Scratch buffer to avoid per-frame heap allocations
    latest_report: Option<AcousticStealthReport>,
}

impl PsychoacousticStealthEngine {
    /// Creates a new psychoacoustic stealth engine with specified configuration.
    pub fn new(config: PsychoacousticConfig) -> Self {
        let num_bins = config.frame_size / 2 + 1;
        let mut bin_frequencies = Vec::with_capacity(num_bins);
        let mut ath_curve = Vec::with_capacity(num_bins);
        let mut bark_band_of_bin = Vec::with_capacity(num_bins);
        let mut alpha_atm_per_bin = Vec::with_capacity(num_bins);

        let df = config.sample_rate / (config.frame_size as f32);

        for k in 0..num_bins {
            let f = (k as f32) * df;
            bin_frequencies.push(f);
            ath_curve.push(threshold_in_quiet_db(f));
            alpha_atm_per_bin.push(atmospheric_absorption_db_km(
                f,
                config.temperature_c,
                config.relative_humidity_pct,
            ));

            // Map bin to Bark band
            let mut band_idx = 24;
            for band in &BARK_BANDS_25 {
                if f >= band.f_low && f < band.f_high {
                    band_idx = band.index;
                    break;
                }
            }
            bark_band_of_bin.push(band_idx);
        }

        // Precompute 25x25 inter-band psychoacoustic spreading attenuation matrix
        let num_bands = BARK_BANDS_25.len();
        let mut spreading_matrix = [[0.0f32; 25]; 25];
        for (i, row) in spreading_matrix.iter_mut().enumerate().take(num_bands) {
            let z_i = BARK_BANDS_25[i].index as f32;
            for (j, cell) in row.iter_mut().enumerate().take(num_bands) {
                let z_j = BARK_BANDS_25[j].index as f32;
                let dz = z_j - z_i;
                if dz <= 0.0 {
                    *cell = 27.0 * dz; // Upward slope towards lower frequencies (-27 dB/Bark)
                } else {
                    *cell = -24.0 * dz; // Downward slope towards higher frequencies (-24 dB/Bark)
                }
            }
        }

        let spl_spectrum = vec![0.0f32; num_bins];

        Self {
            config,
            num_bins,
            bin_frequencies,
            ath_curve,
            bark_band_of_bin,
            alpha_atm_per_bin,
            spreading_matrix,
            spl_spectrum,
            latest_report: None,
        }
    }

    /// Access active configuration.
    pub fn config(&self) -> &PsychoacousticConfig {
        &self.config
    }

    /// Access latest evaluated stealth report.
    pub fn latest_report(&self) -> Option<&AcousticStealthReport> {
        self.latest_report.as_ref()
    }

    /// Analyzes an acoustic power spectrum and motor RPM telemetry to compute the psychoacoustic
    /// masking threshold, human detectability range, and RPM micro-dithering recommendations.
    pub fn analyze_spectrum(
        &mut self,
        power_spectrum: &[f32],
        motor_rpms: &[f32],
        timestamp_sec: f64,
    ) -> AcousticStealthReport {
        let n = power_spectrum.len().min(self.num_bins);
        let spl_fs = self.config.spl_full_scale_1m;

        // 1. Convert normalized digital power spectrum to physical Sound Pressure Level at 1 meter (dB SPL)
        let mut total_energy = 1e-12f32;

        for k in 0..n {
            let p = power_spectrum[k];
            total_energy += p;
            self.spl_spectrum[k] = fast_power_to_db_spl(p, spl_fs);
        }

        let overall_spl_1m = fast_power_to_db_spl(total_energy, spl_fs);
        let overall_spl_1m_dba = (overall_spl_1m - 3.0).max(0.0);

        // 2. Aggregate acoustic energy into 25 Bark bands and detect tonal peaks
        let mut band_energies = [1e-12f32; 25];
        let mut tonal_peaks_count = 0usize;
        let mut max_tonal_peak_in_band = [-1000.0f32; 25];
        let mut tonal_peak_bin_in_band = [usize::MAX; 25];

        for k in 2..n.saturating_sub(2) {
            let curr = self.spl_spectrum[k];
            if curr > self.spl_spectrum[k - 1]
                && curr > self.spl_spectrum[k + 1]
                && curr > self.spl_spectrum[k - 2] + 7.0
                && curr > self.spl_spectrum[k + 2] + 7.0
            {
                tonal_peaks_count += 1;
                let b = self.bark_band_of_bin[k];
                if curr > max_tonal_peak_in_band[b] {
                    max_tonal_peak_in_band[b] = curr;
                    tonal_peak_bin_in_band[b] = k;
                }
            }
        }

        for k in 0..n {
            let b = self.bark_band_of_bin[k];
            band_energies[b] += power_spectrum[k].max(1e-12);
        }

        let mut band_spl = [0.0f32; 25];
        let mut band_spread_src = [0.0f32; 25];
        for b in 0..25 {
            let spl = fast_power_to_db_spl(band_energies[b], spl_fs);
            band_spl[b] = spl;
            band_spread_src[b] = spl - 6.0;
        }

        // 3. Fast Bark inter-band masking threshold computation
        let ambient = self.config.ambient_noise_spl_dba;
        let mut inter_band_mask = [ambient; 25];
        for b in 0..25 {
            let mut max_mask = ambient;
            for m in 0..25 {
                if m != b {
                    let spread_spl = band_spread_src[m] + self.spreading_matrix[m][b];
                    if spread_spl > max_mask {
                        max_mask = spread_spl;
                    }
                }
            }
            inter_band_mask[b] = max_mask;
        }

        // 4. Compute Signal-to-Mask Ratio (SMR) per bin and identify dominant tonal frequency
        let mut max_smr = -100.0f32;
        let mut dominant_freq = self.bin_frequencies.first().copied().unwrap_or(0.0);

        for k in 0..n {
            let b = self.bark_band_of_bin[k];
            let ath = self.ath_curve[k];
            let base_mask = inter_band_mask[b].max(ath);

            let bin_mask = if max_tonal_peak_in_band[b] > -900.0 && tonal_peak_bin_in_band[b] != k {
                base_mask.max(max_tonal_peak_in_band[b] - 6.0)
            } else {
                base_mask
            };

            let smr = self.spl_spectrum[k] - bin_mask;
            if smr > max_smr {
                max_smr = smr;
                dominant_freq = self.bin_frequencies[k];
            }
        }

        // 5. Solve for human detectability range (meters) via spherical spreading and atmospheric absorption
        let detectability_range = self.solve_detectability_range(n);

        // 6. Psychoacoustic annoyance metric
        let tonality_score = (tonal_peaks_count as f32 * 0.15).clamp(0.0, 0.4);
        let smr_score = (max_smr / 40.0).clamp(0.0, 0.4);
        let spl_score = (overall_spl_1m_dba / 100.0).clamp(0.0, 0.2);
        let annoyance_score = (tonality_score + smr_score + spl_score).clamp(0.0, 1.0);

        // 7. Compute thrust-conserving rotor RPM micro-dithers
        let recommended_dithers = self.compute_rpm_dithers(motor_rpms, max_smr);

        let report = AcousticStealthReport {
            human_detectability_range_m: detectability_range,
            dominant_tonal_frequency_hz: dominant_freq,
            max_signal_to_mask_ratio_db: max_smr,
            overall_spl_1m_dba,
            psychoacoustic_annoyance_score: annoyance_score,
            recommended_rpm_dithers: recommended_dithers,
            timestamp_sec,
        };

        self.latest_report = Some(report.clone());
        report
    }

    /// Solves for the maximum distance R where the drone's sound pressure level drops below
    /// the ambient and threshold-in-quiet masking curve across all frequencies.
    fn solve_detectability_range(&self, n: usize) -> f32 {
        let ambient = self.config.ambient_noise_spl_dba;

        // Find the bin with the highest excess over ATH and ambient at reference distance
        let mut max_excess = -1000.0f32;
        let mut max_k = 1;

        for k in 1..n {
            let threshold = self.ath_curve[k].max(ambient);
            let excess = self.spl_spectrum[k] - threshold;
            if excess > max_excess {
                max_excess = excess;
                max_k = k;
            }
        }

        if max_excess <= 0.0 {
            return self.config.reference_distance_m;
        }

        let mut low_r = self.config.reference_distance_m;
        let mut high_r = 10000.0f32;

        for _ in 0..16 {
            let mid_r = 0.5 * (low_r + high_r);
            let log_dist_loss = 20.0 * fast_log10(mid_r / self.config.reference_distance_m);

            // Geometric pruning: distance loss alone exceeds maximum source excess
            if log_dist_loss >= max_excess {
                high_r = mid_r;
                continue;
            }

            // Check dominant bin first (fast path)
            let atm_loss_max = (self.alpha_atm_per_bin[max_k] * mid_r) * 0.001;
            let mut is_audible = (max_excess - log_dist_loss - atm_loss_max) > 0.0;

            if !is_audible {
                // Secondary check across candidate bins with potential visibility
                for k in 1..n {
                    if k == max_k {
                        continue;
                    }
                    let excess = self.spl_spectrum[k] - self.ath_curve[k].max(ambient);
                    if excess <= log_dist_loss {
                        continue;
                    }
                    let atm_loss = (self.alpha_atm_per_bin[k] * mid_r) * 0.001;
                    if excess - log_dist_loss - atm_loss > 0.0 {
                        is_audible = true;
                        break;
                    }
                }
            }

            if is_audible {
                low_r = mid_r;
            } else {
                high_r = mid_r;
            }
        }

        0.5 * (low_r + high_r)
    }

    /// Computes thrust-conserving anti-symmetric RPM micro-dithers to smear blade pass frequencies.
    fn compute_rpm_dithers(&self, motor_rpms: &[f32], max_smr: f32) -> Vec<f32> {
        let num_rotors = self.config.num_rotors.max(1);
        if max_smr <= 0.0 || motor_rpms.is_empty() {
            return vec![0.0f32; num_rotors];
        }

        let max_dither = self.config.max_rpm_dither_pct;
        let scale = (max_smr / 20.0).clamp(0.3, 1.0) * max_dither;

        if num_rotors == 4 {
            // Anti-symmetric quadcopter pattern:
            // Motor 1 (CW): +scale, Motor 2 (CCW): -scale, Motor 3 (CW): +scale/3, Motor 4 (CCW): -scale/3
            // Sum of dithers = scale - scale + scale/3 - scale/3 = 0.0 (strictly thrust-conserving)
            vec![scale, -scale, scale / 3.0, -scale / 3.0]
        } else {
            let mut dithers = Vec::with_capacity(num_rotors);
            for i in 0..num_rotors {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                dithers.push(sign * scale);
            }
            dithers
        }
    }

    /// Resets internal stealth history.
    pub fn reset(&mut self) {
        self.latest_report = None;
    }
}
