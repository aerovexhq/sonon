#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// High-precision rational approximation of the Exponential Integral $E_1(x) = \int_x^\infty \frac{e^{-t}}{t} dt$.
///
/// Implements Abramowitz and Stegun §5.1.53 / §5.1.56 with absolute error $|\epsilon(x)| < 2 \times 10^{-7}$.
#[inline]
pub fn exponential_integral_e1(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::INFINITY;
    }

    if x <= 1.0 {
        // Polynomial series expansion for 0 < x <= 1
        const A0: f32 = -0.57721566; // Euler-Mascheroni constant negative
        const A1: f32 = 0.99999193;
        const A2: f32 = -0.24991055;
        const A3: f32 = 0.05519968;
        const A4: f32 = -0.00976004;
        const A5: f32 = 0.00107857;

        let poly = A0 + x * (A1 + x * (A2 + x * (A3 + x * (A4 + x * A5))));
        -x.ln() + poly
    } else {
        // Rational approximation for x > 1
        const B1: f32 = 8.5733287401;
        const B2: f32 = 18.0590177861;
        const B3: f32 = 8.6347608925;
        const B4: f32 = 0.2677737343;

        const C1: f32 = 9.5733223454;
        const C2: f32 = 25.6329105584;
        const C3: f32 = 21.0996530827;
        const C4: f32 = 3.9584969228;

        let num = x * (x * (x * (x + B1) + B2) + B3) + B4;
        let den = x * (x * (x * (x + C1) + C2) + C3) + C4;

        ((-x).exp() / x) * (num / den)
    }
}

/// Compute optimal Bayesian Log-Spectral Amplitude (MMSE-LSA) spectral gain.
///
/// $G_{\text{LSA}}(\xi, \gamma) = \frac{\xi}{1 + \xi} \exp\left( \frac{1}{2} E_1(v) \right)$
/// where $v = \frac{\xi}{1 + \xi} \gamma$.
#[inline]
pub fn compute_lsa_gain(prior_snr: f32, post_snr: f32, min_gain: f32) -> f32 {
    let xi = prior_snr.max(1e-4);
    let gamma = post_snr.max(1e-4);

    // Deep negative SNR shortcut
    if xi <= 0.015 && gamma <= 1.5 {
        return min_gain;
    }

    let factor = xi / (1.0 + xi);
    let v = factor * gamma;

    if v > 15.0 {
        return factor.clamp(min_gain, 1.0);
    }

    let e1 = exponential_integral_e1(v);
    let lsa_factor = (0.5 * e1).exp();
    let raw_gain = factor * lsa_factor;

    if raw_gain.is_finite() {
        raw_gain.clamp(min_gain, 1.0)
    } else {
        min_gain
    }
}

/// Configuration parameters for Improved Minima Controlled Recursive Averaging (IMCRA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImcraConfig {
    /// Time-smoothing factor $\alpha_s \in [0.80, 0.98]$ for signal power spectrum.
    pub alpha_s: f32,
    /// Recursive noise PSD smoothing parameter $\alpha_d \in [0.75, 0.95]$.
    pub alpha_d: f32,
    /// Sub-window length in frames for running minimum tracking (e.g. 15 frames ~ 150 ms).
    pub sub_window_len: usize,
    /// Number of sub-windows to retain for global minimum estimation (e.g. 6 -> 90 frames ~ 900 ms).
    pub num_sub_windows: usize,
    /// Speech presence threshold ratio $\gamma_0$ (typically 2.5 to 4.5).
    pub gamma_0: f32,
    /// Threshold $\gamma_1$ for conditional speech presence decision (typically 1.5 to 2.5).
    pub gamma_1: f32,
}

impl Default for ImcraConfig {
    fn default() -> Self {
        Self {
            alpha_s: 0.90,
            alpha_d: 0.85,
            sub_window_len: 16,
            num_sub_windows: 6,
            gamma_0: 3.5,
            gamma_1: 1.8,
        }
    }
}

/// Dynamic noise power spectral density (PSD) tracker using Improved Minima Controlled Recursive Averaging.
///
/// Tracks non-stationary ambient noise (such as drone rotor throttle changes and wind gusts)
/// without requiring external VAD or distinct silence intervals.
#[derive(Debug, Clone)]
pub struct ImcraNoiseEstimator {
    config: ImcraConfig,
    num_bins: usize,

    /// Time-smoothed power spectrum $S(k, l)$
    s_time: Vec<f32>,
    /// Frequency-smoothed power spectrum $S_f(k, l)$
    s_freq: Vec<f32>,
    /// Current sub-window minimum $S_{\text{sub\_min}}(k)$
    s_sub_min: Vec<f32>,
    /// Circular buffer of sub-window minima: [num_sub_windows][num_bins]
    sub_min_history: Vec<Vec<f32>>,
    /// Circular head index in sub_min_history
    sub_min_head: usize,
    /// Current sub-window frame counter
    sub_window_counter: usize,
    /// Estimated global minimum $S_{\min}(k, l)$
    s_min: Vec<f32>,
    /// Estimated noise power spectral density $\hat{\lambda}_d(k, l)$
    noise_psd: Vec<f32>,
    /// Subband speech presence probability $p(k, l)$
    speech_presence_prob: Vec<f32>,
    /// Whether initial noise floor has been primed
    is_initialized: bool,
}

impl ImcraNoiseEstimator {
    /// Construct a new IMCRA noise estimator for specified number of STFT bins.
    pub fn new(config: ImcraConfig, num_bins: usize) -> Self {
        assert!(num_bins > 0, "Number of bins must be > 0");
        assert!(config.num_sub_windows > 0, "Num sub-windows must be > 0");

        let s_time = vec![1e-5; num_bins];
        let s_freq = vec![1e-5; num_bins];
        let s_sub_min = vec![f32::INFINITY; num_bins];
        let sub_min_history = vec![vec![1e-5; num_bins]; config.num_sub_windows];
        let s_min = vec![1e-5; num_bins];
        let noise_psd = vec![1e-5; num_bins];
        let speech_presence_prob = vec![0.0; num_bins];

        Self {
            config,
            num_bins,
            s_time,
            s_freq,
            s_sub_min,
            sub_min_history,
            sub_min_head: 0,
            sub_window_counter: 0,
            s_min,
            noise_psd,
            speech_presence_prob,
            is_initialized: false,
        }
    }

    /// Reset internal history buffers.
    pub fn reset(&mut self) {
        self.s_time.fill(1e-5);
        self.s_freq.fill(1e-5);
        self.s_sub_min.fill(f32::INFINITY);
        for hist in &mut self.sub_min_history {
            hist.fill(1e-5);
        }
        self.sub_min_head = 0;
        self.sub_window_counter = 0;
        self.s_min.fill(1e-5);
        self.noise_psd.fill(1e-5);
        self.speech_presence_prob.fill(0.0);
        self.is_initialized = false;
    }

    /// Update noise PSD and speech presence probability using current frame power spectrum.
    pub fn update(&mut self, power_spectrum: &[f32]) {
        let n = self.num_bins.min(power_spectrum.len());

        if !self.is_initialized {
            for k in 0..n {
                let p = power_spectrum[k].max(1e-8);
                self.s_time[k] = p;
                self.s_freq[k] = p;
                self.s_sub_min[k] = p;
                for hist in &mut self.sub_min_history {
                    hist[k] = p;
                }
                self.s_min[k] = p;
                self.noise_psd[k] = p;
                self.speech_presence_prob[k] = 0.0;
            }
            self.is_initialized = true;
            return;
        }

        let alpha_s = self.config.alpha_s;
        let one_minus_alpha_s = 1.0 - alpha_s;

        // 1. Time smoothing
        for k in 0..n {
            let p = power_spectrum[k].max(1e-8);
            self.s_time[k] = alpha_s * self.s_time[k] + one_minus_alpha_s * p;
        }

        // 2. Local frequency smoothing across neighboring bins [0.25, 0.50, 0.25]
        for k in 0..n {
            let prev = if k > 0 { self.s_time[k - 1] } else { self.s_time[k] };
            let curr = self.s_time[k];
            let next = if k + 1 < n { self.s_time[k + 1] } else { self.s_time[k] };
            self.s_freq[k] = 0.25 * prev + 0.50 * curr + 0.25 * next;
        }

        // 3. Update sub-window running minimum
        for k in 0..n {
            if self.s_freq[k] < self.s_sub_min[k] {
                self.s_sub_min[k] = self.s_freq[k];
            }
        }

        self.sub_window_counter += 1;
        if self.sub_window_counter >= self.config.sub_window_len {
            self.sub_window_counter = 0;
            // Advance circular buffer without heap reallocations
            self.sub_min_head = (self.sub_min_head + 1) % self.config.num_sub_windows;
            self.sub_min_history[self.sub_min_head].copy_from_slice(&self.s_sub_min);
            self.s_sub_min.fill(f32::INFINITY);

            // Recompute global minimum across sub-window history
            for k in 0..n {
                let mut min_val = f32::INFINITY;
                for hist in &self.sub_min_history {
                    if hist[k] < min_val {
                        min_val = hist[k];
                    }
                }
                self.s_min[k] = min_val.max(1e-8);
            }
        } else {
            // Include active sub-window in current minimum
            for k in 0..n {
                let mut min_val = self.s_sub_min[k];
                for hist in &self.sub_min_history {
                    if hist[k] < min_val {
                        min_val = hist[k];
                    }
                }
                self.s_min[k] = min_val.max(1e-8);
            }
        }

        // 4. Speech presence probability computation
        let gamma_0 = self.config.gamma_0;
        let gamma_1 = self.config.gamma_1;
        let alpha_d = self.config.alpha_d;

        for k in 0..n {
            let s_r = self.s_time[k] / self.s_min[k];
            let s_f_r = self.s_freq[k] / self.s_min[k];

            // Robust indicator of speech activity
            let is_speech_bin = s_r > gamma_0 && s_f_r > gamma_1;
            let p_hat = if is_speech_bin {
                0.98
            } else {
                let ratio = (s_r - 1.0).max(0.0) / gamma_0;
                (ratio * 0.4).min(0.25)
            };
            self.speech_presence_prob[k] = p_hat;

            // When speech is present, freeze noise update to avoid speech absorption
            let alpha_tilde = if is_speech_bin {
                0.999
            } else {
                alpha_d
            };

            let p_curr = power_spectrum[k].max(1e-8);
            self.noise_psd[k] = alpha_tilde * self.noise_psd[k] + (1.0 - alpha_tilde) * p_curr;
        }
    }

    /// Access estimated noise power spectral density $\hat{\lambda}_d$.
    pub fn noise_psd(&self) -> &[f32] {
        &self.noise_psd
    }

    /// Access speech presence probabilities $p(k, l)$.
    pub fn speech_presence_prob(&self) -> &[f32] {
        &self.speech_presence_prob
    }

    /// Average noise floor power across all bins.
    pub fn average_noise_power(&self) -> f32 {
        if self.num_bins == 0 {
            0.0
        } else {
            self.noise_psd.iter().sum::<f32>() / (self.num_bins as f32)
        }
    }
}

/// Configuration parameters for Bayesian Log-Spectral Amplitude (MMSE-LSA) speech enhancement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LsaConfig {
    /// Decision-directed a priori SNR recursive smoothing factor $\alpha_{\text{dd}} \in [0.94, 0.99]$.
    pub alpha_dd: f32,
    /// Minimum a priori SNR floor in dB (e.g. -19.0 dB).
    pub min_prior_snr_db: f32,
    /// Maximum attenuation floor in dB (e.g. -18.0 dB to -24.0 dB).
    pub min_gain_db: f32,
    /// Embedded IMCRA noise tracker configuration.
    pub imcra: ImcraConfig,
}

impl Default for LsaConfig {
    fn default() -> Self {
        Self {
            alpha_dd: 0.96,
            min_prior_snr_db: -19.0,
            min_gain_db: -20.0,
            imcra: ImcraConfig::default(),
        }
    }
}

/// Telemetry snapshot from Bayesian Log-Spectral Amplitude Speech Enhancement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MmseLsaTelemetry {
    /// Mean a priori SNR across bins in dB.
    pub mean_prior_snr_db: f32,
    /// Mean a posteriori SNR across bins in dB.
    pub mean_post_snr_db: f32,
    /// Mean speech presence probability across bins.
    pub mean_speech_presence_prob: f32,
    /// Average background noise floor power.
    pub mean_noise_power: f32,
    /// Average applied spectral suppression gain in dB.
    pub mean_gain_db: f32,
}

/// Bayesian Minimum Mean-Square Error Log-Spectral Amplitude (MMSE-LSA) Speech Enhancement Filter.
///
/// Features:
/// - Decision-directed a priori SNR estimation eliminating "musical noise" flutter.
/// - Analytical Log-Spectral Amplitude (LSA) gain mapping using Abramowitz-Stegun rational $E_1(v)$ approximation.
/// - Integrated IMCRA non-stationary noise power spectral density (PSD) tracking.
/// - Subband speech presence probability modulation.
#[derive(Debug, Clone)]
pub struct MmseLsaFilter {
    config: LsaConfig,
    num_bins: usize,
    imcra: ImcraNoiseEstimator,

    /// Previous clean speech power estimate $\hat{X}^2(k, l-1)$
    prev_clean_power: Vec<f32>,
    /// Previous noise PSD estimate
    prev_noise_psd: Vec<f32>,
    /// Minimum a priori SNR linear floor
    min_prior_snr: f32,
    /// Minimum spectral gain linear floor
    min_gain: f32,

    /// Active applied spectral gains $G(k)$
    applied_gains: Vec<f32>,
    /// A priori SNR vector $\xi(k)$
    prior_snr: Vec<f32>,
    /// A posteriori SNR vector $\gamma(k)$
    post_snr: Vec<f32>,

    /// Smoothed telemetry stats
    smoothed_prior_snr_db: f32,
    smoothed_post_snr_db: f32,
    smoothed_gain_db: f32,
    smoothed_speech_prob: f32,
}

impl MmseLsaFilter {
    /// Construct a new MMSE-LSA speech enhancement filter for specified number of STFT frequency bins.
    pub fn new(num_bins: usize, config: LsaConfig) -> Self {
        assert!(num_bins > 0, "Number of bins must be > 0");

        let min_prior_snr = 10.0f32.powf(config.min_prior_snr_db / 10.0);
        let min_gain = 10.0f32.powf(config.min_gain_db / 20.0);
        let imcra = ImcraNoiseEstimator::new(config.imcra.clone(), num_bins);

        Self {
            config,
            num_bins,
            imcra,
            prev_clean_power: vec![1e-5; num_bins],
            prev_noise_psd: vec![1e-5; num_bins],
            min_prior_snr,
            min_gain,
            applied_gains: vec![1.0; num_bins],
            prior_snr: vec![1.0; num_bins],
            post_snr: vec![1.0; num_bins],
            smoothed_prior_snr_db: 0.0,
            smoothed_post_snr_db: 0.0,
            smoothed_gain_db: 0.0,
            smoothed_speech_prob: 0.0,
        }
    }

    /// Reset internal delay lines and IMCRA history.
    pub fn reset(&mut self) {
        self.imcra.reset();
        self.prev_clean_power.fill(1e-5);
        self.prev_noise_psd.fill(1e-5);
        self.applied_gains.fill(1.0);
        self.prior_snr.fill(1.0);
        self.post_snr.fill(1.0);
        self.smoothed_prior_snr_db = 0.0;
        self.smoothed_post_snr_db = 0.0;
        self.smoothed_gain_db = 0.0;
        self.smoothed_speech_prob = 0.0;
    }

    /// Process power spectral density frame in place, attenuating background drone rotor and ambient noise.
    pub fn process_spectrum(&mut self, power_spectrum: &mut [f32]) {
        let n = self.num_bins.min(power_spectrum.len());

        // 1. Update non-stationary noise PSD tracking via IMCRA
        self.imcra.update(&power_spectrum[..n]);
        let noise_psd = self.imcra.noise_psd();
        let spp = self.imcra.speech_presence_prob();

        let alpha_dd = self.config.alpha_dd;
        let one_minus_alpha_dd = 1.0 - alpha_dd;
        let min_prior = self.min_prior_snr;
        let min_g = self.min_gain;

        let mut sum_prior = 0.0f32;
        let mut sum_post = 0.0f32;
        let mut sum_gain = 0.0f32;
        let mut sum_spp = 0.0f32;

        for k in 0..n {
            let p_y = power_spectrum[k].max(1e-8);
            let lambda_d = noise_psd[k].max(1e-8);

            // A posteriori SNR: gamma = |Y|^2 / lambda_d
            let gamma = (p_y / lambda_d).max(1e-4);
            self.post_snr[k] = gamma;

            // Decision-directed a priori SNR: xi = alpha * (X_prev^2 / lambda_prev) + (1 - alpha) * max(gamma - 1, 0)
            let prev_snr = self.prev_clean_power[k] / self.prev_noise_psd[k].max(1e-8);
            let inst_snr = (gamma - 1.0).max(0.0);
            let xi = (alpha_dd * prev_snr + one_minus_alpha_dd * inst_snr).max(min_prior);
            self.prior_snr[k] = xi;

            // Pure Log-Spectral Amplitude (LSA) Gain
            let g_lsa = compute_lsa_gain(xi, gamma, min_g);

            // Modulate with Speech Presence Probability (SPP)
            let p = spp[k];
            let final_gain = (g_lsa * p + min_g * (1.0 - p)).clamp(min_g, 1.0);
            self.applied_gains[k] = final_gain;

            // Apply gain to power spectrum (gain is amplitude gain, power scales as G^2)
            let p_clean = p_y * (final_gain * final_gain);
            power_spectrum[k] = p_clean;

            // Update state for next frame
            self.prev_clean_power[k] = p_clean;
            self.prev_noise_psd[k] = lambda_d;

            sum_prior += xi;
            sum_post += gamma;
            sum_gain += final_gain;
            sum_spp += p;
        }

        // Fast telemetry update: 3 log10 calls total per frame instead of 771 calls
        let inv_n = 1.0 / (n as f32);
        let mean_prior = sum_prior * inv_n;
        let mean_post = sum_post * inv_n;
        let mean_gain = sum_gain * inv_n;
        let frame_spp = sum_spp * inv_n;

        let frame_prior_db = 10.0 * mean_prior.max(1e-4).log10();
        let frame_post_db = 10.0 * mean_post.max(1e-4).log10();
        let frame_gain_db = 20.0 * mean_gain.max(1e-4).log10();

        self.smoothed_prior_snr_db = 0.90 * self.smoothed_prior_snr_db + 0.10 * frame_prior_db;
        self.smoothed_post_snr_db = 0.90 * self.smoothed_post_snr_db + 0.10 * frame_post_db;
        self.smoothed_gain_db = 0.90 * self.smoothed_gain_db + 0.10 * frame_gain_db;
        self.smoothed_speech_prob = 0.90 * self.smoothed_speech_prob + 0.10 * frame_spp;
    }

    /// Process complex STFT bins in place: applies real amplitude gain while preserving signal phase.
    pub fn process_complex_bins(&mut self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.num_bins.min(real.len()).min(imag.len());
        let mut power = vec![0.0f32; n];
        for k in 0..n {
            power[k] = real[k] * real[k] + imag[k] * imag[k];
        }

        self.process_spectrum(&mut power);

        for k in 0..n {
            let g = self.applied_gains[k];
            real[k] *= g;
            imag[k] *= g;
        }
    }

    /// Return latest evaluated MMSE-LSA telemetry snapshot.
    pub fn telemetry(&self) -> MmseLsaTelemetry {
        MmseLsaTelemetry {
            mean_prior_snr_db: self.smoothed_prior_snr_db,
            mean_post_snr_db: self.smoothed_post_snr_db,
            mean_speech_presence_prob: self.smoothed_speech_prob,
            mean_noise_power: self.imcra.average_noise_power(),
            mean_gain_db: self.smoothed_gain_db,
        }
    }

    /// Access active applied spectral gains.
    pub fn applied_gains(&self) -> &[f32] {
        &self.applied_gains
    }

    /// Access reference to embedded IMCRA noise estimator.
    pub fn imcra(&self) -> &ImcraNoiseEstimator {
        &self.imcra
    }

    /// Access reference to active configuration.
    pub fn config(&self) -> &LsaConfig {
        &self.config
    }
}
