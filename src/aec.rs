//! Adaptive Acoustic Echo Cancellation (AEC), Normalized Least Mean Squares (NLMS) transversal filter,
//! Partitioned-Block Frequency-Domain Adaptive Filter (PBFDAF), Magnitude Squared Coherence (MSC)
//! Double-Talk Detection (DTD), Residual Echo Suppression (RES), and Echo Return Loss Enhancement (ERLE) telemetry.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use crate::stft::FftProcessor;

/// Configuration parameters for Acoustic Echo Cancellation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecConfig {
    /// Adaptive filter tap length $L$ (e.g., 128 to 512 taps).
    pub filter_length: usize,
    /// Normalized LMS adaptation step size $\mu \in (0.0, 1.0)$.
    pub step_size: f32,
    /// Regularization epsilon $\epsilon$ preventing division by zero during reference silence.
    pub regularization: f32,
    /// Weight leakage factor $\alpha \in [0.999, 1.0]$ preventing tap weight drift.
    pub leakage_factor: f32,
    /// Geigel Double-Talk Detection threshold ratio $\gamma_{\text{dtd}}$.
    pub dtd_threshold: f32,
    /// Hangover duration in samples to sustain weight freezing after double-talk trigger.
    pub dtd_hangover_samples: usize,
    /// Minimum reference signal energy required to engage adaptation.
    pub min_reference_energy: f32,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            filter_length: 256,
            step_size: 0.25,
            regularization: 1e-4,
            leakage_factor: 0.99995,
            dtd_threshold: 0.75,
            dtd_hangover_samples: 160,
            min_reference_energy: 1e-4,
        }
    }
}

/// Operational state of the Double-Talk Detector (DTD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DtdState {
    /// Both reference playback and microphone inputs are in acoustic silence.
    Silence,
    /// Only far-end loudspeaker playback echo is present; adaptive filter updates actively.
    EchoOnly,
    /// Near-end user speech is detected simultaneously with playback; filter weights are frozen.
    DoubleTalk,
}

/// Normalized Least Mean Squares (NLMS) Adaptive Acoustic Echo Canceller with Geigel DTD.
#[derive(Debug, Clone)]
pub struct AcousticEchoCanceller {
    config: AecConfig,
    weights: Vec<f32>,
    ref_history: Vec<f32>,
    ref_head: usize,
    ref_energy: f32,
    dtd_hangover: usize,
    current_dtd_state: DtdState,
    smoothed_mic_power: f32,
    smoothed_err_power: f32,
    power_smoothing_alpha: f32,
}

impl AcousticEchoCanceller {
    /// Construct a new AcousticEchoCanceller with specified configuration.
    pub fn new(config: AecConfig) -> Self {
        assert!(config.filter_length > 0, "Filter length must be positive");
        assert!(
            config.step_size > 0.0 && config.step_size <= 1.0,
            "Step size must be in (0.0, 1.0]"
        );

        let filter_length = config.filter_length;
        Self {
            config,
            weights: vec![0.0; filter_length],
            ref_history: vec![0.0; filter_length * 2],
            ref_head: 0,
            ref_energy: 0.0,
            dtd_hangover: 0,
            current_dtd_state: DtdState::Silence,
            smoothed_mic_power: 1e-6,
            smoothed_err_power: 1e-6,
            power_smoothing_alpha: 0.995,
        }
    }

    /// Reset internal filter weights and reference signal history.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.ref_history.fill(0.0);
        self.ref_head = 0;
        self.ref_energy = 0.0;
        self.dtd_hangover = 0;
        self.current_dtd_state = DtdState::Silence;
        self.smoothed_mic_power = 1e-6;
        self.smoothed_err_power = 1e-6;
    }

    /// Process a single audio sample pair: microphone input `mic_sample` and loudspeaker playback `ref_sample`.
    /// Returns the echo-cancelled clean output sample.
    pub fn process_sample(&mut self, mic_sample: f32, ref_sample: f32) -> f32 {
        let l = self.config.filter_length;

        // 1. Contiguous double-buffering for SIMD vectorized dot product
        let oldest_ref = self.ref_history[self.ref_head];
        self.ref_history[self.ref_head] = ref_sample;
        self.ref_history[self.ref_head + l] = ref_sample;
        let start_idx = self.ref_head + 1;
        self.ref_head = if self.ref_head + 1 == l { 0 } else { self.ref_head + 1 };

        // Bounded recursive energy tracking
        let new_energy = self.ref_energy + ref_sample * ref_sample - oldest_ref * oldest_ref;
        self.ref_energy = new_energy.max(0.0);

        let ref_slice = &self.ref_history[start_idx..start_idx + l];

        // 2. Vectorized echo prediction dot-product
        let mut estimated_echo = 0.0f32;
        for (w, &x_val) in self.weights.iter().zip(ref_slice.iter()) {
            estimated_echo += w * x_val;
        }

        // Peak reference magnitude tracking for Geigel DTD
        let mut max_ref_mag = 0.0f32;
        for &x_val in ref_slice {
            max_ref_mag = max_ref_mag.max(x_val.abs());
        }

        // 3. Compute error signal e[n] = mic_sample - estimated_echo
        let err = mic_sample - estimated_echo;

        // 4. Geigel Double-Talk Detection (DTD)
        let mic_mag = mic_sample.abs();
        let ref_is_active = max_ref_mag > 0.005;

        if !ref_is_active && mic_mag < 0.005 {
            self.current_dtd_state = DtdState::Silence;
        } else if ref_is_active && mic_mag > self.config.dtd_threshold * max_ref_mag {
            self.current_dtd_state = DtdState::DoubleTalk;
            self.dtd_hangover = self.config.dtd_hangover_samples;
        } else if ref_is_active {
            if self.dtd_hangover > 0 {
                self.current_dtd_state = DtdState::DoubleTalk;
            } else {
                self.current_dtd_state = DtdState::EchoOnly;
            }
        } else {
            self.current_dtd_state = DtdState::Silence;
        }

        if self.dtd_hangover > 0 {
            self.dtd_hangover -= 1;
        }

        // 5. Adapt filter weights if not in double-talk and reference energy is sufficient
        let adapt_allowed = self.current_dtd_state == DtdState::EchoOnly
            && self.ref_energy >= self.config.min_reference_energy;

        if adapt_allowed {
            let normalized_step = self.config.step_size / (self.ref_energy + self.config.regularization);
            let delta = normalized_step * err;
            let leakage = self.config.leakage_factor;

            for (w, &x_val) in self.weights.iter_mut().zip(ref_slice.iter()) {
                *w = *w * leakage + delta * x_val;
            }
        }

        // 6. Update smoothed powers for telemetry
        let alpha = self.power_smoothing_alpha;
        self.smoothed_mic_power = alpha * self.smoothed_mic_power + (1.0 - alpha) * (mic_sample * mic_sample);
        self.smoothed_err_power = alpha * self.smoothed_err_power + (1.0 - alpha) * (err * err);

        err
    }

    /// Process continuous audio blocks for microphone and reference playback in batch.
    pub fn process_block(&mut self, mic: &[f32], reference: &[f32], clean_out: &mut [f32]) {
        assert_eq!(
            mic.len(),
            reference.len(),
            "Microphone and reference blocks must have equal length"
        );
        assert_eq!(
            mic.len(),
            clean_out.len(),
            "Output block length must match input block length"
        );

        for i in 0..mic.len() {
            clean_out[i] = self.process_sample(mic[i], reference[i]);
        }
    }

    /// Return the current Echo Return Loss Enhancement (ERLE) measurement in decibels (dB).
    pub fn erle_db(&self) -> f32 {
        let erle_ratio = (self.smoothed_mic_power + 1e-8) / (self.smoothed_err_power + 1e-8);
        (10.0 * erle_ratio.log10()).max(0.0)
    }

    /// Return current Double-Talk Detector state.
    pub fn dtd_state(&self) -> DtdState {
        self.current_dtd_state
    }

    /// Returns true if Double-Talk (near-end user speaking over speaker playback) is active.
    pub fn is_double_talk(&self) -> bool {
        self.current_dtd_state == DtdState::DoubleTalk
    }

    /// Access reference to current adaptive FIR filter tap weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Access reference to active AEC configuration.
    pub fn config(&self) -> &AecConfig {
        &self.config
    }
}

/// Configuration parameters for Partitioned-Block Frequency-Domain Adaptive Filtering (PBFDAF).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PbfdafConfig {
    /// Processing block size $B$ in samples (power of two, e.g. 64 or 128).
    pub block_size: usize,
    /// Number of partitions $P$ (total filter length $L = P \cdot B$, e.g. 16 or 32).
    pub num_partitions: usize,
    /// Frequency-domain LMS adaptation step size $\mu \in (0.0, 1.0]$.
    pub step_size: f32,
    /// Regularization $\epsilon$ for power normalization to avoid division by zero.
    pub regularization: f32,
    /// Weight leakage factor $\lambda \in [0.999, 1.0]$ to prevent drift.
    pub leakage_factor: f32,
    /// Power spectral density smoothing coefficient $\alpha \in [0.7, 0.99]$.
    pub power_smoothing: f32,
    /// Whether to apply time-domain overlap-save constraint on the gradient ($g_p[B..2B] = 0$).
    pub enable_gradient_constraint: bool,
}

impl Default for PbfdafConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_partitions: 16,
            step_size: 0.35,
            regularization: 1e-4,
            leakage_factor: 0.99995,
            power_smoothing: 0.85,
            enable_gradient_constraint: true,
        }
    }
}

/// Configuration for Magnitude Squared Coherence (MSC) Double-Talk Detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtdConfig {
    /// Average magnitude-squared coherence threshold $\theta_{\text{dtd}}$ below which double-talk is flagged.
    pub coherence_threshold: f32,
    /// Minimum reference signal power to engage coherence evaluation.
    pub min_ref_power: f32,
    /// Minimum microphone signal power to consider active.
    pub min_mic_power: f32,
    /// Hangover duration in blocks to sustain frozen filter adaptation after double-talk.
    pub hangover_blocks: usize,
    /// Exponential smoothing factor $\alpha$ for cross-power and auto-power spectral estimation.
    pub smoothing_alpha: f32,
}

impl Default for DtdConfig {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.45,
            min_ref_power: 1e-4,
            min_mic_power: 1e-4,
            hangover_blocks: 6,
            smoothing_alpha: 0.80,
        }
    }
}

/// Configuration for Residual Echo Suppressor (RES) post-filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResConfig {
    /// Enable residual echo suppression post-filter.
    pub enabled: bool,
    /// Over-subtraction suppression factor $\eta \ge 1.0$.
    pub suppression_factor: f32,
    /// Minimum gain floor $G_{\min}$ in linear scale (e.g. 0.05 for -26 dB attenuation).
    pub min_gain: f32,
    /// Leakage factor mapping estimated echo to residual echo.
    pub echo_leakage_ratio: f32,
}

impl Default for ResConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            suppression_factor: 1.5,
            min_gain: 0.05,
            echo_leakage_ratio: 0.20,
        }
    }
}

/// Comprehensive Subband AEC Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubbandAecConfig {
    pub pbfdaf: PbfdafConfig,
    pub dtd: DtdConfig,
    pub res: ResConfig,
    pub sample_rate: f32,
}

impl Default for SubbandAecConfig {
    fn default() -> Self {
        Self {
            pbfdaf: PbfdafConfig::default(),
            dtd: DtdConfig::default(),
            res: ResConfig::default(),
            sample_rate: 16000.0,
        }
    }
}

/// Diagnostic telemetry snapshot from the Subband AEC engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubbandAecTelemetry {
    /// Current Echo Return Loss Enhancement (ERLE) in dB.
    pub erle_db: f32,
    /// Magnitude-squared coherence between reference playback and microphone input.
    pub coherence: f32,
    /// Double-Talk Detector state.
    pub dtd_state: DtdState,
    /// True if double-talk was detected in the latest processing block.
    pub is_double_talk: bool,
}

/// Magnitude Squared Coherence (MSC) Double-Talk Detector.
#[derive(Debug, Clone)]
pub struct CoherenceDtd {
    config: DtdConfig,
    s_xx: Vec<f32>,
    s_dd: Vec<f32>,
    s_xd_real: Vec<f32>,
    s_xd_imag: Vec<f32>,
    avg_coherence: f32,
    hangover: usize,
    state: DtdState,
}

impl CoherenceDtd {
    pub fn new(config: DtdConfig, num_bins: usize) -> Self {
        Self {
            config,
            s_xx: vec![1e-5; num_bins],
            s_dd: vec![1e-5; num_bins],
            s_xd_real: vec![0.0; num_bins],
            s_xd_imag: vec![0.0; num_bins],
            avg_coherence: 0.0,
            hangover: 0,
            state: DtdState::Silence,
        }
    }

    pub fn reset(&mut self) {
        self.s_xx.fill(1e-5);
        self.s_dd.fill(1e-5);
        self.s_xd_real.fill(0.0);
        self.s_xd_imag.fill(0.0);
        self.avg_coherence = 0.0;
        self.hangover = 0;
        self.state = DtdState::Silence;
    }

    pub fn update(
        &mut self,
        x_real: &[f32],
        x_imag: &[f32],
        d_real: &[f32],
        d_imag: &[f32],
    ) {
        let n_bins = self.s_xx.len();
        let alpha = self.config.smoothing_alpha;
        let one_minus_alpha = 1.0 - alpha;

        let mut sum_coherence = 0.0f32;
        let mut active_bins = 0usize;
        let mut total_ref_power = 0.0f32;
        let mut total_mic_power = 0.0f32;

        for k in 0..n_bins {
            let xr = x_real[k];
            let xi = x_imag[k];
            let dr = d_real[k];
            let di = d_imag[k];

            let px = xr * xr + xi * xi;
            let pd = dr * dr + di * di;
            total_ref_power += px;
            total_mic_power += pd;

            let cross_r = xr * dr + xi * di;
            let cross_i = xr * di - xi * dr;

            self.s_xx[k] = alpha * self.s_xx[k] + one_minus_alpha * px;
            self.s_dd[k] = alpha * self.s_dd[k] + one_minus_alpha * pd;
            self.s_xd_real[k] = alpha * self.s_xd_real[k] + one_minus_alpha * cross_r;
            self.s_xd_imag[k] = alpha * self.s_xd_imag[k] + one_minus_alpha * cross_i;

            let num = self.s_xd_real[k] * self.s_xd_real[k] + self.s_xd_imag[k] * self.s_xd_imag[k];
            let den = (self.s_xx[k] * self.s_dd[k]).max(1e-12);
            let coh = (num / den).clamp(0.0, 1.0);

            if self.s_xx[k] > self.config.min_ref_power {
                sum_coherence += coh;
                active_bins += 1;
            }
        }

        let mean_ref = total_ref_power / (n_bins as f32);
        let mean_mic = total_mic_power / (n_bins as f32);

        self.avg_coherence = if active_bins > 0 {
            sum_coherence / (active_bins as f32)
        } else {
            0.0
        };

        let ref_active = mean_ref >= self.config.min_ref_power;
        let mic_active = mean_mic >= self.config.min_mic_power;

        if !ref_active && !mic_active {
            self.state = DtdState::Silence;
        } else if ref_active {
            if mic_active && self.avg_coherence < self.config.coherence_threshold {
                self.state = DtdState::DoubleTalk;
                self.hangover = self.config.hangover_blocks;
            } else if self.hangover > 0 {
                self.hangover -= 1;
                self.state = DtdState::DoubleTalk;
            } else {
                self.state = DtdState::EchoOnly;
            }
        } else {
            self.state = DtdState::Silence;
        }
    }

    pub fn state(&self) -> DtdState {
        self.state
    }

    pub fn avg_coherence(&self) -> f32 {
        self.avg_coherence
    }
}

/// Parametric Residual Echo Suppressor (RES) applying Wiener/spectral gain masking.
#[derive(Debug, Clone)]
pub struct ResidualEchoSuppressor {
    config: ResConfig,
    smoothed_echo_psd: Vec<f32>,
    smoothed_err_psd: Vec<f32>,
}

impl ResidualEchoSuppressor {
    pub fn new(config: ResConfig, num_bins: usize) -> Self {
        Self {
            config,
            smoothed_echo_psd: vec![1e-5; num_bins],
            smoothed_err_psd: vec![1e-5; num_bins],
        }
    }

    pub fn reset(&mut self) {
        self.smoothed_echo_psd.fill(1e-5);
        self.smoothed_err_psd.fill(1e-5);
    }

    pub fn suppress(
        &mut self,
        e_real: &mut [f32],
        e_imag: &mut [f32],
        y_real: &[f32],
        y_imag: &[f32],
    ) {
        if !self.config.enabled {
            return;
        }

        let n = e_real.len();
        let half = n / 2 + 1;
        let alpha = 0.85f32;
        let one_minus_alpha = 0.15f32;

        for k in 0..half {
            let p_echo = y_real[k] * y_real[k] + y_imag[k] * y_imag[k];
            let p_err = e_real[k] * e_real[k] + e_imag[k] * e_imag[k];

            self.smoothed_echo_psd[k] = alpha * self.smoothed_echo_psd[k] + one_minus_alpha * p_echo;
            self.smoothed_err_psd[k] = alpha * self.smoothed_err_psd[k] + one_minus_alpha * p_err;

            let p_res = self.smoothed_echo_psd[k] * self.config.echo_leakage_ratio;
            let ratio = (p_res / (self.smoothed_err_psd[k] + 1e-10)).sqrt();
            let raw_gain = 1.0 - self.config.suppression_factor * ratio;
            let gain = if raw_gain.is_finite() {
                raw_gain.clamp(self.config.min_gain, 1.0)
            } else {
                1.0
            };

            if !gain.is_finite() {
                panic!("gain is NaN! ratio={}, p_res={}, p_err={}, p_echo={}", ratio, p_res, self.smoothed_err_psd[k], p_echo);
            }

            e_real[k] *= gain;
            e_imag[k] *= gain;

            if k > 0 && k < n / 2 {
                let sym = n - k;
                e_real[sym] *= gain;
                e_imag[sym] *= gain;
            }
        }
    }
}

/// Partitioned-Block Frequency-Domain Adaptive Filter (PBFDAF) with Coherence DTD and RES.
#[derive(Debug, Clone)]
pub struct SubbandAec {
    config: SubbandAecConfig,
    fft: FftProcessor,
    block_size: usize,
    fft_size: usize,
    num_partitions: usize,

    weights_real: Vec<Vec<f32>>,
    weights_imag: Vec<Vec<f32>>,

    x_history_real: Vec<Vec<f32>>,
    x_history_imag: Vec<Vec<f32>>,
    history_head: usize,

    p_x: Vec<f32>,

    dtd: CoherenceDtd,
    res: ResidualEchoSuppressor,

    prev_ref_block: Vec<f32>,
    in_mic_buf: Vec<f32>,
    in_ref_buf: Vec<f32>,
    out_clean_buf: Vec<f32>,

    scratch_x_time: Vec<f32>,
    scratch_x_real: Vec<f32>,
    scratch_x_imag: Vec<f32>,
    scratch_y_real: Vec<f32>,
    scratch_y_imag: Vec<f32>,
    scratch_e_time: Vec<f32>,
    scratch_e_real: Vec<f32>,
    scratch_e_imag: Vec<f32>,
    scratch_d_time: Vec<f32>,
    scratch_d_real: Vec<f32>,
    scratch_d_imag: Vec<f32>,
    scratch_g_real: Vec<f32>,
    scratch_g_imag: Vec<f32>,
    scratch_temp_r: Vec<f32>,
    scratch_temp_i: Vec<f32>,

    smoothed_mic_energy: f32,
    smoothed_err_energy: f32,
}

impl SubbandAec {
    /// Construct a new SubbandAec engine with specified configuration.
    pub fn new(config: SubbandAecConfig) -> Self {
        let b = config.pbfdaf.block_size;
        assert!(b > 0 && (b & (b - 1)) == 0, "Block size must be power of two");
        let fft_size = b * 2;
        let num_p = config.pbfdaf.num_partitions;
        assert!(num_p > 0, "Number of partitions must be > 0");

        let fft = FftProcessor::new(fft_size);
        let weights_real = vec![vec![0.0f32; fft_size]; num_p];
        let weights_imag = vec![vec![0.0f32; fft_size]; num_p];

        let x_history_real = vec![vec![0.0f32; fft_size]; num_p];
        let x_history_imag = vec![vec![0.0f32; fft_size]; num_p];

        let p_x = vec![config.pbfdaf.regularization; fft_size];

        let num_bins = fft_size / 2 + 1;
        let dtd = CoherenceDtd::new(config.dtd.clone(), num_bins);
        let res = ResidualEchoSuppressor::new(config.res.clone(), num_bins);

        Self {
            config,
            fft,
            block_size: b,
            fft_size,
            num_partitions: num_p,
            weights_real,
            weights_imag,
            x_history_real,
            x_history_imag,
            history_head: 0,
            p_x,
            dtd,
            res,
            prev_ref_block: vec![0.0f32; b],
            in_mic_buf: Vec::with_capacity(b * 4),
            in_ref_buf: Vec::with_capacity(b * 4),
            out_clean_buf: Vec::with_capacity(b * 4),
            scratch_x_time: vec![0.0f32; fft_size],
            scratch_x_real: vec![0.0f32; fft_size],
            scratch_x_imag: vec![0.0f32; fft_size],
            scratch_y_real: vec![0.0f32; fft_size],
            scratch_y_imag: vec![0.0f32; fft_size],
            scratch_e_time: vec![0.0f32; fft_size],
            scratch_e_real: vec![0.0f32; fft_size],
            scratch_e_imag: vec![0.0f32; fft_size],
            scratch_d_time: vec![0.0f32; fft_size],
            scratch_d_real: vec![0.0f32; fft_size],
            scratch_d_imag: vec![0.0f32; fft_size],
            scratch_g_real: vec![0.0f32; fft_size],
            scratch_g_imag: vec![0.0f32; fft_size],
            scratch_temp_r: vec![0.0f32; fft_size],
            scratch_temp_i: vec![0.0f32; fft_size],
            smoothed_mic_energy: 1e-6,
            smoothed_err_energy: 1e-6,
        }
    }

    /// Reset internal filter weights, delay lines, and FIFO buffers.
    pub fn reset(&mut self) {
        for w in &mut self.weights_real {
            w.fill(0.0);
        }
        for w in &mut self.weights_imag {
            w.fill(0.0);
        }
        for h in &mut self.x_history_real {
            h.fill(0.0);
        }
        for h in &mut self.x_history_imag {
            h.fill(0.0);
        }
        self.history_head = 0;
        self.p_x.fill(self.config.pbfdaf.regularization);
        self.dtd.reset();
        self.res.reset();
        self.prev_ref_block.fill(0.0);
        self.in_mic_buf.clear();
        self.in_ref_buf.clear();
        self.out_clean_buf.clear();
        self.smoothed_mic_energy = 1e-6;
        self.smoothed_err_energy = 1e-6;
    }

    /// Process an individual partitioned block of exactly `block_size` samples.
    fn process_pbfdaf_block(
        &mut self,
        mic_block: &[f32],
        ref_block: &[f32],
        out_clean_block: &mut [f32],
    ) {
        let b = self.block_size;
        let fft_sz = self.fft_size;

        // 1. Overlap-save reference vector: [prev_ref (B), curr_ref (B)]
        self.scratch_x_time[0..b].copy_from_slice(&self.prev_ref_block);
        self.scratch_x_time[b..fft_sz].copy_from_slice(ref_block);
        self.prev_ref_block.copy_from_slice(ref_block);

        // 2. Frequency domain reference X(m)
        self.scratch_x_real.copy_from_slice(&self.scratch_x_time);
        self.scratch_x_imag.fill(0.0);
        self.fft.fft_in_place(&mut self.scratch_x_real, &mut self.scratch_x_imag);

        // 3. Insert into partition history line
        self.x_history_real[self.history_head].copy_from_slice(&self.scratch_x_real);
        self.x_history_imag[self.history_head].copy_from_slice(&self.scratch_x_imag);

        // 4. Update power normalization per bin
        let alpha = self.config.pbfdaf.power_smoothing;
        let one_minus_alpha = 1.0 - alpha;
        for k in 0..fft_sz {
            let p_curr = self.scratch_x_real[k] * self.scratch_x_real[k]
                + self.scratch_x_imag[k] * self.scratch_x_imag[k];
            self.p_x[k] = alpha * self.p_x[k] + one_minus_alpha * p_curr;
        }

        // 5. Accumulate estimated echo Y = sum_p (W_p * X_p)
        self.scratch_y_real.fill(0.0);
        self.scratch_y_imag.fill(0.0);
        for p in 0..self.num_partitions {
            let h_idx = (self.history_head + self.num_partitions - p) % self.num_partitions;
            let wr = &self.weights_real[p];
            let wi = &self.weights_imag[p];
            let xr = &self.x_history_real[h_idx];
            let xi = &self.x_history_imag[h_idx];

            for k in 0..fft_sz {
                self.scratch_y_real[k] += wr[k] * xr[k] - wi[k] * xi[k];
                self.scratch_y_imag[k] += wr[k] * xi[k] + wi[k] * xr[k];
            }
        }

        // 6. Time domain estimated echo via IFFT
        self.scratch_temp_r.copy_from_slice(&self.scratch_y_real);
        self.scratch_temp_i.copy_from_slice(&self.scratch_y_imag);
        self.fft.ifft_in_place(&mut self.scratch_temp_r, &mut self.scratch_temp_i);

        // 7. Time domain linear error e[n] = mic[n] - y[n] in the second half
        for i in 0..b {
            let echo_est = self.scratch_temp_r[b + i];
            let err = mic_block[i] - echo_est;
            out_clean_block[i] = err;
        }

        // 8. Microphone spectrum for DTD
        self.scratch_d_time[0..b].fill(0.0);
        self.scratch_d_time[b..fft_sz].copy_from_slice(mic_block);
        self.scratch_d_real.copy_from_slice(&self.scratch_d_time);
        self.scratch_d_imag.fill(0.0);
        self.fft.fft_in_place(&mut self.scratch_d_real, &mut self.scratch_d_imag);

        self.dtd.update(
            &self.scratch_x_real,
            &self.scratch_x_imag,
            &self.scratch_d_real,
            &self.scratch_d_imag,
        );

        let ref_energy = ref_block.iter().map(|s| s * s).sum::<f32>() / (b as f32);
        let adapt_allowed = (self.dtd.state() == DtdState::EchoOnly)
            && (ref_energy >= self.config.dtd.min_ref_power);

        // 9. Frequency domain error E(m) with zero-padded first half
        self.scratch_e_time[0..b].fill(0.0);
        self.scratch_e_time[b..fft_sz].copy_from_slice(&out_clean_block[..b]);
        self.scratch_e_real.copy_from_slice(&self.scratch_e_time);
        self.scratch_e_imag.fill(0.0);
        self.fft.fft_in_place(&mut self.scratch_e_real, &mut self.scratch_e_imag);

        // 10. Weight adaptation across partitions using unmodified linear error
        if adapt_allowed {
            let mu = self.config.pbfdaf.step_size / (self.num_partitions as f32);
            let lambda = self.config.pbfdaf.leakage_factor;
            let reg = self.config.pbfdaf.regularization;
            let mean_p: f32 = self.p_x.iter().sum::<f32>() / (fft_sz as f32);
            let floor_p = (0.01 * mean_p).max(reg);

            for p in 0..self.num_partitions {
                let h_idx = (self.history_head + self.num_partitions - p) % self.num_partitions;
                let xr = &self.x_history_real[h_idx];
                let xi = &self.x_history_imag[h_idx];

                for k in 0..fft_sz {
                    let er = self.scratch_e_real[k];
                    let ei = self.scratch_e_imag[k];
                    let norm = self.p_x[k].max(floor_p);

                    // Conjugate(X) * E = (xr - j xi) * (er + j ei)
                    let gr = (xr[k] * er + xi[k] * ei) / norm;
                    let gi = (xr[k] * ei - xi[k] * er) / norm;
                    self.scratch_g_real[k] = gr;
                    self.scratch_g_imag[k] = gi;
                }

                if self.config.pbfdaf.enable_gradient_constraint {
                    self.fft.ifft_in_place(&mut self.scratch_g_real, &mut self.scratch_g_imag);
                    self.scratch_g_real[b..fft_sz].fill(0.0);
                    self.scratch_g_imag[b..fft_sz].fill(0.0);
                    self.fft.fft_in_place(&mut self.scratch_g_real, &mut self.scratch_g_imag);
                }

                for k in 0..fft_sz {
                    self.weights_real[p][k] = lambda * self.weights_real[p][k] + mu * self.scratch_g_real[k];
                    self.weights_imag[p][k] = lambda * self.weights_imag[p][k] + mu * self.scratch_g_imag[k];
                }
            }
        }

        // 11. Optional Residual Echo Suppression (RES) post-filter (bypassed during double-talk)
        if self.config.res.enabled && self.dtd.state() != DtdState::DoubleTalk {
            self.scratch_temp_r.copy_from_slice(&self.scratch_e_real);
            self.scratch_temp_i.copy_from_slice(&self.scratch_e_imag);
            self.res.suppress(
                &mut self.scratch_temp_r,
                &mut self.scratch_temp_i,
                &self.scratch_y_real,
                &self.scratch_y_imag,
            );
            self.fft.ifft_in_place(&mut self.scratch_temp_r, &mut self.scratch_temp_i);
            out_clean_block.copy_from_slice(&self.scratch_temp_r[b..fft_sz]);
        }

        // 12. Advance history pointer
        self.history_head = (self.history_head + 1) % self.num_partitions;

        // 13. Telemetry energy tracking
        let mic_energy = mic_block.iter().map(|s| s * s).sum::<f32>() / (b as f32);
        let err_energy = out_clean_block.iter().map(|s| s * s).sum::<f32>() / (b as f32);
        self.smoothed_mic_energy = 0.95 * self.smoothed_mic_energy + 0.05 * mic_energy;
        self.smoothed_err_energy = 0.95 * self.smoothed_err_energy + 0.05 * err_energy;
    }

    /// Process streaming continuous audio blocks of arbitrary length for microphone and playback reference.
    pub fn process_block(&mut self, mic: &[f32], reference: &[f32], clean_out: &mut [f32]) {
        assert_eq!(
            mic.len(),
            reference.len(),
            "Microphone and reference inputs must have identical length"
        );
        assert_eq!(
            mic.len(),
            clean_out.len(),
            "Output buffer length must match input length"
        );

        let n = mic.len();
        if n == 0 {
            return;
        }

        self.in_mic_buf.extend_from_slice(mic);
        self.in_ref_buf.extend_from_slice(reference);

        let b = self.block_size;
        let mut clean_chunk = vec![0.0f32; b];

        while self.in_mic_buf.len() >= b && self.in_ref_buf.len() >= b {
            let mic_chunk: Vec<f32> = self.in_mic_buf.drain(..b).collect();
            let ref_chunk: Vec<f32> = self.in_ref_buf.drain(..b).collect();

            self.process_pbfdaf_block(&mic_chunk, &ref_chunk, &mut clean_chunk);
            self.out_clean_buf.extend_from_slice(&clean_chunk);
        }

        let ready_len = self.out_clean_buf.len().min(n);
        clean_out[..ready_len].copy_from_slice(&self.out_clean_buf[..ready_len]);
        self.out_clean_buf.drain(..ready_len);

        if ready_len < n {
            // Passthrough for the remaining initial delay frames
            clean_out[ready_len..n].copy_from_slice(&mic[ready_len..n]);
        }
    }

    /// Process a single audio sample pair: microphone input `mic_sample` and loudspeaker playback `ref_sample`.
    pub fn process_sample(&mut self, mic_sample: f32, ref_sample: f32) -> f32 {
        let mut out = [0.0f32; 1];
        self.process_block(&[mic_sample], &[ref_sample], &mut out);
        out[0]
    }

    /// Return the current Echo Return Loss Enhancement (ERLE) in dB.
    pub fn erle_db(&self) -> f32 {
        let ratio = (self.smoothed_mic_energy + 1e-8) / (self.smoothed_err_energy + 1e-8);
        (10.0 * ratio.log10()).max(0.0)
    }

    /// Return average magnitude-squared coherence between reference and microphone signals.
    pub fn coherence(&self) -> f32 {
        self.dtd.avg_coherence()
    }

    /// Return active Double-Talk Detector state.
    pub fn dtd_state(&self) -> DtdState {
        self.dtd.state()
    }

    /// Return true if Double-Talk (operator speaking over synthetic speech playback) is detected.
    pub fn is_double_talk(&self) -> bool {
        self.dtd.state() == DtdState::DoubleTalk
    }

    /// Return telemetry snapshot.
    pub fn telemetry(&self) -> SubbandAecTelemetry {
        SubbandAecTelemetry {
            erle_db: self.erle_db(),
            coherence: self.coherence(),
            dtd_state: self.dtd_state(),
            is_double_talk: self.is_double_talk(),
        }
    }

    /// Access reference to active configuration.
    pub fn config(&self) -> &SubbandAecConfig {
        &self.config
    }
}
