//! Adaptive Acoustic Echo Cancellation (AEC), Normalized Least Mean Squares (NLMS) transversal filter,
//! Geigel Double-Talk Detection (DTD), and Echo Return Loss Enhancement (ERLE) telemetry.

#![deny(unsafe_code)]

/// Configuration parameters for Acoustic Echo Cancellation.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
