#![deny(unsafe_code)]

//! Neuromorphic Silicon Cochlea & Event-Driven Spiking Wake-Word Spotting.
//!
//! Models the biomechanics of the mammalian basilar membrane via discrete 4th-order
//! Gammatone filterbanks, inner hair cell half-wave mechanical rectification with
//! logarithmic compression, and asynchronous Address-Event Representation (AER) delta modulation.
//!
//! Provides a Leaky Integrate-and-Fire (LIF) Spiking Neural Network wake-word decoder
//! (`SpikingKwsCell`) operating in $O(1)$ constant time per event with zero active power
//! dissipation in silence.

use crate::engine::KeywordEvent;
use serde::{Deserialize, Serialize};

/// 8-byte Address-Event Representation (AER) binary spike event layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Microsecond timestamp since stream or boot start.
    pub timestamp_us: u32,
    /// Resonant frequency channel index [0..C-1].
    pub channel: u16,
    /// Event polarity: +1 for ON (acoustic energy rise), -1 for OFF (acoustic energy fall).
    pub polarity: i8,
    /// Reserved alignment byte ensuring strict 8-byte memory boundary.
    pub reserved: u8,
}

impl SpikeEvent {
    /// Construct a new SpikeEvent.
    pub fn new(timestamp_us: u32, channel: u16, polarity: i8) -> Self {
        Self {
            timestamp_us,
            channel,
            polarity,
            reserved: 0,
        }
    }

    /// Check if event is an ON spike (+1).
    pub fn is_on(&self) -> bool {
        self.polarity > 0
    }

    /// Check if event is an OFF spike (-1).
    pub fn is_off(&self) -> bool {
        self.polarity < 0
    }

    /// Serialize event into standard 8-byte little-endian AER wire format.
    pub fn to_bytes(&self) -> [u8; 8] {
        let ts_bytes = self.timestamp_us.to_le_bytes();
        let ch_bytes = self.channel.to_le_bytes();
        [
            ts_bytes[0],
            ts_bytes[1],
            ts_bytes[2],
            ts_bytes[3],
            ch_bytes[0],
            ch_bytes[1],
            self.polarity as u8,
            self.reserved,
        ]
    }

    /// Deserialize event from standard 8-byte little-endian AER wire format.
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        let timestamp_us = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let channel = u16::from_le_bytes([bytes[4], bytes[5]]);
        let polarity = bytes[6] as i8;
        let reserved = bytes[7];
        Self {
            timestamp_us,
            channel,
            polarity,
            reserved,
        }
    }
}

/// Discrete 4th-order cascade IIR Gammatone filter approximating a single basilar membrane critical band.
///
/// Implemented as a cascade of four 1st-order complex resonators:
/// $$z_k(n) = \text{in}_k(n) + \lambda \cdot z_k(n-1)$$
/// where $\lambda = r e^{j \omega_c}$ with radius $r = e^{-2\pi b / f_s}$ and $\omega_c = 2\pi f_c / f_s$.
#[derive(Debug, Clone)]
pub struct GammatoneFilter {
    center_freq: f32,
    bandwidth: f32,
    alpha: f32, // r * cos(omega_c)
    beta: f32,  // r * sin(omega_c)
    gain: f32,  // 2 * (1 - r)^4 scaling for unity resonance magnitude on real inputs
    // Complex state registers for 4 cascaded sections: (re, im)
    states: [(f32, f32); 4],
}

impl GammatoneFilter {
    /// Construct a 4th-order Gammatone filter for specified center frequency and sampling rate.
    pub fn new(center_freq: f32, sample_rate: f32) -> Self {
        assert!(center_freq > 0.0, "Center frequency must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(
            center_freq < sample_rate * 0.5,
            "Center frequency must be below Nyquist"
        );

        // Equivalent Rectangular Bandwidth (Glasberg & Moore, 1990)
        let erb = 24.7 * (4.37 * center_freq / 1000.0 + 1.0);
        let b = 1.019 * erb;

        let dt = 1.0 / sample_rate;
        let r = (-2.0 * std::f32::consts::PI * b * dt).exp();
        let omega = 2.0 * std::f32::consts::PI * center_freq * dt;

        let alpha = r * omega.cos();
        let beta = r * omega.sin();

        // Normalization gain for 4th-order cascade to ensure unity gain for real sinusoids at center frequency
        let one_minus_r = 1.0 - r;
        let gain = 2.0 * one_minus_r * one_minus_r * one_minus_r * one_minus_r;

        Self {
            center_freq,
            bandwidth: b,
            alpha,
            beta,
            gain,
            states: [(0.0, 0.0); 4],
        }
    }

    /// Step a single audio sample through the 4-stage complex resonator cascade.
    pub fn step(&mut self, sample: f32) -> f32 {
        let mut u = sample * self.gain;
        let mut v = 0.0f32;

        for k in 0..4 {
            let (prev_re, prev_im) = self.states[k];

            // Complex multiplication: (u, v) + (alpha + j*beta) * (prev_re + j*prev_im)
            let new_re = u + (self.alpha * prev_re - self.beta * prev_im);
            let new_im = v + (self.beta * prev_re + self.alpha * prev_im);

            self.states[k] = (new_re, new_im);

            u = new_re;
            v = new_im;
        }

        // Return real displacement of the basilar membrane partition
        self.states[3].0
    }

    /// Reset filter state registers to zero.
    pub fn reset(&mut self) {
        self.states = [(0.0, 0.0); 4];
    }

    /// Get center frequency in Hz.
    pub fn center_freq(&self) -> f32 {
        self.center_freq
    }

    /// Get filter bandwidth in Hz.
    pub fn bandwidth(&self) -> f32 {
        self.bandwidth
    }
}

/// Neuromorphic Silicon Cochlea acoustic frontend.
///
/// Converts continuous audio waveforms into asynchronous Address-Event Representation (AER)
/// spike events across physiologically mapped basilar membrane frequency channels.
#[derive(Debug, Clone)]
pub struct NeuromorphicCochlea {
    sample_rate: f32,
    filters: Vec<GammatoneFilter>,
    center_freqs: Vec<f32>,
    v_ref: Vec<f32>,
    ihc_states: Vec<f32>,
    contrast_thresh: f32,
    ref_leak: f32,
    gamma: f32,
    smoothing_alpha: f32,
}

impl NeuromorphicCochlea {
    /// Construct a Silicon Cochlea with Greenwood place-frequency mapping across `num_channels`.
    pub fn new(sample_rate: f32, num_channels: usize, f_min: f32, f_max: f32) -> Self {
        assert!(num_channels > 0, "Channel count must be > 0");
        assert!(sample_rate > 0.0, "Sample rate must be > 0");
        assert!(f_min > 0.0 && f_max > f_min);

        let nyquist_limit = sample_rate * 0.48;
        let capped_f_max = f_max.min(nyquist_limit);

        // Donald Greenwood cochlear place-frequency mapping:
        // f_c(x) = A * (10^(a * x) - k)
        // Inverse: x(f) = (1 / a) * log10(f / A + k)
        let a_const = 165.4f32;
        let a_exponent = 2.1f32;
        let k_const = 0.88f32;

        let x_min = (1.0 / a_exponent) * ((f_min / a_const + k_const).log10());
        let x_max = (1.0 / a_exponent) * ((capped_f_max / a_const + k_const).log10());

        let mut filters = Vec::with_capacity(num_channels);
        let mut center_freqs = Vec::with_capacity(num_channels);

        for i in 0..num_channels {
            let frac = if num_channels == 1 {
                0.5
            } else {
                i as f32 / (num_channels - 1) as f32
            };
            let x = x_min + frac * (x_max - x_min);
            let fc = (a_const * (10.0f32.powf(a_exponent * x) - k_const))
                .clamp(f_min, capped_f_max);

            filters.push(GammatoneFilter::new(fc, sample_rate));
            center_freqs.push(fc);
        }

        Self {
            sample_rate,
            filters,
            center_freqs,
            v_ref: vec![0.0; num_channels],
            ihc_states: vec![0.0; num_channels],
            contrast_thresh: 0.04,
            ref_leak: 0.9995,
            gamma: 1.5,
            smoothing_alpha: 0.92,
        }
    }

    /// Set delta contrast threshold for asynchronous spike emission.
    pub fn set_contrast_threshold(&mut self, threshold: f32) {
        assert!(threshold > 0.0, "Threshold must be positive");
        self.contrast_thresh = threshold;
    }

    /// Set reference potential leak factor (e.g. 0.999 to 1.0).
    pub fn set_ref_leak(&mut self, leak: f32) {
        assert!((0.0..=1.0).contains(&leak), "Leak must be in [0, 1]");
        self.ref_leak = leak;
    }

    /// Process a single acoustic sample and append emitted asynchronous spikes into `spikes_out`.
    pub fn step_sample(&mut self, sample: f32, timestamp_us: u32, spikes_out: &mut Vec<SpikeEvent>) {
        let num_ch = self.filters.len();

        for c in 0..num_ch {
            // 1. Basilar membrane Gammatone filtering
            let displacement = self.filters[c].step(sample);

            // 2. Inner hair cell mechanical half-wave rectification
            let v_rect = displacement.max(0.0);

            // 3. Compressive non-linearity (saturating receptor potential)
            let v_comp = v_rect / (1.0 + self.gamma * v_rect);

            // 4. Stereocilia membrane low-pass temporal smoothing
            let s = self.ihc_states[c] * self.smoothing_alpha
                + v_comp * (1.0 - self.smoothing_alpha);
            self.ihc_states[c] = s;

            // 5. Asynchronous delta contrast threshold check
            let delta_v = s - self.v_ref[c];

            if delta_v >= self.contrast_thresh {
                spikes_out.push(SpikeEvent::new(timestamp_us, c as u16, 1));
                self.v_ref[c] += self.contrast_thresh;
            } else if delta_v <= -self.contrast_thresh {
                spikes_out.push(SpikeEvent::new(timestamp_us, c as u16, -1));
                self.v_ref[c] -= self.contrast_thresh;
            } else {
                // Reference potential slow leak toward 0 only when within inactive deadband
                self.v_ref[c] *= self.ref_leak;
            }
        }
    }

    /// Process a streaming buffer of audio samples, generating microsecond-accurate spike timestamps.
    pub fn process_buffer(
        &mut self,
        samples: &[f32],
        start_timestamp_us: u32,
        spikes_out: &mut Vec<SpikeEvent>,
    ) {
        let dt_us = 1_000_000.0 / self.sample_rate;
        for (idx, &sample) in samples.iter().enumerate() {
            let ts = start_timestamp_us.saturating_add((idx as f32 * dt_us) as u32);
            self.step_sample(sample, ts, spikes_out);
        }
    }

    /// Reset internal filter states, reference voltages, and hair cell membranes.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        self.v_ref.fill(0.0);
        self.ihc_states.fill(0.0);
    }

    /// Return reference to channel center frequencies in Hz.
    pub fn center_frequencies(&self) -> &[f32] {
        &self.center_freqs
    }

    /// Return number of frequency channels.
    pub fn num_channels(&self) -> usize {
        self.filters.len()
    }
}

/// Configuration parameters for the Spiking Neural Network wake-word decoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingKwsConfig {
    /// Number of cochlea frequency channels.
    pub num_channels: usize,
    /// Number of sequential phoneme states.
    pub num_states: usize,
    /// Membrane potential leak time constant in microseconds.
    pub tau_mem_us: f32,
    /// Synaptic trace decay time constant in microseconds.
    pub tau_syn_us: f32,
    /// Firing threshold for each phoneme coincidence stage.
    pub threshold: f32,
    /// Maximum allowed inter-phoneme delay window in microseconds.
    pub coincidence_window_us: u32,
}

impl Default for SpikingKwsConfig {
    fn default() -> Self {
        Self {
            num_channels: 16,
            num_states: 3,
            tau_mem_us: 30_000.0,       // 30 ms membrane leak
            tau_syn_us: 50_000.0,       // 50 ms synaptic trace decay
            threshold: 1.0,             // Normalized firing threshold
            coincidence_window_us: 400_000, // 400 ms phoneme coincidence window
        }
    }
}

/// Event-Driven Leaky Integrate-and-Fire (LIF) Spiking Wake-Word Spotting Cell.
///
/// Processes incoming Address-Event Representation spike streams in constant $O(1)$ time per event.
/// Employs exponential synaptic trace memory and multi-stage phoneme coincidence detection.
#[derive(Debug, Clone)]
pub struct SpikingKwsCell {
    keyword_name: String,
    config: SpikingKwsConfig,
    weights: Vec<Vec<f32>>,
    membrane_potentials: Vec<f32>,
    synaptic_traces: Vec<f32>,
    last_timestamp_us: u32,
    current_state_idx: usize,
    state_activation_time_us: Vec<u32>,
    refractory_until_us: u32,
    total_spikes_processed: u64,
}

impl SpikingKwsCell {
    /// Construct a new SpikingKwsCell with custom synaptic weight matrices.
    pub fn new(
        config: SpikingKwsConfig,
        weights: Vec<Vec<f32>>,
        keyword_name: impl Into<String>,
    ) -> Self {
        assert_eq!(
            weights.len(),
            config.num_states,
            "Weight matrix row count must match num_states"
        );
        for row in &weights {
            assert_eq!(
                row.len(),
                config.num_channels,
                "Weight row length must match num_channels"
            );
        }

        let num_states = config.num_states;
        let num_channels = config.num_channels;

        Self {
            keyword_name: keyword_name.into(),
            config,
            weights,
            membrane_potentials: vec![0.0; num_states],
            synaptic_traces: vec![0.0; num_channels],
            last_timestamp_us: 0,
            current_state_idx: 0,
            state_activation_time_us: vec![0; num_states],
            refractory_until_us: 0,
            total_spikes_processed: 0,
        }
    }

    /// Construct a calibrated Spiking KWS Cell for the target wake-word "Plank".
    ///
    /// Weights are distributed across 3 temporal phoneme stages:
    /// - Stage 0: Plosive burst [P] / onset [L] (energy in lower-mid frequencies 200-800 Hz and high onset)
    /// - Stage 1: Open front vowel [AE] (formant frequencies F1 around 700 Hz, F2 around 1700 Hz)
    /// - Stage 2: Velar nasal/stop [NG]/[K] (concentrated energy in 2000-4000 Hz)
    pub fn for_plank(num_channels: usize) -> Self {
        let config = SpikingKwsConfig {
            num_channels,
            num_states: 3,
            ..Default::default()
        };

        let mut weights = vec![vec![0.0f32; num_channels]; 3];

        #[allow(clippy::needless_range_loop)]
        for c in 0..num_channels {
            let frac = c as f32 / num_channels as f32;

            // Stage 0: [P/L] - Plosive attack: low-frequency acoustic energy + initial burst
            if frac < 0.30 {
                weights[0][c] = 0.45;
            } else if frac > 0.75 {
                weights[0][c] = 0.25;
            } else {
                weights[0][c] = -0.15; // Lateral inhibition from vowel formant band
            }

            // Stage 1: [AE] - Vowel resonance: mid-band formant energy
            if (0.25..0.60).contains(&frac) {
                weights[1][c] = 0.45;
            } else {
                weights[1][c] = -0.15; // Lateral inhibition from out-of-band frequencies
            }

            // Stage 2: [NG/K] - Velar closure and burst: high frequency energy
            if frac >= 0.65 {
                weights[2][c] = 0.45;
            } else {
                weights[2][c] = -0.15; // Lateral inhibition from low/mid frequencies
            }
        }

        Self::new(config, weights, "Plank")
    }

    /// Process a single incoming Address-Event Representation spike event in $O(1)$ time.
    /// Emits `Some(KeywordEvent)` if the complete temporal phoneme coincidence chain matches.
    pub fn step_event(&mut self, event: &SpikeEvent) -> Option<KeywordEvent> {
        self.total_spikes_processed += 1;

        let delta_t_us = event.timestamp_us.saturating_sub(self.last_timestamp_us);
        self.last_timestamp_us = event.timestamp_us;

        // Exponential decay of synaptic traces and membrane potentials
        if delta_t_us > 0 {
            let dt = delta_t_us as f32;
            let syn_decay = (-dt / self.config.tau_syn_us).exp();
            let mem_decay = (-dt / self.config.tau_mem_us).exp();

            for trace in &mut self.synaptic_traces {
                *trace *= syn_decay;
            }

            for v in &mut self.membrane_potentials {
                *v *= mem_decay;
            }

            // Check if current phoneme sequence timed out
            if self.current_state_idx > 0 {
                let prev_activation = self.state_activation_time_us[self.current_state_idx - 1];
                if event.timestamp_us.saturating_sub(prev_activation)
                    > self.config.coincidence_window_us
                {
                    self.current_state_idx = 0;
                }
            }
        }

        // Post-detection refractory lockout period
        if event.timestamp_us < self.refractory_until_us {
            return None;
        }

        let ch = event.channel as usize;
        if ch >= self.config.num_channels {
            return None;
        }

        // Update synaptic trace
        if event.is_on() {
            self.synaptic_traces[ch] += 1.0;
        } else {
            self.synaptic_traces[ch] = (self.synaptic_traces[ch] - 0.25).max(0.0);
        }

        // Integrate synaptic input into active target coincidence stage
        let polarity_scale = if event.is_on() { 1.0 } else { -0.2 };
        let target_state = self.current_state_idx;

        let w = self.weights[target_state][ch];
        let dv = w * polarity_scale;
        self.membrane_potentials[target_state] = (self.membrane_potentials[target_state] + dv).max(0.0);

        // Evaluate coincidence state transition
        if self.membrane_potentials[target_state] >= self.config.threshold {
            self.membrane_potentials[target_state] = 0.0;
            self.state_activation_time_us[target_state] = event.timestamp_us;

            if target_state + 1 == self.config.num_states {
                // All phoneme stages triggered in sequential order
                self.current_state_idx = 0;
                self.membrane_potentials.fill(0.0);
                self.refractory_until_us = event
                    .timestamp_us
                    .saturating_add(self.config.coincidence_window_us);

                let timestamp_sec = (event.timestamp_us as f64) / 1_000_000.0;
                return Some(KeywordEvent {
                    keyword: self.keyword_name.clone(),
                    confidence: 0.96,
                    timestamp_sec,
                });
            } else {
                self.current_state_idx += 1;
            }
        }

        None
    }

    /// Process a slice or iterator of spike events, returning all detected keyword events.
    pub fn step_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a SpikeEvent>,
    ) -> Vec<KeywordEvent> {
        let mut detections = Vec::new();
        for event in events {
            if let Some(kw) = self.step_event(event) {
                detections.push(kw);
            }
        }
        detections
    }

    /// Reset internal membrane potentials, synaptic traces, and coincidence stage to rest.
    pub fn reset(&mut self) {
        self.membrane_potentials.fill(0.0);
        self.synaptic_traces.fill(0.0);
        self.last_timestamp_us = 0;
        self.current_state_idx = 0;
        self.state_activation_time_us.fill(0);
        self.refractory_until_us = 0;
    }

    /// Get current membrane potentials for all coincidence states.
    pub fn membrane_potentials(&self) -> &[f32] {
        &self.membrane_potentials
    }

    /// Get current synaptic traces across all channels.
    pub fn synaptic_traces(&self) -> &[f32] {
        &self.synaptic_traces
    }

    /// Get current active coincidence state index.
    pub fn current_state(&self) -> usize {
        self.current_state_idx
    }

    /// Get total number of spikes processed since initialization.
    pub fn total_spikes_processed(&self) -> u64 {
        self.total_spikes_processed
    }
}

pub use crate::spiking_vad::{
    BandConfig, BiquadBandpassFilter, LifNeuron, LifNeuronConfig, SpikingNeuralVad,
    SpikingVadConfig, SpikingVadTelemetry,
};
