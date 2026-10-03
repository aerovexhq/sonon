//! Pitch-Synchronous Overlap-Add (PSOLA) Prosodic Morphing & Glottal Waveform Augmentation.
//!
//! Provides time-domain pitch-synchronous overlap-add (TD-PSOLA) manipulation of speech waveforms,
//! Glottal Closure Instant (GCI) extraction, independent continuous tempo scaling ($0.5\times - 2.5\times$)
//! and $F_0$ pitch shifting ($\pm 12$ semitones) while preserving vocal tract formant envelopes.
//! Used for automated edge KWS synthetic training data augmentation and speaker-invariant template fusion.

#![deny(unsafe_code)]

use std::f32::consts::PI;
use serde::{Deserialize, Serialize};

/// Configuration parameters for Glottal Closure Instant (GCI) detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GciConfig {
    /// Minimum fundamental pitch frequency in Hz (e.g. 65.0 Hz).
    pub min_f0: f32,
    /// Maximum fundamental pitch frequency in Hz (e.g. 450.0 Hz).
    pub max_f0: f32,
    /// Pre-emphasis filtering coefficient for high-frequency excitation peak sharpening.
    pub pre_emphasis: f32,
    /// Local energy integration window duration in milliseconds (e.g. 1.5 ms).
    pub energy_window_ms: f32,
    /// Peak detection threshold ratio relative to local RMS envelope.
    pub peak_threshold_ratio: f32,
}

impl Default for GciConfig {
    fn default() -> Self {
        Self {
            min_f0: 65.0,
            max_f0: 450.0,
            pre_emphasis: 0.95,
            energy_window_ms: 1.5,
            peak_threshold_ratio: 0.35,
        }
    }
}

/// A detected pitch mark representing a Glottal Closure Instant (GCI) in the waveform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchMark {
    /// Sample offset in the input audio buffer.
    pub sample_index: usize,
    /// Estimated local pitch period $T_0$ in samples.
    pub pitch_period_samples: usize,
    /// Flag indicating whether the local acoustic segment is voiced speech.
    pub is_voiced: bool,
}

/// Glottal Closure Instant (GCI) and pitch mark detector operating on raw waveforms.
#[derive(Debug, Clone)]
pub struct GciDetector {
    config: GciConfig,
}

impl GciDetector {
    /// Construct a new GCI detector with specified configuration.
    pub fn new(config: GciConfig) -> Self {
        Self { config }
    }

    /// Estimate fundamental pitch period $T_0$ in samples using normalized autocorrelation around a given center.
    pub fn estimate_pitch_period_at(&self, audio: &[f32], center: usize, sample_rate: f32) -> (usize, f32) {
        let min_period = ((sample_rate / self.config.max_f0).max(8.0)) as usize;
        let max_period = ((sample_rate / self.config.min_f0).min(sample_rate * 0.03)) as usize;
        let nominal_period = ((sample_rate / 130.0).clamp(min_period as f32, max_period as f32)) as usize;

        let n = audio.len();
        let win_len = (max_period * 2).min(512);
        let start = center.saturating_sub(win_len / 2);
        if start + win_len + max_period > n {
            return (nominal_period, 0.0);
        }

        // Energy of reference window
        let mut r0 = 0.0f32;
        for j in 0..win_len {
            let s = audio[start + j];
            r0 += s * s;
        }

        if r0 < 1e-8 {
            return (nominal_period, 0.0);
        }

        let mut best_lag = nominal_period;
        let mut best_norm_corr = -1.0f32;

        for lag in min_period..=max_period {
            let mut r_lag = 0.0f32;
            let mut e_lag = 0.0f32;

            for j in 0..win_len {
                let s1 = audio[start + j];
                let s2 = audio[start + j + lag];
                r_lag += s1 * s2;
                e_lag += s2 * s2;
            }

            let denom = (r0 * e_lag).sqrt().max(1e-12);
            let norm_corr = r_lag / denom;

            if norm_corr > best_norm_corr {
                best_norm_corr = norm_corr;
                best_lag = lag;
            }
        }

        (best_lag, best_norm_corr)
    }

    /// Extract pitch marks and pitch periods across an input speech waveform.
    pub fn extract_pitch_marks(&self, audio: &[f32], sample_rate: f32) -> Vec<PitchMark> {
        let n = audio.len();
        let min_period = ((sample_rate / self.config.max_f0).max(8.0)) as usize;
        let max_period = ((sample_rate / self.config.min_f0).min(sample_rate * 0.03)) as usize;
        let _nominal_period = ((sample_rate / 130.0).clamp(min_period as f32, max_period as f32)) as usize;

        if n < min_period * 2 {
            return Vec::new();
        }

        // Pre-emphasis filter to sharpen pulse peaks
        let mut preemp = vec![0.0f32; n];
        let mut prev = 0.0f32;
        for i in 0..n {
            let curr = audio[i];
            preemp[i] = curr - self.config.pre_emphasis * prev;
            prev = curr;
        }

        let mut marks = Vec::new();
        let mut pos = min_period;

        // Estimate initial pitch period from early audio
        let (mut current_period, mut corr) = self.estimate_pitch_period_at(audio, n / 4, sample_rate);
        if corr < 0.30 {
            let (p_mid, c_mid) = self.estimate_pitch_period_at(audio, n / 2, sample_rate);
            if c_mid > corr {
                current_period = p_mid;
                corr = c_mid;
            }
        }

        // Find initial peak
        let init_search_end = (pos + current_period).min(n);
        let mut initial_peak = pos;
        let mut max_val = f32::NEG_INFINITY;
        for j in pos..init_search_end {
            if preemp[j] > max_val {
                max_val = preemp[j];
                initial_peak = j;
            }
        }
        pos = initial_peak;

        while pos < n.saturating_sub(min_period) {
            // Update local pitch period every few marks
            if marks.len() % 4 == 0 && pos + max_period * 2 < n {
                let (p, c) = self.estimate_pitch_period_at(audio, pos, sample_rate);
                if c > 0.35 {
                    current_period = p;
                    corr = c;
                }
            }

            let is_voiced = corr > 0.30;
            marks.push(PitchMark {
                sample_index: pos,
                pitch_period_samples: current_period,
                is_voiced,
            });

            // Find next peak in window around next expected pulse
            let next_target = pos + current_period;
            let half_search = (current_period / 4).max(3);
            let search_start = (next_target.saturating_sub(half_search)).max(pos + min_period);
            let search_end = (next_target + half_search).min(n);

            if search_start >= search_end {
                break;
            }

            let mut next_peak = next_target.min(n - 1);
            let mut peak_val = f32::NEG_INFINITY;

            for j in search_start..search_end {
                if preemp[j] > peak_val {
                    peak_val = preemp[j];
                    next_peak = j;
                }
            }

            pos = next_peak;
        }

        marks
    }
}

/// Configuration parameters for Time-Domain PSOLA (TD-PSOLA) transformation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsolaConfig {
    /// Speaking rate multiplier $\alpha$ ($0.5\times$ = slow, $1.0\times$ = original, $2.0\times$ = fast).
    pub tempo_scale: f32,
    /// Pitch modification in semitones $\Delta \in [-12.0, +12.0]$ ($+12.0$ = one octave higher).
    pub pitch_shift_semitones: f32,
    /// Audio sample rate in Hz (e.g. 16000.0).
    pub sample_rate: f32,
}

impl Default for PsolaConfig {
    fn default() -> Self {
        Self {
            tempo_scale: 1.0,
            pitch_shift_semitones: 0.0,
            sample_rate: 16000.0,
        }
    }
}

/// Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA) prosodic modifier.
#[derive(Debug, Clone)]
pub struct PsolaModifier {
    detector: GciDetector,
}

impl Default for PsolaModifier {
    fn default() -> Self {
        Self::new(GciConfig::default())
    }
}

impl PsolaModifier {
    /// Construct a new PSOLA modifier with specified GCI detector configuration.
    pub fn new(config: GciConfig) -> Self {
        Self {
            detector: GciDetector::new(config),
        }
    }

    /// Access reference to internal GCI detector.
    pub fn detector(&self) -> &GciDetector {
        &self.detector
    }

    /// Modify speech tempo and pitch using Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA).
    ///
    /// Preserves natural acoustic formants ($F_1, F_2, F_3$) while independently shifting pitch
    /// by $\Delta$ semitones and scaling duration by $1 / \text{tempo\_scale}$.
    pub fn process(&self, audio: &[f32], config: &PsolaConfig) -> Vec<f32> {
        let input_len = audio.len();
        if input_len < 256 {
            return audio.to_vec();
        }

        // Clamp operational parameters to stable ranges
        let tempo = config.tempo_scale.clamp(0.40, 2.80);
        let semitones = config.pitch_shift_semitones.clamp(-14.0, 14.0);
        let pitch_factor = (2.0f32).powf(semitones / 12.0).clamp(0.45, 2.50);

        // Fast path for identity transformation
        if (tempo - 1.0).abs() < 0.005 && semitones.abs() < 0.05 {
            return audio.to_vec();
        }

        // 1. Analysis: Extract pitch marks
        let marks = self.detector.extract_pitch_marks(audio, config.sample_rate);
        if marks.len() < 2 {
            return audio.to_vec();
        }

        // 2. Synthesis pitch mark generation
        let target_len = ((input_len as f32) / tempo).max(64.0) as usize;
        let mut output = vec![0.0f32; target_len];
        let mut window_weights = vec![0.0f32; target_len];

        let mut synth_time = marks[0].sample_index as f32;
        let mut k_analysis = 0usize;

        while (synth_time as usize) < target_len {
            let s_idx = synth_time as usize;

            // Map synthesis time back to analysis time
            let mapped_analysis_time = synth_time * tempo;

            // Find closest analysis pitch mark
            while k_analysis + 1 < marks.len() && (marks[k_analysis + 1].sample_index as f32) <= mapped_analysis_time {
                k_analysis += 1;
            }
            let best_k = if k_analysis + 1 < marks.len() {
                let d1 = (mapped_analysis_time - marks[k_analysis].sample_index as f32).abs();
                let d2 = (marks[k_analysis + 1].sample_index as f32 - mapped_analysis_time).abs();
                if d2 < d1 { k_analysis + 1 } else { k_analysis }
            } else {
                k_analysis
            };

            let mark = &marks[best_k];
            let ana_center = mark.sample_index;
            let period_ana = mark.pitch_period_samples;

            // Synthesis pitch period modified by pitch factor
            let period_synth = ((period_ana as f32) / pitch_factor).clamp(8.0, config.sample_rate * 0.03) as usize;
            let half_win = period_ana.min(period_synth).max(4);

            let h = half_win as isize;
            for d in -h..=h {
                let ana_pos = ana_center as isize + d;
                let out_pos = s_idx as isize + d;

                if ana_pos >= 0 && (ana_pos as usize) < input_len && out_pos >= 0 && (out_pos as usize) < target_len {
                    let win_phase = (d + h) as f32 / (2.0 * h as f32);
                    let w = 0.5 * (1.0 - (2.0 * PI * win_phase).cos());

                    let sample = audio[ana_pos as usize] * w;
                    output[out_pos as usize] += sample;
                    window_weights[out_pos as usize] += w;
                }
            }

            // Advance synthesis time by new pitch period
            synth_time += period_synth as f32;
        }

        // 3. Normalization by overlap-add window weights
        for i in 0..target_len {
            let w_sum = window_weights[i];
            if w_sum > 0.01 {
                output[i] /= w_sum;
            }
        }

        // Apply soft limiter to prevent potential boundary clipping
        let mut peak = 0.0f32;
        for &s in &output {
            if s.abs() > peak {
                peak = s.abs();
            }
        }
        if peak > 0.98 {
            let scale = 0.98 / peak;
            for s in &mut output {
                *s *= scale;
            }
        }

        output
    }
}

/// Specifications for a prosodically morphed synthetic voice variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodicVariantSpec {
    /// Descriptive name of the variant (e.g. "rapid_high_pitch").
    pub name: String,
    /// Speaking tempo rate multiplier (e.g. 0.85 to 1.30).
    pub tempo_scale: f32,
    /// Pitch shift in semitones (e.g. -3.0 to +3.5).
    pub pitch_shift_semitones: f32,
}

/// Multi-speaker prosodic ensemble generator for automatic synthetic voice data augmentation.
#[derive(Debug, Clone)]
pub struct ProsodicEnsembleGenerator {
    modifier: PsolaModifier,
    variants: Vec<ProsodicVariantSpec>,
}

impl Default for ProsodicEnsembleGenerator {
    fn default() -> Self {
        Self::new_standard_ensemble()
    }
}

impl ProsodicEnsembleGenerator {
    /// Construct a new ensemble generator with standard 7-variant multi-speaker prosody specifications.
    pub fn new_standard_ensemble() -> Self {
        let variants = vec![
            ProsodicVariantSpec {
                name: "baseline".to_string(),
                tempo_scale: 1.0,
                pitch_shift_semitones: 0.0,
            },
            ProsodicVariantSpec {
                name: "slow_deep".to_string(),
                tempo_scale: 0.85,
                pitch_shift_semitones: -2.5,
            },
            ProsodicVariantSpec {
                name: "fast_high".to_string(),
                tempo_scale: 1.25,
                pitch_shift_semitones: 3.0,
            },
            ProsodicVariantSpec {
                name: "urgent_rapid".to_string(),
                tempo_scale: 1.35,
                pitch_shift_semitones: 1.5,
            },
            ProsodicVariantSpec {
                name: "deliberate_low".to_string(),
                tempo_scale: 0.78,
                pitch_shift_semitones: -3.5,
            },
            ProsodicVariantSpec {
                name: "elevated_pitch".to_string(),
                tempo_scale: 1.05,
                pitch_shift_semitones: 4.2,
            },
            ProsodicVariantSpec {
                name: "relaxed_slow".to_string(),
                tempo_scale: 0.88,
                pitch_shift_semitones: -1.0,
            },
        ];

        Self {
            modifier: PsolaModifier::default(),
            variants,
        }
    }

    /// Construct with custom prosody specifications.
    pub fn with_variants(variants: Vec<ProsodicVariantSpec>) -> Self {
        Self {
            modifier: PsolaModifier::default(),
            variants,
        }
    }

    /// Access reference to active prosody variant specifications.
    pub fn variants(&self) -> &[ProsodicVariantSpec] {
        &self.variants
    }

    /// Add a custom prosody variant specification to the generator.
    pub fn add_variant(&mut self, spec: ProsodicVariantSpec) {
        self.variants.push(spec);
    }

    /// Generate an ensemble of augmented speech waveforms from a single baseline recording or synthesis.
    pub fn generate_ensemble(&self, baseline_audio: &[f32], sample_rate: f32) -> Vec<(String, Vec<f32>)> {
        let mut ensemble = Vec::with_capacity(self.variants.len());

        for spec in &self.variants {
            let config = PsolaConfig {
                tempo_scale: spec.tempo_scale,
                pitch_shift_semitones: spec.pitch_shift_semitones,
                sample_rate,
            };
            let morphed = self.modifier.process(baseline_audio, &config);
            ensemble.push((spec.name.clone(), morphed));
        }

        ensemble
    }
}
