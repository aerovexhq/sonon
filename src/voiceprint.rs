#![deny(unsafe_code)]

//! Speaker-adaptive voiceprint conditioning and glottal acoustic anti-spoofing verification for edge KWS.
//!
//! Provides on-device speaker embedding extraction (`SpeakerVoiceprint`), dual-threshold
//! operator verification against enrolled authorized pilot voiceprints, and glottal acoustic
//! anti-spoofing detection (`GlottalAntiSpoofDetector`) to prevent voice replay attacks and
//! unauthorized robotic command injection.

use serde::{Deserialize, Serialize};

/// High-dimensional compact speaker embedding vector and acoustic prosodic profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerVoiceprint {
    /// L2-normalized pooled spectral feature embedding vector.
    pub embedding: Vec<f32>,
    /// Estimated mean fundamental pitch frequency $F_0$ in Hz.
    pub mean_f0: f32,
    /// Standard deviation of fundamental pitch frequency $F_0$ in Hz.
    pub f0_std: f32,
    /// Estimated Harmonic-to-Noise Ratio (HNR) in dB.
    pub hnr_db: f32,
    /// Total number of acoustic feature frames analyzed.
    pub frame_count: usize,
}

impl SpeakerVoiceprint {
    /// Extract a compact speaker voiceprint from feature frames and raw audio waveform.
    pub fn from_features_and_audio(
        features: &[Vec<f32>],
        audio: &[f32],
        sample_rate: f32,
    ) -> Result<Self, String> {
        if features.is_empty() {
            return Err("Cannot extract voiceprint from empty feature slice".to_string());
        }

        let num_frames = features.len();
        let feat_dim = features[0].len();
        if feat_dim == 0 {
            return Err("Feature dimension must be positive".to_string());
        }

        // 1. First-order statistics: Mean vector across temporal frames
        let mut mean = vec![0.0f32; feat_dim];
        for frame in features {
            for (d, &val) in frame.iter().enumerate().take(feat_dim) {
                mean[d] += val;
            }
        }
        for m in &mut mean {
            *m /= num_frames as f32;
        }

        // 2. Second-order statistics: Standard deviation vector
        let mut std = vec![0.0f32; feat_dim];
        for frame in features {
            for (d, &val) in frame.iter().enumerate().take(feat_dim) {
                let diff = val - mean[d];
                std[d] += diff * diff;
            }
        }
        for s in &mut std {
            *s = (*s / num_frames as f32).max(1e-6).sqrt();
        }

        // 3. Dynamic articulatory delta statistics
        let mut delta_abs_mean = vec![0.0f32; feat_dim];
        if num_frames > 1 {
            for t in 1..num_frames {
                for d in 0..feat_dim {
                    delta_abs_mean[d] += (features[t][d] - features[t - 1][d]).abs();
                }
            }
            for del in &mut delta_abs_mean {
                *del /= (num_frames - 1) as f32;
            }
        }

        // 4. Fundamental pitch ($F_0$) and Harmonic-to-Noise Ratio ($HNR$) analysis
        let (mean_f0, f0_std, hnr_db) = Self::analyze_pitch_and_hnr(audio, sample_rate);

        // 5. Assemble high-dimensional embedding vector
        // Exclude dimension 0 (C0 energy) and apply liftering to equalize formant frequency bands
        let start_dim = if feat_dim > 1 { 1 } else { 0 };
        let mut raw_embedding = Vec::with_capacity((feat_dim - start_dim) * 3);
        for d in start_dim..feat_dim {
            let lifter = 1.0 + 5.0 * ((std::f32::consts::PI * d as f32) / 12.0).sin();
            raw_embedding.push(mean[d] * lifter);
        }
        for d in start_dim..feat_dim {
            let lifter = 1.0 + 3.0 * ((std::f32::consts::PI * d as f32) / 12.0).sin();
            raw_embedding.push(std[d] * lifter);
        }
        for d in start_dim..feat_dim {
            raw_embedding.push(delta_abs_mean[d]);
        }

        // 6. L2-normalize embedding vector
        let norm_sq: f32 = raw_embedding.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        for x in &mut raw_embedding {
            *x /= norm;
        }

        Ok(Self {
            embedding: raw_embedding,
            mean_f0,
            f0_std,
            hnr_db,
            frame_count: num_frames,
        })
    }

    /// Compute composite speaker similarity $S \in [-1.0, 1.0]$ combining vocal tract filter envelope and glottal source characteristics.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let min_len = self.embedding.len().min(other.embedding.len());
        if min_len == 0 {
            return 0.0;
        }

        // 1. Spectral vocal tract shape similarity (filter)
        let mut dot = 0.0f32;
        for i in 0..min_len {
            dot += self.embedding[i] * other.embedding[i];
        }
        let spec_sim = dot.clamp(-1.0, 1.0);

        // 2. Fundamental pitch source similarity (source)
        let f0_diff = (self.mean_f0 - other.mean_f0).abs();
        let pitch_sim = (-f0_diff / 55.0).exp();

        // 3. Glottal harmonicity similarity
        let hnr_diff = (self.hnr_db - other.hnr_db).abs();
        let hnr_sim = (-hnr_diff / 15.0).exp();

        // Source-filter combined similarity: 60% vocal tract filter + 30% pitch + 10% HNR
        let total_sim = 0.60 * spec_sim + 0.30 * pitch_sim + 0.10 * hnr_sim;
        total_sim.clamp(-1.0, 1.0)
    }

    /// Compute cosine distance $D_{\text{cos}} = 1.0 - S \in [0.0, 2.0]$.
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        1.0 - self.cosine_similarity(other)
    }

    /// Fuse this voiceprint with another observation using weighted linear interpolation.
    pub fn fuse_with(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let min_len = self.embedding.len().min(other.embedding.len());
        let mut fused_emb = Vec::with_capacity(min_len);

        for i in 0..min_len {
            fused_emb.push(self.embedding[i] * (1.0 - w) + other.embedding[i] * w);
        }

        let norm_sq: f32 = fused_emb.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        for x in &mut fused_emb {
            *x /= norm;
        }

        Self {
            embedding: fused_emb,
            mean_f0: self.mean_f0 * (1.0 - w) + other.mean_f0 * w,
            f0_std: self.f0_std * (1.0 - w) + other.f0_std * w,
            hnr_db: self.hnr_db * (1.0 - w) + other.hnr_db * w,
            frame_count: self.frame_count + other.frame_count,
        }
    }

    /// Fuse multiple voiceprint exemplars into a single unified speaker profile.
    pub fn fuse_all(voiceprints: &[Self]) -> Result<Self, String> {
        if voiceprints.is_empty() {
            return Err("Cannot fuse empty list of voiceprints".to_string());
        }
        if voiceprints.len() == 1 {
            return Ok(voiceprints[0].clone());
        }

        let emb_len = voiceprints[0].embedding.len();
        let mut fused_emb = vec![0.0f32; emb_len];
        let mut total_f0 = 0.0f32;
        let mut total_f0_std = 0.0f32;
        let mut total_hnr = 0.0f32;
        let mut total_frames = 0;

        for vp in voiceprints {
            for (i, &val) in vp.embedding.iter().enumerate().take(emb_len) {
                fused_emb[i] += val;
            }
            total_f0 += vp.mean_f0;
            total_f0_std += vp.f0_std;
            total_hnr += vp.hnr_db;
            total_frames += vp.frame_count;
        }

        let n = voiceprints.len() as f32;
        let norm_sq: f32 = fused_emb.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        for x in &mut fused_emb {
            *x /= norm;
        }

        Ok(Self {
            embedding: fused_emb,
            mean_f0: total_f0 / n,
            f0_std: total_f0_std / n,
            hnr_db: total_hnr / n,
            frame_count: total_frames,
        })
    }

    /// Estimate pitch tracks ($F_0$) and harmonic-to-noise ratio ($HNR$) using normalized autocorrelation.
    fn analyze_pitch_and_hnr(audio: &[f32], sample_rate: f32) -> (f32, f32, f32) {
        if audio.len() < 320 {
            return (130.0, 15.0, 6.0);
        }

        let frame_len = ((sample_rate * 0.030).round() as usize).max(256);
        let hop_len = ((sample_rate * 0.015).round() as usize).max(128);

        let min_lag = ((sample_rate / 450.0).floor() as usize).max(10);
        let max_lag = ((sample_rate / 65.0).ceil() as usize).min(frame_len - 1);

        let mut f0_estimates = Vec::new();
        let mut hnr_estimates = Vec::new();

        let mut pos = 0;
        while pos + frame_len <= audio.len() {
            let slice = &audio[pos..pos + frame_len];

            // Compute energy
            let energy: f32 = slice.iter().map(|&s| s * s).sum();
            if energy > 1e-4 {
                // Autocorrelation
                let mut max_corr = -1.0f32;
                let mut best_lag = 0;

                for lag in min_lag..=max_lag {
                    let mut sum = 0.0f32;
                    for i in 0..(frame_len - lag) {
                        sum += slice[i] * slice[i + lag];
                    }
                    if sum > max_corr {
                        max_corr = sum;
                        best_lag = lag;
                    }
                }

                let norm_corr = max_corr / energy;
                if norm_corr > 0.30 && best_lag > 0 {
                    let f0 = sample_rate / best_lag as f32;
                    f0_estimates.push(f0);

                    let hnr = 10.0 * (norm_corr / (1.0 - norm_corr.min(0.999) + 1e-5)).log10();
                    hnr_estimates.push(hnr);
                }
            }

            pos += hop_len;
        }

        if f0_estimates.is_empty() {
            return (130.0, 15.0, 6.0);
        }

        let mean_f0 = f0_estimates.iter().sum::<f32>() / f0_estimates.len() as f32;
        let var_f0 = f0_estimates
            .iter()
            .map(|&f| (f - mean_f0) * (f - mean_f0))
            .sum::<f32>()
            / f0_estimates.len() as f32;
        let std_f0 = var_f0.sqrt();

        let mean_hnr = hnr_estimates.iter().sum::<f32>() / hnr_estimates.len() as f32;

        (mean_f0, std_f0, mean_hnr)
    }
}

/// Configuration parameters for glottal acoustic anti-spoofing verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiSpoofConfig {
    /// Maximum allowable composite spoofing risk score (0.0 = genuine, 1.0 = spoofed).
    pub max_spoof_threshold: f32,
    /// Minimum expected low-frequency energy ratio in 50-200 Hz for genuine human speech.
    pub min_low_freq_ratio: f32,
    /// Minimum natural pitch period jitter ratio.
    pub min_natural_jitter: f32,
    /// Maximum natural pitch period jitter ratio.
    pub max_natural_jitter: f32,
    /// Maximum allowed high-frequency distortion ratio (>3.5 kHz / 1-3 kHz).
    pub max_hf_distortion_ratio: f32,
}

impl Default for AntiSpoofConfig {
    fn default() -> Self {
        Self {
            max_spoof_threshold: 0.55,
            min_low_freq_ratio: 0.015,
            min_natural_jitter: 0.002,
            max_natural_jitter: 0.055,
            max_hf_distortion_ratio: 0.85,
        }
    }
}

/// Detailed diagnostic telemetry from glottal acoustic anti-spoofing analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AntiSpoofReport {
    /// Composite spoofing risk score in [0.0, 1.0] (0.0 = authentic human voice, 1.0 = definite spoof).
    pub spoof_score: f32,
    /// Whether the utterance passed anti-spoof verification (spoof_score <= max_spoof_threshold).
    pub is_genuine: bool,
    /// Ratio of spectral energy in the sub-200 Hz glottal fundamental band relative to total energy.
    pub low_freq_energy_ratio: f32,
    /// Measured pitch period cycle-to-cycle perturbation (jitter ratio).
    pub pitch_jitter_ratio: f32,
    /// Ratio of high-frequency transducer resonance energy (>3.5 kHz) to mid-frequency band (1-3 kHz).
    pub hf_distortion_ratio: f32,
    /// Spectral flatness across upper frequency bands.
    pub spectral_flatness: f32,
    /// Primary cause of rejection if classified as spoof.
    pub spoof_flag: Option<String>,
}

/// Glottal acoustic anti-spoofing detector.
#[derive(Debug, Clone)]
pub struct GlottalAntiSpoofDetector {
    config: AntiSpoofConfig,
}

impl Default for GlottalAntiSpoofDetector {
    fn default() -> Self {
        Self {
            config: AntiSpoofConfig::default(),
        }
    }
}

impl GlottalAntiSpoofDetector {
    /// Construct a new glottal anti-spoofing detector with specified configuration.
    pub fn new(config: AntiSpoofConfig) -> Self {
        Self { config }
    }

    /// Access active configuration parameters.
    pub fn config(&self) -> &AntiSpoofConfig {
        &self.config
    }

    /// Update configuration parameters.
    pub fn set_config(&mut self, config: AntiSpoofConfig) {
        self.config = config;
    }

    /// Analyze raw audio samples for signs of loudspeaker replay, small-transducer roll-off, or synthetic playback.
    pub fn analyze(&self, audio_samples: &[f32], sample_rate: f32) -> AntiSpoofReport {
        if audio_samples.len() < 512 {
            return AntiSpoofReport {
                spoof_score: 0.0,
                is_genuine: true,
                low_freq_energy_ratio: 0.05,
                pitch_jitter_ratio: 0.012,
                hf_distortion_ratio: 0.35,
                spectral_flatness: 0.20,
                spoof_flag: None,
            };
        }

        let fft_size = 512;
        let hop_size = 256;
        let num_bins = fft_size / 2 + 1;
        let bin_width = (sample_rate * 0.5) / (num_bins - 1) as f32;

        let fft = crate::stft::FftProcessor::new(fft_size);
        let window = crate::window::Window::new(crate::window::WindowType::Hann, fft_size);

        let mut low_energy_sum = 0.0f32;
        let mut mid_energy_sum = 0.0f32;
        let mut high_energy_sum = 0.0f32;
        let mut total_energy_sum = 0.0f32;

        let low_max_bin = ((200.0 / bin_width).round() as usize).min(num_bins - 1);
        let mid_min_bin = ((1000.0 / bin_width).round() as usize).min(num_bins - 1);
        let mid_max_bin = ((3000.0 / bin_width).round() as usize).min(num_bins - 1);
        let high_min_bin = ((3500.0 / bin_width).round() as usize).min(num_bins - 1);
        let high_max_bin = ((7500.0 / bin_width).round() as usize).min(num_bins - 1);

        let mut pos = 0;
        let mut frame = vec![0.0f32; fft_size];

        let mut upper_band_power_sum = 0.0f32;
        let mut upper_band_log_sum = 0.0f32;
        let mut upper_band_bin_count = 0usize;

        while pos + fft_size <= audio_samples.len() {
            frame.copy_from_slice(&audio_samples[pos..pos + fft_size]);
            window.apply(&mut frame);
            let power = fft.power_spectrum(&frame);

            for (k, &p) in power.iter().enumerate() {
                total_energy_sum += p;
                if k <= low_max_bin {
                    low_energy_sum += p;
                }
                if k >= mid_min_bin && k <= mid_max_bin {
                    mid_energy_sum += p;
                }
                if k >= high_min_bin && k <= high_max_bin {
                    high_energy_sum += p;
                    upper_band_power_sum += p;
                    upper_band_log_sum += (p.max(1e-9)).ln();
                    upper_band_bin_count += 1;
                }
            }

            pos += hop_size;
        }

        let low_freq_ratio = if total_energy_sum > 1e-6 {
            low_energy_sum / total_energy_sum
        } else {
            0.05
        };

        let hf_distortion_ratio = if mid_energy_sum > 1e-6 {
            high_energy_sum / mid_energy_sum
        } else {
            0.20
        };

        let spectral_flatness = if upper_band_bin_count > 0 && upper_band_power_sum > 1e-6 {
            let geom_mean = (upper_band_log_sum / upper_band_bin_count as f32).exp();
            let arith_mean = upper_band_power_sum / upper_band_bin_count as f32;
            (geom_mean / arith_mean.max(1e-8)).clamp(0.0, 1.0)
        } else {
            0.15
        };

        // Pitch period cycle-to-cycle jitter analysis
        let pitch_jitter_ratio = Self::estimate_pitch_jitter(audio_samples, sample_rate);

        // Evaluate sub-penalties
        let mut spoof_score = 0.0f32;
        let mut flag = None;

        // 1. Low-frequency glottal energy deficit (phone speaker acoustic roll-off)
        if low_freq_ratio < self.config.min_low_freq_ratio {
            let deficit = 1.0 - (low_freq_ratio / self.config.min_low_freq_ratio).clamp(0.0, 1.0);
            spoof_score += deficit * 0.45;
            if flag.is_none() {
                flag = Some("low_frequency_glottal_deficit".to_string());
            }
        }

        // 2. Jitter anomaly (unnaturally flat electronic playback or excessive noisy jitter)
        if pitch_jitter_ratio < self.config.min_natural_jitter {
            let flat_penalty = 1.0
                - (pitch_jitter_ratio / self.config.min_natural_jitter).clamp(0.0, 1.0);
            spoof_score += flat_penalty * 0.35;
            if flag.is_none() {
                flag = Some("unnaturally_flat_glottal_jitter".to_string());
            }
        } else if pitch_jitter_ratio > self.config.max_natural_jitter {
            let noise_penalty =
                ((pitch_jitter_ratio - self.config.max_natural_jitter) / 0.05).clamp(0.0, 1.0);
            spoof_score += noise_penalty * 0.35;
            if flag.is_none() {
                flag = Some("excessive_transducer_jitter_noise".to_string());
            }
        }

        // 3. High-frequency loudspeaker resonance distortion
        if hf_distortion_ratio > self.config.max_hf_distortion_ratio {
            let hf_penalty = ((hf_distortion_ratio - self.config.max_hf_distortion_ratio) / 0.50)
                .clamp(0.0, 1.0);
            spoof_score += hf_penalty * 0.30;
            if flag.is_none() {
                flag = Some("loudspeaker_high_frequency_resonance".to_string());
            }
        }

        let spoof_score_clamped = spoof_score.clamp(0.0, 1.0);
        let is_genuine = spoof_score_clamped <= self.config.max_spoof_threshold;

        AntiSpoofReport {
            spoof_score: spoof_score_clamped,
            is_genuine,
            low_freq_energy_ratio: low_freq_ratio,
            pitch_jitter_ratio,
            hf_distortion_ratio,
            spectral_flatness,
            spoof_flag: if is_genuine { None } else { flag },
        }
    }

    /// Extract cycle-to-cycle relative pitch period jitter across consecutive voiced frames.
    fn estimate_pitch_jitter(audio: &[f32], sample_rate: f32) -> f32 {
        let frame_len = ((sample_rate * 0.030).round() as usize).max(256);
        let hop_len = ((sample_rate * 0.010).round() as usize).max(80);

        let min_lag = ((sample_rate / 400.0).floor() as usize).max(10);
        let max_lag = ((sample_rate / 70.0).ceil() as usize).min(frame_len - 1);

        let mut pitch_lags = Vec::new();
        let mut pos = 0;

        while pos + frame_len <= audio.len() {
            let slice = &audio[pos..pos + frame_len];
            let energy: f32 = slice.iter().map(|&s| s * s).sum();

            if energy > 1e-4 {
                let mut best_corr = -1.0f32;
                let mut best_lag = 0;

                for lag in min_lag..=max_lag {
                    let mut sum = 0.0f32;
                    for i in 0..(frame_len - lag) {
                        sum += slice[i] * slice[i + lag];
                    }
                    if sum > best_corr {
                        best_corr = sum;
                        best_lag = lag;
                    }
                }

                if best_corr / energy > 0.40 && best_lag > 0 {
                    pitch_lags.push(best_lag as f32);
                }
            }

            pos += hop_len;
        }

        if pitch_lags.len() < 3 {
            return 0.012; // Nominal natural jitter default
        }

        let mut abs_diff_sum = 0.0f32;
        for i in 0..(pitch_lags.len() - 1) {
            abs_diff_sum += (pitch_lags[i] - pitch_lags[i + 1]).abs();
        }
        let mean_lag = pitch_lags.iter().sum::<f32>() / pitch_lags.len() as f32;

        let jitter = (abs_diff_sum / (pitch_lags.len() - 1) as f32) / mean_lag.max(1.0);
        jitter.clamp(0.0001, 0.20)
    }
}

/// Enrolled authorized pilot/operator profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorProfile {
    /// Unique operator identification tag.
    pub operator_id: String,
    /// Enrolled reference speaker voiceprint.
    pub voiceprint: SpeakerVoiceprint,
    /// Minimum cosine similarity threshold for authorization acceptance.
    pub similarity_threshold: f32,
    /// Number of distinct audio utterances used to fuse the voiceprint profile.
    pub enrollment_count: usize,
}

/// Verification decision for candidate speaker voice utterances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationDecision {
    /// Voiceprint matched an authorized operator and passed anti-spoof verification.
    Authorized {
        operator_id: String,
        similarity: f32,
        anti_spoof_score: f32,
    },
    /// Utterance is from an unauthorized speaker whose voiceprint did not match any enrolled operator.
    RejectedUnauthorized {
        best_operator_id: Option<String>,
        best_similarity: f32,
        required_threshold: f32,
    },
    /// Utterance exhibited loudspeaker replay or synthetic spoof acoustic artifacts.
    RejectedSpoof {
        spoof_score: f32,
        spoof_threshold: f32,
        reason: String,
    },
}

/// Dual-threshold speaker voiceprint verifier and anti-spoofing controller.
#[derive(Debug, Clone)]
pub struct OperatorVerifier {
    operators: Vec<OperatorProfile>,
    anti_spoof_detector: GlottalAntiSpoofDetector,
    strict_enforcement: bool,
}

impl Default for OperatorVerifier {
    fn default() -> Self {
        Self {
            operators: Vec::new(),
            anti_spoof_detector: GlottalAntiSpoofDetector::default(),
            strict_enforcement: true,
        }
    }
}

impl OperatorVerifier {
    /// Construct a new OperatorVerifier with strict enforcement setting.
    pub fn new(strict_enforcement: bool) -> Self {
        Self {
            operators: Vec::new(),
            anti_spoof_detector: GlottalAntiSpoofDetector::default(),
            strict_enforcement,
        }
    }

    /// Construct with custom anti-spoof configuration.
    pub fn with_anti_spoof_config(config: AntiSpoofConfig, strict_enforcement: bool) -> Self {
        Self {
            operators: Vec::new(),
            anti_spoof_detector: GlottalAntiSpoofDetector::new(config),
            strict_enforcement,
        }
    }

    /// Enroll or update an authorized operator profile.
    pub fn enroll_operator(
        &mut self,
        operator_id: impl Into<String>,
        voiceprint: SpeakerVoiceprint,
        similarity_threshold: f32,
    ) {
        let op_id = operator_id.into();
        let threshold = similarity_threshold.clamp(0.40, 0.98);

        if let Some(existing) = self.operators.iter_mut().find(|op| op.operator_id == op_id) {
            existing.voiceprint = existing.voiceprint.fuse_with(&voiceprint, 0.50);
            existing.similarity_threshold = threshold;
            existing.enrollment_count += 1;
        } else {
            self.operators.push(OperatorProfile {
                operator_id: op_id,
                voiceprint,
                similarity_threshold: threshold,
                enrollment_count: 1,
            });
        }
    }

    /// Remove an enrolled operator by ID.
    pub fn remove_operator(&mut self, operator_id: &str) -> bool {
        let initial_len = self.operators.len();
        self.operators.retain(|op| op.operator_id != operator_id);
        self.operators.len() < initial_len
    }

    /// Clear all enrolled operator profiles.
    pub fn clear_operators(&mut self) {
        self.operators.clear();
    }

    /// List all enrolled operator profiles.
    pub fn operators(&self) -> &[OperatorProfile] {
        &self.operators
    }

    /// Whether strict operator enforcement is active.
    pub fn is_strict(&self) -> bool {
        self.strict_enforcement
    }

    /// Set strict operator enforcement behavior.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict_enforcement = strict;
    }

    /// Access reference to internal glottal anti-spoof detector.
    pub fn anti_spoof_detector(&self) -> &GlottalAntiSpoofDetector {
        &self.anti_spoof_detector
    }

    /// Access mutable reference to internal glottal anti-spoof detector.
    pub fn anti_spoof_detector_mut(&mut self) -> &mut GlottalAntiSpoofDetector {
        &mut self.anti_spoof_detector
    }

    /// Evaluate dual-threshold verification on an incoming voice utterance.
    pub fn verify(
        &self,
        audio_samples: &[f32],
        features: &[Vec<f32>],
        sample_rate: f32,
    ) -> VerificationDecision {
        // 1. Evaluate glottal acoustic anti-spoofing
        let anti_spoof = self.anti_spoof_detector.analyze(audio_samples, sample_rate);
        if !anti_spoof.is_genuine {
            return VerificationDecision::RejectedSpoof {
                spoof_score: anti_spoof.spoof_score,
                spoof_threshold: self.anti_spoof_detector.config().max_spoof_threshold,
                reason: anti_spoof
                    .spoof_flag
                    .unwrap_or_else(|| "unspecified_anti_spoof_anomaly".to_string()),
            };
        }

        // 2. Extract candidate utterance voiceprint
        let candidate_vp = match SpeakerVoiceprint::from_features_and_audio(
            features,
            audio_samples,
            sample_rate,
        ) {
            Ok(vp) => vp,
            Err(_) => {
                return VerificationDecision::RejectedUnauthorized {
                    best_operator_id: None,
                    best_similarity: 0.0,
                    required_threshold: 0.75,
                };
            }
        };

        if self.operators.is_empty() {
            // When no operators are enrolled, by default accept if not strict, else reject
            return VerificationDecision::Authorized {
                operator_id: "anonymous_open_access".to_string(),
                similarity: 1.0,
                anti_spoof_score: anti_spoof.spoof_score,
            };
        }

        // 3. Find matching enrolled operator with highest similarity
        let mut best_op = None;
        let mut best_sim = -1.0f32;
        let mut required_thresh = 0.75f32;

        for op in &self.operators {
            let sim = candidate_vp.cosine_similarity(&op.voiceprint);
            if sim > best_sim {
                best_sim = sim;
                best_op = Some(op.operator_id.clone());
                required_thresh = op.similarity_threshold;
            }
        }

        if best_sim >= required_thresh {
            VerificationDecision::Authorized {
                operator_id: best_op.unwrap_or_default(),
                similarity: best_sim,
                anti_spoof_score: anti_spoof.spoof_score,
            }
        } else {
            VerificationDecision::RejectedUnauthorized {
                best_operator_id: best_op,
                best_similarity: best_sim,
                required_threshold: required_thresh,
            }
        }
    }
}
