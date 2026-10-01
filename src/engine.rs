//! High-level streaming acoustic engine orchestrating VAD, MFCC/PCEN features, and keyword spotting.

use crate::dtw::DtwMatcher;
use crate::mel::MelFilterbank;
use crate::pcen::{PcenConfig, PcenFilter};
use crate::ring_buffer::AudioRingBuffer;
use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};

/// Acoustic feature normalization regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMode {
    /// Classical static logarithmic Mel filterbank energies.
    LogMel,
    /// Per-Channel Energy Normalization with adaptive AGC and dynamic compression.
    Pcen,
}

/// Detection event emitted upon recognized keyword or phrase.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeywordEvent {
    pub keyword: String,
    pub confidence: f32,
    pub timestamp_sec: f64,
}

/// Unified streaming acoustic DSP engine for robotics edge systems.
pub struct SononEngine {
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    num_mfcc: usize,
    ring_buffer: AudioRingBuffer,
    window: Window,
    fft: FftProcessor,
    mel: MelFilterbank,
    pcen: PcenFilter,
    feature_mode: FeatureMode,
    pre_emphasis_alpha: f32,
    last_sample: f32,
    dtw: DtwMatcher,
    feature_history: Vec<Vec<f32>>,
    max_history_frames: usize,
    total_samples_processed: u64,
}

impl SononEngine {
    /// Construct a new Sonon engine with specified sample rate (e.g. 16000.0) and frame sizes.
    pub fn new(sample_rate: f32, frame_size: usize, hop_size: usize, num_mfcc: usize) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(
            frame_size > 0 && (frame_size & (frame_size - 1)) == 0,
            "Frame size must be power of 2"
        );
        assert!(
            hop_size > 0 && hop_size <= frame_size,
            "Hop size must be <= frame size"
        );

        let ring_buffer = AudioRingBuffer::new(frame_size * 16);
        let window = Window::new(WindowType::Hann, frame_size);
        let fft = FftProcessor::new(frame_size);
        let num_mel_filters = 26;
        let mel = MelFilterbank::new(num_mel_filters, frame_size, sample_rate, 80.0, sample_rate / 2.0);
        let pcen = PcenFilter::new(num_mel_filters, PcenConfig::default());
        let dtw = DtwMatcher::new();

        Self {
            sample_rate,
            frame_size,
            hop_size,
            num_mfcc,
            ring_buffer,
            window,
            fft,
            mel,
            pcen,
            feature_mode: FeatureMode::LogMel,
            pre_emphasis_alpha: 0.97,
            last_sample: 0.0,
            dtw,
            feature_history: Vec::with_capacity(128),
            max_history_frames: 64,
            total_samples_processed: 0,
        }
    }

    /// Set feature normalization mode (LogMel or Pcen).
    pub fn set_feature_mode(&mut self, mode: FeatureMode) {
        self.feature_mode = mode;
    }

    /// Return current feature normalization mode.
    pub fn feature_mode(&self) -> FeatureMode {
        self.feature_mode
    }

    /// Set pre-emphasis high-pass filter coefficient (typically 0.95 to 0.98, or 0.0 to disable).
    pub fn set_pre_emphasis(&mut self, alpha: f32) {
        assert!((0.0..=1.0).contains(&alpha), "Alpha must be between 0.0 and 1.0");
        self.pre_emphasis_alpha = alpha;
    }

    /// Enroll a keyword phrase template into the engine using single feature sequence and default band corridor.
    pub fn enroll_keyword(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        self.dtw.add_template(name, features, threshold);
    }

    /// Enroll a keyword phrase template with explicit Sakoe-Chiba corridor band radius `R`.
    pub fn enroll_keyword_banded(
        &mut self,
        name: impl Into<String>,
        features: Vec<Vec<f32>>,
        threshold: f32,
        band_radius: usize,
    ) {
        self.dtw.add_template_banded(name, features, threshold, band_radius);
    }

    /// Enroll a keyword from multiple voice audio exemplars using DBA template fusion
    /// and automatic distance threshold calibration.
    pub fn enroll_keyword_multi(
        &mut self,
        name: impl Into<String>,
        exemplar_audio_slices: &[&[f32]],
        band_radius: usize,
        margin_factor: f32,
    ) -> f32 {
        let exemplars_features: Vec<Vec<Vec<f32>>> = exemplar_audio_slices
            .iter()
            .map(|slice| self.extract_features(slice))
            .filter(|f| !f.is_empty())
            .collect();
        self.dtw
            .add_template_exemplars(name, &exemplars_features, band_radius, margin_factor)
    }

    /// Ingest a slice of raw audio samples (mono float32 [-1.0, 1.0]).
    /// Returns any detected keyword events.
    pub fn ingest_samples(&mut self, samples: &[f32]) -> Vec<KeywordEvent> {
        let mut events = Vec::new();
        self.ring_buffer.push_slice(samples);
        self.total_samples_processed += samples.len() as u64;

        let mut frame_buf = vec![0.0f32; self.frame_size];

        while self.ring_buffer.len() >= self.frame_size {
            if !self.ring_buffer.peek(self.frame_size, &mut frame_buf) {
                break;
            }

            // Apply pre-emphasis filter to boost high-frequency formants and consonants
            let mut preemp = frame_buf.clone();
            if self.pre_emphasis_alpha > 0.0 {
                let mut prev = self.last_sample;
                for sample in &mut preemp {
                    let curr = *sample;
                    *sample = curr - self.pre_emphasis_alpha * prev;
                    prev = curr;
                }
                self.last_sample = prev;
            }

            // Apply windowing function
            self.window.apply(&mut preemp);

            // Compute power spectrum
            let power = self.fft.power_spectrum(&preemp);

            // Compute normalized filterbank energies and MFCCs
            let mfcc = match self.feature_mode {
                FeatureMode::LogMel => {
                    let log_energies = self.mel.compute_log_energies(&power);
                    self.mel.compute_mfcc(&log_energies, self.num_mfcc)
                }
                FeatureMode::Pcen => {
                    let raw_energies = self.mel.compute_energies(&power);
                    let pcen_energies = self.pcen.process_frame(&raw_energies);
                    self.mel.compute_mfcc(&pcen_energies, self.num_mfcc)
                }
            };

            self.feature_history.push(mfcc);
            if self.feature_history.len() > self.max_history_frames {
                self.feature_history.remove(0);
            }

            // Run Sakoe-Chiba corridor DTW match against current sliding observation window
            if let Some((keyword, dist)) = self.dtw.match_window(&self.feature_history) {
                let timestamp_sec =
                    (self.total_samples_processed as f64) / (self.sample_rate as f64);
                let confidence = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);
                events.push(KeywordEvent {
                    keyword,
                    confidence,
                    timestamp_sec,
                });
                self.feature_history.clear(); // Reset history after match to avoid duplicate triggers
            }

            // Advance by hop size
            self.ring_buffer.pop_front(self.hop_size);
        }

        events
    }

    /// Compute feature frames directly for an input audio slice (useful for template generation).
    pub fn extract_features(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        if samples.len() < self.frame_size {
            return features;
        }

        let mut pos = 0;
        let mut frame = vec![0.0f32; self.frame_size];
        let mut pcen_clone = self.pcen.clone();
        pcen_clone.reset();

        while pos + self.frame_size <= samples.len() {
            frame.copy_from_slice(&samples[pos..pos + self.frame_size]);

            // Apply pre-emphasis
            if self.pre_emphasis_alpha > 0.0 {
                let mut prev = 0.0f32;
                for s in &mut frame {
                    let curr = *s;
                    *s = curr - self.pre_emphasis_alpha * prev;
                    prev = curr;
                }
            }

            self.window.apply(&mut frame);
            let power = self.fft.power_spectrum(&frame);

            let mfcc = match self.feature_mode {
                FeatureMode::LogMel => {
                    let log_energies = self.mel.compute_log_energies(&power);
                    self.mel.compute_mfcc(&log_energies, self.num_mfcc)
                }
                FeatureMode::Pcen => {
                    let raw_energies = self.mel.compute_energies(&power);
                    let pcen_energies = pcen_clone.process_frame(&raw_energies);
                    self.mel.compute_mfcc(&pcen_energies, self.num_mfcc)
                }
            };

            features.push(mfcc);
            pos += self.hop_size;
        }

        features
    }

    /// Mutable reference to internal PCEN filter.
    pub fn pcen_mut(&mut self) -> &mut PcenFilter {
        &mut self.pcen
    }

    /// Access internal DTW matcher.
    pub fn dtw(&self) -> &DtwMatcher {
        &self.dtw
    }

    /// Mutable access to internal DTW matcher.
    pub fn dtw_mut(&mut self) -> &mut DtwMatcher {
        &mut self.dtw
    }

    /// Return audio sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}
