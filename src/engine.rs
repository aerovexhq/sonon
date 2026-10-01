//! High-level streaming acoustic engine orchestrating VAD, MFCC/PCEN features, and keyword spotting.

use crate::dtw::DtwMatcher;
use crate::mel::MelFilterbank;
use crate::notch::RotorHarmonicNotchBank;
use crate::pcen::{PcenConfig, PcenFilter};
use crate::ring_buffer::AudioRingBuffer;
use crate::spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
use crate::stft::FftProcessor;
use crate::vad::EnergyVad;
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
    vad: EnergyVad,
    notch_bank: Option<RotorHarmonicNotchBank>,
    spectral_subtraction: Option<SpectralSubtractionSuppressor>,
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
        let vad = EnergyVad::new(2.5, 0.95, 5);
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
            vad,
            notch_bank: None,
            spectral_subtraction: None,
            feature_mode: FeatureMode::LogMel,
            pre_emphasis_alpha: 0.97,
            last_sample: 0.0,
            dtw,
            feature_history: Vec::with_capacity(128),
            max_history_frames: 64,
            total_samples_processed: 0,
        }
    }

    /// Enable drone rotor blade pass frequency (BPF) harmonic notch filtering.
    pub fn enable_rotor_notch(&mut self, num_blades: usize, num_harmonics: usize, q_factor: f32) {
        self.notch_bank = Some(RotorHarmonicNotchBank::new(
            self.sample_rate,
            num_blades,
            num_harmonics,
            q_factor,
        ));
    }

    /// Disable rotor notch filtering.
    pub fn disable_rotor_notch(&mut self) {
        self.notch_bank = None;
    }

    /// Update rotor RPM from autopilot or ESC telemetry.
    pub fn update_motor_rpm(&mut self, rpm: f32) {
        if let Some(ref mut bank) = self.notch_bank {
            bank.update_rpm(rpm);
        }
    }

    /// Update multi-motor RPM telemetry (e.g., 4 motors on a quadcopter).
    pub fn update_multi_motor_rpm(&mut self, motor_rpms: &[f32]) {
        if let Some(ref mut bank) = self.notch_bank {
            bank.update_multi_motor_rpm(motor_rpms);
        }
    }

    /// Return active notch filter frequencies in Hz.
    pub fn active_notch_frequencies(&self) -> Vec<f32> {
        self.notch_bank
            .as_ref()
            .map(|b| b.active_frequencies())
            .unwrap_or_default()
    }

    /// Enable spectral subtraction noise suppression.
    pub fn enable_spectral_subtraction(&mut self, config: SpectralSubtractionConfig) {
        let num_bins = self.frame_size / 2 + 1;
        self.spectral_subtraction = Some(SpectralSubtractionSuppressor::new(num_bins, config));
    }

    /// Disable spectral subtraction noise suppression.
    pub fn disable_spectral_subtraction(&mut self) {
        self.spectral_subtraction = None;
    }

    /// Return reference to internal Voice Activity Detector.
    pub fn vad(&self) -> &EnergyVad {
        &self.vad
    }

    /// Return mutable reference to internal Voice Activity Detector.
    pub fn vad_mut(&mut self) -> &mut EnergyVad {
        &mut self.vad
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

        // Apply rotor notch filtering to incoming samples if enabled
        let mut filtered_samples;
        let input_slice = if let Some(ref mut bank) = self.notch_bank {
            filtered_samples = samples.to_vec();
            bank.process_block(&mut filtered_samples);
            &filtered_samples[..]
        } else {
            samples
        };

        self.ring_buffer.push_slice(input_slice);
        self.total_samples_processed += input_slice.len() as u64;

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

            // Voice activity detection
            let is_speech = self.vad.process_frame(&preemp);

            // Apply windowing function
            self.window.apply(&mut preemp);

            // Compute power spectrum
            let mut power = self.fft.power_spectrum(&preemp);

            // Apply spectral subtraction noise suppression if enabled
            if let Some(ref mut ss) = self.spectral_subtraction {
                ss.process_spectrum(&mut power, is_speech);
            }

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

        // Apply notch filtering if active
        let mut filtered_samples;
        let effective_samples = if let Some(ref bank) = self.notch_bank {
            let mut b_clone = bank.clone();
            filtered_samples = samples.to_vec();
            b_clone.process_block(&mut filtered_samples);
            &filtered_samples[..]
        } else {
            samples
        };

        let mut pos = 0;
        let mut frame = vec![0.0f32; self.frame_size];
        let mut pcen_clone = self.pcen.clone();
        pcen_clone.reset();

        let mut ss_clone = self.spectral_subtraction.clone();

        while pos + self.frame_size <= effective_samples.len() {
            frame.copy_from_slice(&effective_samples[pos..pos + self.frame_size]);

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
            let mut power = self.fft.power_spectrum(&frame);

            if let Some(ref mut ss) = ss_clone {
                ss.process_spectrum(&mut power, true);
            }

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

    /// Clear internal state and history.
    pub fn reset(&mut self) {
        self.ring_buffer.clear();
        self.pcen.reset();
        self.vad = EnergyVad::new(2.5, 0.95, 5);
        if let Some(ref mut bank) = self.notch_bank {
            bank.reset();
        }
        if let Some(ref mut ss) = self.spectral_subtraction {
            ss.reset();
        }
        self.feature_history.clear();
        self.last_sample = 0.0;
        self.total_samples_processed = 0;
    }
}
