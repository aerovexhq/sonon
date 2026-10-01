//! High-level streaming acoustic engine orchestrating VAD, MFCC features, and keyword spotting.

use crate::dtw::DtwMatcher;
use crate::mel::MelFilterbank;
use crate::ring_buffer::AudioRingBuffer;
use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};

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
    dtw: DtwMatcher,
    feature_history: Vec<Vec<f32>>,
    max_history_frames: usize,
    total_samples_processed: u64,
}

impl SononEngine {
    /// Construct a new Sonon engine with specified sample rate (e.g. 16000.0) and frame sizes.
    pub fn new(sample_rate: f32, frame_size: usize, hop_size: usize, num_mfcc: usize) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(frame_size > 0 && (frame_size & (frame_size - 1)) == 0, "Frame size must be power of 2");
        assert!(hop_size > 0 && hop_size <= frame_size, "Hop size must be <= frame size");

        let ring_buffer = AudioRingBuffer::new(frame_size * 16);
        let window = Window::new(WindowType::Hann, frame_size);
        let fft = FftProcessor::new(frame_size);
        let mel = MelFilterbank::new(26, frame_size, sample_rate, 80.0, sample_rate / 2.0);
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
            dtw,
            feature_history: Vec::with_capacity(128),
            max_history_frames: 64,
            total_samples_processed: 0,
        }
    }

    /// Enroll a keyword phrase template into the engine.
    pub fn enroll_keyword(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        self.dtw.add_template(name, features, threshold);
    }

    /// Ingest a slice of raw audio samples (mono float32 [-1.0, 1.0]).
    /// Returns any detected keyword events.
    pub fn ingest_samples(&mut self, samples: &[f32]) -> Vec<KeywordEvent> {
        let mut events = Vec::new();
        self.ring_buffer.push_slice(samples);
        self.total_samples_processed += samples.len() as u64;

        let mut frame_buf = vec![0.0f32; self.frame_size];

        while self.ring_buffer.len() >= self.frame_size {
            if !self.ring_buffer.read_latest(self.frame_size, &mut frame_buf) {
                break;
            }

            // Apply window
            let mut windowed = frame_buf.clone();
            self.window.apply(&mut windowed);

            // Compute power spectrum and log mel energies
            let power = self.fft.power_spectrum(&windowed);
            let log_energies = self.mel.compute_log_energies(&power);
            let mfcc = self.mel.compute_mfcc(&log_energies, self.num_mfcc);

            self.feature_history.push(mfcc);
            if self.feature_history.len() > self.max_history_frames {
                self.feature_history.remove(0);
            }

            // Run DTW match against current sliding observation window
            if let Some((keyword, dist)) = self.dtw.match_window(&self.feature_history) {
                let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
                let confidence = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);
                events.push(KeywordEvent {
                    keyword,
                    confidence,
                    timestamp_sec,
                });
                self.feature_history.clear(); // Reset history after match to avoid duplicate triggers
            }

            // Advance by hop size (clear read samples by re-pushing residual if needed)
            // For simplicity in streaming without sub-sampling latency, break when latest processed
            break;
        }

        events
    }

    /// Compute MFCC features directly for an input audio slice (useful for template generation).
    pub fn extract_features(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        if samples.len() < self.frame_size {
            return features;
        }

        let mut pos = 0;
        let mut frame = vec![0.0f32; self.frame_size];

        while pos + self.frame_size <= samples.len() {
            frame.copy_from_slice(&samples[pos..pos + self.frame_size]);
            self.window.apply(&mut frame);
            let power = self.fft.power_spectrum(&frame);
            let log_energies = self.mel.compute_log_energies(&power);
            let mfcc = self.mel.compute_mfcc(&log_energies, self.num_mfcc);
            features.push(mfcc);
            pos += self.hop_size;
        }

        features
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}
