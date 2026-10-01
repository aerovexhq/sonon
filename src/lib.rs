//! # Sonon (`sonon`)
//!
//! Minimalist embedded acoustic DSP, Voice Activity Detection (VAD), Per-Channel Energy
//! Normalization (PCEN), and few-shot keyword/phrase spotting engine engineered in pure
//! safe Rust for autonomous robotics.

#![deny(unsafe_code)]

pub mod dtw;
pub mod engine;
pub mod mel;
pub mod pcen;
pub mod ring_buffer;
pub mod stft;
pub mod vad;
pub mod window;

pub use dtw::{
    calibrate_threshold, dtw_barycenter_averaging, extract_warping_path_banded, DtwMatcher,
    PhraseTemplate,
};
pub use engine::{FeatureMode, KeywordEvent, SononEngine};
pub use mel::MelFilterbank;
pub use pcen::{PcenConfig, PcenFilter};
pub use ring_buffer::AudioRingBuffer;
pub use stft::FftProcessor;
pub use vad::EnergyVad;
pub use window::{Window, WindowType};
