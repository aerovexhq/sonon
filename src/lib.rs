//! # Sonon (`sonon`)
//!
//! Minimalist embedded acoustic DSP, Voice Activity Detection (VAD), and few-shot
//! keyword/phrase spotting engine engineered in pure safe Rust for autonomous robotics.

#![deny(unsafe_code)]

pub mod dtw;
pub mod engine;
pub mod mel;
pub mod ring_buffer;
pub mod stft;
pub mod vad;
pub mod window;

pub use dtw::{DtwMatcher, PhraseTemplate};
pub use engine::{KeywordEvent, SononEngine};
pub use mel::MelFilterbank;
pub use ring_buffer::AudioRingBuffer;
pub use stft::FftProcessor;
pub use vad::EnergyVad;
pub use window::{Window, WindowType};
