//! # Sonon (`sonon`)
//!
//! Minimalist embedded acoustic DSP, Voice Activity Detection (VAD), Per-Channel Energy
//! Normalization (PCEN), telemetry-coupled rotor notch filtering, spectral subtraction,
//! multi-microphone spatial beamforming, AeroSSM Structured State-Space Models, and
//! anticipatory prefix flight interlock engine engineered in pure safe Rust for autonomous robotics.

#![deny(unsafe_code)]

pub mod aerossm;
pub mod anticipatory;
pub mod beamforming;
pub mod dtw;
pub mod engine;
pub mod health;
pub mod mel;
pub mod notch;
pub mod pcen;
pub mod ring_buffer;
pub mod spectral_subtraction;
pub mod stft;
pub mod vad;
pub mod window;

pub use aerossm::{AeroSsmCell, SincConvFrontend};
pub use anticipatory::{AnticipatoryPrefixDecoder, InterlockState, WaldSprtConfig};
pub use beamforming::{
    ArrayGeometry, DelayAndSumBeamformer, DoaEstimator, GccPhatEstimator, Point3D, SPEED_OF_SOUND,
};
pub use dtw::{
    calibrate_threshold, dtw_barycenter_averaging, extract_warping_path_banded, DtwMatcher,
    PhraseTemplate,
};
pub use engine::{FeatureMode, KeywordEvent, SononEngine};
pub use health::{
    AcousticHealthMonitor, AirframeHealthSnapshot, AnomalySeverity, MavlinkNamedValueFloat,
    MotorHealthConfig, MotorHealthReport,
};
pub use mel::MelFilterbank;
pub use notch::{BiquadNotchFilter, RotorHarmonicNotchBank};
pub use pcen::{PcenConfig, PcenFilter};
pub use ring_buffer::AudioRingBuffer;
pub use spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
pub use stft::FftProcessor;
pub use vad::EnergyVad;
pub use window::{Window, WindowType};
