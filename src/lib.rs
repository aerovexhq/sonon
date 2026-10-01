//! # Sonon (`sonon`)
//!
//! Minimalist embedded acoustic DSP, Voice Activity Detection (VAD), Per-Channel Energy
//! Normalization (PCEN), telemetry-coupled rotor notch filtering, spectral subtraction,
//! multi-microphone spatial beamforming, AeroSSM Structured State-Space Models,
//! neuromorphic silicon cochlea, and spiking wake-word spotting engineered in pure safe Rust for autonomous robotics.

#![deny(unsafe_code)]

pub mod aec;
pub mod aerossm;
pub mod anticipatory;
pub mod beamforming;
pub mod capi;
pub mod dtw;
pub mod engine;
pub mod fixed;
pub mod health;
pub mod mel;
pub mod neuromorphic;
pub mod notch;
pub mod pcen;
pub mod phonetic;
pub mod ring_buffer;
pub mod shm;
pub mod spectral_subtraction;
pub mod stft;
pub mod vad;
pub mod window;

pub use aec::{AcousticEchoCanceller, AecConfig, DtdState};
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
pub use fixed::{FixedDtwMatcher, StaticAudioBuffer, Q15, Q31};
pub use health::{
    AcousticHealthMonitor, AirframeHealthSnapshot, AnomalySeverity, MavlinkNamedValueFloat,
    MotorHealthConfig, MotorHealthReport,
};
pub use mel::MelFilterbank;
pub use neuromorphic::{
    GammatoneFilter, NeuromorphicCochlea, SpikeEvent, SpikingKwsCell, SpikingKwsConfig,
};
pub use notch::{BiquadNotchFilter, RotorHarmonicNotchBank};
pub use pcen::{PcenConfig, PcenFilter};
pub use phonetic::{FormantTarget, G2pEngine, KlattSynthesizer, Phoneme, PhonemeSegment};
pub use ring_buffer::AudioRingBuffer;
pub use shm::{ShmAudioChannel, ShmHeader, DEFAULT_SHM_PATH};
pub use spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
pub use stft::FftProcessor;
pub use vad::EnergyVad;
pub use window::{Window, WindowType};
