//! # Sonon (`sonon`)
//!
//! Minimalist embedded acoustic DSP, Voice Activity Detection (VAD), Per-Channel Energy
//! Normalization (PCEN), telemetry-coupled rotor notch filtering, spectral subtraction,
//! multi-microphone spatial beamforming, AeroSSM Structured State-Space Models,
//! neuromorphic silicon cochlea, and spiking wake-word spotting engineered in pure safe Rust for autonomous robotics.

#![deny(unsafe_code)]

pub mod aec;
pub mod aeroacoustics;
pub mod aerossm;
pub mod anticipatory;
pub mod beamforming;
pub mod capi;
pub mod cwt;
pub mod doppler;
pub mod dtw;
pub mod echolocation;
pub mod engine;
pub mod fixed;
pub mod health;
pub mod mel;
pub mod neuromorphic;
pub mod notch;
pub mod ormia;
pub mod pcen;
pub mod phonetic;
pub mod psychoacoustic;
pub mod riscv_pulp;
pub mod ring_buffer;
pub mod shm;
pub mod spectral_subtraction;
pub mod stft;
pub mod swarm_mesh;
pub mod tse;
pub mod vad;
pub mod wasm;
pub mod wind;
pub mod window;
pub mod zero_shot;

pub use aec::{AcousticEchoCanceller, AecConfig, DtdState};
pub use aeroacoustics::{
    a_weighting_db, bessel_j, AeroacousticConfig, AeroacousticInverter, AeroacousticTelemetry,
    DirectivitySphere3D, GroundNoiseFootprint, RotorGeometry,
};
pub use aerossm::{AeroSsmCell, SincConvFrontend};
pub use anticipatory::{AnticipatoryPrefixDecoder, InterlockState, WaldSprtConfig};
pub use beamforming::{
    ArrayGeometry, DelayAndSumBeamformer, DoaEstimator, GccPhatEstimator, Point3D, SPEED_OF_SOUND,
};
pub use cwt::{
    ContinuousWaveletFilterbank, CwtProfilerConfig, CwtScalogram, RotorDamageProfiler,
    RotorDamageReport, WaveletType,
};
pub use doppler::{speed_of_sound_at_temp, DopplerCompensator, DopplerConfig};
pub use dtw::{
    calibrate_threshold, dtw_barycenter_averaging, extract_warping_path_banded,
    weighted_euclidean_distance, AcousticNoiseClusterTracker, ConfusionMatrix, DtwMatcher,
    PhraseTemplate, QuantizedDtwMatcher, QuantizedFrame, QuantizedPhraseTemplate,
    StreamingDtwConfig, StreamingMatchResult,
};
pub use echolocation::{
    AcousticObstacle, AcousticPointCloud, CaCfarConfig, ChirpConfig, LfmChirpGenerator,
    MultiMicAcousticEcholocator, SubterraneanCaveSimulator,
};
pub use engine::{EvaluationReport, FeatureMode, KeywordEvent, SononEngine};
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
pub use ormia::{
    OrmiaBiquad, OrmiaBridgeFilter, OrmiaConfig, OrmiaDirectionEstimator, OrmiaTelemetry,
};
pub use pcen::{PcenConfig, PcenFilter};
pub use phonetic::{
    encode_wav_16bit, write_wav_file, ExemplarVariationConfig, FormantTarget, G2pEngine,
    IntonationContour, KlattSynthesizer, LiljencrantsFantPulse, Phoneme, PhonemeSegment,
    SyntheticExemplarGenerator, SyntheticExemplarMetadata, VocalAccent,
};
pub use psychoacoustic::{
    atmospheric_absorption_db_km, bark_to_freq, freq_to_bark, threshold_in_quiet_db,
    AcousticStealthReport, BarkBand, PsychoacousticConfig, PsychoacousticStealthEngine,
    BARK_BANDS_25,
};
pub use riscv_pulp::{
    PackedI16x2, PackedI8x4, PulpConfig, PulpMemoryArena, PulpPowerModel, PulpTelemetry,
    PulpVectorEngine, RvvConfig, RvvVectorEngine,
};
pub use ring_buffer::{AudioRingBuffer, FeatureRingBuffer};
pub use shm::{ShmAudioChannel, ShmHeader, DEFAULT_SHM_PATH};
pub use spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
pub use stft::FftProcessor;
pub use swarm_mesh::{
    DEFAULT_SPEED_OF_SOUND, SwarmClockSync, SwarmCovarianceConsensus, SwarmMeshConfig,
    SwarmNodeState, SwarmTargetReport, SyntheticApertureBeamformer,
};
pub use tse::{
    calculate_bearing_from_gps, Complex32, FlightDynamicsSimulator, GpsCoordinate,
    SpatialConditioningTarget, TargetSoundExtractor, TseConfig, TseReport,
};
pub use vad::EnergyVad;
pub use wind::{
    AdaptiveRumbleFilter, AeroacousticWindSimulator, TurbulentBoundaryLayerSuppressor,
    WindNoiseTelemetry, WindTurbulenceConfig,
};
pub use window::{Window, WindowType};
pub use zero_shot::{
    CrossAttentionAligner, CrossAttentionResult, FoilType, MultiLingualG2p,
    PhoneticEmbeddingSpace, PhoneticFoil, PhoneticFoilGenerator, SupportedLanguage,
    ZeroShotCalibrationReport, ZeroShotCalibrator,
};
