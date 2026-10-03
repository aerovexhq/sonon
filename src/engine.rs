use crate::adaptation::{ActiveLearningCandidate, AdaptationTelemetry, ContinualAdaptationEngine};
use crate::aec::{AcousticEchoCanceller, AecConfig};
use crate::aeroacoustics::{
    AeroacousticConfig, AeroacousticInverter, AeroacousticTelemetry, DirectivitySphere3D,
    GroundNoiseFootprint, RotorGeometry,
};
use crate::beamforming::{ArrayGeometry, Point3D};
use crate::cwt::{CwtProfilerConfig, RotorDamageProfiler, RotorDamageReport};
use crate::doppler::{DopplerCompensator, DopplerConfig};
use crate::dtw::{
    AcousticNoiseClusterTracker, ConfusionMatrix, DtwMatcher, QuantizedDtwMatcher,
    QuantizedPhraseTemplate, StreamingDtwConfig,
};
use crate::echolocation::{AcousticPointCloud, CaCfarConfig, ChirpConfig, MultiMicAcousticEcholocator};
use crate::health::{AcousticHealthMonitor, AirframeHealthSnapshot, MotorHealthConfig};
use crate::mel::MelFilterbank;
use crate::notch::RotorHarmonicNotchBank;
use crate::ormia::{OrmiaConfig, OrmiaDirectionEstimator, OrmiaTelemetry};
use crate::pcen::{PcenConfig, PcenFilter};
use crate::phonetic::{G2pEngine, KlattSynthesizer, SyntheticExemplarGenerator, VocalAccent};
use crate::psychoacoustic::{AcousticStealthReport, PsychoacousticConfig, PsychoacousticStealthEngine};
use crate::zero_shot::{
    CrossAccentCalibrationReport, MultiAccentCalibrator, SupportedLanguage,
    ZeroShotCalibrationReport, ZeroShotCalibrator,
};
use crate::riscv_pulp::{PulpConfig, PulpPowerModel, PulpTelemetry};
use crate::ring_buffer::{AudioRingBuffer, FeatureRingBuffer};
use crate::spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
use crate::stft::FftProcessor;
use crate::subbyte::{SubByteBitWidth, SubByteDtwMatcher, SubBytePhraseTemplate};
use crate::swarm_mesh::{SwarmMeshConfig, SwarmNodeState, SwarmTargetReport, SyntheticApertureBeamformer};
use crate::tse::{GpsCoordinate, TargetSoundExtractor, TseConfig, TseReport};
use crate::ctc_beam_search::{CommandRecognitionResult, CtcCommandDecoder, CtcDecoderConfig};
use crate::spiking_vad::{SpikingNeuralVad, SpikingVadConfig, SpikingVadTelemetry};
use crate::vad::EnergyVad;
use crate::voiceprint::{OperatorVerifier, SpeakerVoiceprint, VerificationDecision};
use crate::wind::{TurbulentBoundaryLayerSuppressor, WindNoiseTelemetry, WindTurbulenceConfig};
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
    pub authorized_operator: Option<String>,
    pub voiceprint_similarity: Option<f32>,
    pub is_anti_spoof_verified: bool,
}

impl KeywordEvent {
    /// Construct a basic keyword event without operator authentication.
    pub fn new(keyword: impl Into<String>, confidence: f32, timestamp_sec: f64) -> Self {
        Self {
            keyword: keyword.into(),
            confidence,
            timestamp_sec,
            authorized_operator: None,
            voiceprint_similarity: None,
            is_anti_spoof_verified: false,
        }
    }
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
    health_monitor: Option<AcousticHealthMonitor>,
    latest_health_snapshot: Option<AirframeHealthSnapshot>,
    cwt_profiler: Option<RotorDamageProfiler>,
    latest_cwt_report: Option<RotorDamageReport>,
    feature_mode: FeatureMode,
    pre_emphasis_alpha: f32,
    last_sample: f32,
    dtw: DtwMatcher,
    feature_history: Vec<Vec<f32>>,
    max_history_frames: usize,
    total_samples_processed: u64,
    aec: Option<AcousticEchoCanceller>,
    doppler: Option<DopplerCompensator>,
    tse: Option<TargetSoundExtractor>,
    latest_tse_report: Option<TseReport>,
    ormia: Option<OrmiaDirectionEstimator>,
    latest_ormia_telemetry: Option<OrmiaTelemetry>,
    psychoacoustic: Option<PsychoacousticStealthEngine>,
    latest_stealth_report: Option<AcousticStealthReport>,
    pulp_model: Option<PulpPowerModel>,
    latest_pulp_telemetry: Option<PulpTelemetry>,
    wind_suppressor: Option<TurbulentBoundaryLayerSuppressor>,
    latest_wind_telemetry: Option<WindNoiseTelemetry>,
    echolocator: Option<MultiMicAcousticEcholocator>,
    latest_point_cloud: Option<AcousticPointCloud>,
    aeroacoustic_inverter: Option<AeroacousticInverter>,
    latest_directivity_sphere: Option<DirectivitySphere3D>,
    latest_ground_footprint: Option<GroundNoiseFootprint>,
    latest_aeroacoustic_telemetry: Option<AeroacousticTelemetry>,
    swarm_beamformer: Option<SyntheticApertureBeamformer>,
    latest_swarm_target_report: Option<SwarmTargetReport>,
    current_motor_rpms: Vec<f32>,
    num_mel_filters: usize,
    streaming_dtw_config: StreamingDtwConfig,
    refractory_lockout_remaining: usize,
    feature_ring_buffer: FeatureRingBuffer,
    noise_tracker: AcousticNoiseClusterTracker,
    continual_adaptation: Option<ContinualAdaptationEngine>,
    latest_adaptation_telemetry: Option<AdaptationTelemetry>,
    spiking_vad: Option<SpikingNeuralVad>,
    spiking_vad_gating: bool,
    latest_spiking_vad_telemetry: Option<SpikingVadTelemetry>,
    vtln_alpha: f32,
    cached_vtln_mel: Option<MelFilterbank>,
    operator_verifier: Option<OperatorVerifier>,
    latest_verification_decision: Option<VerificationDecision>,
    rolling_audio_cache: AudioRingBuffer,
    ctc_decoder: Option<CtcCommandDecoder>,
    latest_recognized_command: Option<CommandRecognitionResult>,
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
        let rolling_audio_cache = AudioRingBuffer::new(((sample_rate * 3.0) as usize).max(frame_size * 32));
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
            health_monitor: None,
            latest_health_snapshot: None,
            cwt_profiler: None,
            latest_cwt_report: None,
            feature_mode: FeatureMode::LogMel,
            pre_emphasis_alpha: 0.97,
            last_sample: 0.0,
            dtw,
            feature_history: Vec::with_capacity(128),
            max_history_frames: 64,
            total_samples_processed: 0,
            aec: None,
            doppler: None,
            tse: None,
            latest_tse_report: None,
            ormia: None,
            latest_ormia_telemetry: None,
            psychoacoustic: None,
            latest_stealth_report: None,
            pulp_model: None,
            latest_pulp_telemetry: None,
            wind_suppressor: None,
            latest_wind_telemetry: None,
            echolocator: None,
            latest_point_cloud: None,
            aeroacoustic_inverter: None,
            latest_directivity_sphere: None,
            latest_ground_footprint: None,
            latest_aeroacoustic_telemetry: None,
            swarm_beamformer: None,
            latest_swarm_target_report: None,
            current_motor_rpms: Vec::new(),
            num_mel_filters,
            streaming_dtw_config: StreamingDtwConfig::default(),
            refractory_lockout_remaining: 0,
            feature_ring_buffer: FeatureRingBuffer::new(128, num_mfcc),
            noise_tracker: AcousticNoiseClusterTracker::new(num_mfcc, 0.05),
            continual_adaptation: None,
            latest_adaptation_telemetry: None,
            spiking_vad: None,
            spiking_vad_gating: true,
            latest_spiking_vad_telemetry: None,
            vtln_alpha: 1.0,
            cached_vtln_mel: None,
            operator_verifier: None,
            latest_verification_decision: None,
            rolling_audio_cache,
            ctc_decoder: None,
            latest_recognized_command: None,
        }
    }

    /// Enable Continuous Wavelet Transform (CWT) rotor micro-damage profiler.
    pub fn enable_cwt_profiler(&mut self, config: CwtProfilerConfig, num_blades: usize) {
        self.cwt_profiler = Some(RotorDamageProfiler::new(self.sample_rate, config, num_blades));
    }

    /// Disable CWT rotor micro-damage profiler.
    pub fn disable_cwt_profiler(&mut self) {
        self.cwt_profiler = None;
        self.latest_cwt_report = None;
    }

    /// Access reference to active CWT rotor micro-damage profiler if enabled.
    pub fn cwt_profiler(&self) -> Option<&RotorDamageProfiler> {
        self.cwt_profiler.as_ref()
    }

    /// Access mutable reference to active CWT rotor micro-damage profiler if enabled.
    pub fn cwt_profiler_mut(&mut self) -> Option<&mut RotorDamageProfiler> {
        self.cwt_profiler.as_mut()
    }

    /// Return latest evaluated CWT rotor micro-damage diagnostic report.
    pub fn latest_cwt_report(&self) -> Option<&RotorDamageReport> {
        self.latest_cwt_report.as_ref()
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

    /// Enable acoustic health monitoring and blade anomaly diagnostics.
    pub fn enable_health_monitoring(&mut self, config: MotorHealthConfig) {
        self.health_monitor = Some(AcousticHealthMonitor::new(
            self.sample_rate,
            self.frame_size,
            config,
        ));
    }

    /// Disable acoustic health monitoring.
    pub fn disable_health_monitoring(&mut self) {
        self.health_monitor = None;
        self.latest_health_snapshot = None;
    }

    /// Return latest evaluated airframe health snapshot.
    pub fn latest_health_snapshot(&self) -> Option<&AirframeHealthSnapshot> {
        self.latest_health_snapshot.as_ref()
    }

    /// Update rotor RPM from autopilot or ESC telemetry.
    pub fn update_motor_rpm(&mut self, rpm: f32) {
        self.current_motor_rpms = vec![rpm];
        if let Some(ref mut bank) = self.notch_bank {
            bank.update_rpm(rpm);
        }
        if let Some(ref mut monitor) = self.health_monitor {
            monitor.update_motor_rpm(0, rpm);
        }
        if let Some(ref mut cwt) = self.cwt_profiler {
            cwt.update_rpm(rpm);
        }
        if let Some(ref mut extractor) = self.tse {
            extractor.set_motor_rpms(&[rpm], 2);
        }
    }

    /// Update multi-motor RPM telemetry (e.g., 4 motors on a quadcopter).
    pub fn update_multi_motor_rpm(&mut self, motor_rpms: &[f32]) {
        self.current_motor_rpms = motor_rpms.to_vec();
        if let Some(ref mut bank) = self.notch_bank {
            bank.update_multi_motor_rpm(motor_rpms);
        }
        if let Some(ref mut monitor) = self.health_monitor {
            monitor.update_motor_rpms(motor_rpms);
        }
        if let Some(ref mut cwt) = self.cwt_profiler {
            if let Some(&rpm0) = motor_rpms.first() {
                cwt.update_rpm(rpm0);
            }
        }
        if let Some(ref mut extractor) = self.tse {
            extractor.set_motor_rpms(motor_rpms, 2);
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

    /// Enable Acoustic Echo Cancellation (AEC) with specified configuration.
    pub fn enable_aec(&mut self, config: AecConfig) {
        self.aec = Some(AcousticEchoCanceller::new(config));
    }

    /// Disable Acoustic Echo Cancellation.
    pub fn disable_aec(&mut self) {
        self.aec = None;
    }

    /// Access reference to active Acoustic Echo Canceller if enabled.
    pub fn aec(&self) -> Option<&AcousticEchoCanceller> {
        self.aec.as_ref()
    }

    /// Access mutable reference to active Acoustic Echo Canceller if enabled.
    pub fn aec_mut(&mut self) -> Option<&mut AcousticEchoCanceller> {
        self.aec.as_mut()
    }

    /// Enable Doppler shift compensation and kinematic velocity frequency warping.
    pub fn enable_doppler_compensation(&mut self, config: DopplerConfig) {
        self.doppler = Some(DopplerCompensator::new(
            config,
            self.num_mel_filters,
            self.frame_size,
            self.sample_rate,
            80.0,
            self.sample_rate / 2.0,
        ));
    }

    /// Disable Doppler shift compensation.
    pub fn disable_doppler_compensation(&mut self) {
        self.doppler = None;
    }

    /// Access reference to active Doppler compensator if enabled.
    pub fn doppler(&self) -> Option<&DopplerCompensator> {
        self.doppler.as_ref()
    }

    /// Access mutable reference to active Doppler compensator if enabled.
    pub fn doppler_mut(&mut self) -> Option<&mut DopplerCompensator> {
        self.doppler.as_mut()
    }

    /// Update drone 3D flight velocity vector (vx, vy, vz in m/s) from autopilot/MAVLink telemetry.
    pub fn update_kinematic_velocity(&mut self, vx: f32, vy: f32, vz: f32) {
        if let Some(ref mut d) = self.doppler {
            d.update_velocity_3d(vx, vy, vz);
        }
    }

    /// Enable Vocal Tract Length Normalization (VTLN) frequency warping with warping factor alpha.
    /// Warping factor $\alpha \in [0.70, 1.40]$ normalizes vocal tract length variations across
    /// speakers ($\alpha > 1.0$ compresses high formant frequencies for shorter vocal tracts,
    /// $\alpha < 1.0$ expands lower formant frequencies for longer vocal tracts).
    pub fn enable_vtln(&mut self, alpha: f32) {
        let alpha_clamped = alpha.clamp(0.51, 1.99);
        self.vtln_alpha = alpha_clamped;
        self.cached_vtln_mel = Some(MelFilterbank::new_with_vtln(
            self.num_mel_filters,
            self.frame_size,
            self.sample_rate,
            80.0,
            self.sample_rate / 2.0,
            alpha_clamped,
        ));
    }

    /// Disable Vocal Tract Length Normalization (VTLN).
    pub fn disable_vtln(&mut self) {
        self.vtln_alpha = 1.0;
        self.cached_vtln_mel = None;
    }

    /// Check whether Vocal Tract Length Normalization (VTLN) is active.
    pub fn is_vtln_active(&self) -> bool {
        self.cached_vtln_mel.is_some()
    }

    /// Current VTLN frequency warping factor alpha.
    pub fn vtln_alpha(&self) -> f32 {
        self.vtln_alpha
    }

    /// Enable Acoustic Directional Target Sound Extraction (TSE) with steered MVDR and spatial gating.
    pub fn enable_target_sound_extractor(&mut self, geometry: ArrayGeometry, config: TseConfig) {
        self.tse = Some(TargetSoundExtractor::new(geometry, self.sample_rate, config));
    }

    /// Disable Target Sound Extraction.
    pub fn disable_target_sound_extractor(&mut self) {
        self.tse = None;
        self.latest_tse_report = None;
    }

    /// Access reference to active Target Sound Extractor if enabled.
    pub fn target_sound_extractor(&self) -> Option<&TargetSoundExtractor> {
        self.tse.as_ref()
    }

    /// Access mutable reference to active Target Sound Extractor if enabled.
    pub fn target_sound_extractor_mut(&mut self) -> Option<&mut TargetSoundExtractor> {
        self.tse.as_mut()
    }

    /// Return latest evaluated Target Sound Extraction diagnostic report.
    pub fn latest_tse_report(&self) -> Option<&TseReport> {
        self.latest_tse_report.as_ref()
    }

    /// Update target operator line-of-sight bearing from azimuth and elevation angles in radians.
    pub fn update_target_bearing(&mut self, azimuth_rad: f32, elevation_rad: f32) {
        if let Some(ref mut d) = self.doppler {
            d.update_target_bearing(azimuth_rad, elevation_rad);
        }
        if let Some(ref mut extractor) = self.tse {
            extractor.set_target_bearing(azimuth_rad, elevation_rad);
        }
    }

    /// Dynamically update steered target using GPS drone and operator coordinates with drone yaw heading.
    pub fn update_target_gps(
        &mut self,
        drone_gps: GpsCoordinate,
        operator_gps: GpsCoordinate,
        drone_yaw_rad: f32,
    ) {
        if let Some(ref mut extractor) = self.tse {
            extractor.set_target_gps(drone_gps, operator_gps, drone_yaw_rad);
        }
    }

    /// Configure motor null coordinates in the airframe body frame for the Target Sound Extractor.
    pub fn set_tse_motor_null_positions(&mut self, motor_positions: &[Point3D]) {
        if let Some(ref mut extractor) = self.tse {
            extractor.set_motor_null_positions(motor_positions);
        }
    }

    /// Configure explicit spatial null directions (azimuth_rad, elevation_rad) for the Target Sound Extractor.
    pub fn set_tse_spatial_null_directions(&mut self, null_directions: &[(f32, f32)]) {
        if let Some(ref mut extractor) = self.tse {
            extractor.set_spatial_null_directions(null_directions);
        }
    }

    /// Disable spatial null constraints in the Target Sound Extractor.
    pub fn clear_tse_spatial_nulls(&mut self) {
        if let Some(ref mut extractor) = self.tse {
            extractor.clear_spatial_nulls();
        }
    }

    /// Ingest multi-channel audio through steered MVDR and spatial mask gating, feeding extracted target stream into keyword spotting.
    pub fn ingest_multi_channel_tse(
        &mut self,
        multi_channel_inputs: &[&[f32]],
    ) -> Result<Vec<KeywordEvent>, String> {
        let (hop, num_mics) = {
            let tse = self.tse.as_ref().ok_or_else(|| "TSE is not enabled".to_string())?;
            (tse.config().hop_size, tse.num_mics())
        };

        let num_channels = multi_channel_inputs.len();
        if num_channels < num_mics {
            return Err(format!(
                "TSE requires at least {} channels, but only {} provided",
                num_mics, num_channels
            ));
        }

        let input_len = multi_channel_inputs[0].len();
        for ch in multi_channel_inputs {
            if ch.len() != input_len {
                return Err("All multi-channel inputs must have identical length".to_string());
            }
        }

        let mut events = Vec::new();
        let mut offset = 0;
        let mut extracted_mono = vec![0.0f32; hop];

        while offset + hop <= input_len {
            let mut slices = Vec::with_capacity(num_channels);
            for ch in multi_channel_inputs {
                slices.push(&ch[offset..offset + hop]);
            }
            let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
            let report = {
                let tse = self.tse.as_mut().unwrap();
                tse.process_block(&slices, &mut extracted_mono, timestamp_sec)
            };
            self.latest_tse_report = Some(report);

            let block_events = self.ingest_samples(&extracted_mono);
            events.extend(block_events);
            offset += hop;
        }

        Ok(events)
    }

    /// Enable bio-inspired Ormia ochracea micro-tympanum differential microphone emulation.
    pub fn enable_ormia_bridge(&mut self, config: OrmiaConfig) {
        self.ormia = Some(OrmiaDirectionEstimator::new(config));
    }

    /// Disable Ormia micro-tympanum bridge.
    pub fn disable_ormia_bridge(&mut self) {
        self.ormia = None;
        self.latest_ormia_telemetry = None;
    }

    /// Access reference to active Ormia direction estimator if enabled.
    pub fn ormia(&self) -> Option<&OrmiaDirectionEstimator> {
        self.ormia.as_ref()
    }

    /// Access mutable reference to active Ormia direction estimator if enabled.
    pub fn ormia_mut(&mut self) -> Option<&mut OrmiaDirectionEstimator> {
        self.ormia.as_mut()
    }

    /// Return latest evaluated Ormia micro-tympanum direction and spatial amplification telemetry.
    pub fn latest_ormia_telemetry(&self) -> Option<&OrmiaTelemetry> {
        self.latest_ormia_telemetry.as_ref()
    }

    /// Ingest dual-microphone audio streams through Ormia ochracea inter-tympanic bridge,
    /// amplifying sub-2mm IID/ITD cues, estimating 3D line-of-sight source azimuth,
    /// and feeding the spatially enhanced audio stream into the keyword spotter.
    pub fn process_dual_mic_ormia(
        &mut self,
        mic1: &[f32],
        mic2: &[f32],
    ) -> Result<Vec<KeywordEvent>, String> {
        if mic1.len() != mic2.len() {
            return Err("Dual microphone inputs must have identical length".to_string());
        }

        let mut out1 = vec![0.0f32; mic1.len()];
        let mut out2 = vec![0.0f32; mic1.len()];

        let telemetry = {
            let ormia = self.ormia.as_mut().ok_or_else(|| "Ormia bridge is not enabled".to_string())?;
            ormia.process_block(mic1, mic2, &mut out1, &mut out2)
        };
        self.latest_ormia_telemetry = Some(telemetry.clone());

        // Construct spatially enhanced mono stream:
        // Weight towards ipsilateral channel with highest amplification
        let mut enhanced = vec![0.0f32; mic1.len()];
        let w1 = (0.5 * (1.0 + (telemetry.azimuth_deg / 90.0).clamp(-1.0, 1.0))).clamp(0.0, 1.0);
        let w2 = 1.0 - w1;
        for i in 0..mic1.len() {
            enhanced[i] = w1 * out1[i] + w2 * out2[i];
        }

        let events = self.ingest_samples(&enhanced);
        Ok(events)
    }

    /// Enable ISO/IEC 11172-3 psychoacoustic masking model and active drone acoustic stealth engine.
    pub fn enable_psychoacoustic_stealth(&mut self, config: PsychoacousticConfig) {
        self.psychoacoustic = Some(PsychoacousticStealthEngine::new(config));
    }

    /// Disable psychoacoustic masking and stealth engine.
    pub fn disable_psychoacoustic_stealth(&mut self) {
        self.psychoacoustic = None;
        self.latest_stealth_report = None;
    }

    /// Access reference to active psychoacoustic stealth engine if enabled.
    pub fn psychoacoustic(&self) -> Option<&PsychoacousticStealthEngine> {
        self.psychoacoustic.as_ref()
    }

    /// Access mutable reference to active psychoacoustic stealth engine if enabled.
    pub fn psychoacoustic_mut(&mut self) -> Option<&mut PsychoacousticStealthEngine> {
        self.psychoacoustic.as_mut()
    }

    /// Return latest evaluated acoustic stealth and human detectability diagnostic report.
    pub fn latest_stealth_report(&self) -> Option<&AcousticStealthReport> {
        self.latest_stealth_report.as_ref()
    }

    /// Enable ultra-low-power PULP / RISC-V edge surveillance energy model.
    pub fn enable_pulp_acceleration(&mut self, config: PulpConfig) {
        self.pulp_model = Some(PulpPowerModel::new(config));
    }

    /// Disable PULP / RISC-V acceleration energy model.
    pub fn disable_pulp_acceleration(&mut self) {
        self.pulp_model = None;
        self.latest_pulp_telemetry = None;
    }

    /// Access reference to active PULP power model if enabled.
    pub fn pulp_model(&self) -> Option<&PulpPowerModel> {
        self.pulp_model.as_ref()
    }

    /// Access latest evaluated PULP surveillance telemetry.
    pub fn latest_pulp_telemetry(&self) -> Option<&PulpTelemetry> {
        self.latest_pulp_telemetry.as_ref()
    }

    /// Enable aerodynamic wind buffeting and turbulent boundary layer (TBL) suppression.
    pub fn enable_wind_suppression(&mut self, config: WindTurbulenceConfig) {
        self.wind_suppressor = Some(TurbulentBoundaryLayerSuppressor::new(config));
    }

    /// Disable aerodynamic wind suppression.
    pub fn disable_wind_suppression(&mut self) {
        self.wind_suppressor = None;
        self.latest_wind_telemetry = None;
    }

    /// Access reference to active turbulent boundary layer wind suppressor if enabled.
    pub fn wind_suppressor(&self) -> Option<&TurbulentBoundaryLayerSuppressor> {
        self.wind_suppressor.as_ref()
    }

    /// Access mutable reference to active turbulent boundary layer wind suppressor if enabled.
    pub fn wind_suppressor_mut(&mut self) -> Option<&mut TurbulentBoundaryLayerSuppressor> {
        self.wind_suppressor.as_mut()
    }

    /// Return latest evaluated wind buffeting and TBL telemetry snapshot.
    pub fn latest_wind_telemetry(&self) -> Option<&WindNoiseTelemetry> {
        self.latest_wind_telemetry.as_ref()
    }

    /// Update flight vehicle forward airspeed (in m/s) to tune the adaptive aerodynamic rumble filter.
    pub fn update_flight_airspeed(&mut self, airspeed_mps: f32) {
        if let Some(ref mut wind) = self.wind_suppressor {
            wind.update_airspeed(airspeed_mps);
        }
    }

    /// Process dual-microphone audio streams through the turbulent boundary layer suppressor,
    /// separating aerodynamic convective pseudosound from true propagating acoustic sound waves,
    /// and feeding the restored speech stream directly into the keyword spotter.
    pub fn process_dual_mic_wind_suppression(
        &mut self,
        mic1: &[f32],
        mic2: &[f32],
    ) -> Result<Vec<KeywordEvent>, String> {
        if mic1.len() != mic2.len() {
            return Err("Dual microphone inputs must have identical length".to_string());
        }

        let hop = {
            let suppressor = self
                .wind_suppressor
                .as_ref()
                .ok_or_else(|| "Wind suppressor is not enabled".to_string())?;
            suppressor.config().hop_size
        };

        let mut events = Vec::new();
        let mut offset = 0;
        let mut clean_block = vec![0.0f32; hop];

        while offset + hop <= mic1.len() {
            let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
            let telemetry = {
                let suppressor = self.wind_suppressor.as_mut().unwrap();
                suppressor.process_block(
                    &mic1[offset..offset + hop],
                    &mic2[offset..offset + hop],
                    &mut clean_block,
                    timestamp_sec,
                )
            };
            self.latest_wind_telemetry = Some(telemetry);

            let block_events = self.ingest_samples(&clean_block);
            events.extend(block_events);
            offset += hop;
        }

        Ok(events)
    }

    /// Enable active acoustic echolocation and 3D obstacle point cloud mapping.
    pub fn enable_acoustic_echolocation(
        &mut self,
        geometry: ArrayGeometry,
        chirp_config: ChirpConfig,
        cfar_config: CaCfarConfig,
    ) {
        self.echolocator = Some(MultiMicAcousticEcholocator::new(
            geometry,
            chirp_config,
            cfar_config,
        ));
    }

    /// Disable acoustic echolocation.
    pub fn disable_acoustic_echolocation(&mut self) {
        self.echolocator = None;
        self.latest_point_cloud = None;
    }

    /// Access reference to active acoustic echolocator if enabled.
    pub fn echolocator(&self) -> Option<&MultiMicAcousticEcholocator> {
        self.echolocator.as_ref()
    }

    /// Access mutable reference to active acoustic echolocator if enabled.
    pub fn echolocator_mut(&mut self) -> Option<&mut MultiMicAcousticEcholocator> {
        self.echolocator.as_mut()
    }

    /// Return latest evaluated 3D acoustic obstacle point cloud.
    pub fn latest_point_cloud(&self) -> Option<&AcousticPointCloud> {
        self.latest_point_cloud.as_ref()
    }

    /// Process multi-channel audio recordings through the acoustic echolocator,
    /// executing matched filter pulse compression, CA-CFAR detection, and 3D point cloud triangulation.
    pub fn process_multi_channel_echolocation(
        &mut self,
        multi_channel_inputs: &[&[f32]],
        tx_reference: Option<&[f32]>,
    ) -> Result<AcousticPointCloud, String> {
        let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
        let cloud = {
            let echolocator = self
                .echolocator
                .as_mut()
                .ok_or_else(|| "Acoustic echolocator is not enabled".to_string())?;
            echolocator.process_ping(multi_channel_inputs, tx_reference, timestamp_sec)
        };
        self.latest_point_cloud = Some(cloud.clone());
        Ok(cloud)
    }

    /// Enable physics-informed aeroacoustic inverse source reconstruction and far-field directivity mapping.
    pub fn enable_aeroacoustic_inversion(
        &mut self,
        rotors: Vec<RotorGeometry>,
        mic_positions: Vec<Point3D>,
        config: AeroacousticConfig,
    ) {
        self.aeroacoustic_inverter = Some(AeroacousticInverter::new(rotors, mic_positions, config));
    }

    /// Disable aeroacoustic inversion.
    pub fn disable_aeroacoustic_inversion(&mut self) {
        self.aeroacoustic_inverter = None;
        self.latest_directivity_sphere = None;
        self.latest_ground_footprint = None;
        self.latest_aeroacoustic_telemetry = None;
    }

    /// Access reference to active aeroacoustic inverter if enabled.
    pub fn aeroacoustic_inverter(&self) -> Option<&AeroacousticInverter> {
        self.aeroacoustic_inverter.as_ref()
    }

    /// Access mutable reference to active aeroacoustic inverter if enabled.
    pub fn aeroacoustic_inverter_mut(&mut self) -> Option<&mut AeroacousticInverter> {
        self.aeroacoustic_inverter.as_mut()
    }

    /// Return latest evaluated 3D radiation directivity sphere.
    pub fn latest_directivity_sphere(&self) -> Option<&DirectivitySphere3D> {
        self.latest_directivity_sphere.as_ref()
    }

    /// Return latest evaluated 2D ground noise footprint projection.
    pub fn latest_ground_noise_footprint(&self) -> Option<&GroundNoiseFootprint> {
        self.latest_ground_footprint.as_ref()
    }

    /// Return latest evaluated aeroacoustic telemetry snapshot.
    pub fn latest_aeroacoustic_telemetry(&self) -> Option<&AeroacousticTelemetry> {
        self.latest_aeroacoustic_telemetry.as_ref()
    }

    /// Process multi-channel microphone audio recordings through the aeroacoustic inverter,
    /// reconstructing unsteady blade forces, calculating 3D radiation directivity, and projecting ground dB(A) noise footprints.
    pub fn process_aeroacoustic_frame(
        &mut self,
        channels: &[&[f32]],
        altitude_agl_m: f32,
        target_ground_pos_m: Option<(f32, f32)>,
        current_yaw_rad: f32,
    ) -> Result<AeroacousticTelemetry, String> {
        let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
        let rpms = self.current_motor_rpms.clone();
        let inverter = self
            .aeroacoustic_inverter
            .as_mut()
            .ok_or_else(|| "Aeroacoustic inverter is not enabled".to_string())?;

        let telem = inverter.process_frame(
            channels,
            &rpms,
            altitude_agl_m,
            target_ground_pos_m,
            current_yaw_rad,
            timestamp_sec,
        );

        if let Some(sphere) = inverter.latest_directivity_sphere() {
            self.latest_directivity_sphere = Some(sphere.clone());
        }
        if let Some(footprint) = inverter.latest_ground_noise_footprint() {
            self.latest_ground_footprint = Some(footprint.clone());
        }
        self.latest_aeroacoustic_telemetry = Some(telem.clone());

        Ok(telem)
    }

    /// Enable distributed multi-UAV swarm acoustic mesh beamforming and synthetic aperture radar.
    pub fn enable_swarm_mesh_beamforming(
        &mut self,
        local_node_id: usize,
        config: SwarmMeshConfig,
    ) {
        self.swarm_beamformer = Some(SyntheticApertureBeamformer::new(local_node_id, config));
    }

    /// Disable swarm mesh beamforming.
    pub fn disable_swarm_mesh_beamforming(&mut self) {
        self.swarm_beamformer = None;
        self.latest_swarm_target_report = None;
    }

    /// Access reference to active swarm synthetic aperture beamformer if enabled.
    pub fn swarm_beamformer(&self) -> Option<&SyntheticApertureBeamformer> {
        self.swarm_beamformer.as_ref()
    }

    /// Access mutable reference to active swarm synthetic aperture beamformer if enabled.
    pub fn swarm_beamformer_mut(&mut self) -> Option<&mut SyntheticApertureBeamformer> {
        self.swarm_beamformer.as_mut()
    }

    /// Update active swarm node coordinates and topologies across the mesh network.
    pub fn update_swarm_nodes(&mut self, nodes: Vec<SwarmNodeState>) {
        if let Some(ref mut beamformer) = self.swarm_beamformer {
            beamformer.update_swarm_nodes(nodes);
        }
    }

    /// Return latest evaluated swarm target localization report.
    pub fn latest_swarm_target_report(&self) -> Option<&SwarmTargetReport> {
        self.latest_swarm_target_report.as_ref()
    }

    /// Process multi-node acoustic recordings through the distributed synthetic aperture beamformer,
    /// tracking distant ground vehicles and localizing targets with sub-degree angular precision.
    pub fn process_swarm_mesh_frame(
        &mut self,
        node_audio_slices: &[&[f32]],
    ) -> Option<SwarmTargetReport> {
        let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
        let beamformer = self.swarm_beamformer.as_mut()?;
        let report = beamformer.process_swarm_frame(node_audio_slices, timestamp_sec);
        if let Some(ref r) = report {
            self.latest_swarm_target_report = Some(r.clone());
        }
        report
    }

    /// Return current relativistic acoustic Doppler scale factor (1.0 if disabled or stationary).
    pub fn doppler_scale_factor(&self) -> f32 {
        self.doppler.as_ref().map_or(1.0, |d| d.doppler_factor())
    }

    /// Return reference to internal Voice Activity Detector.
    pub fn vad(&self) -> &EnergyVad {
        &self.vad
    }

    /// Return mutable reference to internal Voice Activity Detector.
    pub fn vad_mut(&mut self) -> &mut EnergyVad {
        &mut self.vad
    }

    /// Enable Hardware-Accelerated Streaming Spiking Neural VAD with Neuromorphic Latency.
    pub fn enable_spiking_vad(&mut self, config: SpikingVadConfig) {
        self.spiking_vad = Some(SpikingNeuralVad::new(config));
    }

    /// Disable Spiking Neural VAD.
    pub fn disable_spiking_vad(&mut self) {
        self.spiking_vad = None;
        self.latest_spiking_vad_telemetry = None;
    }

    /// Access reference to active Spiking Neural VAD if enabled.
    pub fn spiking_vad(&self) -> Option<&SpikingNeuralVad> {
        self.spiking_vad.as_ref()
    }

    /// Access mutable reference to active Spiking Neural VAD if enabled.
    pub fn spiking_vad_mut(&mut self) -> Option<&mut SpikingNeuralVad> {
        self.spiking_vad.as_mut()
    }

    /// Return latest evaluated Spiking Neural VAD telemetry snapshot.
    pub fn latest_spiking_vad_telemetry(&self) -> Option<&SpikingVadTelemetry> {
        self.latest_spiking_vad_telemetry.as_ref()
    }

    /// Configure whether Spiking Neural VAD gates DTW matching during quiescent periods.
    pub fn set_spiking_vad_gating(&mut self, enabled: bool) {
        self.spiking_vad_gating = enabled;
    }

    /// Check if Spiking Neural VAD gating is enabled.
    pub fn is_spiking_vad_gating_enabled(&self) -> bool {
        self.spiking_vad_gating
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

    /// Sampling rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Access current feature history window.
    pub fn feature_history(&self) -> &[Vec<f32>] {
        &self.feature_history
    }

    /// Access reference to internal DTW phrase matcher.
    pub fn dtw(&self) -> &DtwMatcher {
        &self.dtw
    }

    /// Access mutable reference to internal DTW phrase matcher.
    pub fn dtw_mut(&mut self) -> &mut DtwMatcher {
        &mut self.dtw
    }

    /// Access reference to online acoustic noise tracker.
    pub fn noise_tracker(&self) -> &AcousticNoiseClusterTracker {
        &self.noise_tracker
    }

    /// Access mutable reference to online acoustic noise tracker.
    pub fn noise_tracker_mut(&mut self) -> &mut AcousticNoiseClusterTracker {
        &mut self.noise_tracker
    }

    /// Active feature reliability weights derived from running noise variance.
    pub fn noise_feature_weights(&self) -> &[f32] {
        self.noise_tracker.weights()
    }

    /// Spectral flatness of the background noise floor.
    pub fn noise_spectral_flatness(&self) -> f32 {
        self.noise_tracker.spectral_flatness()
    }

    /// Export 8-bit quantized DTW phrase matcher initialized with currently enrolled templates.
    pub fn create_quantized_matcher(&self) -> QuantizedDtwMatcher {
        let mut q_matcher = QuantizedDtwMatcher::new();
        for template in self.dtw.templates() {
            q_matcher.add_template(
                &template.name,
                &template.features,
                template.threshold,
                template.band_radius,
            );
        }
        q_matcher
    }

    /// Export quantized phrase templates for embedded serialization.
    pub fn export_quantized_templates(&self) -> Vec<QuantizedPhraseTemplate> {
        let q_matcher = self.create_quantized_matcher();
        q_matcher.templates().to_vec()
    }

    /// Export an enrolled keyword template as a sub-byte packed template (1-bit, 2-bit, or 4-bit).
    pub fn export_subbyte_template(
        &self,
        keyword: &str,
        bit_width: SubByteBitWidth,
    ) -> Result<SubBytePhraseTemplate, String> {
        let template = self
            .dtw
            .templates()
            .iter()
            .find(|t| t.name == keyword)
            .ok_or_else(|| format!("Keyword '{}' not enrolled in engine", keyword))?;

        Ok(SubBytePhraseTemplate::from_features(
            &template.name,
            &template.features,
            bit_width,
            template.threshold,
            template.band_radius,
        ))
    }

    /// Export an ultra-low-bitrate sub-byte DTW phrase matcher initialized with currently enrolled templates.
    pub fn create_subbyte_matcher(&self, bit_width: SubByteBitWidth) -> SubByteDtwMatcher {
        let mut matcher = SubByteDtwMatcher::new();
        for template in self.dtw.templates() {
            let subbyte_tmpl = SubBytePhraseTemplate::from_features(
                &template.name,
                &template.features,
                bit_width,
                template.threshold,
                template.band_radius,
            );
            matcher.add_template(subbyte_tmpl);
        }
        matcher
    }

    /// Enroll a keyword phrase template into the engine using single feature sequence and default band corridor.
    pub fn enroll_keyword(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        self.max_history_frames = self.max_history_frames.max(features.len() + 32);
        self.dtw.add_template(name, features, threshold);
    }

    /// Clear all enrolled keyword templates from the engine.
    pub fn clear_keywords(&mut self) {
        self.dtw.clear_templates();
    }

    /// Enroll a keyword phrase template with explicit Sakoe-Chiba corridor band radius `R`.
    pub fn enroll_keyword_banded(
        &mut self,
        name: impl Into<String>,
        features: Vec<Vec<f32>>,
        threshold: f32,
        band_radius: usize,
    ) {
        self.max_history_frames = self.max_history_frames.max(features.len() + 32);
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
        for ex in &exemplars_features {
            self.max_history_frames = self.max_history_frames.max(ex.len() + 32);
        }
        self.dtw
            .add_template_exemplars(name, &exemplars_features, band_radius, margin_factor)
    }

    /// Synthesize speech audio waveform from plain text using rule-based G2P and Klatt formant synthesis.
    pub fn synthesize_speech_from_text(&self, text: &str) -> Vec<f32> {
        let segments = G2pEngine::text_to_phonemes(text);
        let synth = KlattSynthesizer::new(self.sample_rate);
        synth.synthesize(&segments)
    }

    /// Enroll a keyword directly from plain text without prior voice recording.
    /// Synthesizes acoustic waveform, extracts feature frames, and registers into the DTW template library.
    pub fn enroll_keyword_from_text(
        &mut self,
        name: impl Into<String>,
        text: &str,
        threshold: f32,
    ) -> usize {
        let audio = self.synthesize_speech_from_text(text);
        let features = self.extract_features(&audio);
        let frame_count = features.len();
        self.enroll_keyword(name, features, threshold);
        frame_count
    }

    /// Multi-modal template fusion: combines a zero-shot synthesized text template with
    /// 1-shot or few-shot human voice exemplars using DTW Barycenter Averaging (DBA).
    pub fn enroll_keyword_hybrid(
        &mut self,
        name: impl Into<String>,
        text: &str,
        user_exemplar_audio_slices: &[&[f32]],
        band_radius: usize,
        margin_factor: f32,
    ) -> f32 {
        let synth_audio = self.synthesize_speech_from_text(text);
        let synth_features = self.extract_features(&synth_audio);

        let mut all_exemplars = Vec::new();
        if !synth_features.is_empty() {
            all_exemplars.push(synth_features);
        }

        for slice in user_exemplar_audio_slices {
            let feats = self.extract_features(slice);
            if !feats.is_empty() {
                all_exemplars.push(feats);
            }
        }

        self.dtw
            .add_template_exemplars(name, &all_exemplars, band_radius, margin_factor)
    }

    /// Enroll a keyword phrase using the automated synthetic speech exemplar pipeline.
    /// Programmatically generates multiple pitch/rate/tract synthetic audio variations, fuses them with DBA,
    /// and automatically calibrates the recognition distance threshold.
    pub fn enroll_keyword_synthetic_pipeline(
        &mut self,
        name: impl Into<String>,
        phrase: &str,
        num_exemplars: usize,
        band_radius: usize,
        margin_factor: f32,
    ) -> f32 {
        let name_str = name.into();
        let generator = SyntheticExemplarGenerator::new(self.sample_rate);
        let audio_variants = generator.generate_exemplars(phrase, num_exemplars.max(3));
        let mut feature_variants = Vec::with_capacity(audio_variants.len());

        for audio in &audio_variants {
            let feats = self.extract_features(audio);
            if !feats.is_empty() {
                feature_variants.push(feats);
            }
        }

        for feats in &feature_variants {
            self.max_history_frames = self.max_history_frames.max(feats.len() + 32);
        }

        self.dtw
            .add_template_exemplars(name_str, &feature_variants, band_radius, margin_factor)
    }

    /// Enroll a wake-word phrase zero-shot directly from text using cross-attention phonetic alignment
    /// and automated minimal-pair foil discrimination margin threshold calibration.
    pub fn enroll_keyword_zero_shot(
        &mut self,
        name: impl Into<String>,
        phrase: &str,
        language: SupportedLanguage,
        accent: VocalAccent,
    ) -> Result<ZeroShotCalibrationReport, String> {
        let name_str = name.into();
        let calibrator = ZeroShotCalibrator::new();
        let (reference_template, report) = calibrator.calibrate(phrase, language, accent, self)?;

        self.max_history_frames = self.max_history_frames.max(reference_template.len() + 32);
        self.enroll_keyword_banded(
            name_str,
            reference_template,
            report.calibrated_threshold,
            8,
        );

        Ok(report)
    }

    /// Evaluate zero-shot phonetic discrimination margins and confusion matrix for a keyword phrase.
    pub fn evaluate_zero_shot_discrimination(
        &self,
        phrase: &str,
        language: SupportedLanguage,
        accent: VocalAccent,
    ) -> Result<ZeroShotCalibrationReport, String> {
        let calibrator = ZeroShotCalibrator::new();
        let (_, report) = calibrator.calibrate(phrase, language, accent, self)?;
        Ok(report)
    }

    /// Enroll a keyword phrase with multi-accent active articulatory synthesis and joint threshold calibration.
    pub fn enroll_keyword_multi_accent(
        &mut self,
        name: impl Into<String>,
        phrase: &str,
        language: SupportedLanguage,
        accents: &[VocalAccent],
    ) -> Result<CrossAccentCalibrationReport, String> {
        let name_str = name.into();
        let calibrator = MultiAccentCalibrator::new();
        let (reference_template, report) = calibrator.calibrate(phrase, language, accents, self)?;

        self.max_history_frames = self.max_history_frames.max(reference_template.len() + 32);
        self.enroll_keyword_banded(
            name_str,
            reference_template,
            report.calibrated_threshold,
            8,
        );

        Ok(report)
    }

    /// Evaluate multi-accent calibration report for a keyword phrase across specified accents.
    pub fn evaluate_multi_accent_discrimination(
        &self,
        phrase: &str,
        language: SupportedLanguage,
        accents: &[VocalAccent],
    ) -> Result<CrossAccentCalibrationReport, String> {
        let calibrator = MultiAccentCalibrator::new();
        let (_, report) = calibrator.calibrate(phrase, language, accents, self)?;
        Ok(report)
    }

    /// Enable dual-threshold operator verification and anti-spoofing gating.
    pub fn enable_operator_verification(&mut self, strict: bool) {
        if let Some(ref mut verifier) = self.operator_verifier {
            verifier.set_strict(strict);
        } else {
            self.operator_verifier = Some(OperatorVerifier::new(strict));
        }
    }

    /// Disable operator verification gating.
    pub fn disable_operator_verification(&mut self) {
        self.operator_verifier = None;
        self.latest_verification_decision = None;
    }

    /// Check whether operator verification is enabled.
    pub fn is_operator_verification_enabled(&self) -> bool {
        self.operator_verifier.is_some()
    }

    /// Enroll an authorized operator using one or more voice audio slices.
    pub fn enroll_authorized_operator(
        &mut self,
        operator_id: impl Into<String>,
        voice_audio_slices: &[&[f32]],
        similarity_threshold: f32,
    ) -> Result<SpeakerVoiceprint, String> {
        if voice_audio_slices.is_empty() {
            return Err("At least one voice audio slice required for enrollment".to_string());
        }

        let mut vps = Vec::with_capacity(voice_audio_slices.len());
        for &slice in voice_audio_slices {
            let feats = self.extract_features(slice);
            if feats.is_empty() {
                continue;
            }
            let vp = SpeakerVoiceprint::from_features_and_audio(&feats, slice, self.sample_rate)?;
            vps.push(vp);
        }

        if vps.is_empty() {
            return Err("Failed to extract voiceprint features from provided audio slices".to_string());
        }

        let fused = SpeakerVoiceprint::fuse_all(&vps)?;
        let op_id = operator_id.into();

        if let Some(ref mut verifier) = self.operator_verifier {
            verifier.enroll_operator(op_id, fused.clone(), similarity_threshold);
        } else {
            let mut verifier = OperatorVerifier::new(true);
            verifier.enroll_operator(op_id, fused.clone(), similarity_threshold);
            self.operator_verifier = Some(verifier);
        }

        Ok(fused)
    }

    /// Remove an authorized operator by ID.
    pub fn remove_authorized_operator(&mut self, operator_id: &str) -> bool {
        if let Some(ref mut verifier) = self.operator_verifier {
            verifier.remove_operator(operator_id)
        } else {
            false
        }
    }

    /// Clear all enrolled authorized operators.
    pub fn clear_authorized_operators(&mut self) {
        if let Some(ref mut verifier) = self.operator_verifier {
            verifier.clear_operators();
        }
    }

    /// Return list of all enrolled operator IDs.
    pub fn authorized_operators(&self) -> Vec<String> {
        if let Some(ref verifier) = self.operator_verifier {
            verifier.operators().iter().map(|op| op.operator_id.clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Access reference to active OperatorVerifier if enabled.
    pub fn operator_verifier(&self) -> Option<&OperatorVerifier> {
        self.operator_verifier.as_ref()
    }

    /// Access mutable reference to active OperatorVerifier if enabled.
    pub fn operator_verifier_mut(&mut self) -> Option<&mut OperatorVerifier> {
        self.operator_verifier.as_mut()
    }

    /// Access latest verification decision from streaming keyword spotting.
    pub fn latest_verification_decision(&self) -> Option<&VerificationDecision> {
        self.latest_verification_decision.as_ref()
    }

    /// Verify speaker voiceprint and glottal anti-spoofing on an arbitrary audio slice.
    pub fn verify_speaker_audio(&self, audio_samples: &[f32]) -> Option<VerificationDecision> {
        if let Some(ref verifier) = self.operator_verifier {
            let feats = self.extract_features(audio_samples);
            Some(verifier.verify(audio_samples, &feats, self.sample_rate))
        } else {
            None
        }
    }

    /// Enable continuous phonetic CTC beam search command decoder.
    pub fn enable_ctc_command_decoder(&mut self, config: CtcDecoderConfig) {
        self.ctc_decoder = Some(CtcCommandDecoder::new(config));
    }

    /// Disable CTC command decoder.
    pub fn disable_ctc_command_decoder(&mut self) {
        self.ctc_decoder = None;
        self.latest_recognized_command = None;
    }

    /// Check whether CTC command decoder is enabled.
    pub fn is_ctc_command_decoder_enabled(&self) -> bool {
        self.ctc_decoder.is_some()
    }

    /// Access reference to active CTC command decoder if enabled.
    pub fn ctc_decoder(&self) -> Option<&CtcCommandDecoder> {
        self.ctc_decoder.as_ref()
    }

    /// Access mutable reference to active CTC command decoder if enabled.
    pub fn ctc_decoder_mut(&mut self) -> Option<&mut CtcCommandDecoder> {
        self.ctc_decoder.as_mut()
    }

    /// Access latest recognized multi-word flight command result if available.
    pub fn latest_recognized_command(&self) -> Option<&CommandRecognitionResult> {
        self.latest_recognized_command.as_ref()
    }

    /// Recognize a multi-word flight command from an audio sample buffer using continuous phonetic CTC beam search
    /// and on-device language model rescoring.
    pub fn recognize_command_stream(&mut self, audio: &[f32]) -> Option<CommandRecognitionResult> {
        let start_time = std::time::Instant::now();
        let feats = self.extract_features(audio);
        if feats.is_empty() {
            return None;
        }

        let decoder = if let Some(ref mut d) = self.ctc_decoder {
            d
        } else {
            self.ctc_decoder = Some(CtcCommandDecoder::new(CtcDecoderConfig::default()));
            self.ctc_decoder.as_mut().unwrap()
        };

        decoder.reset();

        for frame in &feats {
            let is_speech = frame[0] > -80.0;
            decoder.step_frame(frame, is_speech);
        }

        let mut res = decoder.finalize()?;
        res.latency_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.latest_recognized_command = Some(res.clone());
        Some(res)
    }

    /// Access reference to active streaming DTW configuration.
    pub fn streaming_dtw_config(&self) -> &StreamingDtwConfig {
        &self.streaming_dtw_config
    }

    /// Access mutable reference to active streaming DTW configuration.
    pub fn streaming_dtw_config_mut(&mut self) -> &mut StreamingDtwConfig {
        &mut self.streaming_dtw_config
    }

    /// Replace active streaming DTW configuration.
    pub fn set_streaming_dtw_config(&mut self, config: StreamingDtwConfig) {
        self.streaming_dtw_config = config;
    }

    /// Access reference to internal zero-heap feature ring buffer.
    pub fn feature_ring_buffer(&self) -> &FeatureRingBuffer {
        &self.feature_ring_buffer
    }

    /// Enable continual domain adaptation for an enrolled keyword.
    pub fn enable_continual_adaptation(
        &mut self,
        keyword: &str,
        memory_capacity: usize,
    ) -> Result<(), String> {
        let template = self
            .dtw
            .templates()
            .iter()
            .find(|t| t.name == keyword)
            .ok_or_else(|| format!("Keyword '{}' not found in enrolled templates", keyword))?;

        let engine = ContinualAdaptationEngine::new(
            keyword,
            template.features.clone(),
            template.threshold,
            memory_capacity,
            template.band_radius,
        );
        self.continual_adaptation = Some(engine);
        Ok(())
    }

    /// Disable continual domain adaptation.
    pub fn disable_continual_adaptation(&mut self) {
        self.continual_adaptation = None;
        self.latest_adaptation_telemetry = None;
    }

    /// Access reference to active continual adaptation engine if enabled.
    pub fn continual_adaptation(&self) -> Option<&ContinualAdaptationEngine> {
        self.continual_adaptation.as_ref()
    }

    /// Access mutable reference to active continual adaptation engine if enabled.
    pub fn continual_adaptation_mut(&mut self) -> Option<&mut ContinualAdaptationEngine> {
        self.continual_adaptation.as_mut()
    }

    /// Access latest evaluated adaptation telemetry report.
    pub fn latest_adaptation_telemetry(&self) -> Option<AdaptationTelemetry> {
        self.continual_adaptation.as_ref().map(|a| a.telemetry())
    }

    /// Process an observed speech segment through continual adaptation and sync template in DTW matcher.
    pub fn adapt_keyword_observation(
        &mut self,
        features: &[Vec<f32>],
        match_dist: f32,
        snr_db: f32,
        timestamp_sec: f32,
        raw_audio: &[f32],
    ) -> Result<Option<ActiveLearningCandidate>, String> {
        if let Some(ref mut engine) = self.continual_adaptation {
            let candidate = engine.process_observation(
                features,
                match_dist,
                snr_db,
                timestamp_sec,
                raw_audio,
            )?;

            // Sync updated canonical centroid back into DTW template
            let updated_centroid = engine.canonical_centroid().to_vec();
            let kw_name = engine.telemetry().keyword;
            self.dtw.update_template_features(&kw_name, updated_centroid);

            self.latest_adaptation_telemetry = Some(engine.telemetry());
            Ok(candidate)
        } else {
            Ok(None)
        }
    }

    /// Ingest streaming microphone samples alongside far-end loudspeaker reference samples.
    /// Cancels acoustic echo before running VAD, feature extraction, and wake-word spotting.
    pub fn ingest_samples_with_reference(
        &mut self,
        mic_samples: &[f32],
        ref_samples: &[f32],
    ) -> Vec<KeywordEvent> {
        let mut clean_samples = vec![0.0f32; mic_samples.len()];
        let effective_mic = if let Some(ref mut aec) = self.aec {
            aec.process_block(mic_samples, ref_samples, &mut clean_samples);
            &clean_samples[..]
        } else {
            mic_samples
        };

        self.ingest_samples_internal(effective_mic)
    }

    /// Ingest a slice of raw audio samples (mono float32 [-1.0, 1.0]).
    /// Returns any detected keyword events.
    pub fn ingest_samples(&mut self, samples: &[f32]) -> Vec<KeywordEvent> {
        self.ingest_samples_internal(samples)
    }

    fn ingest_samples_internal(&mut self, samples: &[f32]) -> Vec<KeywordEvent> {
        let mut events = Vec::new();

        // Apply Doppler compensation if enabled and moving
        let doppler_resampled;
        let effective_input = if let Some(ref d) = self.doppler {
            let factor = d.doppler_factor();
            if (factor - 1.0).abs() >= 0.005 {
                doppler_resampled = DopplerCompensator::resample_audio(samples, factor);
                &doppler_resampled[..]
            } else {
                samples
            }
        } else {
            samples
        };

        // Apply rotor notch filtering to incoming samples if enabled
        let mut filtered_samples;
        let input_slice = if let Some(ref mut bank) = self.notch_bank {
            filtered_samples = effective_input.to_vec();
            bank.process_block(&mut filtered_samples);
            &filtered_samples[..]
        } else {
            effective_input
        };

        let mut preemp = vec![0.0f32; self.frame_size];
        let chunk_size = (self.ring_buffer.capacity() - self.frame_size).max(self.hop_size);

        for chunk in input_slice.chunks(chunk_size) {
            // Process chunk through Spiking Neural VAD if enabled
            let is_spiking_active = if let Some(ref mut svad) = self.spiking_vad {
                svad.process_buffer(chunk);
                let telem = svad.telemetry();
                let active = telem.is_speech_active;
                self.latest_spiking_vad_telemetry = Some(telem);
                active
            } else {
                true
            };

            self.ring_buffer.push_slice(chunk);
            self.rolling_audio_cache.push_slice(chunk);
            self.total_samples_processed += chunk.len() as u64;

            while self.ring_buffer.len() >= self.frame_size {
                if !self.ring_buffer.peek(self.frame_size, &mut preemp) {
                    break;
                }

                // Run acoustic health monitoring if enabled (using un-emphasized physical frame)
                if let Some(ref mut monitor) = self.health_monitor {
                    let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
                    self.latest_health_snapshot = Some(monitor.analyze_frame(&preemp, timestamp_sec));
                }

                // Run CWT non-stationary rotor micro-damage profiler if enabled
                if let Some(ref mut cwt) = self.cwt_profiler {
                    let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
                    self.latest_cwt_report = Some(cwt.analyze_frame(&preemp, timestamp_sec));
                }

                // Apply pre-emphasis filter to boost high-frequency formants and consonants
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
                let is_energy_speech = self.vad.process_frame(&preemp);
                let is_speech = if self.spiking_vad.is_some() && self.spiking_vad_gating {
                    is_spiking_active
                } else {
                    is_energy_speech
                };

                // Apply windowing function
                self.window.apply(&mut preemp);

                // Compute power spectrum
                let mut power = self.fft.power_spectrum(&preemp);

                // Run psychoacoustic stealth and detectability analysis if enabled
                if let Some(ref mut stealth) = self.psychoacoustic {
                    let timestamp_sec = (self.total_samples_processed as f64) / (self.sample_rate as f64);
                    self.latest_stealth_report = Some(stealth.analyze_spectrum(
                        &power,
                        &self.current_motor_rpms,
                        timestamp_sec,
                    ));
                }

                // Apply spectral subtraction noise suppression if enabled
                if let Some(ref mut ss) = self.spectral_subtraction {
                    ss.process_spectrum(&mut power, is_speech);
                }

                let active_mel = if let Some(ref vtln_mel) = self.cached_vtln_mel {
                    vtln_mel
                } else if let Some(ref d) = self.doppler {
                    d.cached_filterbank()
                } else {
                    &self.mel
                };

                // Compute normalized filterbank energies and MFCCs
                let mfcc = match self.feature_mode {
                    FeatureMode::LogMel => {
                        let log_energies = active_mel.compute_log_energies(&power);
                        active_mel.compute_mfcc(&log_energies, self.num_mfcc)
                    }
                    FeatureMode::Pcen => {
                        let raw_energies = active_mel.compute_energies(&power);
                        let pcen_energies = self.pcen.process_frame(&raw_energies);
                        active_mel.compute_mfcc(&pcen_energies, self.num_mfcc)
                    }
                };

                if !is_speech {
                    self.noise_tracker.update_noise(&mfcc);
                }

                self.feature_ring_buffer.push_frame(&mfcc);
                self.feature_history.push(mfcc.clone());
                if self.feature_history.len() > self.max_history_frames {
                    self.feature_history.remove(0);
                }

                // Step streaming CTC command decoder if enabled
                if let Some(ref mut decoder) = self.ctc_decoder {
                    let ctc_speech = is_speech || mfcc[0] > -80.0;
                    decoder.step_frame(&mfcc, ctc_speech);
                    if !ctc_speech && decoder.consecutive_silence_frames() >= decoder.config().silence_cutoff_frames {
                        if let Some(cmd) = decoder.finalize() {
                            self.latest_recognized_command = Some(cmd);
                            decoder.reset();
                        }
                    }
                }

                if self.refractory_lockout_remaining > 0 {
                    self.refractory_lockout_remaining -= 1;
                } else if !self.spiking_vad_gating || self.spiking_vad.is_none() || is_spiking_active {
                    let noise_floor = self.vad.noise_floor();
                    let mut matched = false;

                    // Match candidate window lengths ending at current frame with noise-floor adaptation
                    if let Some(res) = self.dtw.match_streaming_window(
                        &self.feature_history,
                        noise_floor,
                        &self.streaming_dtw_config,
                    ) {
                        let timestamp_sec =
                            (self.total_samples_processed as f64) / (self.sample_rate as f64);

                        let mut is_authorized = true;
                        let mut authorized_op = None;
                        let mut voiceprint_sim = None;
                        let mut anti_spoof_ok = false;

                        if let Some(ref verifier) = self.operator_verifier {
                            let num_audio_samples = (res.matched_frames * self.hop_size + self.frame_size)
                                .min(self.rolling_audio_cache.len());
                            let mut matched_audio = vec![0.0f32; num_audio_samples];
                            if self.rolling_audio_cache.read_latest(num_audio_samples, &mut matched_audio) {
                                let start_idx = self.feature_history.len().saturating_sub(res.matched_frames);
                                let matched_features = &self.feature_history[start_idx..];
                                let decision = verifier.verify(&matched_audio, matched_features, self.sample_rate);
                                self.latest_verification_decision = Some(decision.clone());
                                match decision {
                                    VerificationDecision::Authorized { operator_id, similarity, .. } => {
                                        is_authorized = true;
                                        authorized_op = Some(operator_id);
                                        voiceprint_sim = Some(similarity);
                                        anti_spoof_ok = true;
                                    }
                                    VerificationDecision::RejectedUnauthorized { best_similarity, .. } => {
                                        voiceprint_sim = Some(best_similarity);
                                        if verifier.is_strict() {
                                            is_authorized = false;
                                        }
                                    }
                                    VerificationDecision::RejectedSpoof { .. } => {
                                        if verifier.is_strict() {
                                            is_authorized = false;
                                        }
                                    }
                                }
                            }
                        }

                        if is_authorized {
                            events.push(KeywordEvent {
                                keyword: res.keyword.clone(),
                                confidence: res.confidence,
                                timestamp_sec,
                                authorized_operator: authorized_op,
                                voiceprint_similarity: voiceprint_sim,
                                is_anti_spoof_verified: anti_spoof_ok,
                            });
                        }

                        // Hook continual domain adaptation if active for this keyword
                        if let Some(ref mut adapt) = self.continual_adaptation {
                            if adapt.telemetry().keyword == res.keyword {
                                let start_idx = self.feature_history.len().saturating_sub(res.matched_frames);
                                let matched_slice = &self.feature_history[start_idx..];
                                let snr_db = self.noise_tracker.estimate_snr_db(&mfcc).max(12.0);
                                let _ = adapt.process_observation(
                                    matched_slice,
                                    res.distance,
                                    snr_db,
                                    timestamp_sec as f32,
                                    &[],
                                );
                                let updated = adapt.canonical_centroid().to_vec();
                                self.dtw.update_template_features(&res.keyword, updated);
                                self.latest_adaptation_telemetry = Some(adapt.telemetry());
                            }
                        }

                        self.refractory_lockout_remaining = self.streaming_dtw_config.refractory_frames;
                        self.feature_history.clear();
                        self.feature_ring_buffer.clear();
                        matched = true;
                    }

                    // Fallback to match_window for legacy full-observation cases
                    if !matched {
                        if let Some((keyword, dist)) = self.dtw.match_window(&self.feature_history) {
                            let timestamp_sec =
                                (self.total_samples_processed as f64) / (self.sample_rate as f64);
                            let confidence = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);

                            let mut is_authorized = true;
                            let mut authorized_op = None;
                            let mut voiceprint_sim = None;
                            let mut anti_spoof_ok = false;

                            if let Some(ref verifier) = self.operator_verifier {
                                let num_audio_samples = (self.feature_history.len() * self.hop_size + self.frame_size)
                                    .min(self.rolling_audio_cache.len());
                                let mut matched_audio = vec![0.0f32; num_audio_samples];
                                if self.rolling_audio_cache.read_latest(num_audio_samples, &mut matched_audio) {
                                    let decision = verifier.verify(&matched_audio, &self.feature_history, self.sample_rate);
                                    self.latest_verification_decision = Some(decision.clone());
                                    match decision {
                                        VerificationDecision::Authorized { operator_id, similarity, .. } => {
                                            is_authorized = true;
                                            authorized_op = Some(operator_id);
                                            voiceprint_sim = Some(similarity);
                                            anti_spoof_ok = true;
                                        }
                                        VerificationDecision::RejectedUnauthorized { best_similarity, .. } => {
                                            voiceprint_sim = Some(best_similarity);
                                            if verifier.is_strict() {
                                                is_authorized = false;
                                            }
                                        }
                                        VerificationDecision::RejectedSpoof { .. } => {
                                            if verifier.is_strict() {
                                                is_authorized = false;
                                            }
                                        }
                                    }
                                }
                            }

                            if is_authorized {
                                events.push(KeywordEvent {
                                    keyword,
                                    confidence,
                                    timestamp_sec,
                                    authorized_operator: authorized_op,
                                    voiceprint_similarity: voiceprint_sim,
                                    is_anti_spoof_verified: anti_spoof_ok,
                                });
                            }
                            self.refractory_lockout_remaining = self.streaming_dtw_config.refractory_frames;
                            self.feature_history.clear();
                            self.feature_ring_buffer.clear();
                        }
                    }
                }

                // Run PULP surveillance energy modeling if enabled
                if let Some(ref model) = self.pulp_model {
                    let mut active_cycles = 15_000u32;
                    if is_speech {
                        active_cycles += 12_000;
                    }
                    if self.psychoacoustic.is_some() {
                        active_cycles += 4_000;
                    }
                    if self.notch_bank.is_some() {
                        active_cycles += 2_000;
                    }
                    self.latest_pulp_telemetry = Some(model.evaluate_frame_power(active_cycles, 0.45));
                }

                // Advance by hop size
                self.ring_buffer.pop_front(self.hop_size);
            }
        }

        events
    }

    /// Compute feature frames directly for an input audio slice (useful for template generation).
    pub fn extract_features(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.extract_features_with_mel(samples, None)
    }

    /// Compute feature frames directly for an input audio slice using an explicit VTLN warping factor alpha.
    pub fn extract_features_with_vtln(&self, samples: &[f32], alpha: f32) -> Vec<Vec<f32>> {
        let vtln_mel = MelFilterbank::new_with_vtln(
            self.num_mel_filters,
            self.frame_size,
            self.sample_rate,
            80.0,
            self.sample_rate / 2.0,
            alpha.clamp(0.51, 1.99),
        );
        self.extract_features_with_mel(samples, Some(&vtln_mel))
    }

    fn extract_features_with_mel(
        &self,
        samples: &[f32],
        mel_override: Option<&MelFilterbank>,
    ) -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        if samples.len() < self.frame_size {
            return features;
        }

        // Apply Doppler compensation if enabled and moving
        let doppler_resampled;
        let effective_samples = if let Some(ref d) = self.doppler {
            let factor = d.doppler_factor();
            if (factor - 1.0).abs() >= 0.005 {
                doppler_resampled = DopplerCompensator::resample_audio(samples, factor);
                &doppler_resampled[..]
            } else {
                samples
            }
        } else {
            samples
        };

        // Apply notch filtering if active
        let mut filtered_samples;
        let final_samples = if let Some(ref bank) = self.notch_bank {
            let mut b_clone = bank.clone();
            filtered_samples = effective_samples.to_vec();
            b_clone.process_block(&mut filtered_samples);
            &filtered_samples[..]
        } else {
            effective_samples
        };

        let mut pos = 0;
        let mut frame = vec![0.0f32; self.frame_size];
        let mut pcen_clone = self.pcen.clone();
        pcen_clone.reset();

        let mut ss_clone = self.spectral_subtraction.clone();

        while pos + self.frame_size <= final_samples.len() {
            frame.copy_from_slice(&final_samples[pos..pos + self.frame_size]);

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

            let active_mel = if let Some(m) = mel_override {
                m
            } else if let Some(ref vtln_mel) = self.cached_vtln_mel {
                vtln_mel
            } else if let Some(ref d) = self.doppler {
                d.cached_filterbank()
            } else {
                &self.mel
            };

            let mfcc = match self.feature_mode {
                FeatureMode::LogMel => {
                    let log_energies = active_mel.compute_log_energies(&power);
                    active_mel.compute_mfcc(&log_energies, self.num_mfcc)
                }
                FeatureMode::Pcen => {
                    let raw_energies = active_mel.compute_energies(&power);
                    let pcen_energies = pcen_clone.process_frame(&raw_energies);
                    active_mel.compute_mfcc(&pcen_energies, self.num_mfcc)
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
        if let Some(ref mut d) = self.doppler {
            d.reset();
        }
        if let Some(ref mut ormia) = self.ormia {
            ormia.reset();
        }
        if let Some(ref mut stealth) = self.psychoacoustic {
            stealth.reset();
        }
        if let Some(ref mut wind) = self.wind_suppressor {
            wind.reset();
        }
        if let Some(ref mut echo) = self.echolocator {
            echo.reset();
        }
        if let Some(ref mut svad) = self.spiking_vad {
            svad.reset();
        }
        self.latest_spiking_vad_telemetry = None;
        self.latest_cwt_report = None;
        self.latest_ormia_telemetry = None;
        self.latest_stealth_report = None;
        self.latest_pulp_telemetry = None;
        self.latest_wind_telemetry = None;
        self.latest_point_cloud = None;
        self.latest_directivity_sphere = None;
        self.latest_ground_footprint = None;
        self.latest_aeroacoustic_telemetry = None;
        self.latest_swarm_target_report = None;
        self.current_motor_rpms.clear();
        self.feature_history.clear();
        self.noise_tracker.reset();
        self.latest_adaptation_telemetry = None;
        self.cached_vtln_mel = None;
        self.vtln_alpha = 1.0;
        self.rolling_audio_cache.clear();
        self.latest_verification_decision = None;
        self.last_sample = 0.0;
        self.total_samples_processed = 0;
    }

    /// Returns standardized minimal-pair phonetic foil phrases for key autonomous robotics keywords.
    pub fn phonetic_foils_for_keyword(keyword: &str) -> Vec<&'static str> {
        let norm = keyword.trim().to_lowercase();
        match norm.as_str() {
            "take off" | "take_off" => vec!["shake off", "make off", "lake loft", "fake off"],
            "land" => vec!["hand", "band", "sand", "stand", "grand"],
            "hold" | "hold position" => vec!["cold", "bold", "fold", "gold", "sold"],
            "abort" => vec!["report", "support", "sport", "court"],
            "emergency" => vec!["urgency", "agency", "clergy"],
            _ => vec!["negative", "standby", "weather", "altitude"],
        }
    }

    /// Evaluates wake-word spotting discrimination on target positive utterances vs negative out-of-vocabulary / phonetic foil utterances.
    pub fn evaluate_keyword_discrimination(
        &mut self,
        keyword: &str,
        positive_utterances: &[Vec<f32>],
        negative_utterances: &[Vec<f32>],
    ) -> EvaluationReport {
        let mut tp = 0;
        let mut fn_count = 0;
        let mut fp = 0;
        let mut tn = 0;

        let mut pos_dists = Vec::new();
        let mut neg_dists = Vec::new();

        for pos in positive_utterances {
            self.reset();
            let events = self.ingest_samples(pos);
            let matched = events.iter().any(|e| e.keyword == keyword);
            if matched {
                tp += 1;
            } else {
                fn_count += 1;
            }

            // Extract best matching DTW distance for statistical margin analysis
            let feats = self.extract_features(pos);
            if let Some(template) = self.dtw.templates().iter().find(|t| t.name == keyword) {
                let dist = DtwMatcher::compute_distance_banded(
                    &feats,
                    &template.features,
                    template.band_radius,
                );
                if dist.is_finite() {
                    pos_dists.push(dist);
                }
            }
        }

        for neg in negative_utterances {
            self.reset();
            let events = self.ingest_samples(neg);
            let matched = events.iter().any(|e| e.keyword == keyword);
            if matched {
                fp += 1;
            } else {
                tn += 1;
            }

            let feats = self.extract_features(neg);
            if let Some(template) = self.dtw.templates().iter().find(|t| t.name == keyword) {
                let dist = DtwMatcher::compute_distance_banded(
                    &feats,
                    &template.features,
                    template.band_radius,
                );
                if dist.is_finite() {
                    neg_dists.push(dist);
                }
            }
        }

        let mean_pos = if pos_dists.is_empty() {
            0.0
        } else {
            pos_dists.iter().sum::<f32>() / (pos_dists.len() as f32)
        };

        let mean_neg = if neg_dists.is_empty() {
            0.0
        } else {
            neg_dists.iter().sum::<f32>() / (neg_dists.len() as f32)
        };

        EvaluationReport {
            keyword: keyword.to_string(),
            matrix: ConfusionMatrix::new(tp, fp, tn, fn_count),
            mean_positive_distance: mean_pos,
            mean_negative_distance: mean_neg,
            discrimination_margin: mean_neg - mean_pos,
        }
    }
}

/// Comprehensive keyword evaluation report detailing empirical confusion matrix and acoustic discrimination margins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Evaluated keyword name.
    pub keyword: String,
    /// Confusion matrix statistics (TP, FP, TN, FN).
    pub matrix: ConfusionMatrix,
    /// Mean DTW distance of true positive utterances to reference template.
    pub mean_positive_distance: f32,
    /// Mean DTW distance of negative foil utterances to reference template.
    pub mean_negative_distance: f32,
    /// Acoustic discrimination margin: mean negative distance minus mean positive distance.
    pub discrimination_margin: f32,
}
