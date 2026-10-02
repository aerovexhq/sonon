use crate::aec::{AcousticEchoCanceller, AecConfig};
use crate::beamforming::ArrayGeometry;
use crate::cwt::{CwtProfilerConfig, RotorDamageProfiler, RotorDamageReport};
use crate::doppler::{DopplerCompensator, DopplerConfig};
use crate::dtw::DtwMatcher;
use crate::health::{AcousticHealthMonitor, AirframeHealthSnapshot, MotorHealthConfig};
use crate::mel::MelFilterbank;
use crate::notch::RotorHarmonicNotchBank;
use crate::pcen::{PcenConfig, PcenFilter};
use crate::phonetic::{G2pEngine, KlattSynthesizer};
use crate::ring_buffer::AudioRingBuffer;
use crate::spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractionSuppressor};
use crate::stft::FftProcessor;
use crate::tse::{GpsCoordinate, TargetSoundExtractor, TseConfig, TseReport};
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
    num_mel_filters: usize,
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
            num_mel_filters,
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
        if let Some(ref mut bank) = self.notch_bank {
            bank.update_rpm(rpm);
        }
        if let Some(ref mut monitor) = self.health_monitor {
            monitor.update_motor_rpm(0, rpm);
        }
        if let Some(ref mut cwt) = self.cwt_profiler {
            cwt.update_rpm(rpm);
        }
    }

    /// Update multi-motor RPM telemetry (e.g., 4 motors on a quadcopter).
    pub fn update_multi_motor_rpm(&mut self, motor_rpms: &[f32]) {
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

    /// Enroll a keyword phrase template into the engine using single feature sequence and default band corridor.
    pub fn enroll_keyword(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        self.max_history_frames = self.max_history_frames.max(features.len() + 32);
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
            self.ring_buffer.push_slice(chunk);
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
                let is_speech = self.vad.process_frame(&preemp);

                // Apply windowing function
                self.window.apply(&mut preemp);

                // Compute power spectrum
                let mut power = self.fft.power_spectrum(&preemp);

                // Apply spectral subtraction noise suppression if enabled
                if let Some(ref mut ss) = self.spectral_subtraction {
                    ss.process_spectrum(&mut power, is_speech);
                }

                let active_mel = if let Some(ref d) = self.doppler {
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
        }

        events
    }

    /// Compute feature frames directly for an input audio slice (useful for template generation).
    pub fn extract_features(&self, samples: &[f32]) -> Vec<Vec<f32>> {
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

            let active_mel = if let Some(ref d) = self.doppler {
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
        self.latest_cwt_report = None;
        self.feature_history.clear();
        self.last_sample = 0.0;
        self.total_samples_processed = 0;
    }
}
