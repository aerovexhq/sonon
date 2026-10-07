//! # Metacognitive Voice AI & Self-State-Aware Spoken Intelligence Engine
//!
//! Implements continuous self-state-awareness, metacognitive introspection,
//! conversational floor tracking, full-duplex barge-in handling, ahead-of-time
//! inner monologue planning, 3D affective momentum modeling, epistemic uncertainty
//! self-repair, and dynamic hardware compute budget scaling for edge robotics voice interaction.
//!
//! Designed for real-time edge companion computers, robotics microcontrollers, and aerospace autopilots.
//! Complies strictly with pure safe Rust (`#![deny(unsafe_code)]`).

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::phonetic::AffectiveVector;

/// Current state of the conversational floor and spoken interaction lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationalFloorState {
    /// Quiescent listening; tracking ambient acoustic noise floor.
    Idle,
    /// User actively speaking; ingesting and buffering incoming voice stream.
    Listening,
    /// Evaluating conversational intent and formulating ahead-of-time inner monologue.
    Thinking,
    /// Actively generating and playing synthesized acoustic output.
    Speaking,
    /// Emitting low-latency affirmative backchannel vocalization while user retains the primary floor.
    Backchanneling,
    /// Overlapping user speech detected during active agent speech playback.
    Interrupted,
    /// Floor surrendered to user; active synthesis fading out with sub-50ms latency.
    Yielding,
}

/// Nature and severity of detected user barge-in during active agent speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BargeInType {
    /// Low-intensity listener acknowledgment (e.g. "uh-huh", "copy", "okay"). Agent continues speaking.
    PassiveBackchannel,
    /// High-energy command or explicit conversational interruption (e.g. "stop", "abort", "wait").
    /// Agent yields the floor immediately.
    ActiveOverride,
}

/// Prosodic inflection intent predicted for upcoming inner monologue tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodicIntent {
    /// Standard declarative statement cadence with subtle terminal pitch drop.
    DeclarativeCadence,
    /// Inquisitive question cadence with terminal pitch rise.
    InquisitiveRise,
    /// High-urgency alert or commanding callout with elevated pitch and steep formant attacks.
    UrgentExclamation,
    /// Natural cognitive hesitation or micro-pause before complex technical descriptions.
    HesitationPause,
    /// Empathetic or collaborative affirmative cadence.
    EmpatheticAffirmation,
}

/// Individual semantic and prosodic token in the ahead-of-time Inner Monologue buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InnerMonologueToken {
    /// Token sequence identifier.
    pub token_id: usize,
    /// Orthographic or phonetic text fragment.
    pub text: String,
    /// Lookahead temporal offset in milliseconds relative to current acoustic rendering clock.
    pub lookahead_offset_ms: f32,
    /// Intended prosodic inflection style.
    pub prosodic_intent: ProsodicIntent,
    /// Target 3D affective coordinate for this token.
    pub target_affect: AffectiveVector,
    /// Epistemic predictive entropy / uncertainty (higher means more uncertain).
    pub entropy: f32,
}

/// Corrective action directive triggered by epistemic uncertainty or error self-repair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelfRepairDirective {
    /// No corrective intervention required; generation proceeds fluidly.
    None,
    /// Inject natural hesitation marker or micro-pause to synchronize internal cognitive planning.
    InjectHesitation {
        /// Suggested hesitation tag (e.g., "[hesitation]", "uh", "umm").
        hesitation_token: String,
        /// Pause duration in milliseconds.
        duration_ms: f32,
    },
    /// Substitute or reformulate an uncertain lexical item.
    SubstituteLexical {
        /// Erroneous or uncertain original token.
        original: String,
        /// Corrected target token.
        replacement: String,
    },
    /// Explicit aerospace or operational correction sequence (e.g., "correction, [phrase]").
    AcousticSelfCorrection {
        /// Erroneous phrase.
        erroneous_phrase: String,
        /// Corrected phrase.
        corrected_phrase: String,
    },
}

/// Acoustic modulation parameters derived from the continuous 3D Affective State.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AffectiveAcousticModulation {
    /// Speech playback speed / tempo multiplier (e.g. 0.85 to 1.35).
    pub tempo_multiplier: f32,
    /// Fundamental frequency pitch shift in semitones (-4.0 to +6.0).
    pub pitch_shift_semitones: f32,
    /// Pitch dynamic range variance scale factor (0.7 to 1.8).
    pub pitch_variance_scale: f32,
    /// Spectral tilt adjustment in dB (-4.0 dB warm/mellow to +4.0 dB bright/urgent).
    pub spectral_tilt_db: f32,
    /// Glottal vocal fold tension parameter [0.1, 1.0].
    pub glottal_tension: f32,
    /// Subglottal breathiness / turbulence aspiration ratio [0.0, 0.6].
    pub breathiness_ratio: f32,
}

/// Real-time hardware budget scaling recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeScalingMode {
    /// Full computational headroom; execute full 16-step high-fidelity ODE integration.
    HighFidelity16Step,
    /// Standard operation; execute 8-step Sway-sampled Euler integration.
    Standard8Step,
    /// Elevated system load; execute 4-step fast Consistency Flow integration.
    FastConsistency4Step,
    /// Critical compute throttling or buffer exhaustion; fallback to 2-step or quantized INT8 edge model.
    EmergencyFallback2Step,
}

/// Comprehensive telemetry snapshot from the Metacognitive Voice Engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetacognitiveTelemetry {
    /// Current conversational floor state.
    pub floor_state: ConversationalFloorState,
    /// Estimated probability that the agent owns the floor [0.0, 1.0].
    pub agent_floor_probability: f32,
    /// Estimated probability that the user owns the floor [0.0, 1.0].
    pub user_floor_probability: f32,
    /// Estimated probability of conversational overlap / double-talk [0.0, 1.0].
    pub overlap_probability: f32,
    /// Current 3D affective coordinate in Valence-Arousal-Dominance space.
    pub current_affect: AffectiveVector,
    /// Current acoustic modulation parameters derived from affect.
    pub acoustic_modulation: AffectiveAcousticModulation,
    /// Number of tokens currently buffered in ahead-of-time lookahead monologue.
    pub lookahead_token_count: usize,
    /// Mean epistemic entropy across active generation horizon.
    pub mean_entropy: f32,
    /// Active self-repair directive, if any.
    pub active_repair: SelfRepairDirective,
    /// Measured Real-Time Factor (inference duration / audio playback duration).
    pub real_time_factor: f32,
    /// Recommended ODE integration scaling mode based on hardware headroom.
    pub recommended_compute_scaling: ComputeScalingMode,
    /// Audio playback ring buffer fill percentage [0.0, 1.0].
    pub audio_buffer_fill_ratio: f32,
}

/// Configuration parameters for the Conversational Floor & Turn State Manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnManagerConfig {
    /// Energy threshold (in linear RMS) above which user mic activity is detected.
    pub user_speech_rms_threshold: f32,
    /// Duration of silence in milliseconds to declare turn completion by user.
    pub turn_endpoint_silence_ms: f32,
    /// Maximum duration in milliseconds to classify an interruption as a passive backchannel.
    pub max_passive_backchannel_ms: f32,
    /// Minimum energy in linear RMS to trigger an active barge-in override.
    pub active_override_rms_threshold: f32,
    /// Maximum allowed time in milliseconds to yield the floor when interrupted (< 50ms standard).
    pub yield_fadeout_target_ms: f32,
}

impl Default for TurnManagerConfig {
    fn default() -> Self {
        Self {
            user_speech_rms_threshold: 0.025,
            turn_endpoint_silence_ms: 450.0,
            max_passive_backchannel_ms: 350.0,
            active_override_rms_threshold: 0.080,
            yield_fadeout_target_ms: 35.0,
        }
    }
}

/// Conversational Floor and Turn State Manager.
#[derive(Debug, Clone)]
pub struct TurnStateManager {
    config: TurnManagerConfig,
    current_state: ConversationalFloorState,
    agent_floor_prob: f32,
    user_floor_prob: f32,
    overlap_prob: f32,
    user_speech_duration_ms: f32,
    user_silence_duration_ms: f32,
    total_floor_switches: u32,
    barge_in_count: u32,
}

impl TurnStateManager {
    /// Create new turn state manager with specified configuration.
    pub fn new(config: TurnManagerConfig) -> Self {
        Self {
            config,
            current_state: ConversationalFloorState::Idle,
            agent_floor_prob: 0.0,
            user_floor_prob: 0.0,
            overlap_prob: 0.0,
            user_speech_duration_ms: 0.0,
            user_silence_duration_ms: 0.0,
            total_floor_switches: 0,
            barge_in_count: 0,
        }
    }

    /// Update conversational floor state given incoming user mic RMS and agent playback status.
    pub fn update(
        &mut self,
        user_mic_rms: f32,
        is_agent_speaking: bool,
        frame_duration_ms: f32,
    ) -> (ConversationalFloorState, Option<BargeInType>) {
        let is_user_active = user_mic_rms >= self.config.user_speech_rms_threshold;

        if is_user_active {
            self.user_speech_duration_ms += frame_duration_ms;
            self.user_silence_duration_ms = 0.0;
        } else {
            self.user_silence_duration_ms += frame_duration_ms;
            if self.user_silence_duration_ms > self.config.turn_endpoint_silence_ms {
                self.user_speech_duration_ms = 0.0;
            }
        }

        // Calculate continuous floor probabilities
        let user_activity_factor = (user_mic_rms / (self.config.user_speech_rms_threshold * 2.0)).clamp(0.0, 1.0);
        if is_agent_speaking && is_user_active {
            self.overlap_prob = (self.overlap_prob * 0.70 + 0.30).clamp(0.0, 1.0);
            self.agent_floor_prob = (self.agent_floor_prob * 0.70 + 0.15).clamp(0.0, 1.0);
            self.user_floor_prob = (self.user_floor_prob * 0.70 + user_activity_factor * 0.30).clamp(0.0, 1.0);
        } else if is_agent_speaking {
            self.overlap_prob = (self.overlap_prob * 0.80).clamp(0.0, 1.0);
            self.agent_floor_prob = (self.agent_floor_prob * 0.80 + 0.20).clamp(0.0, 1.0);
            self.user_floor_prob = (self.user_floor_prob * 0.80).clamp(0.0, 1.0);
        } else if is_user_active {
            self.overlap_prob = (self.overlap_prob * 0.80).clamp(0.0, 1.0);
            self.agent_floor_prob = (self.agent_floor_prob * 0.80).clamp(0.0, 1.0);
            self.user_floor_prob = (self.user_floor_prob * 0.80 + user_activity_factor * 0.20).clamp(0.0, 1.0);
        } else {
            self.overlap_prob *= 0.85;
            self.agent_floor_prob *= 0.85;
            self.user_floor_prob *= 0.85;
        }

        let mut detected_barge_in = None;

        // State machine transition logic
        let next_state = match self.current_state {
            ConversationalFloorState::Idle => {
                if is_user_active {
                    self.total_floor_switches += 1;
                    ConversationalFloorState::Listening
                } else if is_agent_speaking {
                    self.total_floor_switches += 1;
                    ConversationalFloorState::Speaking
                } else {
                    ConversationalFloorState::Idle
                }
            }
            ConversationalFloorState::Listening => {
                if self.user_silence_duration_ms > self.config.turn_endpoint_silence_ms {
                    ConversationalFloorState::Thinking
                } else {
                    ConversationalFloorState::Listening
                }
            }
            ConversationalFloorState::Thinking => {
                if is_agent_speaking {
                    self.total_floor_switches += 1;
                    ConversationalFloorState::Speaking
                } else if is_user_active {
                    ConversationalFloorState::Listening
                } else {
                    ConversationalFloorState::Thinking
                }
            }
            ConversationalFloorState::Speaking => {
                if is_user_active {
                    // Evaluate barge-in nature
                    if user_mic_rms >= self.config.active_override_rms_threshold
                        || self.user_speech_duration_ms > self.config.max_passive_backchannel_ms
                    {
                        self.barge_in_count += 1;
                        detected_barge_in = Some(BargeInType::ActiveOverride);
                        ConversationalFloorState::Interrupted
                    } else {
                        detected_barge_in = Some(BargeInType::PassiveBackchannel);
                        ConversationalFloorState::Speaking
                    }
                } else if !is_agent_speaking {
                    ConversationalFloorState::Idle
                } else {
                    ConversationalFloorState::Speaking
                }
            }
            ConversationalFloorState::Interrupted => {
                // Instantly transition to Yielding to begin sub-50ms fadeout
                ConversationalFloorState::Yielding
            }
            ConversationalFloorState::Yielding => {
                if is_user_active {
                    ConversationalFloorState::Listening
                } else {
                    ConversationalFloorState::Idle
                }
            }
            ConversationalFloorState::Backchanneling => {
                if !is_agent_speaking {
                    if is_user_active {
                        ConversationalFloorState::Listening
                    } else {
                        ConversationalFloorState::Idle
                    }
                } else {
                    ConversationalFloorState::Backchanneling
                }
            }
        };

        self.current_state = next_state;
        (self.current_state, detected_barge_in)
    }

    /// Access current floor state.
    pub fn current_state(&self) -> ConversationalFloorState {
        self.current_state
    }

    /// Access agent floor probability.
    pub fn agent_floor_probability(&self) -> f32 {
        self.agent_floor_prob
    }

    /// Access user floor probability.
    pub fn user_floor_probability(&self) -> f32 {
        self.user_floor_prob
    }

    /// Access double-talk overlap probability.
    pub fn overlap_probability(&self) -> f32 {
        self.overlap_prob
    }

    /// Total count of detected barge-in events.
    pub fn barge_in_count(&self) -> u32 {
        self.barge_in_count
    }

    /// Force state transition (e.g. on external prompt start).
    pub fn set_state(&mut self, state: ConversationalFloorState) {
        self.current_state = state;
    }
}

/// Latent Inner Monologue Engine modeling ahead-of-time cognitive anticipation.
#[derive(Debug, Clone)]
pub struct InnerMonologueEngine {
    lookahead_buffer: VecDeque<InnerMonologueToken>,
    lookahead_target_ms: f32,
    next_token_id: usize,
    accumulated_playback_time_ms: f32,
}

impl InnerMonologueEngine {
    /// Create new inner monologue engine with specified lookahead time in milliseconds (default 250ms).
    pub fn new(lookahead_target_ms: f32) -> Self {
        Self {
            lookahead_buffer: VecDeque::new(),
            lookahead_target_ms,
            next_token_id: 1,
            accumulated_playback_time_ms: 0.0,
        }
    }

    /// Queue a prospective inner monologue token into the ahead-of-time planning buffer.
    pub fn push_token(
        &mut self,
        text: &str,
        prosodic_intent: ProsodicIntent,
        target_affect: AffectiveVector,
        entropy: f32,
    ) {
        let lookahead_offset_ms = match self.lookahead_buffer.back() {
            Some(last) => last.lookahead_offset_ms + 150.0,
            None => self.lookahead_target_ms,
        };

        let token = InnerMonologueToken {
            token_id: self.next_token_id,
            text: text.to_string(),
            lookahead_offset_ms,
            prosodic_intent,
            target_affect,
            entropy,
        };
        self.next_token_id += 1;
        self.lookahead_buffer.push_back(token);
    }

    /// Advance the acoustic playback clock, consuming lookahead time.
    pub fn advance_clock(&mut self, elapsed_ms: f32) {
        self.accumulated_playback_time_ms += elapsed_ms;
        for token in &mut self.lookahead_buffer {
            token.lookahead_offset_ms -= elapsed_ms;
        }

        // Drop tokens that have completed acoustic articulation (offset <= -100ms)
        while let Some(front) = self.lookahead_buffer.front() {
            if front.lookahead_offset_ms < -100.0 {
                self.lookahead_buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Peek upcoming tokens in lookahead window.
    pub fn peek_lookahead(&self) -> &VecDeque<InnerMonologueToken> {
        &self.lookahead_buffer
    }

    /// Inspect the immediate next token about to be voiced (offset closest to 0ms).
    pub fn current_articulation_token(&self) -> Option<&InnerMonologueToken> {
        self.lookahead_buffer
            .iter()
            .min_by(|a, b| a.lookahead_offset_ms.abs().partial_cmp(&b.lookahead_offset_ms.abs()).unwrap())
    }

    /// Check if an urgent alert or high-consequence semantic token is coming within lookahead horizon.
    pub fn has_impending_urgency(&self) -> bool {
        self.lookahead_buffer.iter().any(|t| {
            t.prosodic_intent == ProsodicIntent::UrgentExclamation
                || t.target_affect.arousal >= 0.75
                || t.target_affect.valence <= -0.40
        })
    }

    /// Number of buffered lookahead tokens.
    pub fn token_count(&self) -> usize {
        self.lookahead_buffer.len()
    }

    /// Clear inner monologue queue (e.g. upon interruption).
    pub fn clear(&mut self) {
        self.lookahead_buffer.clear();
    }
}

/// Dynamic 3D Affective State Trajectory Tracker.
#[derive(Debug, Clone)]
pub struct AffectiveTrajectoryTracker {
    current_affect: AffectiveVector,
    target_affect: AffectiveVector,
    inertia_alpha: f32,
}

impl AffectiveTrajectoryTracker {
    /// Create new tracker with initial affective coordinates and inertia coefficient (e.g., 0.85).
    pub fn new(initial: AffectiveVector, inertia_alpha: f32) -> Self {
        Self {
            current_affect: initial,
            target_affect: initial,
            inertia_alpha: inertia_alpha.clamp(0.1, 0.99),
        }
    }

    /// Set conversational target emotion toward which the model will smoothly trajectory-track.
    pub fn set_target(&mut self, target: AffectiveVector) {
        self.target_affect = target;
    }

    /// Advance time step, smoothing current affect toward target with inertia and external impulse.
    pub fn step(&mut self, external_impulse: Option<AffectiveVector>) {
        let alpha = self.inertia_alpha;

        let mut next_v = alpha * self.current_affect.valence + (1.0 - alpha) * self.target_affect.valence;
        let mut next_a = alpha * self.current_affect.arousal + (1.0 - alpha) * self.target_affect.arousal;
        let mut next_d = alpha * self.current_affect.dominance + (1.0 - alpha) * self.target_affect.dominance;

        if let Some(impulse) = external_impulse {
            next_v += 0.20 * impulse.valence;
            next_a += 0.20 * impulse.arousal;
            next_d += 0.20 * impulse.dominance;
        }

        self.current_affect = AffectiveVector::new(next_v, next_a, next_d);
    }

    /// Derive acoustic modulation parameters from current 3D affective state coordinates.
    pub fn derive_modulation(&self) -> AffectiveAcousticModulation {
        let v = self.current_affect.valence;
        let a = self.current_affect.arousal;
        let d = self.current_affect.dominance;

        // Tempo scales with arousal: high arousal -> fast rate (up to 1.35x), calm -> measured rate (down to 0.85x)
        let tempo_multiplier = (1.0 + 0.35 * (a - 0.5)).clamp(0.80, 1.40);

        // Pitch shift: positive valence and elevated arousal lift F0; distress and low arousal lower it
        let pitch_shift_semitones = (2.5 * v + 3.5 * (a - 0.5)).clamp(-4.0, 6.0);

        // Pitch dynamic range variance: higher arousal expands the pitch inflection excursion
        let pitch_variance_scale = (1.0 + 0.6 * a).clamp(0.7, 1.8);

        // Spectral tilt: pleasantness (high V) and high dominance soften harsh high frequencies;
        // high urgency / low dominance / negative valence increases high-frequency bite
        let spectral_tilt_db = (-2.0 * v - 2.5 * (1.0 - d)).clamp(-4.0, 4.0);

        // Vocal fold glottal tension: dominance and arousal firm up the glottis
        let glottal_tension = (0.5 + 0.3 * d + 0.2 * a).clamp(0.1, 1.0);

        // Subglottal breathiness / whisper component: high intimacy/low dominance/low arousal adds breath
        let breathiness_ratio = (0.15 + 0.35 * (1.0 - d) * (1.0 - a)).clamp(0.0, 0.6);

        AffectiveAcousticModulation {
            tempo_multiplier,
            pitch_shift_semitones,
            pitch_variance_scale,
            spectral_tilt_db,
            glottal_tension,
            breathiness_ratio,
        }
    }

    /// Access current affective state coordinates.
    pub fn current_affect(&self) -> AffectiveVector {
        self.current_affect
    }
}

/// Epistemic Uncertainty Monitor and Prosodic Self-Repair Evaluator.
#[derive(Debug, Clone)]
pub struct EpistemicUncertaintyMonitor {
    hesitation_entropy_threshold: f32,
    repair_entropy_threshold: f32,
    consecutive_uncertain_tokens: u32,
}

impl EpistemicUncertaintyMonitor {
    /// Create new uncertainty monitor with calibrated entropy thresholds.
    pub fn new(hesitation_threshold: f32, repair_threshold: f32) -> Self {
        Self {
            hesitation_entropy_threshold: hesitation_threshold,
            repair_entropy_threshold: repair_threshold,
            consecutive_uncertain_tokens: 0,
        }
    }

    /// Evaluate generation uncertainty and formulate corrective self-repair directives.
    pub fn evaluate_token(&mut self, token: &InnerMonologueToken) -> SelfRepairDirective {
        if token.entropy >= self.repair_entropy_threshold {
            self.consecutive_uncertain_tokens += 1;
            SelfRepairDirective::AcousticSelfCorrection {
                erroneous_phrase: token.text.clone(),
                corrected_phrase: format!("correction, {}", token.text),
            }
        } else if token.entropy >= self.hesitation_entropy_threshold {
            self.consecutive_uncertain_tokens += 1;
            let hesitation_token = if self.consecutive_uncertain_tokens > 1 {
                "umm".to_string()
            } else {
                "uh".to_string()
            };
            SelfRepairDirective::InjectHesitation {
                hesitation_token,
                duration_ms: 120.0,
            }
        } else {
            self.consecutive_uncertain_tokens = 0;
            SelfRepairDirective::None
        }
    }

    /// Calculate Shannon entropy from a discrete probability distribution.
    pub fn calculate_shannon_entropy(probabilities: &[f32]) -> f32 {
        let mut entropy = 0.0;
        for &p in probabilities {
            if p > 1e-12 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

/// Computational Budget and Real-Time Factor (RTF) Autoregulator.
#[derive(Debug, Clone)]
pub struct ComputeBudgetAutoregulator {
    smoothed_rtf: f32,
    smoothing_factor: f32,
    buffer_capacity_samples: usize,
    current_buffer_samples: usize,
}

impl ComputeBudgetAutoregulator {
    /// Create new compute budget autoregulator with specified audio buffer capacity.
    pub fn new(buffer_capacity_samples: usize) -> Self {
        Self {
            smoothed_rtf: 0.15,
            smoothing_factor: 0.85,
            buffer_capacity_samples,
            current_buffer_samples: buffer_capacity_samples / 2,
        }
    }

    /// Record an inference cycle execution duration vs produced audio duration.
    pub fn record_cycle(&mut self, inference_duration_ms: f32, audio_duration_ms: f32, current_buffer_samples: usize) {
        if audio_duration_ms > 0.0 {
            let instantaneous_rtf = inference_duration_ms / audio_duration_ms;
            self.smoothed_rtf = self.smoothing_factor * self.smoothed_rtf + (1.0 - self.smoothing_factor) * instantaneous_rtf;
        }
        self.current_buffer_samples = current_buffer_samples;
    }

    /// Current smoothed Real-Time Factor.
    pub fn real_time_factor(&self) -> f32 {
        self.smoothed_rtf
    }

    /// Current buffer fill ratio [0.0, 1.0].
    pub fn buffer_fill_ratio(&self) -> f32 {
        if self.buffer_capacity_samples == 0 {
            return 0.0;
        }
        (self.current_buffer_samples as f32 / self.buffer_capacity_samples as f32).clamp(0.0, 1.0)
    }

    /// Derive recommended Flow Matching DiT ODE solver step scaling based on hardware headroom.
    pub fn recommended_scaling(&self) -> ComputeScalingMode {
        let rtf = self.smoothed_rtf;
        let fill = self.buffer_fill_ratio();

        // If buffer is starving (< 20% fill) or RTF is high (> 0.85), drop to emergency mode
        if fill < 0.20 || rtf > 0.85 {
            ComputeScalingMode::EmergencyFallback2Step
        } else if fill < 0.35 || rtf > 0.65 {
            ComputeScalingMode::FastConsistency4Step
        } else if rtf > 0.40 {
            ComputeScalingMode::Standard8Step
        } else {
            ComputeScalingMode::HighFidelity16Step
        }
    }
}

/// Unified Metacognitive Voice Engine integrating all five self-state-aware pillars.
#[derive(Debug, Clone)]
pub struct MetacognitiveVoiceEngine {
    turn_manager: TurnStateManager,
    inner_monologue: InnerMonologueEngine,
    affect_tracker: AffectiveTrajectoryTracker,
    uncertainty_monitor: EpistemicUncertaintyMonitor,
    compute_autoregulator: ComputeBudgetAutoregulator,
}

impl MetacognitiveVoiceEngine {
    /// Create a new Metacognitive Voice Engine with sensible edge defaults.
    pub fn new(sample_rate: u32) -> Self {
        let buffer_capacity_samples = (sample_rate as f32 * 1.5) as usize; // 1.5 second ring buffer
        Self {
            turn_manager: TurnStateManager::new(TurnManagerConfig::default()),
            inner_monologue: InnerMonologueEngine::new(250.0), // 250ms cognitive lookahead
            affect_tracker: AffectiveTrajectoryTracker::new(AffectiveVector::default(), 0.88),
            uncertainty_monitor: EpistemicUncertaintyMonitor::new(1.20, 2.40),
            compute_autoregulator: ComputeBudgetAutoregulator::new(buffer_capacity_samples),
        }
    }

    /// Process a duplex audio frame tick.
    pub fn process_duplex_tick(
        &mut self,
        user_mic_rms: f32,
        is_agent_speaking: bool,
        frame_duration_ms: f32,
        current_buffer_samples: usize,
    ) -> (ConversationalFloorState, Option<BargeInType>) {
        // 1. Advance monologue clock
        self.inner_monologue.advance_clock(frame_duration_ms);

        // 2. Step affective trajectory
        self.affect_tracker.step(None);

        // 3. Update compute autoregulator buffer tracking
        self.compute_autoregulator.current_buffer_samples = current_buffer_samples;

        // 4. Update turn manager
        let (state, barge_in) = self.turn_manager.update(user_mic_rms, is_agent_speaking, frame_duration_ms);

        // If interrupted by active override, purge monologue and clear speech buffer
        if let Some(BargeInType::ActiveOverride) = barge_in {
            self.inner_monologue.clear();
        }

        (state, barge_in)
    }

    /// Enqueue an ahead-of-time semantic inner monologue token.
    pub fn plan_upcoming_token(
        &mut self,
        text: &str,
        prosodic_intent: ProsodicIntent,
        target_affect: AffectiveVector,
        entropy: f32,
    ) -> SelfRepairDirective {
        let token = InnerMonologueToken {
            token_id: self.inner_monologue.next_token_id,
            text: text.to_string(),
            lookahead_offset_ms: self.inner_monologue.lookahead_target_ms,
            prosodic_intent,
            target_affect,
            entropy,
        };

        let repair_directive = self.uncertainty_monitor.evaluate_token(&token);
        self.inner_monologue.push_token(text, prosodic_intent, target_affect, entropy);
        self.affect_tracker.set_target(target_affect);

        repair_directive
    }

    /// Record compute cycle latency to maintain RTF autoregulation.
    pub fn record_compute_cycle(&mut self, inference_ms: f32, audio_ms: f32, current_buffer_samples: usize) {
        self.compute_autoregulator.record_cycle(inference_ms, audio_ms, current_buffer_samples);
    }

    /// Access live diagnostic telemetry snapshot.
    pub fn telemetry(&self) -> MetacognitiveTelemetry {
        let modulation = self.affect_tracker.derive_modulation();
        let lookahead = self.inner_monologue.peek_lookahead();
        let mean_entropy = if lookahead.is_empty() {
            0.0
        } else {
            lookahead.iter().map(|t| t.entropy).sum::<f32>() / lookahead.len() as f32
        };

        MetacognitiveTelemetry {
            floor_state: self.turn_manager.current_state(),
            agent_floor_probability: self.turn_manager.agent_floor_probability(),
            user_floor_probability: self.turn_manager.user_floor_probability(),
            overlap_probability: self.turn_manager.overlap_probability(),
            current_affect: self.affect_tracker.current_affect(),
            acoustic_modulation: modulation,
            lookahead_token_count: self.inner_monologue.token_count(),
            mean_entropy,
            active_repair: SelfRepairDirective::None,
            real_time_factor: self.compute_autoregulator.real_time_factor(),
            recommended_compute_scaling: self.compute_autoregulator.recommended_scaling(),
            audio_buffer_fill_ratio: self.compute_autoregulator.buffer_fill_ratio(),
        }
    }

    /// Access mutable reference to turn manager.
    pub fn turn_manager_mut(&mut self) -> &mut TurnStateManager {
        &mut self.turn_manager
    }

    /// Access mutable reference to inner monologue engine.
    pub fn inner_monologue_mut(&mut self) -> &mut InnerMonologueEngine {
        &mut self.inner_monologue
    }

    /// Access mutable reference to affect tracker.
    pub fn affect_tracker_mut(&mut self) -> &mut AffectiveTrajectoryTracker {
        &mut self.affect_tracker
    }

    /// Access mutable reference to compute autoregulator.
    pub fn compute_autoregulator_mut(&mut self) -> &mut ComputeBudgetAutoregulator {
        &mut self.compute_autoregulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_state_manager_idle_to_listening_and_speaking() {
        let mut manager = TurnStateManager::new(TurnManagerConfig::default());
        assert_eq!(manager.current_state(), ConversationalFloorState::Idle);

        // User speaks with sufficient RMS
        let (state, barge_in) = manager.update(0.045, false, 20.0);
        assert_eq!(state, ConversationalFloorState::Listening);
        assert!(barge_in.is_none());

        // User stops speaking and silence persists past endpoint threshold
        let mut final_state = state;
        for _ in 0..30 {
            let (s, _) = manager.update(0.005, false, 20.0);
            final_state = s;
        }
        assert_eq!(final_state, ConversationalFloorState::Thinking);

        // Agent begins speaking
        let (state, _) = manager.update(0.005, true, 20.0);
        assert_eq!(state, ConversationalFloorState::Speaking);
    }

    #[test]
    fn test_barge_in_active_override_vs_passive_backchannel() {
        let mut manager = TurnStateManager::new(TurnManagerConfig::default());
        manager.set_state(ConversationalFloorState::Speaking);

        // 1. Passive low-energy backchannel
        let (state, barge_in) = manager.update(0.030, true, 50.0);
        assert_eq!(state, ConversationalFloorState::Speaking);
        assert_eq!(barge_in, Some(BargeInType::PassiveBackchannel));

        // 2. High-energy active override command
        let (state, barge_in) = manager.update(0.095, true, 50.0);
        assert_eq!(state, ConversationalFloorState::Interrupted);
        assert_eq!(barge_in, Some(BargeInType::ActiveOverride));

        // Next tick transitions to Yielding
        let (state, _) = manager.update(0.095, true, 20.0);
        assert_eq!(state, ConversationalFloorState::Yielding);
    }

    #[test]
    fn test_inner_monologue_ahead_of_time_queue_and_clock_advance() {
        let mut engine = InnerMonologueEngine::new(250.0);
        engine.push_token("Radar", ProsodicIntent::DeclarativeCadence, AffectiveVector::default(), 0.15);
        engine.push_token("Warning", ProsodicIntent::UrgentExclamation, AffectiveVector::new(-0.5, 0.9, 0.8), 0.25);

        assert_eq!(engine.token_count(), 2);
        assert!(engine.has_impending_urgency());

        // Advance clock by 100ms
        engine.advance_clock(100.0);
        let first = engine.peek_lookahead().front().unwrap();
        assert!((first.lookahead_offset_ms - 150.0).abs() < 1e-4);

        // Advance clock to articulate first token
        engine.advance_clock(260.0);
        assert_eq!(engine.token_count(), 1);
        assert_eq!(engine.peek_lookahead().front().unwrap().text, "Warning");
    }

    #[test]
    fn test_affective_trajectory_tracker_and_acoustic_modulation() {
        let initial = AffectiveVector::new(0.0, 0.2, 0.5);
        let mut tracker = AffectiveTrajectoryTracker::new(initial, 0.80);

        // Transition toward high-arousal urgent warning
        tracker.set_target(AffectiveVector::new(-0.8, 0.9, 0.8));
        for _ in 0..15 {
            tracker.step(None);
        }

        let current = tracker.current_affect();
        assert!(current.valence < -0.4);
        assert!(current.arousal > 0.6);

        let mod_params = tracker.derive_modulation();
        // High arousal should increase tempo
        assert!(mod_params.tempo_multiplier > 1.0);
        // Elevated arousal should expand pitch dynamic variance
        assert!(mod_params.pitch_variance_scale > 1.2);
    }

    #[test]
    fn test_epistemic_uncertainty_and_self_repair_triggers() {
        let mut monitor = EpistemicUncertaintyMonitor::new(1.0, 2.0);

        // 1. Confident token -> None
        let confident_token = InnerMonologueToken {
            token_id: 1,
            text: "Heading".to_string(),
            lookahead_offset_ms: 200.0,
            prosodic_intent: ProsodicIntent::DeclarativeCadence,
            target_affect: AffectiveVector::default(),
            entropy: 0.45,
        };
        assert_eq!(monitor.evaluate_token(&confident_token), SelfRepairDirective::None);

        // 2. Hesitant token -> InjectHesitation
        let hesitant_token = InnerMonologueToken {
            token_id: 2,
            text: "Waypoint".to_string(),
            lookahead_offset_ms: 150.0,
            prosodic_intent: ProsodicIntent::HesitationPause,
            target_affect: AffectiveVector::default(),
            entropy: 1.35,
        };
        match monitor.evaluate_token(&hesitant_token) {
            SelfRepairDirective::InjectHesitation { hesitation_token, duration_ms } => {
                assert_eq!(hesitation_token, "uh");
                assert_eq!(duration_ms, 120.0);
            }
            other => panic!("Expected hesitation, got {:?}", other),
        }

        // 3. Highly uncertain token -> AcousticSelfCorrection
        let uncertain_token = InnerMonologueToken {
            token_id: 3,
            text: "Bravo".to_string(),
            lookahead_offset_ms: 100.0,
            prosodic_intent: ProsodicIntent::DeclarativeCadence,
            target_affect: AffectiveVector::default(),
            entropy: 2.55,
        };
        match monitor.evaluate_token(&uncertain_token) {
            SelfRepairDirective::AcousticSelfCorrection { erroneous_phrase, corrected_phrase } => {
                assert_eq!(erroneous_phrase, "Bravo");
                assert_eq!(corrected_phrase, "correction, Bravo");
            }
            other => panic!("Expected self correction, got {:?}", other),
        }
    }

    #[test]
    fn test_compute_budget_autoregulator_scaling() {
        let mut regulator = ComputeBudgetAutoregulator::new(16000);

        // Under light load (RTF = 0.15), recommend 16 steps
        regulator.record_cycle(15.0, 100.0, 12000);
        assert_eq!(regulator.recommended_scaling(), ComputeScalingMode::HighFidelity16Step);

        // Under heavy load (RTF = 0.75), recommend 4 steps
        for _ in 0..10 {
            regulator.record_cycle(80.0, 100.0, 8000);
        }
        assert_eq!(regulator.recommended_scaling(), ComputeScalingMode::FastConsistency4Step);

        // Starving buffer (< 20% fill), recommend EmergencyFallback2Step
        regulator.record_cycle(50.0, 100.0, 2000);
        assert_eq!(regulator.recommended_scaling(), ComputeScalingMode::EmergencyFallback2Step);
    }

    #[test]
    fn test_unified_metacognitive_engine_telemetry() {
        let mut engine = MetacognitiveVoiceEngine::new(16000);
        let directive = engine.plan_upcoming_token(
            "Altimeter",
            ProsodicIntent::DeclarativeCadence,
            AffectiveVector::new(0.1, 0.4, 0.6),
            0.50,
        );
        assert_eq!(directive, SelfRepairDirective::None);

        let (state, barge) = engine.process_duplex_tick(0.010, true, 20.0, 8000);
        assert_eq!(state, ConversationalFloorState::Speaking);
        assert!(barge.is_none());

        let telemetry = engine.telemetry();
        assert_eq!(telemetry.floor_state, ConversationalFloorState::Speaking);
        assert_eq!(telemetry.lookahead_token_count, 1);
        assert!(telemetry.real_time_factor > 0.0);
    }
}
