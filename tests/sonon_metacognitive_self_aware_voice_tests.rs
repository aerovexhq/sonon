//! # Metacognitive Voice AI & Self-State-Aware Spoken Intelligence Tests
//!
//! Verifies the five architectural pillars of acoustic and conversational metacognition:
//! 1. Conversational Floor & Turn State Tracking (Barge-In)
//! 2. Latent Inner Monologue (Ahead-of-Time Cognitive Anticipation)
//! 3. Dynamic Affective Trajectory in 3D VAD Space
//! 4. Epistemic Uncertainty & Prosodic Self-Repair
//! 5. Hardware & Computational Budget Self-Awareness (RTF Autoregulation)
//!
//! Complies with pure safe Rust (`#![deny(unsafe_code)]`).

#![deny(unsafe_code)]

use sonon::engine::SononEngine;
use sonon::metacognitive::{
    AffectiveTrajectoryTracker, BargeInType, ComputeBudgetAutoregulator, ComputeScalingMode,
    ConversationalFloorState, EpistemicUncertaintyMonitor, InnerMonologueEngine,
    InnerMonologueToken, ProsodicIntent, SelfRepairDirective, TurnManagerConfig, TurnStateManager,
};
use sonon::phonetic::AffectiveVector;

#[test]
fn test_conversational_floor_tracking_and_turn_taking() {
    let mut manager = TurnStateManager::new(TurnManagerConfig::default());
    assert_eq!(manager.current_state(), ConversationalFloorState::Idle);
    assert_eq!(manager.agent_floor_probability(), 0.0);
    assert_eq!(manager.user_floor_probability(), 0.0);

    // 1. User starts speaking: state switches from Idle to Listening
    let (state, barge) = manager.update(0.040, false, 20.0);
    assert_eq!(state, ConversationalFloorState::Listening);
    assert!(barge.is_none());
    assert!(manager.user_floor_probability() > 0.0);

    // 2. User finishes speaking: silence accumulates past threshold (450ms)
    let mut current_state = state;
    for _ in 0..25 {
        let (s, _) = manager.update(0.002, false, 20.0); // 25 * 20ms = 500ms
        current_state = s;
    }
    assert_eq!(current_state, ConversationalFloorState::Thinking);

    // 3. Agent begins speaking: state transitions to Speaking
    let (state, _) = manager.update(0.002, true, 20.0);
    assert_eq!(state, ConversationalFloorState::Speaking);
    assert!(manager.agent_floor_probability() > 0.0);
}

#[test]
fn test_barge_in_active_override_sub_50ms_yielding() {
    let mut manager = TurnStateManager::new(TurnManagerConfig {
        user_speech_rms_threshold: 0.025,
        turn_endpoint_silence_ms: 450.0,
        max_passive_backchannel_ms: 300.0,
        active_override_rms_threshold: 0.075,
        yield_fadeout_target_ms: 35.0,
    });
    manager.set_state(ConversationalFloorState::Speaking);

    // User loudly shouts "Abort!" while agent is speaking (RMS = 0.120)
    let (state, barge_in) = manager.update(0.120, true, 20.0);
    assert_eq!(state, ConversationalFloorState::Interrupted);
    assert_eq!(barge_in, Some(BargeInType::ActiveOverride));

    // Next tick immediately transitions to Yielding
    let (state, _) = manager.update(0.120, true, 15.0);
    assert_eq!(state, ConversationalFloorState::Yielding);
    assert_eq!(manager.barge_in_count(), 1);
}

#[test]
fn test_barge_in_passive_backchannel_continuation() {
    let mut manager = TurnStateManager::new(TurnManagerConfig::default());
    manager.set_state(ConversationalFloorState::Speaking);

    // User softly says "uh-huh" (RMS = 0.035, well below active override threshold 0.080)
    let (state, barge_in) = manager.update(0.035, true, 50.0);
    assert_eq!(state, ConversationalFloorState::Speaking);
    assert_eq!(barge_in, Some(BargeInType::PassiveBackchannel));

    // Agent remains Speaking without interruption
    let (state, barge_in) = manager.update(0.005, true, 20.0);
    assert_eq!(state, ConversationalFloorState::Speaking);
    assert!(barge_in.is_none());
}

#[test]
fn test_inner_monologue_ahead_of_time_anticipation_and_cadence() {
    let mut monologue = InnerMonologueEngine::new(250.0); // 250ms lookahead

    // Enqueue ahead-of-time semantic tokens
    monologue.push_token("Approaching", ProsodicIntent::DeclarativeCadence, AffectiveVector::default(), 0.10);
    monologue.push_token("Waypoint", ProsodicIntent::DeclarativeCadence, AffectiveVector::default(), 0.12);
    monologue.push_token("Alpha", ProsodicIntent::DeclarativeCadence, AffectiveVector::default(), 0.08);
    monologue.push_token("Caution", ProsodicIntent::UrgentExclamation, AffectiveVector::new(-0.6, 0.85, 0.7), 0.30);

    assert_eq!(monologue.token_count(), 4);
    assert!(monologue.has_impending_urgency());

    // Advance clock by 100ms
    monologue.advance_clock(100.0);
    assert_eq!(monologue.token_count(), 4);

    // Advance clock by 260ms (first token "Approaching" has completed articulation and is evicted)
    monologue.advance_clock(260.0);
    assert_eq!(monologue.token_count(), 3);
    assert_eq!(monologue.peek_lookahead().front().unwrap().text, "Waypoint");
}

#[test]
fn test_continuous_3d_affective_momentum_and_acoustic_modulation() {
    let calm_state = AffectiveVector::new(0.2, 0.1, 0.5);
    let mut tracker = AffectiveTrajectoryTracker::new(calm_state, 0.85);

    // Initial calm modulation
    let calm_mod = tracker.derive_modulation();
    assert!(calm_mod.tempo_multiplier < 1.0);
    assert!(calm_mod.pitch_shift_semitones < 1.0);

    // External event: Rotor blade anomaly detected (high arousal, negative valence, high dominance)
    let emergency_target = AffectiveVector::new(-0.8, 0.95, 0.85);
    tracker.set_target(emergency_target);

    // Step the dynamical system over 20 iterations
    for _ in 0..20 {
        tracker.step(None);
    }

    let urgent_state = tracker.current_affect();
    assert!(urgent_state.valence < -0.4);
    assert!(urgent_state.arousal > 0.7);

    // Urgent modulation: rapid tempo, expanded pitch excursion, elevated high-frequency bite
    let urgent_mod = tracker.derive_modulation();
    assert!(urgent_mod.tempo_multiplier > 1.10);
    assert!(urgent_mod.pitch_variance_scale > 1.30);
    assert!(urgent_mod.glottal_tension > 0.70);
}

#[test]
fn test_epistemic_uncertainty_monitoring_and_prosodic_self_repair() {
    let mut monitor = EpistemicUncertaintyMonitor::new(1.0, 2.2);

    // Low uncertainty
    let token_confident = InnerMonologueToken {
        token_id: 1,
        text: "Altitude".to_string(),
        lookahead_offset_ms: 200.0,
        prosodic_intent: ProsodicIntent::DeclarativeCadence,
        target_affect: AffectiveVector::default(),
        entropy: 0.30,
    };
    assert_eq!(monitor.evaluate_token(&token_confident), SelfRepairDirective::None);

    // Moderate uncertainty -> Triggers hesitation
    let token_hesitant = InnerMonologueToken {
        token_id: 2,
        text: "Vector".to_string(),
        lookahead_offset_ms: 150.0,
        prosodic_intent: ProsodicIntent::HesitationPause,
        target_affect: AffectiveVector::default(),
        entropy: 1.45,
    };
    match monitor.evaluate_token(&token_hesitant) {
        SelfRepairDirective::InjectHesitation { hesitation_token, duration_ms } => {
            assert!(hesitation_token == "uh" || hesitation_token == "umm");
            assert!(duration_ms >= 80.0);
        }
        other => panic!("Expected hesitation directive, got {:?}", other),
    }

    // High uncertainty -> Triggers explicit self-repair
    let token_confused = InnerMonologueToken {
        token_id: 3,
        text: "Heading three five zero".to_string(),
        lookahead_offset_ms: 100.0,
        prosodic_intent: ProsodicIntent::DeclarativeCadence,
        target_affect: AffectiveVector::default(),
        entropy: 2.75,
    };
    match monitor.evaluate_token(&token_confused) {
        SelfRepairDirective::AcousticSelfCorrection { erroneous_phrase, corrected_phrase } => {
            assert_eq!(erroneous_phrase, "Heading three five zero");
            assert!(corrected_phrase.starts_with("correction,"));
        }
        other => panic!("Expected self correction directive, got {:?}", other),
    }
}

#[test]
fn test_computational_budget_autoregulation_and_ode_scaling() {
    let mut autoreg = ComputeBudgetAutoregulator::new(24000); // 1.5s at 16 kHz

    // 1. Ample compute headroom (RTF = 0.12, 80% buffer fill)
    autoreg.record_cycle(12.0, 100.0, 19200);
    assert_eq!(autoreg.recommended_scaling(), ComputeScalingMode::HighFidelity16Step);

    // 2. Compute throttling (RTF jumps to 0.55)
    for _ in 0..15 {
        autoreg.record_cycle(55.0, 100.0, 15000);
    }
    assert_eq!(autoreg.recommended_scaling(), ComputeScalingMode::Standard8Step);

    // 3. Heavy compute pressure (RTF = 0.78)
    for _ in 0..15 {
        autoreg.record_cycle(78.0, 100.0, 10000);
    }
    assert_eq!(autoreg.recommended_scaling(), ComputeScalingMode::FastConsistency4Step);

    // 4. Critical buffer depletion (< 20% capacity) -> Emergency Fallback
    autoreg.record_cycle(40.0, 100.0, 3000);
    assert_eq!(autoreg.recommended_scaling(), ComputeScalingMode::EmergencyFallback2Step);
}

#[test]
fn test_sonon_engine_metacognitive_lifecycle_and_telemetry() {
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);
    assert!(engine.metacognitive_engine().is_none());

    // Enable metacognitive voice engine
    engine.enable_metacognitive_engine();
    assert!(engine.metacognitive_engine().is_some());

    // Plan inner monologue tokens
    let repair = engine.plan_inner_monologue_token(
        "Climb to flight level two four zero",
        ProsodicIntent::DeclarativeCadence,
        AffectiveVector::new(0.2, 0.4, 0.6),
        0.40,
    ).unwrap();
    assert_eq!(repair, SelfRepairDirective::None);

    // Process duplex frame: agent speaking, quiet user mic
    let (state, barge) = engine.process_metacognitive_duplex_tick(0.005, true, 20.0, 12000).unwrap();
    assert_eq!(state, ConversationalFloorState::Speaking);
    assert!(barge.is_none());

    // Query comprehensive telemetry
    let telemetry = engine.metacognitive_telemetry().unwrap();
    assert_eq!(telemetry.floor_state, ConversationalFloorState::Speaking);
    assert!(telemetry.agent_floor_probability > 0.0);
    assert_eq!(telemetry.lookahead_token_count, 1);
    assert!(telemetry.real_time_factor > 0.0);
    assert_eq!(telemetry.recommended_compute_scaling, ComputeScalingMode::HighFidelity16Step);

    // Disable metacognitive engine
    engine.disable_metacognitive_engine();
    assert!(engine.metacognitive_engine().is_none());
    assert!(engine.metacognitive_telemetry().is_err());
}
