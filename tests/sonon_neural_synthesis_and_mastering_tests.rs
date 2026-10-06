//! Integration tests for Sonon Neural Speech Synthesis, Aerospace Normalization,
//! and Acoustic Signal Mastering Chain.

#![deny(unsafe_code)]

use sonon::{NeuralSynthOptions, SononAcousticMaster, SononEngine, SononVoiceProfile};
use std::f32::consts::PI;

#[test]
fn test_engine_aerospace_text_normalization() {
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);

    // Test standard flight deck expansions
    let raw_text = "FL350 heading HDG270 cleared RWY28R";
    let normalized = engine.normalize_aerospace_text(raw_text);
    assert_eq!(
        normalized,
        "flight level three five zero heading heading two seven zero cleared runway two eight right"
    );

    // Test tactical avionics acronyms
    let tactical = "UAV in VTOL mode monitoring TCAS and ILS glideslope.";
    let norm_tactical = engine.normalize_aerospace_text(tactical);
    assert!(norm_tactical.contains("U A V"));
    assert!(norm_tactical.contains("V-TOL"));
    assert!(norm_tactical.contains("tee-cas"));
    assert!(norm_tactical.contains("I L S"));

    // Test custom acronym registration
    engine
        .aerospace_normalizer_mut()
        .register_acronym("AEROVEX", "arrow vex");
    assert_eq!(
        engine.normalize_aerospace_text("Welcome to AEROVEX flight systems"),
        "Welcome to arrow vex flight systems"
    );
}

#[test]
fn test_engine_acoustic_mastering_pipeline() {
    let engine = SononEngine::new(16000.0, 512, 160, 13);
    let sample_rate = 24000u32;
    let num_samples = 24000; // 1.0 second of audio

    // Create synthetic signal with high DC offset (0.8), 30 Hz sub-bass rumble, and 1 kHz vocal tone
    let mut audio = vec![0.8f32; num_samples];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let rumble = (2.0 * PI * 30.0 * t).sin() * 0.4;
        let tone = (2.0 * PI * 1000.0 * t).sin() * 0.2;
        audio[i] += rumble + tone;
    }

    // Apply Sonon mastering
    engine.master_speech_audio(&mut audio, sample_rate, true);

    // 1. Verify steady-state DC offset (after 100 ms filter settling) is eliminated
    let steady_state = &audio[(sample_rate as usize / 10)..];
    let dc_mean: f32 = steady_state.iter().sum::<f32>() / steady_state.len() as f32;
    assert!(
        dc_mean.abs() < 1e-3,
        "Steady-state DC mean must be zeroed out by highpass filter"
    );

    // 2. Verify peak amplitude is clamped precisely to -1.0 dBFS true peak (0.89125)
    let max_abs = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        (max_abs - 0.89125).abs() < 1e-3,
        "Peak amplitude must equal -1.0 dBFS normalization target"
    );
}

#[test]
fn test_voice_profile_and_synth_options_serde() {
    let profile = SononVoiceProfile::CustomBlended(vec![
        ("af_heart".to_string(), 0.6),
        ("am_adam".to_string(), 0.4),
    ]);

    let serialized = serde_json::to_string(&profile).expect("Serialization failed");
    let deserialized: SononVoiceProfile =
        serde_json::from_str(&serialized).expect("Deserialization failed");
    assert_eq!(profile, deserialized);

    let options = NeuralSynthOptions {
        speed: 1.05,
        apply_aerospace_normalization: true,
        apply_acoustic_mastering: true,
    };
    let opt_json = serde_json::to_string(&options).expect("Serialization failed");
    let opt_back: NeuralSynthOptions =
        serde_json::from_str(&opt_json).expect("Deserialization failed");
    assert_eq!(options.speed, opt_back.speed);
    assert_eq!(
        options.apply_aerospace_normalization,
        opt_back.apply_aerospace_normalization
    );
}

#[test]
fn test_acoustic_master_impulse_response_stability() {
    let mut master = SononAcousticMaster::new(24000);
    let mut impulse = vec![0.0f32; 3000];
    impulse[0] = 1.0;

    master.process_in_place(&mut impulse, false);

    // Verify filter stability: output must decay to near zero after 2000 samples
    let tail = &impulse[2000..3000];
    for &sample in tail {
        assert!(
            sample.abs() < 1e-4,
            "Impulse response tail must decay smoothly to zero (stable IIR)"
        );
    }
}

#[test]
fn test_engine_paralinguistic_intent_and_tokens() {
    let engine = SononEngine::new(16000.0, 512, 160, 13);

    // 1. Explicit bracket tags parsing
    let tagged_text = "Well [laughter] that was completely unexpected, wasn't it? [giggle]";
    let chunks = engine.parse_paralinguistic_tags(tagged_text);
    assert_eq!(chunks.len(), 4);
    assert_eq!(
        chunks[1],
        sonon::ParalinguisticChunk::Vocalization(sonon::ParalinguisticTag::Laughter)
    );
    assert_eq!(
        chunks[3],
        sonon::ParalinguisticChunk::Vocalization(sonon::ParalinguisticTag::Giggle)
    );

    // 2. Raw English intent inference
    let raw_text = "Haha, look at the engine RPM. Phew, we recovered.";
    let inferred = engine.infer_paralinguistic_intent(raw_text);
    assert!(inferred.contains("[laughter],"));
    assert!(inferred.contains("[sigh],"));
}

#[test]
fn test_affective_vector_and_structured_intent_inference() {
    let engine = SononEngine::new(16000.0, 512, 160, 13);

    let v1 = sonon::AffectiveVector::new(0.8, 0.7, 0.5);
    let v2 = sonon::AffectiveVector::new(0.0, 0.1, 0.5);
    assert!(v1.distance(&v2) > 0.9);

    let result = engine.infer_conversational_intent("Haha, what a remarkable maneuver!");
    assert_eq!(result.detected_tag, Some(sonon::ParalinguisticTag::Laughter));
    assert!(result.confidence >= 0.8);
    assert!(result.affective_state.valence > 0.5);
    assert!(result.affective_state.arousal > 0.5);
    assert!(result.injected_text.contains("[laughter],"));
}

