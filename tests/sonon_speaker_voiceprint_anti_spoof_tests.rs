use sonon::engine::{FeatureMode, SononEngine};
use sonon::phonetic::{G2pEngine, KlattSynthesizer};
use sonon::voiceprint::{
    GlottalAntiSpoofDetector, OperatorVerifier, SpeakerVoiceprint, VerificationDecision,
};

fn synthesize_test_voice(
    phrase: &str,
    f0: f32,
    vocal_tract_scale: f32,
    sample_rate: f32,
) -> Vec<f32> {
    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_f0(f0);
    synth.set_vocal_tract_scale(vocal_tract_scale);
    let phonemes = G2pEngine::text_to_phonemes(phrase);
    synth.synthesize(&phonemes)
}

#[test]
fn test_speaker_voiceprint_extraction_and_properties() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);

    // Speaker 1: Adult male operator (F0 = 115 Hz, vocal tract scale = 1.05)
    let audio_spk1_a = synthesize_test_voice("take off now", 115.0, 1.05, sample_rate);
    let audio_spk1_b = synthesize_test_voice("abort flight now", 118.0, 1.05, sample_rate);

    // Speaker 2: Adult female operator (F0 = 215 Hz, vocal tract scale = 0.90)
    let audio_spk2_a = synthesize_test_voice("take off now", 215.0, 0.90, sample_rate);
    let audio_spk2_b = synthesize_test_voice("abort flight now", 212.0, 0.90, sample_rate);

    let feats_1a = engine.extract_features(&audio_spk1_a);
    let feats_1b = engine.extract_features(&audio_spk1_b);
    let feats_2a = engine.extract_features(&audio_spk2_a);
    let feats_2b = engine.extract_features(&audio_spk2_b);

    let vp_1a = SpeakerVoiceprint::from_features_and_audio(&feats_1a, &audio_spk1_a, sample_rate)
        .expect("Voiceprint extraction must succeed");
    let vp_1b = SpeakerVoiceprint::from_features_and_audio(&feats_1b, &audio_spk1_b, sample_rate)
        .expect("Voiceprint extraction must succeed");
    let vp_2a = SpeakerVoiceprint::from_features_and_audio(&feats_2a, &audio_spk2_a, sample_rate)
        .expect("Voiceprint extraction must succeed");
    let vp_2b = SpeakerVoiceprint::from_features_and_audio(&feats_2b, &audio_spk2_b, sample_rate)
        .expect("Voiceprint extraction must succeed");

    // Verify L2 normalization
    let norm_1a: f32 = vp_1a.embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_2a: f32 = vp_2a.embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm_1a - 1.0).abs() < 1e-4, "L2 norm must be 1.0, got {}", norm_1a);
    assert!((norm_2a - 1.0).abs() < 1e-4, "L2 norm must be 1.0, got {}", norm_2a);

    // Intra-speaker similarity (same speaker, different phrases)
    let sim_spk1_intra = vp_1a.cosine_similarity(&vp_1b);
    let sim_spk2_intra = vp_2a.cosine_similarity(&vp_2b);
    let sim_inter_1a_2a = vp_1a.cosine_similarity(&vp_2a);

    assert!(
        sim_spk1_intra >= 0.85,
        "Speaker 1 intra-speaker similarity should be >= 0.85, got {}",
        sim_spk1_intra
    );
    assert!(
        sim_spk2_intra >= 0.85,
        "Speaker 2 intra-speaker similarity should be >= 0.85, got {}",
        sim_spk2_intra
    );

    // Inter-speaker similarity (different speakers)
    let sim_inter_1b_2b = vp_1b.cosine_similarity(&vp_2b);

    assert!(
        sim_inter_1a_2a <= 0.75,
        "Inter-speaker similarity should be <= 0.75, got {}",
        sim_inter_1a_2a
    );
    assert!(
        sim_inter_1b_2b <= 0.75,
        "Inter-speaker similarity should be <= 0.75, got {}",
        sim_inter_1b_2b
    );

    // Separation margin
    let margin = sim_spk1_intra - sim_inter_1a_2a;
    assert!(
        margin >= 0.15,
        "Speaker discrimination margin must be >= 0.15, got {}",
        margin
    );
}

#[test]
fn test_voiceprint_multi_exemplar_fusion() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);

    let audio1 = synthesize_test_voice("take off", 120.0, 1.0, sample_rate);
    let audio2 = synthesize_test_voice("land now", 124.0, 1.0, sample_rate);
    let audio3 = synthesize_test_voice("hold position", 122.0, 1.0, sample_rate);

    let vp1 = SpeakerVoiceprint::from_features_and_audio(&engine.extract_features(&audio1), &audio1, sample_rate).unwrap();
    let vp2 = SpeakerVoiceprint::from_features_and_audio(&engine.extract_features(&audio2), &audio2, sample_rate).unwrap();
    let vp3 = SpeakerVoiceprint::from_features_and_audio(&engine.extract_features(&audio3), &audio3, sample_rate).unwrap();

    let fused = SpeakerVoiceprint::fuse_all(&[vp1.clone(), vp2.clone(), vp3.clone()]).unwrap();

    let s1 = fused.cosine_similarity(&vp1);
    let s2 = fused.cosine_similarity(&vp2);
    let s3 = fused.cosine_similarity(&vp3);
    println!("fused similarities: s1={}, s2={}, s3={}", s1, s2, s3);

    // Fused profile should maintain high similarity with all constituent exemplars
    assert!(s1 >= 0.82);
    assert!(s2 >= 0.82);
    assert!(s3 >= 0.82);
}

#[test]
fn test_glottal_anti_spoof_genuine_voice() {
    let sample_rate = 16000.0f32;
    let detector = GlottalAntiSpoofDetector::default();

    // Generate genuine human speech with natural glottal fundamental pulses
    let genuine_audio = synthesize_test_voice("take off", 130.0, 1.0, sample_rate);
    let report = detector.analyze(&genuine_audio, sample_rate);

    assert!(
        report.is_genuine,
        "Authentic human speech must pass anti-spoof verification (score = {})",
        report.spoof_score
    );
    assert!(report.spoof_score < 0.40);
    assert!(report.spoof_flag.is_none());
    assert!(
        report.low_freq_energy_ratio > 0.015,
        "Genuine speech must possess low-frequency glottal energy: {}",
        report.low_freq_energy_ratio
    );
}

#[test]
fn test_glottal_anti_spoof_replay_and_transducer_rejection() {
    let sample_rate = 16000.0f32;
    let detector = GlottalAntiSpoofDetector::default();

    let genuine_audio = synthesize_test_voice("take off", 130.0, 1.0, sample_rate);

    // Simulate mobile phone / small loudspeaker replay attack:
    // 1. High-pass filter at 350 Hz (severe low-frequency cutoff of phone speaker)
    // 2. High-frequency harmonic distortion (> 3.5 kHz resonance boosting)
    let mut spoofed_audio = genuine_audio.clone();

    // High-pass filtering using simple 1st-order difference filter (attenuates < 300 Hz heavily)
    let alpha_hp = 0.92f32;
    let mut prev_in = 0.0f32;
    let mut prev_out = 0.0f32;
    for sample in &mut spoofed_audio {
        let curr_in = *sample;
        let curr_out = alpha_hp * (prev_out + curr_in - prev_in);
        prev_in = curr_in;
        prev_out = curr_out;
        *sample = curr_out;
    }

    // Add high-frequency transducer harmonic resonance distortion (4500 Hz carrier intermodulation)
    for (i, sample) in spoofed_audio.iter_mut().enumerate() {
        let t = i as f32 / sample_rate;
        let hf_noise = (2.0 * std::f32::consts::PI * 4600.0 * t).sin() * 0.08;
        *sample += hf_noise;
    }

    let report = detector.analyze(&spoofed_audio, sample_rate);

    assert!(
        !report.is_genuine,
        "Replay attack with loudspeaker roll-off must be flagged as spoof (score = {})",
        report.spoof_score
    );
    assert!(report.spoof_score > detector.config().max_spoof_threshold);
    assert!(report.spoof_flag.is_some());
}

#[test]
fn test_dual_threshold_operator_verifier() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);
    let mut verifier = OperatorVerifier::new(true);

    let auth_enrollment = synthesize_test_voice("take off", 120.0, 1.05, sample_rate);
    let auth_vp = SpeakerVoiceprint::from_features_and_audio(
        &engine.extract_features(&auth_enrollment),
        &auth_enrollment,
        sample_rate,
    )
    .unwrap();

    verifier.enroll_operator("pilot_commander", auth_vp, 0.82);

    // Test 1: Authorized pilot speaking different command
    let auth_test = synthesize_test_voice("abort flight", 122.0, 1.05, sample_rate);
    let decision_auth = verifier.verify(
        &auth_test,
        &engine.extract_features(&auth_test),
        sample_rate,
    );

    match decision_auth {
        VerificationDecision::Authorized { operator_id, similarity, .. } => {
            assert_eq!(operator_id, "pilot_commander");
            assert!(similarity >= 0.82, "Similarity {} below threshold 0.82", similarity);
        }
        _ => panic!("Authorized pilot was rejected: {:?}", decision_auth),
    }

    // Test 2: Unauthorized impostor speaker (female 220 Hz, scale 0.90)
    let impostor_test = synthesize_test_voice("abort flight", 220.0, 0.90, sample_rate);
    let decision_impostor = verifier.verify(
        &impostor_test,
        &engine.extract_features(&impostor_test),
        sample_rate,
    );

    match decision_impostor {
        VerificationDecision::RejectedUnauthorized { best_similarity, .. } => {
            assert!(
                best_similarity < 0.82,
                "Impostor similarity {} exceeded threshold",
                best_similarity
            );
        }
        _ => panic!("Unauthorized speaker was erroneously accepted: {:?}", decision_impostor),
    }
}

#[test]
fn test_sonon_engine_streaming_kws_operator_gating() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    // Enroll keyword "take off" with generous DTW threshold to ensure spotting across speakers
    let keyword_ref = synthesize_test_voice("take off", 120.0, 1.05, sample_rate);
    let feats_ref = engine.extract_features(&keyword_ref);
    engine.enroll_keyword_banded("take_off", feats_ref.clone(), 10.0, 8);

    // Enroll authorized operator "chief_pilot" using Speaker 1 voice
    let enroll_sample1 = synthesize_test_voice("take off", 120.0, 1.05, sample_rate);
    let enroll_sample2 = synthesize_test_voice("abort now", 122.0, 1.05, sample_rate);
    engine
        .enroll_authorized_operator("chief_pilot", &[&enroll_sample1, &enroll_sample2], 0.82)
        .expect("Enroll operator should succeed");

    assert!(engine.is_operator_verification_enabled());
    assert_eq!(engine.authorized_operators(), vec!["chief_pilot"]);

    // Test A: Authorized pilot streams "take off"
    let pilot_audio = synthesize_test_voice("take off", 121.0, 1.05, sample_rate);
    let silence_prefix = vec![0.0f32; 1600];
    let silence_suffix = vec![0.0f32; 3200];

    let mut detected_events = Vec::new();
    engine.ingest_samples(&silence_prefix);
    for chunk in pilot_audio.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        detected_events.extend(evs);
    }
    for chunk in silence_suffix.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        detected_events.extend(evs);
    }

    assert_eq!(detected_events.len(), 1, "Authorized pilot should trigger exactly 1 detection");
    let ev = &detected_events[0];
    assert_eq!(ev.keyword, "take_off");
    assert_eq!(ev.authorized_operator.as_deref(), Some("chief_pilot"));
    assert!(ev.is_anti_spoof_verified);
    assert!(ev.voiceprint_similarity.unwrap() >= 0.82);

    // Test B: Impostor speaker streams "take off" under strict operator enforcement
    engine.reset();
    let impostor_audio = synthesize_test_voice("take off", 225.0, 0.88, sample_rate);

    let mut impostor_events = Vec::new();
    engine.ingest_samples(&silence_prefix);
    for chunk in impostor_audio.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        impostor_events.extend(evs);
    }
    for chunk in silence_suffix.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        impostor_events.extend(evs);
    }

    assert_eq!(
        impostor_events.len(),
        0,
        "Impostor speaker must be strictly rejected by operator verifier"
    );

    // Verify engine telemetry recorded the rejection
    let decision = engine.latest_verification_decision();
    assert!(decision.is_some());
    match decision.unwrap() {
        VerificationDecision::RejectedUnauthorized { best_similarity, .. } => {
            assert!(*best_similarity < 0.82);
        }
        _ => panic!("Expected RejectedUnauthorized, got {:?}", decision),
    }

    // Test C: Non-strict mode emits event annotated as unverified
    engine.reset();
    engine.enable_operator_verification(false); // Non-strict

    let mut non_strict_events = Vec::new();
    engine.ingest_samples(&silence_prefix);
    for chunk in impostor_audio.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        non_strict_events.extend(evs);
    }
    for chunk in silence_suffix.chunks(160) {
        let evs = engine.ingest_samples(chunk);
        non_strict_events.extend(evs);
    }

    assert_eq!(
        non_strict_events.len(),
        1,
        "Non-strict mode should emit keyword event for external policy handling"
    );
    let ev_unauth = &non_strict_events[0];
    assert_eq!(ev_unauth.keyword, "take_off");
    assert_eq!(ev_unauth.authorized_operator, None);
    assert!(!ev_unauth.is_anti_spoof_verified);
}
