//! Phase 15 Verification: Zero-Shot Text-to-Template Phonetic Engine,
//! Rule-Based Grapheme-to-Phoneme (G2P), Klatt Acoustic Formant Synthesizer,
//! and Multi-Modal Hybrid DBA Template Fusion.

#![deny(unsafe_code)]

use sonon::engine::SononEngine;
use sonon::phonetic::{G2pEngine, KlattSynthesizer, Phoneme};
use std::time::Instant;

#[test]
fn test_g2p_deterministic_transcription() {
    // 1. Verify primary flight commands map to exact expected ARPAbet phonemes
    let abort_phonemes = G2pEngine::text_to_phonemes("abort");
    let abort_expected = [Phoneme::AH, Phoneme::B, Phoneme::AO, Phoneme::R, Phoneme::T];
    assert_eq!(
        abort_phonemes.len(),
        abort_expected.len(),
        "Abort must have 5 phonemes"
    );
    for (seg, &exp) in abort_phonemes.iter().zip(abort_expected.iter()) {
        assert_eq!(seg.phoneme, exp);
        assert!(seg.duration_ms > 0.0);
    }

    let land_phonemes = G2pEngine::text_to_phonemes("land");
    let land_expected = [Phoneme::L, Phoneme::AE, Phoneme::N, Phoneme::D];
    assert_eq!(land_phonemes.len(), land_expected.len());
    for (seg, &exp) in land_phonemes.iter().zip(land_expected.iter()) {
        assert_eq!(seg.phoneme, exp);
    }

    let plank_phonemes = G2pEngine::text_to_phonemes("plank");
    let plank_expected = [Phoneme::P, Phoneme::L, Phoneme::AE, Phoneme::NG, Phoneme::K];
    assert_eq!(plank_phonemes.len(), plank_expected.len());
    for (seg, &exp) in plank_phonemes.iter().zip(plank_expected.iter()) {
        assert_eq!(seg.phoneme, exp);
    }

    // 2. Multi-word phrase with inter-word pause
    let takeoff_phonemes = G2pEngine::text_to_phonemes("take off");
    assert!(
        takeoff_phonemes.iter().any(|s| s.phoneme == Phoneme::SIL),
        "Multi-word phrase must contain inter-word silence boundary"
    );
}

#[test]
fn test_klatt_formant_resonator_and_spectrum() {
    let sample_rate = 16000.0f32;
    let synth = KlattSynthesizer::new(sample_rate);

    let segments = G2pEngine::text_to_phonemes("abort");
    let audio = synth.synthesize(&segments);

    assert!(
        !audio.is_empty(),
        "Synthesized audio waveform must not be empty"
    );

    // Verify amplitude bounds: float32 in [-1.0, 1.0] with non-zero energy
    let max_amp = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / (audio.len() as f32)).sqrt();

    assert!(
        max_amp > 0.05 && max_amp <= 1.0,
        "Max amplitude must be normalized, got {}",
        max_amp
    );
    assert!(
        rms > 0.01,
        "Audio RMS energy must be positive and audible, got {}",
        rms
    );
}

#[test]
fn test_zero_shot_text_enrollment_and_spotting() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(sonon::FeatureMode::Pcen);

    // Enroll "abort" directly from text without any human voice recording!
    let frames = engine.enroll_keyword_from_text("abort", "abort", 4.0);
    assert!(frames > 0, "Enrolled text keyword must produce feature frames");
    assert_eq!(engine.dtw().template_count(), 1);

    // Synthesize test utterance matching the enrolled command
    let test_audio = engine.synthesize_speech_from_text("abort");
    assert!(!test_audio.is_empty());

    // Ingest the synthesized speech stream
    let events = engine.ingest_samples(&test_audio);

    assert!(
        events.iter().any(|e| e.keyword == "abort"),
        "Engine must spot zero-shot text-enrolled keyword 'abort'"
    );

    let matched_event = events.iter().find(|e| e.keyword == "abort").unwrap();
    assert!(
        matched_event.confidence > 0.70,
        "Confidence for exact zero-shot keyword match must exceed 0.70, got {:.2}",
        matched_event.confidence
    );
}

#[test]
fn test_multi_modal_hybrid_text_and_voice_fusion() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Simulate 2 slightly varying human voice exemplars
    let voice_ex1 = engine.synthesize_speech_from_text("land");
    let mut voice_ex2 = voice_ex1.clone();
    for s in &mut voice_ex2 {
        *s *= 0.90; // Slight gain variation
    }

    // Fuse zero-shot text synthesis with user exemplars via DBA
    let threshold = engine.enroll_keyword_hybrid(
        "land_hybrid",
        "land",
        &[&voice_ex1, &voice_ex2],
        10,
        1.35,
    );

    assert!(
        threshold > 0.0,
        "Hybrid enrollment must compute positive calibrated threshold, got {:.2}",
        threshold
    );
    assert_eq!(engine.dtw().template_count(), 1);

    // Verify recognition of test utterance against hybrid DBA template
    let test_audio = engine.synthesize_speech_from_text("land");
    let events = engine.ingest_samples(&test_audio);

    assert!(
        events.iter().any(|e| e.keyword == "land_hybrid"),
        "Multi-modal hybrid template must successfully spot keyword"
    );
}

#[test]
fn test_g2p_unseen_word_rule_heuristics() {
    // Unseen / novel vocabulary words outside fixed lexicon must transcribe deterministically
    let words = ["beacon", "radar", "payload", "propeller", "stabilizer", "vector"];

    for word in words {
        let phonemes = G2pEngine::text_to_phonemes(word);
        assert!(
            !phonemes.is_empty(),
            "G2P must generate phonemes for unseen word '{}'",
            word
        );

        // Ensure every generated segment has valid acoustic targets
        for seg in &phonemes {
            let target = seg.phoneme.acoustic_targets();
            assert!(target.f1 > 0.0 && target.f2 > target.f1);
            assert!(target.default_duration_ms > 0.0);
        }
    }
}

#[test]
fn test_text_to_speech_synthesis_throughput_benchmark() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 256, 128, 13);

    let phrase = "abort emergency return home hold position";

    let start = Instant::now();
    let mut total_samples = 0;
    for _ in 0..20 {
        let audio = engine.synthesize_speech_from_text(phrase);
        total_samples += audio.len();
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();

    assert!(
        samples_per_sec > 500_000.0,
        "Klatt speech synthesis throughput must exceed 500,000 samples/sec (> 30x real-time), achieved {:.2}",
        samples_per_sec
    );
}
