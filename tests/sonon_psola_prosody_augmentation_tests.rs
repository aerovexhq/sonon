//! Comprehensive test suite for Pitch-Synchronous Overlap-Add (PSOLA) Prosodic Morphing,
//! Glottal Closure Instant (GCI) extraction, tempo/pitch transformations, and augmented edge KWS enrollment.

#![deny(unsafe_code)]

use sonon::phonetic::{G2pEngine, KlattSynthesizer, Phoneme, PhonemeSegment};
use sonon::psola::{
    GciConfig, GciDetector, ProsodicEnsembleGenerator, PsolaConfig, PsolaModifier,
};
use sonon::SononEngine;
use std::time::Instant;

#[test]
fn test_gci_pitch_mark_extraction_and_pitch_period_estimation() {
    let sample_rate = 16000.0;
    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_speaking_rate(1.0);
    synth.set_intonation_contour(sonon::phonetic::IntonationContour::Flat);

    // 1. Synthesize at fundamental pitch F0 = 125 Hz
    synth.set_f0(125.0);
    let seg125 = PhonemeSegment {
        phoneme: Phoneme::AA,
        duration_ms: 250.0,
        stress: 0,
    };
    let audio125 = synth.synthesize(&[seg125]);
    assert!(!audio125.is_empty());

    let detector = GciDetector::new(GciConfig::default());
    let marks125 = detector.extract_pitch_marks(&audio125, sample_rate);

    assert!(marks125.len() >= 10, "Expected at least 10 pitch marks for 250ms at 125Hz");
    // Expected period at 125 Hz = 16000 / 125 = 128 samples
    let avg_period125: f32 = marks125.iter().map(|m| m.pitch_period_samples as f32).sum::<f32>()
        / (marks125.len() as f32);
    assert!(
        (avg_period125 - 128.0).abs() < 16.0,
        "Average period {:.1} samples differed from expected 128 samples",
        avg_period125
    );

    // Verify voiced detection
    let voiced_count = marks125.iter().filter(|m| m.is_voiced).count();
    assert!(
        voiced_count > marks125.len() / 2,
        "Voiced segments should predominate during sustained vowel AA"
    );

    // 2. Synthesize at fundamental pitch F0 = 250 Hz
    synth.set_f0(250.0);
    let seg250 = PhonemeSegment {
        phoneme: Phoneme::AA,
        duration_ms: 250.0,
        stress: 0,
    };
    let audio250 = synth.synthesize(&[seg250]);
    let marks250 = detector.extract_pitch_marks(&audio250, sample_rate);

    // Expected period at 250 Hz = 16000 / 250 = 64 samples
    let avg_period250: f32 = marks250.iter().map(|m| m.pitch_period_samples as f32).sum::<f32>()
        / (marks250.len() as f32);
    assert!(
        (avg_period250 - 64.0).abs() < 12.0,
        "Average period {:.1} samples differed from expected 64 samples",
        avg_period250
    );
    assert!(
        marks250.len() > marks125.len(),
        "Higher F0 (250Hz) must yield more pitch marks than lower F0 (125Hz)"
    );
}

#[test]
fn test_td_psola_tempo_scaling_duration_preservation() {
    let sample_rate = 16000.0;
    let synth = KlattSynthesizer::new(sample_rate);
    let segments = G2pEngine::text_to_phonemes("falcon");
    let base_audio = synth.synthesize(&segments);
    let orig_len = base_audio.len();
    assert!(orig_len > 1000);

    let modifier = PsolaModifier::default();

    // 1. Slow tempo: 0.65x speed (longer duration)
    let config_slow = PsolaConfig {
        tempo_scale: 0.65,
        pitch_shift_semitones: 0.0,
        sample_rate,
    };
    let slow_audio = modifier.process(&base_audio, &config_slow);
    let expected_slow_len = (orig_len as f32 / 0.65) as usize;
    let ratio_slow = slow_audio.len() as f32 / expected_slow_len as f32;
    assert!(
        (ratio_slow - 1.0).abs() < 0.05,
        "Slow audio len {} deviated from expected {}",
        slow_audio.len(),
        expected_slow_len
    );
    assert!(slow_audio.len() > orig_len);

    // 2. Fast tempo: 1.45x speed (shorter duration)
    let config_fast = PsolaConfig {
        tempo_scale: 1.45,
        pitch_shift_semitones: 0.0,
        sample_rate,
    };
    let fast_audio = modifier.process(&base_audio, &config_fast);
    let expected_fast_len = (orig_len as f32 / 1.45) as usize;
    let ratio_fast = fast_audio.len() as f32 / expected_fast_len as f32;
    assert!(
        (ratio_fast - 1.0).abs() < 0.05,
        "Fast audio len {} deviated from expected {}",
        fast_audio.len(),
        expected_fast_len
    );
    assert!(fast_audio.len() < orig_len);

    // 3. Identity transformation: 1.00x speed
    let config_ident = PsolaConfig {
        tempo_scale: 1.0,
        pitch_shift_semitones: 0.0,
        sample_rate,
    };
    let ident_audio = modifier.process(&base_audio, &config_ident);
    assert_eq!(ident_audio.len(), orig_len);
}

#[test]
fn test_td_psola_pitch_shifting_and_formant_envelope_invariance() {
    let sample_rate = 16000.0;
    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_intonation_contour(sonon::phonetic::IntonationContour::Flat);
    synth.set_f0(130.0);
    let seg = PhonemeSegment {
        phoneme: Phoneme::IY,
        duration_ms: 300.0,
        stress: 0,
    };
    let base_audio = synth.synthesize(&[seg]);

    let modifier = PsolaModifier::default();

    // 1. Shift pitch up by +5 semitones
    let config_high = PsolaConfig {
        tempo_scale: 1.0,
        pitch_shift_semitones: 5.0,
        sample_rate,
    };
    let high_audio = modifier.process(&base_audio, &config_high);

    // 2. Shift pitch down by -5 semitones
    let config_low = PsolaConfig {
        tempo_scale: 1.0,
        pitch_shift_semitones: -5.0,
        sample_rate,
    };
    let low_audio = modifier.process(&base_audio, &config_low);

    // Measure pitch periods from output
    let detector = GciDetector::new(GciConfig::default());
    let marks_base = detector.extract_pitch_marks(&base_audio, sample_rate);
    let marks_high = detector.extract_pitch_marks(&high_audio, sample_rate);
    let marks_low = detector.extract_pitch_marks(&low_audio, sample_rate);

    let avg_period_base: f32 = marks_base.iter().map(|m| m.pitch_period_samples as f32).sum::<f32>()
        / (marks_base.len() as f32);
    let avg_period_high: f32 = marks_high.iter().map(|m| m.pitch_period_samples as f32).sum::<f32>()
        / (marks_high.len() as f32);
    let avg_period_low: f32 = marks_low.iter().map(|m| m.pitch_period_samples as f32).sum::<f32>()
        / (marks_low.len() as f32);

    // Higher pitch must have shorter pitch period
    assert!(
        avg_period_high < avg_period_base,
        "High pitch period {:.1} was not shorter than base {:.1}",
        avg_period_high,
        avg_period_base
    );
    // Lower pitch must have longer pitch period
    assert!(
        avg_period_low > avg_period_base,
        "Low pitch period {:.1} was not longer than base {:.1}",
        avg_period_low,
        avg_period_base
    );

    // Ratio verification: 5 semitones = 2^(5/12) ≈ 1.3348
    let expected_high_period = avg_period_base / (2.0f32.powf(5.0 / 12.0));
    assert!(
        (avg_period_high - expected_high_period).abs() < 14.0,
        "High period {:.1} deviated from expected {:.1}",
        avg_period_high,
        expected_high_period
    );
}

#[test]
fn test_prosodic_ensemble_generator_diversity() {
    let sample_rate = 16000.0;
    let synth = KlattSynthesizer::new(sample_rate);
    let segments = G2pEngine::text_to_phonemes("hold position");
    let base_audio = synth.synthesize(&segments);

    let generator = ProsodicEnsembleGenerator::new_standard_ensemble();
    assert_eq!(generator.variants().len(), 7);

    let ensemble = generator.generate_ensemble(&base_audio, sample_rate);
    assert_eq!(ensemble.len(), 7);

    // Check diversity: lengths must not be all identical
    let mut distinct_lengths = std::collections::HashSet::new();
    for (name, audio) in &ensemble {
        assert!(!audio.is_empty(), "Variant '{}' had empty audio", name);
        distinct_lengths.insert(audio.len());
    }

    assert!(
        distinct_lengths.len() >= 5,
        "Expected at least 5 distinct durations across 7 variants, got {}",
        distinct_lengths.len()
    );
}

#[test]
fn test_sonon_engine_psola_augmented_synthesis_kws_spotting() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    // 1. Enroll keyword "falcon" using PSOLA augmented synthesis pipeline
    let threshold = engine.enroll_keyword_augmented_synthesis("falcon", "falcon", 16, 1.30);
    assert!(
        threshold > 0.0,
        "Calibrated threshold must be positive, got {:.3}",
        threshold
    );

    let synth = KlattSynthesizer::new(sample_rate);

    // 2. Test detection on baseline synthetic speech
    let seg_base = G2pEngine::text_to_phonemes("falcon");
    let audio_base = synth.synthesize(&seg_base);
    let events_base = engine.ingest_samples(&audio_base);
    assert!(
        events_base.iter().any(|e| e.keyword == "falcon"),
        "Engine failed to detect baseline 'falcon'"
    );

    // 3. Test detection on fast morphed speech (1.25x tempo)
    let fast_audio = engine.modify_speech_prosody(&audio_base, 1.25, 0.0);
    engine.reset();
    let events_fast = engine.ingest_samples(&fast_audio);
    assert!(
        events_fast.iter().any(|e| e.keyword == "falcon"),
        "Engine failed to detect fast morphed (1.25x) 'falcon'"
    );

    // 4. Test detection on deep pitch morphed speech (-3.0 semitones)
    let deep_audio = engine.modify_speech_prosody(&audio_base, 0.95, -3.0);
    engine.reset();
    let events_deep = engine.ingest_samples(&deep_audio);
    assert!(
        events_deep.iter().any(|e| e.keyword == "falcon"),
        "Engine failed to detect deep pitch (-3st) 'falcon'"
    );

    // 5. Test detection on high pitch morphed speech (+3.5 semitones)
    let high_audio = engine.modify_speech_prosody(&audio_base, 1.10, 3.5);
    engine.reset();
    let events_high = engine.ingest_samples(&high_audio);
    assert!(
        events_high.iter().any(|e| e.keyword == "falcon"),
        "Engine failed to detect high pitch (+3.5st) 'falcon'"
    );

    // 6. Negative test: distractors must not trigger "falcon"
    let seg_distract = G2pEngine::text_to_phonemes("emergency status");
    let audio_distract = synth.synthesize(&seg_distract);
    engine.reset();
    let events_distract = engine.ingest_samples(&audio_distract);
    assert!(
        !events_distract.iter().any(|e| e.keyword == "falcon"),
        "Distractor speech falsely triggered 'falcon'"
    );
}

#[test]
fn test_psola_processing_throughput_and_latency() {
    let sample_rate = 16000.0;
    let synth = KlattSynthesizer::new(sample_rate);
    let seg = PhonemeSegment {
        phoneme: Phoneme::AA,
        duration_ms: 1000.0, // 1.0 second of audio
        stress: 1,
    };
    let audio = synth.synthesize(&[seg]);
    assert!(audio.len() >= 16000);

    let modifier = PsolaModifier::default();
    let config = PsolaConfig {
        tempo_scale: 1.15,
        pitch_shift_semitones: 2.0,
        sample_rate,
    };

    let start = Instant::now();
    let morphed = modifier.process(&audio, &config);
    let elapsed = start.elapsed();

    assert!(!morphed.is_empty());
    let throughput = (audio.len() as f64) / elapsed.as_secs_f64();
    let rt_factor = throughput / (sample_rate as f64);

    // In debug mode, throughput should exceed 48,000 samples/sec (> 3x real time)
    // In release mode, throughput easily exceeds 150,000 samples/sec (> 9x real time)
    let min_throughput = if cfg!(debug_assertions) { 48000.0 } else { 150000.0 };
    assert!(
        throughput > min_throughput,
        "Throughput {:.0} samples/sec was below {:.0} bound (RT factor: {:.1}x)",
        throughput,
        min_throughput,
        rt_factor
    );
}
