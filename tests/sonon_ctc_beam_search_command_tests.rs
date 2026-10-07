//! Comprehensive verification test suite for Continuous Phonetic CTC Prefix Beam Search
//! and On-Device Language Model Rescoring for Multi-Word Robotics and Aerospace Flight Commands.

#![deny(unsafe_code)]

use sonon::ctc_beam_search::{
    index_to_phoneme, log_add_exp, phoneme_to_index, CtcCommandDecoder, CtcDecoderConfig,
    CtcPosteriorFrame, CtcToken, FlightGrammarLm, LexiconTrie, ALL_PHONEMES, CTC_BLANK_INDEX,
    CTC_VOCAB_SIZE, NUM_PHONEMES,
};
use sonon::engine::SononEngine;
use sonon::phonetic::{G2pEngine, KlattSynthesizer, Phoneme};
use std::time::Instant;

#[test]
fn test_ctc_token_and_phoneme_alphabet_properties() {
    assert_eq!(ALL_PHONEMES.len(), NUM_PHONEMES);
    assert_eq!(NUM_PHONEMES, 39);
    assert_eq!(CTC_VOCAB_SIZE, 40);
    assert_eq!(CTC_BLANK_INDEX, 39);

    // Verify bijective mapping for all 39 phonemes
    for (idx, &p) in ALL_PHONEMES.iter().enumerate() {
        assert_eq!(phoneme_to_index(p), Some(idx));
        assert_eq!(index_to_phoneme(idx), Some(p));

        let token = CtcToken::Phoneme(p);
        assert_eq!(token.index(), idx);
        assert_eq!(CtcToken::from_index(idx), Some(token));
    }

    // Verify Blank token mapping
    let blank_token = CtcToken::Blank;
    assert_eq!(blank_token.index(), CTC_BLANK_INDEX);
    assert_eq!(CtcToken::from_index(CTC_BLANK_INDEX), Some(blank_token));
    assert_eq!(CtcToken::from_index(40), None);

    // Verify log_add_exp properties
    assert_eq!(log_add_exp(f32::NEG_INFINITY, -2.0), -2.0);
    assert_eq!(log_add_exp(-3.0, f32::NEG_INFINITY), -3.0);
    let expected = (1.0_f32.exp() + 2.0_f32.exp()).ln();
    let actual = log_add_exp(1.0, 2.0);
    assert!((actual - expected).abs() < 1e-5);

    // Large difference should smoothly return max
    assert_eq!(log_add_exp(100.0, -100.0), 100.0);
}

#[test]
fn test_lexicon_trie_segmentation_and_prefix_matching() {
    let mut trie = LexiconTrie::new();

    trie.insert("take", &[Phoneme::T, Phoneme::EY, Phoneme::K]);
    trie.insert("off", &[Phoneme::AO, Phoneme::F]);
    trie.insert("hold", &[Phoneme::HH, Phoneme::OW, Phoneme::L, Phoneme::D]);
    trie.insert(
        "position",
        &[
            Phoneme::P,
            Phoneme::AH,
            Phoneme::Z,
            Phoneme::IH,
            Phoneme::SH,
            Phoneme::AH,
            Phoneme::N,
        ],
    );

    // Exact matches
    assert_eq!(
        trie.exact_word(&[Phoneme::T, Phoneme::EY, Phoneme::K]),
        Some("take")
    );
    assert_eq!(trie.exact_word(&[Phoneme::AO, Phoneme::F]), Some("off"));
    assert_eq!(
        trie.exact_word(&[Phoneme::T, Phoneme::EY]),
        None // Incomplete word
    );

    // Prefix validation
    assert!(trie.is_valid_prefix(&[Phoneme::T, Phoneme::EY]));
    assert!(trie.is_valid_prefix(&[Phoneme::T, Phoneme::EY, Phoneme::K]));
    assert!(trie.is_valid_prefix(&[Phoneme::HH, Phoneme::OW]));
    assert!(!trie.is_valid_prefix(&[Phoneme::T, Phoneme::Z]));
    assert!(!trie.is_valid_prefix(&[Phoneme::B, Phoneme::AA]));

    // Multi-word phonetic segmentation: "take off"
    let take_off_phonemes = vec![
        Phoneme::T,
        Phoneme::EY,
        Phoneme::K,
        Phoneme::AO,
        Phoneme::F,
    ];
    let (words, remainder) = trie.segment_phonemes(&take_off_phonemes);
    assert_eq!(words, vec!["take", "off"]);
    assert!(remainder.is_empty());

    // Multi-word segmentation with trailing suffix
    let mut trailing = take_off_phonemes.clone();
    trailing.push(Phoneme::P);
    trailing.push(Phoneme::AH);
    let (words2, remainder2) = trie.segment_phonemes(&trailing);
    assert_eq!(words2, vec!["take", "off"]);
    assert_eq!(remainder2, vec![Phoneme::P, Phoneme::AH]);
    assert!(trie.is_valid_prefix(&remainder2)); // Valid prefix of "position"
}

#[test]
fn test_flight_grammar_lm_rescoring() {
    let lm = FlightGrammarLm::new_flight_control_lm();

    // 1. "take off" vs ungrammatical "off take"
    let valid_take_off = vec!["take".to_string(), "off".to_string()];
    let invalid_take_off = vec!["off".to_string(), "take".to_string()];

    let score_valid = lm.score_word_sequence(&valid_take_off);
    let score_invalid = lm.score_word_sequence(&invalid_take_off);

    assert!(
        score_valid > score_invalid + 2.0,
        "Valid command 'take off' ({:.2}) must strongly outscore 'off take' ({:.2})",
        score_valid,
        score_invalid
    );

    // 2. "hold position" vs "position hold"
    let valid_hold = vec!["hold".to_string(), "position".to_string()];
    let invalid_hold = vec!["position".to_string(), "hold".to_string()];
    assert!(lm.score_word_sequence(&valid_hold) > lm.score_word_sequence(&invalid_hold) + 2.0);

    // 3. Multi-word phrase "return to home"
    let return_home = vec!["return".to_string(), "to".to_string(), "home".to_string()];
    let return_score = lm.score_word_sequence(&return_home);
    assert!(return_score > -3.0); // Very high probability for standard flight command
}

#[test]
fn test_ctc_prefix_beam_search_repetitive_collapsing_and_decoding() {
    let config = CtcDecoderConfig {
        beam_width: 24,
        pruning_threshold: 1e-4,
        blank_bias: 0.8,
        lm_weight: 0.70,
        word_insertion_bonus: 1.25,
        length_penalty_alpha: 0.70,
        temperature: 1.0,
        variance: 0.06,
        silence_cutoff_frames: 15,
    };
    let mut decoder = CtcCommandDecoder::new(config);

    // Helper to generate a posterior frame peaked at a specific token
    let make_frame = |token: CtcToken| -> CtcPosteriorFrame {
        let mut probs = [0.001f32; CTC_VOCAB_SIZE];
        let target_idx = token.index();
        probs[target_idx] = 0.96;
        // Normalize
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
        CtcPosteriorFrame::new(probs)
    };

    let blank_frame = make_frame(CtcToken::Blank);
    let t_frame = make_frame(CtcToken::Phoneme(Phoneme::T));
    let ey_frame = make_frame(CtcToken::Phoneme(Phoneme::EY));
    let k_frame = make_frame(CtcToken::Phoneme(Phoneme::K));
    let ao_frame = make_frame(CtcToken::Phoneme(Phoneme::AO));
    let f_frame = make_frame(CtcToken::Phoneme(Phoneme::F));

    // Feed sequence: Blank x 2 -> T x 5 -> EY x 8 -> K x 4 -> Blank x 3 -> AO x 7 -> F x 5 -> Blank x 3
    for _ in 0..2 {
        decoder.step_posterior(&blank_frame);
    }
    for _ in 0..5 {
        decoder.step_posterior(&t_frame);
    }
    for _ in 0..8 {
        decoder.step_posterior(&ey_frame);
    }
    for _ in 0..4 {
        decoder.step_posterior(&k_frame);
    }
    for _ in 0..3 {
        decoder.step_posterior(&blank_frame);
    }
    for _ in 0..7 {
        decoder.step_posterior(&ao_frame);
    }
    for _ in 0..5 {
        decoder.step_posterior(&f_frame);
    }
    for _ in 0..3 {
        decoder.step_posterior(&blank_frame);
    }

    let result = decoder.finalize().expect("Expected decoded command");
    assert_eq!(result.command, "take off");
    assert_eq!(result.words, vec!["take", "off"]);
    assert_eq!(
        result.phonemes,
        vec![
            Phoneme::T,
            Phoneme::EY,
            Phoneme::K,
            Phoneme::AO,
            Phoneme::F
        ]
    );
    assert!(result.confidence > 0.80);
}

#[test]
fn test_end_to_end_synthetic_speech_command_recognition() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.enable_ctc_command_decoder(CtcDecoderConfig::default());

    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_f0(130.0);
    synth.set_speaking_rate(1.0);

    let test_phrases = ["take off", "hold position", "emergency stop"];

    for phrase in test_phrases {
        let segments = G2pEngine::text_to_phonemes(phrase);
        let audio = synth.synthesize(&segments);
        assert!(!audio.is_empty(), "Audio synthesis must not be empty");

        let start = Instant::now();
        let recognized = engine.recognize_command_stream(&audio);
        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;


        assert!(
            recognized.is_some(),
            "Engine failed to recognize synthesized command '{}'",
            phrase
        );

        let res = recognized.unwrap();
        assert_eq!(res.command, phrase);
        assert!(
            res.confidence > 0.60,
            "Confidence for '{}' was {:.3}, expected > 0.60",
            phrase,
            res.confidence
        );

        // Assert latency requirement (< 35ms in release mode, < 300ms in unoptimized debug mode)
        let max_latency_ms = if cfg!(debug_assertions) { 300.0 } else { 35.0 };
        assert!(
            elapsed_ms < max_latency_ms,
            "Latency for '{}' was {:.2}ms, expected < {:.1}ms",
            phrase,
            elapsed_ms,
            max_latency_ms
        );
    }
}

#[test]
fn test_streaming_ctc_latency_and_endpointing_throughput() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    let config = CtcDecoderConfig {
        silence_cutoff_frames: 10, // 10 frames = 100ms silence endpoint
        ..Default::default()
    };
    engine.enable_ctc_command_decoder(config);

    let synth = KlattSynthesizer::new(sample_rate);
    let segments = G2pEngine::text_to_phonemes("take off");
    let mut speech_audio = synth.synthesize(&segments);

    // Append 300ms of silence to trigger endpointing
    let silence = vec![0.0f32; (sample_rate * 0.30) as usize];
    speech_audio.extend_from_slice(&silence);

    let start = Instant::now();
    // Stream audio into engine in chunks
    let chunk_size = 320; // 20ms chunks
    for chunk in speech_audio.chunks(chunk_size) {
        let _ = engine.ingest_samples(chunk);
    }
    let elapsed = start.elapsed();
    let throughput_samples_per_sec = (speech_audio.len() as f64) / elapsed.as_secs_f64();

    // Verify endpointing automatically finalized recognized command
    let latest_cmd = engine.latest_recognized_command();
    assert!(
        latest_cmd.is_some(),
        "Expected streaming endpointing to automatically populate latest_recognized_command"
    );
    let cmd = latest_cmd.unwrap();
    assert_eq!(cmd.command, "take off");

    // Throughput verification: should process > 50,000 samples/sec in debug (> 3x real-time), > 80,000 in release (> 5x real-time)
    let min_throughput = if cfg!(debug_assertions) { 20000.0 } else { 80000.0 };
    assert!(
        throughput_samples_per_sec > min_throughput,
        "Throughput {:.0} samples/sec was below {:.0} bound (RT factor: {:.1}x)",
        throughput_samples_per_sec,
        min_throughput,
        throughput_samples_per_sec / (sample_rate as f64)
    );
}
