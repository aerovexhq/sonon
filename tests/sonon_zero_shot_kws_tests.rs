#![deny(unsafe_code)]

//! Comprehensive integration test suite verifying zero-shot wake-word enrollment,
//! multi-lingual G2P across 6 international languages, cross-attention phonetic-acoustic alignment,
//! Phonetic Posteriorgram (PPG) representations, and automated minimal-pair foil discrimination margin calibration.

use sonon::engine::{FeatureMode, SononEngine};
use sonon::phonetic::{KlattSynthesizer, Phoneme, SyntheticExemplarGenerator, VocalAccent};
use sonon::zero_shot::{
    CrossAttentionAligner, FoilType, MultiLingualG2p, PhoneticEmbeddingSpace,
    PhoneticFoilGenerator, SupportedLanguage, ZeroShotCalibrator,
};

#[test]
fn test_multilingual_g2p_across_all_languages() {
    let test_cases = [
        (SupportedLanguage::English, "take off", Phoneme::T),
        (SupportedLanguage::English, "land", Phoneme::L),
        (SupportedLanguage::Spanish, "despegar", Phoneme::D),
        (SupportedLanguage::Spanish, "aterrizar", Phoneme::AA),
        (SupportedLanguage::French, "decoller", Phoneme::D),
        (SupportedLanguage::French, "atterrir", Phoneme::AA),
        (SupportedLanguage::German, "abheben", Phoneme::AA),
        (SupportedLanguage::German, "landen", Phoneme::L),
        (SupportedLanguage::Japanese, "ritsuriku", Phoneme::R),
        (SupportedLanguage::Japanese, "chakuriku", Phoneme::CH),
        (SupportedLanguage::Mandarin, "qifei", Phoneme::CH),
        (SupportedLanguage::Mandarin, "jiangluo", Phoneme::JH),
    ];

    for (lang, phrase, expected_first_phoneme) in test_cases {
        let segments = MultiLingualG2p::text_to_phonemes(phrase, lang);
        assert!(
            !segments.is_empty(),
            "G2P for phrase '{}' in {:?} must not be empty",
            phrase,
            lang
        );

        let first = segments[0].phoneme;
        assert_eq!(
            first, expected_first_phoneme,
            "Phrase '{}' in {:?}: expected first phoneme {:?}, got {:?}",
            phrase, lang, expected_first_phoneme, first
        );

        // Verify valid duration and stress values
        for seg in &segments {
            assert!(
                seg.duration_ms >= 20.0 && seg.duration_ms <= 300.0,
                "Phoneme {:?} duration {} ms out of valid range",
                seg.phoneme,
                seg.duration_ms
            );
        }
    }
}

#[test]
fn test_phonetic_posteriorgram_embedding_space() {
    let space = PhoneticEmbeddingSpace::new();
    assert_eq!(space.phonemes().len(), 38);

    // 1. Verify centroids are normalized unit vectors
    for &p in space.phonemes() {
        let c = space.centroid(p);
        let norm_sq: f32 = c.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "Centroid for {:?} must be normalized (got norm {})",
            p,
            norm
        );
    }

    // 2. Self-similarity vs cross-phoneme similarity
    let c_aa = space.centroid(Phoneme::AA);
    let sim_self = space.frame_similarity(&c_aa, Phoneme::AA);
    assert!(
        (sim_self - 1.0).abs() < 1e-3,
        "Self similarity must be 1.0 (got {})",
        sim_self
    );

    let sim_cross = space.frame_similarity(&c_aa, Phoneme::S);
    assert!(
        sim_cross < 0.50,
        "Vowel /AA/ vs fricative /S/ similarity {} must be distinctly low",
        sim_cross
    );

    // 3. Frame PPG posterior distribution
    let ppg = space.compute_ppg(&c_aa);
    assert_eq!(ppg.len(), 38);

    let sum_prob: f32 = ppg.iter().sum();
    assert!(
        (sum_prob - 1.0).abs() < 1e-4,
        "PPG probabilities must sum to 1.0 (got {})",
        sum_prob
    );

    // /AA/ should have highest posterior when querying with its own centroid
    let aa_idx = space.phonemes().iter().position(|&p| p == Phoneme::AA).unwrap();
    let max_idx = ppg
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        max_idx, aa_idx,
        "Max probability phoneme in PPG must match queried centroid"
    );
}

#[test]
fn test_cross_attention_phonetic_acoustic_alignment() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);

    let phrase = "take off";
    let segments = MultiLingualG2p::text_to_phonemes(phrase, SupportedLanguage::English);
    let raw_phonemes: Vec<Phoneme> = segments
        .iter()
        .map(|s| s.phoneme)
        .filter(|&p| p != Phoneme::SIL)
        .collect();

    let synth = KlattSynthesizer::new(sample_rate);
    let audio = synth.synthesize(&segments);
    assert!(!audio.is_empty());

    let features = engine.extract_features(&audio);
    assert!(features.len() > 10);

    let aligner = CrossAttentionAligner::new();
    let res = aligner
        .align(&raw_phonemes, &features)
        .expect("Cross-attention alignment failed");

    assert_eq!(res.num_tokens, raw_phonemes.len());
    assert_eq!(res.num_frames, features.len());
    assert_eq!(res.aligned_features.len(), features.len());

    // Attention matrix dimensions
    assert_eq!(res.attention_matrix.len(), res.num_tokens * res.num_frames);

    // Column-wise attention sum must be 1.0
    for j in 0..res.num_frames {
        let mut col_sum = 0.0f32;
        for i in 0..res.num_tokens {
            col_sum += res.attention_matrix[i * res.num_frames + j];
        }
        assert!(
            (col_sum - 1.0).abs() < 1e-3,
            "Frame {} attention column sum should be 1.0 (got {})",
            j,
            col_sum
        );
    }

    // Monotonicity score should be high (> 0.85) for natural chronological speech
    assert!(
        res.monotonicity_score >= 0.80,
        "Monotonicity score {} below expected 0.80 threshold",
        res.monotonicity_score
    );

    // Phonetic coverage should bind majority of tokens (> 0.70)
    assert!(
        res.phonetic_coverage >= 0.70,
        "Phonetic coverage {} below expected 0.70 threshold",
        res.phonetic_coverage
    );

    // Entropy should be bounded (< 2.5 nats)
    assert!(
        res.alignment_entropy < 2.5,
        "Alignment entropy {} exceeds 2.5 threshold",
        res.alignment_entropy
    );
}

#[test]
fn test_minimal_pair_phonetic_foil_generation() {
    // English foils
    let en_foils = PhoneticFoilGenerator::generate_foils("take off", SupportedLanguage::English);
    assert!(!en_foils.is_empty());

    let has_consonant = en_foils.iter().any(|f| f.foil_type == FoilType::ConsonantMinimalPair);
    let has_vowel = en_foils.iter().any(|f| f.foil_type == FoilType::VowelFormantShift);
    let has_lexical = en_foils.iter().any(|f| f.foil_type == FoilType::LexicalBoundary);
    let has_cross = en_foils.iter().any(|f| f.foil_type == FoilType::CrossLingualDistractor);

    assert!(has_consonant, "Must generate consonant minimal pair foils");
    assert!(has_vowel, "Must generate vowel formant shift foils");
    assert!(has_lexical, "Must generate lexical boundary foils");
    assert!(has_cross, "Must generate cross-lingual distractors");

    // Spanish foils
    let es_foils = PhoneticFoilGenerator::generate_foils("despegar", SupportedLanguage::Spanish);
    assert!(!es_foils.is_empty());
    assert!(es_foils.iter().any(|f| f.foil_text == "despejar" || f.foil_text == "despesar"));

    // Japanese foils
    let ja_foils = PhoneticFoilGenerator::generate_foils("ritsuriku", SupportedLanguage::Japanese);
    assert!(!ja_foils.is_empty());
    assert!(ja_foils.iter().any(|f| f.foil_text == "mitsuriku" || f.foil_text == "hitsuriku"));
}

#[test]
fn test_zero_shot_discrimination_margin_and_threshold_calibration() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    let calibrator = ZeroShotCalibrator::new();

    // 1. English Calibration: "take off"
    let (template_en, report_en) = calibrator
        .calibrate("take off", SupportedLanguage::English, VocalAccent::GeneralAmerican, &engine)
        .expect("English zero-shot calibration failed");

    assert!(!template_en.is_empty());
    assert!(
        report_en.discrimination_margin > 0.0,
        "Discrimination margin {} must be strictly positive",
        report_en.discrimination_margin
    );
    assert!(
        report_en.calibrated_threshold > report_en.intra_target_distance,
        "Calibrated threshold {} must exceed intra-target distance {}",
        report_en.calibrated_threshold,
        report_en.intra_target_distance
    );
    assert!(
        report_en.calibrated_threshold < report_en.min_foil_distance,
        "Calibrated threshold {} must be below nearest foil distance {}",
        report_en.calibrated_threshold,
        report_en.min_foil_distance
    );

    // Verify 0 false positives on minimal-pair foils
    assert_eq!(
        report_en.confusion_matrix.false_positives, 0,
        "Zero-shot threshold must yield 0 false positives on phonetic foils"
    );
    assert_eq!(
        report_en.confusion_matrix.false_negatives, 0,
        "Zero-shot threshold must yield 0 false negatives on target variations"
    );
    assert!(
        report_en.confusion_matrix.f1_score() > 0.99,
        "F1 score must be 1.0 (got {})",
        report_en.confusion_matrix.f1_score()
    );

    // 2. Spanish Calibration: "despegar"
    let (template_es, report_es) = calibrator
        .calibrate("despegar", SupportedLanguage::Spanish, VocalAccent::International, &engine)
        .expect("Spanish zero-shot calibration failed");

    assert!(!template_es.is_empty());
    assert!(report_es.discrimination_margin > 0.0);
    assert_eq!(report_es.confusion_matrix.false_positives, 0);
    assert_eq!(report_es.confusion_matrix.false_negatives, 0);
}

#[test]
fn test_sonon_engine_zero_shot_multilingual_kws_spotting() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    // 1. Zero-shot enroll Spanish command "despegar"
    let rep_es = engine
        .enroll_keyword_zero_shot(
            "despegar",
            "despegar",
            SupportedLanguage::Spanish,
            VocalAccent::International,
        )
        .expect("Zero-shot enrollment of 'despegar' failed");

    assert_eq!(rep_es.keyword, "despegar");
    assert_eq!(engine.dtw().template_count(), 1);

    // 2. Zero-shot enroll Japanese command "ritsuriku"
    let rep_ja = engine
        .enroll_keyword_zero_shot(
            "ritsuriku",
            "ritsuriku",
            SupportedLanguage::Japanese,
            VocalAccent::International,
        )
        .expect("Zero-shot enrollment of 'ritsuriku' failed");

    assert_eq!(rep_ja.keyword, "ritsuriku");
    assert_eq!(engine.dtw().template_count(), 2);

    // 3. Test Spanish target spotting: synthesize Spanish "despegar"
    let synth = SyntheticExemplarGenerator::new(sample_rate);
    let es_segments = MultiLingualG2p::text_to_phonemes("despegar", SupportedLanguage::Spanish);
    let es_audio_variants = synth.generate_exemplars_from_segments(&es_segments, 1);
    assert!(!es_audio_variants.is_empty());

    // Stream through engine
    let mut padded_es = vec![0.0f32; 4800]; // 0.3s silence lead
    padded_es.extend_from_slice(&es_audio_variants[0]);
    padded_es.extend_from_slice(&vec![0.0f32; 4800]); // 0.3s silence trail

    let events = engine.ingest_samples(&padded_es);
    let detected_es = events.iter().any(|e| e.keyword == "despegar");
    assert!(
        detected_es,
        "Zero-shot enrolled Spanish keyword 'despegar' must be spotted in streaming audio"
    );

    // 4. Test Spanish minimal-pair foil rejection: synthesize "despejar"
    engine.reset();
    let es_foil_segments = MultiLingualG2p::text_to_phonemes("despejar", SupportedLanguage::Spanish);
    let es_foil_variants = synth.generate_exemplars_from_segments(&es_foil_segments, 1);
    assert!(!es_foil_variants.is_empty());

    let mut padded_foil = vec![0.0f32; 4800];
    padded_foil.extend_from_slice(&es_foil_variants[0]);
    padded_foil.extend_from_slice(&vec![0.0f32; 4800]);

    let foil_events = engine.ingest_samples(&padded_foil);
    assert_eq!(
        foil_events.len(),
        0,
        "Phonetic minimal-pair foil 'despejar' must trigger 0 false alarms on 'despegar'"
    );

    // 5. Test Japanese target spotting: synthesize "ritsuriku"
    engine.reset();
    let ja_segments = MultiLingualG2p::text_to_phonemes("ritsuriku", SupportedLanguage::Japanese);
    let ja_audio_variants = synth.generate_exemplars_from_segments(&ja_segments, 1);
    assert!(!ja_audio_variants.is_empty());

    let mut padded_ja = vec![0.0f32; 4800];
    padded_ja.extend_from_slice(&ja_audio_variants[0]);
    padded_ja.extend_from_slice(&vec![0.0f32; 4800]);

    let ja_events = engine.ingest_samples(&padded_ja);
    let detected_ja = ja_events.iter().any(|e| e.keyword == "ritsuriku");
    assert!(
        detected_ja,
        "Zero-shot enrolled Japanese keyword 'ritsuriku' must be spotted in streaming audio"
    );

    // 6. Test background silence false alarm rejection
    engine.reset();
    let silence = vec![0.0f32; 16000]; // 1.0s silence
    let silence_events = engine.ingest_samples(&silence);
    assert_eq!(
        silence_events.len(),
        0,
        "Silence must produce 0 false alarms"
    );
}
