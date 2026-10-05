#![deny(unsafe_code)]

//! Comprehensive unit and integration test suite for Distribution-Free Conformal Prediction
//! in embedded acoustic wake-word spotting (Sonon crate).
//!
//! Validates:
//! 1. Finite-sample coverage guarantees under exchangeable non-conformity score distributions.
//! 2. Exact ceiling quantile indexing across diverse calibration sizes and risk levels.
//! 3. Multi-hypothesis conformal prediction set semantics (singleton, empty, ambiguous).
//! 4. Phonetic minimal-pair foil separation bounds ("falcon" vs "walcon").
//! 5. SononEngine conformal verification streaming integration flow.
//! 6. SononEngine strict conformal gating against out-of-distribution acoustic triggers.
//! 7. Mathematical properties, monotonicity, Serde serialization, and lifecycle edge cases.

use sonon::conformal::{
    ConformalCalibrationReport, ConformalConfig, ConformalKwsPredictor, ConformalPredictionSet,
};
use sonon::dtw::DtwMatcher;
use sonon::engine::{KeywordEvent, SononEngine};
use sonon::phonetic::{G2pEngine, KlattSynthesizer};

/// Deterministic Linear Congruential Generator (LCG) for reproducible pseudo-random numbers
/// without requiring external crate dependencies.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    /// Sample from an exponential distribution with specified mean.
    /// Represents non-negative acoustic distance scores.
    fn next_exponential(&mut self, mean: f32) -> f32 {
        let u = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
        -mean * u.ln()
    }
}

/// Helper to synthesize a speech phrase using the internal phonetic engine.
fn synthesize_speech_phrase(phrase: &str, f0: f32, sample_rate: f32) -> Vec<f32> {
    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_f0(f0);
    let phonemes = G2pEngine::text_to_phonemes(phrase);
    synth.synthesize(&phonemes)
}


// ---------------------------------------------------------------------------
// Test 1: Finite-Sample Coverage Guarantee
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_finite_sample_coverage_guarantee() {
    let mut rng = DeterministicRng::new(20261005);
    let alpha = 0.10f32;
    let n_calibration = 100usize;
    let n_test = 2000usize;
    let num_trials = 50usize;

    let config = ConformalConfig {
        risk_level: alpha,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    };

    let mut trial_coverages = Vec::with_capacity(num_trials);

    for _ in 0..num_trials {
        let mut predictor = ConformalKwsPredictor::new(config.clone());

        // Generate exchangeable calibration non-conformity scores from Exponential(mean = 2.0)
        let mut calib_scores = Vec::with_capacity(n_calibration);
        for _ in 0..n_calibration {
            calib_scores.push(rng.next_exponential(2.0));
        }

        // Foils placed comfortably beyond positive cluster so threshold is purely quantile-driven
        let foil_scores = vec![50.0f32, 60.0, 70.0];

        let report = predictor
            .calibrate("kw_test", &calib_scores, &foil_scores)
            .expect("Calibration must succeed");

        assert_eq!(report.positive_samples, n_calibration);
        assert!((report.coverage_guarantee - (1.0 - alpha)).abs() < 1e-5);
        // Quantile index for n=100, alpha=0.10: ceil(101 * 0.90) = ceil(90.9) = 91
        assert_eq!(report.quantile_index, 91);

        // Generate unseen test samples from the exact same data-generating distribution
        let mut covered_count = 0usize;
        for _ in 0..n_test {
            let test_sample = rng.next_exponential(2.0);
            if predictor.is_verified_detection("kw_test", test_sample) {
                covered_count += 1;
            }
        }

        let empirical_coverage = (covered_count as f32) / (n_test as f32);
        trial_coverages.push(empirical_coverage);
    }

    // Mathematical theorem: E[coverage] = ceil((n+1)(1-alpha)) / (n+1) >= 1 - alpha
    // Allowing tight +/-0.5% finite Monte Carlo sampling variance on 100k generated samples
    let mean_coverage: f32 = trial_coverages.iter().sum::<f32>() / (num_trials as f32);
    assert!(
        mean_coverage >= (1.0 - alpha) - 0.005,
        "Mean empirical coverage {} must meet formal guarantee within Monte Carlo margin",
        mean_coverage
    );

    // Verify for a single larger calibration set (n=199), empirical coverage strictly satisfies >= 1 - alpha
    let mut predictor_large = ConformalKwsPredictor::new(config);
    let mut large_calib = Vec::with_capacity(199);
    for _ in 0..199 {
        large_calib.push(rng.next_exponential(2.0));
    }
    let rep_large = predictor_large
        .calibrate("kw_large", &large_calib, &[100.0, 120.0])
        .expect("Calibration of large sample set must succeed");
    // ceil(200 * 0.90) = 180
    assert_eq!(rep_large.quantile_index, 180);

    let mut covered_large = 0usize;
    let n_large_test = 5000usize;
    for _ in 0..n_large_test {
        let sample = rng.next_exponential(2.0);
        if predictor_large.is_verified_detection("kw_large", sample) {
            covered_large += 1;
        }
    }
    let large_coverage = (covered_large as f32) / (n_large_test as f32);
    assert!(
        large_coverage >= 1.0 - alpha,
        "Empirical coverage {} on 5000 unseen test points must satisfy guarantee >= {}",
        large_coverage,
        1.0 - alpha
    );
}

// ---------------------------------------------------------------------------
// Test 2: Exact Quantile Indexing Calculation
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_exact_quantile_indexing() {
    // Test exact ceil((n + 1) * (1 - alpha)) index calculation across diverse sample sizes and risk levels

    // Case 1: n = 10
    // alpha = 0.10: ceil(11 * 0.90) = ceil(9.90) = 10
    // alpha = 0.20: ceil(11 * 0.80) = ceil(8.80) = 9
    // alpha = 0.35: ceil(11 * 0.65) = ceil(7.15) = 8
    // alpha = 0.05: ceil(11 * 0.95) = ceil(10.45) = 11 -> clamped to 10
    let test_matrix_n10 = [
        (0.10f32, 10usize),
        (0.20f32, 9usize),
        (0.35f32, 8usize),
        (0.05f32, 10usize),
    ];
    let scores_10: Vec<f32> = (1..=10).map(|i| i as f32).collect();

    for (alpha, expected_idx) in test_matrix_n10 {
        let mut pred = ConformalKwsPredictor::new(ConformalConfig {
            risk_level: alpha,
            min_calibration_samples: 5,
            foil_penalty_weight: 1.25,
        });
        let rep = pred
            .calibrate("kws", &scores_10, &[])
            .expect("Calibration must succeed");
        assert_eq!(
            rep.quantile_index, expected_idx,
            "For n=10, alpha={}, expected index {}, got {}",
            alpha, expected_idx, rep.quantile_index
        );
        assert_eq!(
            rep.conformal_threshold, scores_10[expected_idx - 1],
            "Threshold must match ordered score at quantile index"
        );
    }

    // Case 2: n = 19
    // alpha = 0.10: ceil(20 * 0.90) = 18
    // alpha = 0.05: ceil(20 * 0.95) = 19
    // alpha = 0.25: ceil(20 * 0.75) = 15
    let test_matrix_n19 = [
        (0.10f32, 18usize),
        (0.05f32, 19usize),
        (0.25f32, 15usize),
    ];
    let scores_19: Vec<f32> = (1..=19).map(|i| i as f32).collect();

    for (alpha, expected_idx) in test_matrix_n19 {
        let mut pred = ConformalKwsPredictor::new(ConformalConfig {
            risk_level: alpha,
            min_calibration_samples: 5,
            foil_penalty_weight: 1.25,
        });
        let rep = pred
            .calibrate("kws", &scores_19, &[])
            .expect("Calibration must succeed");
        assert_eq!(
            rep.quantile_index, expected_idx,
            "For n=19, alpha={}, expected index {}, got {}",
            alpha, expected_idx, rep.quantile_index
        );
        assert_eq!(rep.conformal_threshold, scores_19[expected_idx - 1]);
    }

    // Case 3: n = 99
    // alpha = 0.10: ceil(100 * 0.90) = 90
    // alpha = 0.05: ceil(100 * 0.95) = 95
    // alpha = 0.01: ceil(100 * 0.99) = 99
    // alpha = 0.02: ceil(100 * 0.98) = 98
    let test_matrix_n99 = [
        (0.10f32, 90usize),
        (0.05f32, 95usize),
        (0.01f32, 99usize),
        (0.02f32, 98usize),
    ];
    let scores_99: Vec<f32> = (1..=99).map(|i| (i as f32) * 0.5).collect();

    for (alpha, expected_idx) in test_matrix_n99 {
        let mut pred = ConformalKwsPredictor::new(ConformalConfig {
            risk_level: alpha,
            min_calibration_samples: 10,
            foil_penalty_weight: 1.25,
        });
        let rep = pred
            .calibrate("kws", &scores_99, &[])
            .expect("Calibration must succeed");
        assert_eq!(
            rep.quantile_index, expected_idx,
            "For n=99, alpha={}, expected index {}, got {}",
            alpha, expected_idx, rep.quantile_index
        );
        assert_eq!(rep.conformal_threshold, scores_99[expected_idx - 1]);
    }

    // Edge risk levels: alpha = 0.001 (near zero risk) and alpha = 0.50 (maximum risk)
    let mut pred_edge_min = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.001,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    });
    let rep_edge_min = pred_edge_min
        .calibrate("kws", &scores_19, &[])
        .expect("Calibration with small alpha must succeed");
    // (19 + 1) * (1 - 0.001) = 19.98 -> ceil is 20 -> clamped to n = 19
    assert_eq!(rep_edge_min.quantile_index, 19);

    let mut pred_edge_half = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.50,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    });
    let rep_edge_half = pred_edge_half
        .calibrate("kws", &scores_19, &[])
        .expect("Calibration with alpha 0.50 must succeed");
    // (19 + 1) * 0.50 = 10.0 -> ceil is 10
    assert_eq!(rep_edge_half.quantile_index, 10);

    // Minimum calibration sample count enforcement
    let mut pred_strict = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 15,
        foil_penalty_weight: 1.25,
    });
    let err = pred_strict.calibrate("kws", &scores_10, &[]);
    assert!(err.is_err(), "Must reject when positive count < min_calibration_samples");
    let err_msg = err.err().unwrap();
    assert!(err_msg.contains("minimum required is 15"));
}

// ---------------------------------------------------------------------------
// Test 3: Prediction Set Semantics (Singleton, Empty, Ambiguous)
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_prediction_set_semantics() {
    let mut predictor = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    });

    // Calibrate three distinct command models
    // "falcon": thresholds around 2.4
    let falcon_pos = vec![1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4];
    predictor
        .calibrate("falcon", &falcon_pos, &[10.0])
        .expect("Falcon calibration must succeed");

    // "eagle": thresholds around 3.2
    let eagle_pos = vec![1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2];
    predictor
        .calibrate("eagle", &eagle_pos, &[10.0])
        .expect("Eagle calibration must succeed");

    // "hawk": thresholds around 1.6
    let hawk_pos = vec![0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6];
    predictor
        .calibrate("hawk", &hawk_pos, &[10.0])
        .expect("Hawk calibration must succeed");

    assert_eq!(predictor.calibrated_count(), 3);

    // Case A: Singleton Prediction Set
    // Observed distances: falcon=1.5 (<=2.4 ok), eagle=4.2 (>3.2 fail), hawk=2.8 (>1.6 fail)
    let query_singleton = [("falcon", 1.5f32), ("eagle", 4.2f32), ("hawk", 2.8f32)];
    let pred_singleton = predictor.predict_set(&query_singleton);

    assert!(pred_singleton.is_singleton());
    assert!(!pred_singleton.is_empty());
    assert!(!pred_singleton.is_ambiguous());
    assert_eq!(pred_singleton.confirmed_keyword(), Some("falcon"));
    assert_eq!(pred_singleton.candidates, vec!["falcon".to_string()]);
    assert_eq!(pred_singleton.best_distance, 1.5);
    assert_eq!(pred_singleton.p_values.len(), 1);
    assert!(pred_singleton.p_values[0] > 0.0 && pred_singleton.p_values[0] <= 1.0);

    // Case B: Empty Prediction Set (Severe Out-of-Distribution Noise)
    // Observed distances all exceed conformal safety bounds
    let query_empty = [("falcon", 7.8f32), ("eagle", 9.1f32), ("hawk", 6.5f32)];
    let pred_empty = predictor.predict_set(&query_empty);

    assert!(pred_empty.is_empty());
    assert!(!pred_singleton.is_empty() != false);
    assert!(!pred_empty.is_singleton());
    assert!(!pred_empty.is_ambiguous());
    assert_eq!(pred_empty.confirmed_keyword(), None);
    assert!(pred_empty.candidates.is_empty());
    assert!(pred_empty.p_values.is_empty());
    assert_eq!(pred_empty.best_distance, f32::INFINITY);

    // Case C: Ambiguous Prediction Set (Multiple Admissible Hypotheses)
    // Observed distances: falcon=1.4 (<=2.4 ok), eagle=2.2 (<=3.2 ok), hawk=1.1 (<=1.6 ok)
    let query_ambiguous = [("falcon", 1.4f32), ("eagle", 2.2f32), ("hawk", 1.1f32)];
    let pred_ambiguous = predictor.predict_set(&query_ambiguous);

    assert!(pred_ambiguous.is_ambiguous());
    assert!(!pred_ambiguous.is_singleton());
    assert!(!pred_ambiguous.is_empty());
    assert_eq!(pred_ambiguous.confirmed_keyword(), None);
    assert_eq!(pred_ambiguous.candidates.len(), 3);
    assert_eq!(pred_ambiguous.p_values.len(), 3);
    assert_eq!(pred_ambiguous.best_distance, 1.1);

    // Case D: Partial ambiguity (2 out of 3 match)
    let query_dual = [("falcon", 1.8f32), ("eagle", 2.5f32), ("hawk", 3.0f32)];
    let pred_dual = predictor.predict_set(&query_dual);
    assert!(pred_dual.is_ambiguous());
    assert_eq!(pred_dual.candidates.len(), 2);
    assert!(pred_dual.candidates.contains(&"falcon".to_string()));
    assert!(pred_dual.candidates.contains(&"eagle".to_string()));
    assert!(!pred_dual.candidates.contains(&"hawk".to_string()));
    assert_eq!(pred_dual.best_distance, 1.8);
}

// ---------------------------------------------------------------------------
// Test 4: Minimal-Pair Foil Separation Bounds ("falcon" vs "walcon")
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_minimal_pair_foil_separation() {
    let mut predictor = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    });

    // Positive calibration distances for target "falcon"
    let pos_falcon = vec![1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2];

    // Minimal-pair negative foil distances for "walcon"
    // Note closest foil is at 2.6
    let foil_walcon = vec![2.6, 2.9, 3.1, 3.8, 4.2];

    // Under alpha = 0.10, n = 10, quantile index = 10, raw positive quantile q_val = 3.2.
    // However, 3.2 >= min_foil (2.6).
    // Conformal safety bounding must clamp the effective threshold to (min_foil * 0.95).max(scores[0])
    // 2.6 * 0.95 = 2.47
    let report = predictor
        .calibrate("falcon", &pos_falcon, &foil_walcon)
        .expect("Calibration with minimal pair foils must succeed");

    let expected_clamped_threshold = 2.6 * 0.95;
    assert!(
        (report.conformal_threshold - expected_clamped_threshold).abs() < 1e-4,
        "Conformal threshold {} must be clamped strictly below closest foil {}",
        report.conformal_threshold,
        expected_clamped_threshold
    );
    assert_eq!(report.positive_samples, 10);
    assert_eq!(report.foil_samples, 5);

    // Verify separation holds:
    // 1. Target "falcon" exemplar at distance 2.0 is verified
    assert!(predictor.is_verified_detection("falcon", 2.0));

    // 2. Minimal-pair foil "walcon" at distance 2.6 is strictly rejected
    assert!(
        !predictor.is_verified_detection("falcon", 2.6),
        "Closest minimal pair foil distance 2.6 must NOT be a verified detection"
    );

    // 3. Higher foil distances are also rejected
    assert!(!predictor.is_verified_detection("falcon", 2.9));
    assert!(!predictor.is_verified_detection("falcon", 3.1));

    // Prediction set test: query containing minimal pair foil should yield empty set
    let foil_query = [("falcon", 2.6f32)];
    let pred_foil = predictor.predict_set(&foil_query);
    assert!(
        pred_foil.is_empty(),
        "Minimal pair foil score must produce an empty prediction set"
    );

    // Acoustic verification with synthesized phonetic utterances
    let sample_rate = 16000.0f32;
    let falcon_ref = synthesize_speech_phrase("falcon", 120.0, sample_rate);
    let walcon_ref = synthesize_speech_phrase("walcon", 120.0, sample_rate);

    let engine = SononEngine::new(sample_rate, 512, 160, 13);
    let feats_falcon = engine.extract_features(&falcon_ref);
    let feats_walcon = engine.extract_features(&walcon_ref);

    let dtw_dist = DtwMatcher::compute_distance_banded(&feats_walcon, &feats_falcon, 8);
    assert!(
        dtw_dist.is_finite() && dtw_dist > 0.0,
        "Synthesized minimal pair foil must yield finite acoustic distance"
    );
}

// ---------------------------------------------------------------------------
// Test 5: SononEngine Conformal Verification Streaming Integration Flow
// ---------------------------------------------------------------------------

#[test]
fn test_sonon_engine_conformal_verification_flow() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    // 1. Synthesize reference template for keyword "falcon"
    let template_audio = synthesize_speech_phrase("falcon", 120.0, sample_rate);
    let features = engine.extract_features(&template_audio);
    assert!(!features.is_empty());

    // Enroll with generous raw DTW threshold to ensure streaming spotting triggers
    engine.enroll_keyword_banded("falcon", features.clone(), 3.5, 8);
    assert_eq!(engine.dtw().template_count(), 1);

    // 2. Enable Conformal Guarantees in SononEngine
    let conf_cfg = ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    };
    engine.enable_conformal_guarantees(conf_cfg);
    assert!(engine.conformal_predictor().is_some());

    // 3. Calibrate conformal predictor using positive waveforms and foils
    let mut pos_waveforms = Vec::new();
    for f0 in [105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0] {
        pos_waveforms.push(synthesize_speech_phrase("falcon", f0, sample_rate));
    }

    let mut foil_waveforms = Vec::new();
    for f0 in [112.0, 116.0, 120.0, 124.0] {
        foil_waveforms.push(synthesize_speech_phrase("walcon", f0, sample_rate));
    }

    for (i, pos) in pos_waveforms.iter().enumerate() {
        let f = engine.extract_features(pos);
        let d = DtwMatcher::compute_distance_banded(&f, &features, 8);
        println!("pos {}: dist = {}", i, d);
    }
    for (i, foil) in foil_waveforms.iter().enumerate() {
        let f = engine.extract_features(foil);
        let d = DtwMatcher::compute_distance_banded(&f, &features, 8);
        println!("foil {}: dist = {}", i, d);
    }
    let report = engine
        .calibrate_keyword_conformal("falcon", &pos_waveforms, &foil_waveforms)
        .expect("Calibration of 'falcon' must succeed");

    assert_eq!(report.keyword, "falcon");
    assert_eq!(report.positive_samples, 8);
    assert_eq!(report.foil_samples, 4);
    assert!((report.coverage_guarantee - 0.90).abs() < 1e-4);

    // 4. Ingest matching stream and verify keyword spotting
    engine.reset();
    let events = engine.ingest_samples(&template_audio);
    println!("Direct events: {:?}", events);

    assert!(
        !events.is_empty(),
        "Ingesting matching audio must emit at least one KeywordEvent"
    );

    let matched_event = events
        .iter()
        .find(|e| e.keyword == "falcon")
        .expect("Must find 'falcon' event");

    // 5. Verify conformal metadata attached to KeywordEvent
    assert!(
        matched_event.conformal_p_value.is_some(),
        "Conformal p-value must be populated"
    );
    let p_val = matched_event.conformal_p_value.unwrap();
    assert!(
        p_val > 0.0 && p_val <= 1.0,
        "Conformal p-value {} must be in (0.0, 1.0]",
        p_val
    );
    assert!(
        matched_event.is_conformal_verified,
        "Matching audio within calibrated cluster must be conformal verified. Event: {:?}",
        matched_event
    );
}

// ---------------------------------------------------------------------------
// Test 6: SononEngine Strict Conformal Gating
// ---------------------------------------------------------------------------

#[test]
fn test_sonon_engine_conformal_strict_gating() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    // 1. Enroll template "falcon" with a permissive raw DTW threshold (6.5)
    // allowing close minimal pair "walcon" (distance ~6.1) to trigger raw DTW
    let template_audio = synthesize_speech_phrase("falcon", 120.0, sample_rate);
    let features = engine.extract_features(&template_audio);
    engine.enroll_keyword_banded("falcon", features, 6.5, 8);

    // 2. Enable conformal guarantees
    engine.enable_conformal_guarantees(ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 8,
        foil_penalty_weight: 1.25,
    });

    // 3. Calibrate conformal predictor with tight positive exemplars
    // Conformal threshold will be around 1.3
    let mut pos_waveforms = Vec::new();
    for f0 in [105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0] {
        pos_waveforms.push(synthesize_speech_phrase("falcon", f0, sample_rate));
    }
    let mut foil_waveforms = Vec::new();
    for f0 in [112.0, 116.0, 120.0, 124.0] {
        foil_waveforms.push(synthesize_speech_phrase("walcon", f0, sample_rate));
    }
    let calib_rep = engine
        .calibrate_keyword_conformal("falcon", &pos_waveforms, &foil_waveforms)
        .expect("Calibration must succeed");
    assert!(
        calib_rep.conformal_threshold < 2.0,
        "Calibrated conformal threshold {} should be well below 2.0",
        calib_rep.conformal_threshold
    );

    // Audio A: Valid in-distribution "falcon" audio
    let audio_valid = template_audio.clone();

    // Audio B: Minimal-pair foil "walcon" audio (false trigger in raw DTW)
    let audio_foil = synthesize_speech_phrase("walcon", 120.0, sample_rate);

    // Verify raw DTW distances
    let feats_foil = engine.extract_features(&audio_foil);
    let foil_dist = DtwMatcher::compute_distance_banded(
        &feats_foil,
        &engine.dtw().templates()[0].features,
        8,
    );
    assert!(
        foil_dist < 6.5 && foil_dist > calib_rep.conformal_threshold,
        "Foil distance {} must be < raw DTW threshold (6.5) but > conformal threshold ({})",
        foil_dist,
        calib_rep.conformal_threshold
    );

    // Part A: Strict gating DISABLED (conformal_gating = false)
    engine.reset();
    engine.set_conformal_gating(false);
    let events_ungated = engine.ingest_samples(&audio_foil);
    assert!(
        !events_ungated.is_empty(),
        "Without conformal gating, permissive DTW allows foil 'walcon' to trigger an event"
    );
    let ev_ungated = &events_ungated[0];
    assert_eq!(ev_ungated.keyword, "falcon");
    assert!(
        !ev_ungated.is_conformal_verified,
        "Foil trigger must NOT be conformally verified"
    );
    assert!(ev_ungated.conformal_p_value.is_some());

    // Part B: Strict gating ENABLED (conformal_gating = true)
    // The unverified foil match must be strictly suppressed
    engine.reset();
    engine.set_conformal_gating(true);
    let events_gated = engine.ingest_samples(&audio_foil);
    assert_eq!(
        events_gated.len(),
        0,
        "Strict conformal gating must completely suppress unverified foil trigger"
    );

    // Part C: Ingest valid, in-distribution audio under strict gating: must be emitted
    engine.reset();
    let events_valid = engine.ingest_samples(&audio_valid);
    assert!(
        !events_valid.is_empty(),
        "Valid in-distribution audio must be emitted under strict conformal gating"
    );
    let ev_valid = &events_valid[0];
    assert_eq!(ev_valid.keyword, "falcon");
    assert!(
        ev_valid.is_conformal_verified,
        "Valid detection must be verified"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Conformal p-Value Bounds and Monotonicity
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_p_value_monotonicity_and_bounds() {
    let mut predictor = ConformalKwsPredictor::new(ConformalConfig {
        risk_level: 0.10,
        min_calibration_samples: 5,
        foil_penalty_weight: 1.25,
    });

    let scores = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let n = scores.len();
    predictor
        .calibrate("cmd", &scores, &[])
        .expect("Calibration must succeed");

    // Formula: p(d) = (1 + count(s_i >= d)) / (n + 1)
    // Test point strictly below minimum score (d = 0.5)
    let p_min = predictor.compute_p_value("cmd", 0.5);
    // count(s_i >= 0.5) = 5 -> (1 + 5) / (5 + 1) = 6/6 = 1.0
    assert_eq!(p_min, 1.0);

    // Test point at exact median (d = 3.0)
    let p_med = predictor.compute_p_value("cmd", 3.0);
    // count(s_i >= 3.0) = 3 -> (1 + 3) / 6 = 4/6 = 2/3
    assert!((p_med - (4.0 / 6.0)).abs() < 1e-6);

    // Test point strictly above maximum score (d = 10.0)
    let p_max = predictor.compute_p_value("cmd", 10.0);
    // count(s_i >= 10.0) = 0 -> (1 + 0) / 6 = 1/6
    let min_possible_p = 1.0 / ((n + 1) as f32);
    assert_eq!(p_max, min_possible_p);

    // Verify strict monotonicity: as distance increases, p-value is non-increasing
    let test_distances = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 10.0];
    let mut last_p = 1.0f32;
    for &d in &test_distances {
        let p = predictor.compute_p_value("cmd", d);
        assert!(
            p <= last_p + 1e-7,
            "p-value must be monotonically non-increasing with distance: at d={}, p={} > previous {}",
            d, p, last_p
        );
        assert!(p >= min_possible_p, "p-value {} cannot fall below 1 / (n + 1)", p);
        assert!(p <= 1.0, "p-value {} cannot exceed 1.0", p);
        last_p = p;
    }

    // Uncalibrated keyword must yield 0.0
    assert_eq!(predictor.compute_p_value("uncalibrated_kw", 1.0), 0.0);
}

// ---------------------------------------------------------------------------
// Test 8: Engine Conformal Lifecycle, Recalibration, and Disable
// ---------------------------------------------------------------------------

#[test]
fn test_sonon_engine_conformal_lifecycle_and_error_handling() {
    let mut engine = SononEngine::new(16000.0, 256, 128, 13);

    // Attempting calibration before enabling guarantees must return an Err
    let err_uninit = engine.calibrate_distances_conformal("kw", &[1.0; 10], &[]);
    assert!(err_uninit.is_err());
    assert!(err_uninit.err().unwrap().contains("not enabled"));

    // Enable guarantees
    let cfg = ConformalConfig::default();
    assert_eq!(cfg.risk_level, 0.05);
    assert_eq!(cfg.min_calibration_samples, 8);
    engine.enable_conformal_guarantees(cfg);
    assert!(engine.conformal_predictor().is_some());

    // Attempting to calibrate un-enrolled keyword via calibrate_keyword_conformal
    let dummy_audio = vec![vec![0.0f32; 512]; 8];
    let err_unenrolled = engine.calibrate_keyword_conformal("un_enrolled", &dummy_audio, &[]);
    assert!(err_unenrolled.is_err());
    assert!(err_unenrolled.err().unwrap().contains("not enrolled"));

    // Mutable access to predictor
    if let Some(pred_mut) = engine.conformal_predictor_mut() {
        assert_eq!(pred_mut.calibrated_count(), 0);
    } else {
        panic!("conformal_predictor_mut must return Some");
    }

    // Disable conformal guarantees
    engine.disable_conformal_guarantees();
    assert!(engine.conformal_predictor().is_none());
}

// ---------------------------------------------------------------------------
// Test 9: Serde Serialization and Deserialization Roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_conformal_serialization_roundtrip() {
    // 1. ConformalConfig roundtrip
    let cfg = ConformalConfig {
        risk_level: 0.02,
        min_calibration_samples: 16,
        foil_penalty_weight: 1.5,
    };
    let json_cfg = serde_json::to_string(&cfg).expect("Config serialization must succeed");
    let deser_cfg: ConformalConfig =
        serde_json::from_str(&json_cfg).expect("Config deserialization must succeed");
    assert_eq!(deser_cfg.risk_level, cfg.risk_level);
    assert_eq!(deser_cfg.min_calibration_samples, cfg.min_calibration_samples);
    assert_eq!(deser_cfg.foil_penalty_weight, cfg.foil_penalty_weight);

    // 2. ConformalCalibrationReport roundtrip
    let report = ConformalCalibrationReport {
        keyword: "falcon_prime".to_string(),
        conformal_threshold: 2.345,
        positive_samples: 42,
        foil_samples: 12,
        coverage_guarantee: 0.98,
        quantile_index: 41,
    };
    let json_rep = serde_json::to_string(&report).expect("Report serialization must succeed");
    let deser_rep: ConformalCalibrationReport =
        serde_json::from_str(&json_rep).expect("Report deserialization must succeed");
    assert_eq!(report, deser_rep);

    // 3. ConformalPredictionSet roundtrip
    let pred_set = ConformalPredictionSet {
        candidates: vec!["alpha".to_string(), "bravo".to_string()],
        p_values: vec![0.85, 0.42],
        best_distance: 1.23,
    };
    let json_set = serde_json::to_string(&pred_set).expect("Set serialization must succeed");
    let deser_set: ConformalPredictionSet =
        serde_json::from_str(&json_set).expect("Set deserialization must succeed");
    assert_eq!(pred_set, deser_set);

    // 4. KeywordEvent with conformal fields roundtrip
    let mut event = KeywordEvent::new("takeoff", 0.95, 12.34);
    event.conformal_p_value = Some(0.78);
    event.is_conformal_verified = true;
    let json_ev = serde_json::to_string(&event).expect("Event serialization must succeed");
    let deser_ev: KeywordEvent =
        serde_json::from_str(&json_ev).expect("Event deserialization must succeed");
    assert_eq!(event, deser_ev);
}
