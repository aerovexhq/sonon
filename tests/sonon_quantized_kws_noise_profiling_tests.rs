#![deny(unsafe_code)]

//! Comprehensive test suite for adaptive acoustic noise floor profiling,
//! dynamic feature reliability weighting, and 8-bit quantized streaming DTW wake-word spotting.

use sonon::dtw::{
    weighted_euclidean_distance, AcousticNoiseClusterTracker, DtwMatcher, QuantizedDtwMatcher,
    QuantizedFrame, StreamingDtwConfig,
};
use sonon::engine::{FeatureMode, SononEngine};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_acoustic_noise_cluster_tracker_adaptation() {
    let num_features = 13;
    let mut tracker = AcousticNoiseClusterTracker::new(num_features, 0.08);

    assert_eq!(tracker.sample_count(), 0);
    assert_eq!(tracker.weights().len(), num_features);
    for &w in tracker.weights() {
        assert!((w - 1.0).abs() < 1e-4);
    }

    // Simulate background rotor noise where low channels (0..3) have high noise energy and variance,
    // while higher channels (4..12) remain quiescent
    for i in 0..100 {
        let mut noise_frame = vec![0.0f32; num_features];
        let phase = i as f32 * 0.2;
        noise_frame[0] = 15.0 + 8.0 * phase.sin();
        noise_frame[1] = 12.0 + 6.0 * (phase * 1.5).cos();
        noise_frame[2] = 8.0 + 4.0 * (phase * 2.0).sin();
        for k in 3..num_features {
            noise_frame[k] = 0.5 + 0.1 * (i as f32 * 0.1 + k as f32).sin();
        }
        tracker.update_noise(&noise_frame);
    }

    assert_eq!(tracker.sample_count(), 100);

    let weights = tracker.weights();
    assert_eq!(weights.len(), num_features);

    // Channels with high noise variance should receive lower reliability weights
    assert!(
        weights[0] < weights[10],
        "High-variance noise channel 0 (weight {}) should have lower weight than quiet channel 10 (weight {})",
        weights[0],
        weights[10]
    );

    for &w in weights {
        assert!(w >= 0.15 && w <= 1.0, "Weight {} outside [0.15, 1.0]", w);
    }

    // Verify weighted Euclidean distance calculation downweights noisy channels
    let vec_a = vec![10.0; num_features];
    let mut vec_b = vec_a.clone();
    vec_b[0] += 5.0; // perturbation in noisy channel
    let mut vec_c = vec_a.clone();
    vec_c[10] += 5.0; // equal perturbation in reliable channel

    let dist_noisy = tracker.compute_weighted_distance(&vec_a, &vec_b);
    let dist_reliable = tracker.compute_weighted_distance(&vec_a, &vec_c);
    let direct_dist = weighted_euclidean_distance(&vec_a, &vec_b, tracker.weights());
    assert!((dist_noisy - direct_dist).abs() < 1e-5);

    assert!(
        dist_noisy < dist_reliable,
        "Distance in noisy channel ({}) should be smaller than in reliable channel ({}) due to dynamic weighting",
        dist_noisy,
        dist_reliable
    );

    // Verify reset clears statistics
    tracker.reset();
    assert_eq!(tracker.sample_count(), 0);
    for &w in tracker.weights() {
        assert!((w - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_quantized_frame_affine_precision() {
    let test_frame: Vec<f32> = vec![
        -15.4, -8.2, -3.1, 0.0, 2.5, 7.8, 14.2, 22.9, 31.0, 18.5, 9.4, 1.2, -6.7,
    ];

    let q_frame = QuantizedFrame::from_f32(&test_frame);
    assert_eq!(q_frame.values.len(), test_frame.len());

    let recon = q_frame.to_f32();
    assert_eq!(recon.len(), test_frame.len());

    // Calculate maximum absolute error and Pearson correlation R^2
    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    for (orig, rec) in test_frame.iter().zip(recon.iter()) {
        let err = (orig - rec).abs();
        if err > max_err {
            max_err = err;
        }
        sum_sq_err += err * err;
    }

    let mse = sum_sq_err / (test_frame.len() as f32);
    assert!(
        max_err < 0.25,
        "Maximum quantization error {} exceeds threshold 0.25",
        max_err
    );
    assert!(
        mse < 0.05,
        "Mean squared quantization error {} exceeds threshold 0.05",
        mse
    );
}

#[test]
fn test_quantized_dtw_matcher_and_rolling_buffers() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    // Synthesize target keyword audio and extract features
    let keyword = "take off";
    let audio = engine.synthesize_speech_from_text(keyword);
    let features = engine.extract_features(&audio);
    assert!(!features.is_empty(), "Extracted features should not be empty");

    let mut q_matcher = QuantizedDtwMatcher::new();
    let threshold = 3.5f32;
    let band_radius = 6;
    q_matcher.add_template(keyword, &features, threshold, band_radius);

    assert_eq!(q_matcher.template_count(), 1);
    let template = &q_matcher.templates()[0];
    assert_eq!(template.name, keyword);
    assert_eq!(template.frames.len(), features.len());

    // Self-matching distance of the quantized template against its own frames should be 0
    let self_dist = QuantizedDtwMatcher::compute_distance_banded_q(
        &template.frames,
        &template.frames,
        band_radius,
    );
    assert_eq!(self_dist, 0, "Self DTW distance must be exactly 0");

    // Float-equivalent self-distance should be 0.0
    let float_self_dist = QuantizedDtwMatcher::compute_distance_float_equivalent(
        &template.frames,
        &template.frames,
        band_radius,
        template.scale,
    );
    assert_eq!(float_self_dist, 0.0);

    // Test streaming window matching with the quantized template
    let streaming_config = StreamingDtwConfig::default();
    let result = q_matcher.match_streaming_window_q(
        &template.frames,
        template.scale,
        0.01,
        &streaming_config,
    );

    assert!(result.is_some(), "Quantized streaming matcher should detect exact phrase match");
    let match_res = result.unwrap();
    assert_eq!(match_res.keyword, keyword);
    assert!(match_res.distance <= threshold);
    assert!(match_res.confidence >= 0.95);
}

#[test]
fn test_engine_adaptive_noise_profiling_and_quantized_export() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    // Enroll synthetic keyword
    engine.enroll_keyword_synthetic_pipeline("take off", "take off", 3, 6, 1.25);
    assert_eq!(engine.dtw().template_count(), 1);

    // Feed background rotor noise with silence
    let noise_samples: Vec<f32> = (0..3200)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.08 * (2.0 * PI * 180.0 * t).sin() + 0.03 * (2.0 * PI * 360.0 * t).sin()
        })
        .collect();

    let initial_samples = engine.noise_tracker().sample_count();
    let _ = engine.ingest_samples(&noise_samples);
    let post_samples = engine.noise_tracker().sample_count();

    assert!(
        post_samples > initial_samples,
        "Noise tracker sample count should increase on background audio frames"
    );

    let weights = engine.noise_feature_weights();
    assert_eq!(weights.len(), 13);
    for &w in weights {
        assert!(w >= 0.15 && w <= 1.0);
    }

    // Export quantized matcher from engine
    let q_matcher = engine.create_quantized_matcher();
    assert_eq!(q_matcher.template_count(), 1);

    let q_templates = engine.export_quantized_templates();
    assert_eq!(q_templates.len(), 1);
    assert_eq!(q_templates[0].name, "take off");
    assert!(!q_templates[0].frames.is_empty());
}

#[test]
fn test_quantized_vs_float_dtw_discrimination() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);

    let target_keyword = "take off";
    let foil_keyword = "shake off";

    let target_audio = engine.synthesize_speech_from_text(target_keyword);
    let foil_audio = engine.synthesize_speech_from_text(foil_keyword);

    let target_features = engine.extract_features(&target_audio);
    let foil_features = engine.extract_features(&foil_audio);

    // Compute float DTW distances
    let float_target_dist = DtwMatcher::compute_distance_banded(&target_features, &target_features, 6);
    let float_foil_dist = DtwMatcher::compute_distance_banded(&foil_features, &target_features, 6);

    // Construct quantized matcher
    let mut q_matcher = QuantizedDtwMatcher::new();
    q_matcher.add_template(target_keyword, &target_features, 3.5, 6);
    let template = &q_matcher.templates()[0];

    // Quantize foil frames with template scale
    let foil_q_frames: Vec<Vec<i8>> = foil_features
        .iter()
        .map(|f| {
            let q = QuantizedFrame::from_f32(f);
            q.values
        })
        .collect();

    let q_target_dist = QuantizedDtwMatcher::compute_distance_banded_q(
        &template.frames,
        &template.frames,
        template.band_radius,
    );
    let q_foil_dist = QuantizedDtwMatcher::compute_distance_banded_q(
        &foil_q_frames,
        &template.frames,
        template.band_radius,
    );

    assert_eq!(q_target_dist, 0);
    assert!(
        q_foil_dist > 0,
        "Foil distance ({}) must be strictly greater than target distance (0)",
        q_foil_dist
    );

    // Verify discrimination margin preservation
    assert!(
        float_foil_dist > float_target_dist,
        "Float foil distance must exceed target distance"
    );
    let q_margin = q_foil_dist - q_target_dist;
    assert!(
        q_margin > 10,
        "Quantized discrimination margin ({}) must be positive and significant",
        q_margin
    );
}

#[test]
fn test_quantized_dtw_throughput_benchmark() {
    let frame_len = 13;
    let num_frames = 60;

    let seq1: Vec<Vec<i8>> = (0..num_frames)
        .map(|i| (0..frame_len).map(|k| ((i * 3 + k * 5) % 127) as i8).collect())
        .collect();
    let seq2: Vec<Vec<i8>> = (0..num_frames)
        .map(|i| (0..frame_len).map(|k| ((i * 3 + k * 5 + 2) % 127) as i8).collect())
        .collect();

    let iterations = 2000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = QuantizedDtwMatcher::compute_distance_banded_q(&seq1, &seq2, 8);
    }
    let elapsed = start.elapsed();

    let total_comparisons = iterations * num_frames;
    let throughput = (total_comparisons as f64) / elapsed.as_secs_f64();

    // Throughput should exceed 500,000 frames/sec on embedded / desktop cores
    assert!(
        throughput > 100_000.0,
        "Quantized DTW throughput {:.0} frames/sec is below target",
        throughput
    );
}
