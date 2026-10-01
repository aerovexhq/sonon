use sonon::{
    calibrate_threshold, dtw_barycenter_averaging, FeatureMode, PcenConfig, PcenFilter,
    SononEngine,
};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_pcen_adaptive_agc_and_dynamic_compression() {
    let num_channels = 26;
    let config = PcenConfig::default();
    let mut pcen = PcenFilter::new(num_channels, config);

    // Feed quiet baseline energy
    let quiet_frame = vec![0.1f32; num_channels];
    for _ in 0..20 {
        let _ = pcen.process_frame(&quiet_frame);
    }

    // Sudden loud transient spike (simulating consonant burst or rotor spool-up)
    let loud_frame = vec![10.0f32; num_channels];
    let first_loud_out = pcen.process_frame(&loud_frame);

    // Continued loud input: AGC should adaptively normalize it downward
    let mut adapted_loud_out = vec![0.0f32; num_channels];
    for _ in 0..100 {
        adapted_loud_out = pcen.process_frame(&loud_frame);
    }

    // Adaptive AGC check: initial transient contrast should be higher than adapted steady-state
    assert!(
        first_loud_out[0] > adapted_loud_out[0],
        "PCEN adaptive AGC must compress continuous steady-state noise over time"
    );
}

#[test]
fn test_sakoe_chiba_banded_dtw_corridor() {
    let sample_rate = 16000.0;
    let engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Generate 0.3s tone sequence
    let n = (0.3 * sample_rate) as usize;
    let signal1: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 600.0 * (i as f32) / sample_rate).sin())
        .collect();

    // Slightly time-stretched (0.33s) tone sequence
    let m = (0.33 * sample_rate) as usize;
    let signal2: Vec<f32> = (0..m)
        .map(|i| (2.0 * PI * 600.0 * (i as f32) / sample_rate).sin())
        .collect();

    let feat1 = engine.extract_features(&signal1);
    let feat2 = engine.extract_features(&signal2);

    let unconstrained_dist = sonon::dtw::DtwMatcher::compute_distance(&feat1, &feat2);
    let banded_dist = sonon::dtw::DtwMatcher::compute_distance_banded(&feat1, &feat2, 10);

    assert!(unconstrained_dist.is_finite(), "Unconstrained dist must be finite");
    assert!(banded_dist.is_finite(), "Banded dist must be finite");
    // Banded distance with generous radius should be very close to unconstrained distance
    assert!(
        (unconstrained_dist - banded_dist).abs() < 1e-3,
        "Banded DTW with radius 10 should closely approximate unconstrained for similar duration"
    );
}

#[test]
fn test_dtw_barycenter_averaging_convergence() {
    let sample_rate = 16000.0;
    let engine = SononEngine::new(sample_rate, 256, 128, 13);

    let make_audio = |freq: f32, duration: f32| -> Vec<f32> {
        let count = (duration * sample_rate) as usize;
        (0..count)
            .map(|i| (2.0 * PI * freq * (i as f32) / sample_rate).sin())
            .collect()
    };

    let a1 = engine.extract_features(&make_audio(700.0, 0.25));
    let a2 = engine.extract_features(&make_audio(710.0, 0.27));
    let a3 = engine.extract_features(&make_audio(690.0, 0.24));

    let exemplars = vec![a1.clone(), a2.clone(), a3.clone()];
    let centroid = dtw_barycenter_averaging(&exemplars, 5, 12);

    assert!(!centroid.is_empty(), "Centroid must be non-empty");

    // Centroid distance to all exemplars should be tightly bounded
    for ex in &exemplars {
        let d = sonon::dtw::DtwMatcher::compute_distance_banded(&centroid, ex, 12);
        assert!(d < 6.0, "Centroid distance to exemplar must be bounded, got {}", d);
    }

    let threshold = calibrate_threshold(&exemplars, 12, 1.3);
    assert!(threshold > 0.0, "Calibrated threshold must be positive");
}

#[test]
fn test_multi_exemplar_dba_enrollment_and_detection() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    let make_audio = |freq: f32, duration: f32| -> Vec<f32> {
        let count = (duration * sample_rate) as usize;
        (0..count)
            .map(|i| (2.0 * PI * freq * (i as f32) / sample_rate).sin())
            .collect()
    };

    let ex1 = make_audio(850.0, 0.25);
    let ex2 = make_audio(860.0, 0.27);
    let ex3 = make_audio(840.0, 0.24);

    let threshold = engine.enroll_keyword_multi("abort", &[&ex1, &ex2, &ex3], 12, 1.4);
    assert!(threshold > 0.0, "Threshold must be positive");

    // Ingest test audio matching the keyword
    let test_audio = make_audio(855.0, 0.26);
    let events = engine.ingest_samples(&test_audio);

    assert!(
        events.iter().any(|e| e.keyword == "abort"),
        "Multi-exemplar DBA enrolled keyword 'abort' must be recognized"
    );
}

#[test]
fn test_partial_keyword_early_detection_70_percent() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Full phrase: 0.4s multi-tone command ("take off")
    // Part 1 (70%): 0.28s tone at 400 Hz
    // Part 2 (30%): 0.12s tone at 800 Hz
    let total_samples = (0.4 * sample_rate) as usize;
    let part1_samples = (0.28 * sample_rate) as usize;

    let mut full_phrase = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = (i as f32) / sample_rate;
        if i < part1_samples {
            full_phrase.push((2.0 * PI * 400.0 * t).sin());
        } else {
            full_phrase.push((2.0 * PI * 800.0 * t).sin());
        }
    }

    // Enroll template with first 70% of the phrase (anticipatory / early trigger)
    let prefix_70 = &full_phrase[0..part1_samples];
    let prefix_features = engine.extract_features(prefix_70);

    engine.enroll_keyword("takeoff_early", prefix_features, 4.0);

    // Stream the audio frame by frame in 160-sample (10ms) chunks
    let mut early_detected = false;
    let mut detected_at_sample = 0;

    for (chunk_idx, chunk) in full_phrase.chunks(160).enumerate() {
        let events = engine.ingest_samples(chunk);
        if events.iter().any(|e| e.keyword == "takeoff_early") {
            early_detected = true;
            detected_at_sample = (chunk_idx + 1) * 160;
            break;
        }
    }

    assert!(early_detected, "70% prefix keyword must trigger early recognition");
    assert!(
        detected_at_sample <= part1_samples + 320,
        "Detection must fire within the first 70% of the phrase plus pipeline latency window"
    );
}

#[test]
fn test_throughput_phase2_benchmark() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Enroll a template
    let template_audio: Vec<f32> = (0..3200)
        .map(|i| (2.0 * PI * 500.0 * (i as f32) / sample_rate).sin())
        .collect();
    let feats = engine.extract_features(&template_audio);
    engine.enroll_keyword_banded("stream_word", feats, 3.5, 12);

    // Stream 160,000 samples (10 seconds of audio)
    let test_stream: Vec<f32> = (0..160_000)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate).sin())
        .collect();

    let start = Instant::now();
    for chunk in test_stream.chunks(320) {
        let _ = engine.ingest_samples(chunk);
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (160_000.0) / elapsed.as_secs_f64();
    let min_threshold = if cfg!(debug_assertions) { 500_000.0 } else { 1_000_000.0 };

    assert!(
        samples_per_sec > min_threshold,
        "Throughput was {:.0} samples/sec, below {} threshold",
        samples_per_sec,
        min_threshold
    );
}
