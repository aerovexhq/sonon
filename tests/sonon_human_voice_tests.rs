#![deny(unsafe_code)]

//! Real Human Voice Integration & Empirical Validation Test Suite.
//!
//! Evaluates few-shot DTW keyword spotting, Dynamic Barycenter Averaging (DBA),
//! and 70% anticipatory prefix early-exit on user-provided acoustic voice recordings.
//!
//! Note on User Privacy:
//! In accordance with strict user privacy policies, raw human voice datasets are never
//! stored in the public repository. Provide a local directory via the `SONON_VOICE_DATASET_DIR`
//! environment variable to execute these tests against real-world voice samples.

use sonon::{calibrate_threshold, dtw_barycenter_averaging, FeatureMode, SononEngine};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Helper function to locate optional user-provided voice dataset.
fn get_voice_fixtures_dir() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("SONON_VOICE_DATASET_DIR") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let candidates = [
        "/root/voice_dataset",
        "/tmp/sonon_voice",
        "tests/fixtures",
    ];
    for cand in &candidates {
        let p = PathBuf::from(cand);
        if p.exists() && p.join("plank_exemplars").exists() {
            return Some(p);
        }
    }
    None
}

/// Helper function to load a canonical 16-bit 16 kHz Mono PCM WAV file in safe Rust.
fn load_wav_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open {path:?}: {e}"))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| format!("Failed to read {path:?}: {e}"))?;

    if buffer.len() < 44 {
        return Err(format!("File {path:?} is too small to be a WAV file"));
    }

    if &buffer[0..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        return Err(format!("Invalid WAV header in {path:?}"));
    }

    // Locate data chunk
    let mut offset = 12;
    while offset + 8 <= buffer.len() {
        let chunk_id = &buffer[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            buffer[offset + 4],
            buffer[offset + 5],
            buffer[offset + 6],
            buffer[offset + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let data_start = offset + 8;
            let data_end = (data_start + chunk_size).min(buffer.len());
            let raw_samples = &buffer[data_start..data_end];

            let num_samples = raw_samples.len() / 2;
            let mut samples = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let sample_bytes = [raw_samples[2 * i], raw_samples[2 * i + 1]];
                let sample_i16 = i16::from_le_bytes(sample_bytes);
                samples.push(sample_i16 as f32 / 32768.0);
            }
            return Ok(samples);
        }

        offset += 8 + chunk_size;
    }

    Err(format!("No data chunk found in WAV {path:?}"))
}

#[test]
fn test_load_and_process_human_voice_exemplars() {
    let Some(fixtures_dir) = get_voice_fixtures_dir() else {
        println!("Notice: Skipping human voice test (provide SONON_VOICE_DATASET_DIR to run).");
        return;
    };

    let engine = SononEngine::new(16000.0, 512, 160, 26);

    for idx in 1..=15 {
        let path = fixtures_dir
            .join("plank_exemplars")
            .join(format!("plank_{idx:02}.wav"));
        if !path.exists() {
            continue;
        }
        let samples = load_wav_16k_mono(&path).expect("Failed to load exemplar WAV");
        assert!(!samples.is_empty(), "Exemplar {idx} audio must not be empty");

        let features = engine.extract_features(&samples);
        assert!(
            !features.is_empty(),
            "Exemplar {idx} must yield valid feature frames"
        );
        for frame in &features {
            assert_eq!(frame.len(), 26, "Each frame must contain 26 Mel features");
            for &val in frame {
                assert!(!val.is_nan(), "Features must not contain NaN");
                assert!(!val.is_infinite(), "Features must not contain Inf");
            }
        }
    }
}

#[test]
fn test_human_voice_dtw_barycenter_averaging() {
    let Some(fixtures_dir) = get_voice_fixtures_dir() else {
        println!("Notice: Skipping human voice test (provide SONON_VOICE_DATASET_DIR to run).");
        return;
    };

    let engine = SononEngine::new(16000.0, 512, 160, 26);

    // Load first 4 natural human voice exemplars
    let mut exemplar_features = Vec::new();
    for idx in 1..=4 {
        let path = fixtures_dir
            .join("plank_exemplars")
            .join(format!("plank_{idx:02}.wav"));
        if !path.exists() {
            continue;
        }
        let samples = load_wav_16k_mono(&path).expect("Failed to load exemplar WAV");
        let feat = engine.extract_features(&samples);
        exemplar_features.push(feat);
    }

    if exemplar_features.len() < 2 {
        return;
    }

    // Run DBA centroid synthesis
    let dba_centroid = dtw_barycenter_averaging(&exemplar_features, 6, 12);
    assert!(
        !dba_centroid.is_empty(),
        "DBA centroid must contain synthesized frames"
    );

    // Verify DBA centroid is closer on average to peer exemplars
    let mut avg_dist_to_centroid = 0.0;
    for ex in &exemplar_features {
        avg_dist_to_centroid +=
            sonon::dtw::DtwMatcher::compute_distance_banded(ex, &dba_centroid, 12);
    }
    avg_dist_to_centroid /= exemplar_features.len() as f32;

    assert!(
        avg_dist_to_centroid < 3.5,
        "DBA centroid distance must be bounded, got {avg_dist_to_centroid}"
    );

    let threshold = calibrate_threshold(&exemplar_features, 12, 1.3);
    assert!(threshold > 0.0, "Threshold must be positive");
}

#[test]
fn test_human_voice_medoid_enrollment_and_recognition() {
    let Some(fixtures_dir) = get_voice_fixtures_dir() else {
        println!("Notice: Skipping human voice test (provide SONON_VOICE_DATASET_DIR to run).");
        return;
    };

    let mut engine = SononEngine::new(16000.0, 512, 160, 26);
    engine.set_feature_mode(FeatureMode::LogMel);

    // Load Medoid exemplar (#02) as reference
    let ref_path = fixtures_dir
        .join("plank_exemplars")
        .join("plank_02.wav");
    if !ref_path.exists() {
        return;
    }

    let ref_samples = load_wav_16k_mono(&ref_path).expect("Failed to load reference WAV");
    let ref_features = engine.extract_features(&ref_samples);

    engine.enroll_keyword("plank", ref_features.clone(), 3.5);

    // Test recognition on peer human voice exemplars #01, #03, #05
    for idx in [1, 3, 5] {
        let path = fixtures_dir
            .join("plank_exemplars")
            .join(format!("plank_{idx:02}.wav"));
        if !path.exists() {
            continue;
        }
        let test_samples = load_wav_16k_mono(&path).expect("Failed to load test WAV");
        let test_features = engine.extract_features(&test_samples);

        let dist =
            sonon::dtw::DtwMatcher::compute_distance_banded(&test_features, &ref_features, 12);
        assert!(
            dist < 3.5,
            "Distance to peer human exemplar {idx} must be low, got {dist}"
        );
    }
}

#[test]
fn test_human_voice_70_percent_early_detection() {
    let Some(fixtures_dir) = get_voice_fixtures_dir() else {
        println!("Notice: Skipping human voice test (provide SONON_VOICE_DATASET_DIR to run).");
        return;
    };

    let engine = SononEngine::new(16000.0, 512, 160, 26);

    // Load human voice exemplar #04 (360ms)
    let p4 = fixtures_dir
        .join("plank_exemplars")
        .join("plank_04.wav");
    let p2 = fixtures_dir
        .join("plank_exemplars")
        .join("plank_02.wav");

    if !p4.exists() || !p2.exists() {
        return;
    }

    let samples = load_wav_16k_mono(&p4).expect("Failed to load exemplar WAV");
    let full_features = engine.extract_features(&samples);
    let total_frames = full_features.len();

    // 70% Prefix
    let prefix_len = (total_frames as f32 * 0.70).round() as usize;
    let prefix_features = &full_features[0..prefix_len];

    // Reference template 70% prefix
    let ref_samples = load_wav_16k_mono(&p2).expect("Failed to load reference WAV");
    let ref_features = engine.extract_features(&ref_samples);
    let ref_prefix_len = (ref_features.len() as f32 * 0.70).round() as usize;
    let ref_prefix = &ref_features[0..ref_prefix_len];

    let early_distance =
        sonon::dtw::DtwMatcher::compute_distance_banded(prefix_features, ref_prefix, 10);
    assert!(
        early_distance < 3.5,
        "70% partial human phrase early distance must be bounded, got {early_distance}"
    );
}

#[test]
fn test_human_voice_streaming_throughput() {
    let Some(fixtures_dir) = get_voice_fixtures_dir() else {
        println!("Notice: Skipping human voice test (provide SONON_VOICE_DATASET_DIR to run).");
        return;
    };

    let mut engine = SononEngine::new(16000.0, 512, 160, 26);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Enroll reference
    let ref_path = fixtures_dir
        .join("plank_exemplars")
        .join("plank_02.wav");
    let stream_path = fixtures_dir.join("random_recording_16k.wav");

    if !ref_path.exists() || !stream_path.exists() {
        return;
    }

    let ref_samples = load_wav_16k_mono(&ref_path).expect("Failed to load reference WAV");
    let ref_features = engine.extract_features(&ref_samples);
    engine.enroll_keyword("plank", ref_features, 3.0);

    // Load full 40-second continuous human speech stream
    let continuous_samples =
        load_wav_16k_mono(&stream_path).expect("Failed to load continuous stream WAV");
    let total_samples = continuous_samples.len();

    let start_instant = std::time::Instant::now();
    let events = engine.ingest_samples(&continuous_samples);
    let elapsed = start_instant.elapsed();

    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / 16000.0;

    println!(
        "Continuous human speech throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time), detected events: {}",
        events.len()
    );

    assert!(
        samples_per_sec > 500_000.0,
        "Streaming throughput must exceed 500,000 samples/sec, got {samples_per_sec:.0}"
    );
}
