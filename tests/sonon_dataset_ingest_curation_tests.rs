//! Comprehensive Unit and Integration Test Suite for
//! Scalable Dataset Ingestion, Audio Quality Inspection, and CTC Forced Alignment.
//!
//! Verifies:
//! 1. Clean speech acoustic quality metrics (WADA-SNR, clipping, crest factor, spectral flatness).
//! 2. DC offset removal and peak normalization restoration.
//! 3. Digital clipping and noisy speech quality gating rejection.
//! 4. Continuous Viterbi CTC trellis forced alignment and phoneme timestamp monotonicity.
//! 5. Sharded WebDataset binary archive serialization and streaming roundtrip.
//! 6. End-to-end `SononEngine::ingest_curated_sample` pipeline.

#![deny(unsafe_code)]

use sonon::phonetic::Phoneme;
use sonon::dataset_ingest::{
    AudioQualityConfig, AudioSignalInspector, CtcForcedAligner, DatasetSample, DatasetShardReader,
    DatasetShardWriter,
};
use sonon::engine::SononEngine;
use sonon::phonetic::{G2pEngine, KlattSynthesizer};
use std::f32::consts::PI;
use std::io::Cursor;

/// Deterministic Pseudo-Random Number Generator for synthetic test noise.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Synthesize a clean human-like speech phrase using G2P and Klatt cascade synthesis.
fn synthesize_speech(phrase: &str, f0: f32, sample_rate: f32) -> Vec<f32> {
    let segments = G2pEngine::text_to_phonemes(phrase);
    let mut synth = KlattSynthesizer::new(sample_rate);
    synth.set_f0(f0);
    synth.synthesize(&segments)
}

#[test]
fn test_audio_signal_inspector_clean_speech_quality() {
    let sample_rate = 16000.0f32;
    let audio = synthesize_speech("falcon", 125.0, sample_rate);
    assert!(!audio.is_empty(), "Synthesized audio must not be empty");

    let inspector = AudioSignalInspector::new(sample_rate, AudioQualityConfig::default());
    let report = inspector.inspect(&audio);

    println!("Clean speech quality report: {:?}", report);

    assert!(report.is_acceptable, "Clean speech must pass quality inspection");
    assert!(report.snr_db >= 15.0, "Clean speech SNR must be >= 15 dB (got {})", report.snr_db);
    assert!(report.clipping_ratio < 0.001, "Clean speech clipping ratio must be near 0 (got {})", report.clipping_ratio);
    assert!(report.crest_factor >= 2.0, "Crest factor must be >= 2.0 (got {})", report.crest_factor);
    assert!(report.dc_offset < 0.02, "DC offset must be minimal (got {})", report.dc_offset);
    assert!(report.spectral_flatness < 0.40, "Speech spectral flatness must be < 0.40 (got {})", report.spectral_flatness);
}

#[test]
fn test_audio_signal_inspector_clipping_and_dc_offset_rejection() {
    let sample_rate = 16000.0f32;
    let base_audio = synthesize_speech("abort", 120.0, sample_rate);

    // Corrupt with high DC bias (+0.12) and severe clipping (clamp to +-0.6 and scale up)
    let mut corrupted = Vec::with_capacity(base_audio.len());
    for s in &base_audio {
        let clipped = (s * 3.5 + 0.12).clamp(-0.999, 0.999);
        corrupted.push(clipped);
    }

    let inspector = AudioSignalInspector::new(sample_rate, AudioQualityConfig::default());
    let bad_report = inspector.inspect(&corrupted);

    println!("Corrupted audio report: {:?}", bad_report);

    assert!(!bad_report.is_acceptable, "Corrupted audio must be rejected by quality gating");
    assert!(bad_report.dc_offset > 0.03, "DC offset must be detected");
    assert!(bad_report.clipping_ratio > 0.01, "Clipping ratio must be flagged");

    // Apply digital signal restoration filterbank
    let cleaned = inspector.clean_audio(&corrupted);
    let cleaned_report = inspector.inspect(&cleaned);

    println!("Restored audio report: {:?}", cleaned_report);
    assert!(cleaned_report.dc_offset < 0.02, "DC removal filter must eliminate DC bias");
    let peak = cleaned.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(peak <= 0.90, "Cleaned audio must be peak-normalized");
}

#[test]
fn test_wada_snr_degradation_under_gaussian_noise() {
    let sample_rate = 16000.0f32;
    let clean = synthesize_speech("land", 130.0, sample_rate);
    let inspector = AudioSignalInspector::new(sample_rate, AudioQualityConfig::default());

    let mut rng = DeterministicRng::new(42);

    // Clean speech
    let clean_snr = inspector.inspect(&clean).snr_db;

    // Moderate noise (sigma = 0.05)
    let mut mod_noisy = clean.clone();
    for s in &mut mod_noisy {
        *s += rng.next_gaussian() * 0.05;
    }
    let mod_snr = inspector.inspect(&mod_noisy).snr_db;

    // Severe noise (sigma = 0.25)
    let mut severe_noisy = clean.clone();
    for s in &mut severe_noisy {
        *s += rng.next_gaussian() * 0.25;
    }
    let severe_snr = inspector.inspect(&severe_noisy).snr_db;

    println!("WADA-SNR: clean={:.1} dB, moderate={:.1} dB, severe={:.1} dB", clean_snr, mod_snr, severe_snr);

    assert!(clean_snr > mod_snr, "Clean SNR must be higher than moderate noise SNR");
    assert!(mod_snr > severe_snr, "Moderate noise SNR must be higher than severe noise SNR");
    assert!(severe_snr < 10.0, "Severe noise must result in low estimated SNR");
}

#[test]
fn test_ctc_forced_aligner_word_and_phoneme_timestamps() {
    let sample_rate = 16000.0f32;
    let audio = synthesize_speech("falcon", 120.0, sample_rate);
    let aligner = CtcForcedAligner::new(sample_rate);

    let report = aligner
        .align(&audio, "falcon")
        .expect("Forced alignment of 'falcon' must succeed");

    println!("Alignment report: {:?}", report);

    assert_eq!(report.transcript, "falcon");
    assert_eq!(report.words.len(), 1);
    assert_eq!(report.words[0].word, "falcon");
    assert!(report.is_valid_alignment, "Alignment must be valid");
    assert!(report.total_duration_sec > 0.2, "Duration must be positive");

    let word = &report.words[0];
    assert!(word.start_time_sec < word.end_time_sec, "Word start must precede end");
    assert!(!word.phonemes.is_empty(), "Phonemes must be extracted");

    // Verify phoneme sequence matches ARPAbet for "falcon": [F, AE, L, K, AH, N]
    let expected_phonemes = [Phoneme::F, Phoneme::AE, Phoneme::L, Phoneme::K, Phoneme::AH, Phoneme::N];
    assert_eq!(word.phonemes.len(), expected_phonemes.len());

    for (idx, &exp_p) in expected_phonemes.iter().enumerate() {
        assert_eq!(word.phonemes[idx].phoneme, exp_p);
        assert!(word.phonemes[idx].start_time_sec <= word.phonemes[idx].end_time_sec);
        if idx > 0 {
            assert!(
                word.phonemes[idx].start_time_sec >= word.phonemes[idx - 1].start_time_sec,
                "Phoneme start times must be monotonically non-decreasing"
            );
        }
    }
}

#[test]
fn test_ctc_forced_aligner_multi_word_phrase() {
    let sample_rate = 16000.0f32;
    let audio = synthesize_speech("abort land", 125.0, sample_rate);
    let aligner = CtcForcedAligner::new(sample_rate);

    let report = aligner
        .align(&audio, "abort land")
        .expect("Multi-word alignment must succeed");

    assert_eq!(report.words.len(), 2);
    assert_eq!(report.words[0].word, "abort");
    assert_eq!(report.words[1].word, "land");

    assert!(report.words[0].end_time_sec <= report.words[1].start_time_sec + 0.1);
    assert!(report.is_valid_alignment);
}

#[test]
fn test_sharded_dataset_writer_and_reader_roundtrip() {
    let sample_rate = 16000.0f32;
    let audio1 = synthesize_speech("takeoff", 120.0, sample_rate);
    let audio2 = synthesize_speech("hover", 130.0, sample_rate);

    let sample1 = DatasetSample {
        sample_id: "drone_cmd_001".to_string(),
        audio: audio1,
        sample_rate,
        transcript: "takeoff".to_string(),
        speaker_id: Some("operator_alpha".to_string()),
        quality_report: None,
    };

    let sample2 = DatasetSample {
        sample_id: "drone_cmd_002".to_string(),
        audio: audio2,
        sample_rate,
        transcript: "hover".to_string(),
        speaker_id: Some("operator_bravo".to_string()),
        quality_report: None,
    };

    // Serialize samples into in-memory shard archive
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = DatasetShardWriter::new(&mut buffer).expect("Writer init must succeed");
        writer.write_sample(&sample1).expect("Write sample 1 must succeed");
        writer.write_sample(&sample2).expect("Write sample 2 must succeed");
        writer.flush().expect("Flush must succeed");
        assert_eq!(writer.samples_written(), 2);
    }

    // Read back through streaming ShardReader
    buffer.set_position(0);
    let mut reader = DatasetShardReader::new(buffer).expect("Reader init must succeed");

    let read_sample1 = reader.read_next_sample().expect("Read 1 must succeed").expect("Must have sample 1");
    assert_eq!(read_sample1.sample_id, "drone_cmd_001");
    assert_eq!(read_sample1.transcript, "takeoff");
    assert_eq!(read_sample1.audio.len(), sample1.audio.len());
    assert_eq!(read_sample1.speaker_id.as_deref(), Some("operator_alpha"));

    let read_sample2 = reader.read_next_sample().expect("Read 2 must succeed").expect("Must have sample 2");
    assert_eq!(read_sample2.sample_id, "drone_cmd_002");
    assert_eq!(read_sample2.transcript, "hover");
    assert_eq!(read_sample2.audio.len(), sample2.audio.len());

    let eof = reader.read_next_sample().expect("Read 3 must succeed");
    assert!(eof.is_none(), "Must return None at EOF");
    assert_eq!(reader.samples_read(), 2);
}

#[test]
fn test_sonon_engine_curated_dataset_ingestion_pipeline() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.enable_audio_quality_gating(AudioQualityConfig::default());
    engine.enable_forced_aligner();

    assert!(engine.audio_inspector().is_some());
    assert!(engine.forced_aligner().is_some());

    // Ingest valid clean synthesized utterance
    let audio = synthesize_speech("falcon", 125.0, sample_rate);
    let valid_sample = DatasetSample {
        sample_id: "valid_001".to_string(),
        audio,
        sample_rate,
        transcript: "falcon".to_string(),
        speaker_id: None,
        quality_report: None,
    };

    let result = engine.ingest_curated_sample(&valid_sample);
    assert!(result.is_ok(), "Valid sample ingestion must succeed");
    let (quality, alignment) = result.unwrap();
    assert!(quality.is_acceptable);
    assert!(alignment.is_valid_alignment);
    assert_eq!(alignment.words[0].word, "falcon");

    // Ingest corrupted noise sample
    let bad_audio = vec![0.999f32; 8000]; // pure clipped rail
    let bad_sample = DatasetSample {
        sample_id: "bad_001".to_string(),
        audio: bad_audio,
        sample_rate,
        transcript: "falcon".to_string(),
        speaker_id: None,
        quality_report: None,
    };

    let bad_result = engine.ingest_curated_sample(&bad_sample);
    assert!(bad_result.is_err(), "Corrupted sample must be rejected by quality gating");
}
