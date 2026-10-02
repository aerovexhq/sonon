#![deny(unsafe_code)]

//! Phase 20 Test Suite: WebAssembly (WASM) Real-Time Browser AudioWorklet & Interactive Web Engine.
//!
//! Validates handle-based FFI lifecycle, zero-allocation buffers, Klatt speech synthesis,
//! zero-shot text enrollment, continuous streaming ingestion, and CWT telemetry via safe WASM exports.

use sonon::wasm::{
    sonon_wasm_clear_templates, sonon_wasm_create, sonon_wasm_destroy,
    sonon_wasm_enable_cwt_profiler, sonon_wasm_enable_health_monitoring,
    sonon_wasm_enable_rotor_notch, sonon_wasm_enroll_audio_buffer, sonon_wasm_enroll_text,
    sonon_wasm_get_cwt_fatigue_index, sonon_wasm_get_cwt_kurtosis, sonon_wasm_get_health_score,
    sonon_wasm_get_input_buffer_capacity, sonon_wasm_get_input_sample,
    sonon_wasm_get_last_keyword_byte, sonon_wasm_get_last_keyword_confidence,
    sonon_wasm_get_last_keyword_len, sonon_wasm_get_template_count, sonon_wasm_get_vad_active,
    sonon_wasm_ingest, sonon_wasm_reset, sonon_wasm_set_input_sample, sonon_wasm_set_string_byte,
    sonon_wasm_synthesize_text, sonon_wasm_update_rpm,
};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_wasm_engine_lifecycle() {
    let _lock = TEST_LOCK.lock().unwrap();
    // Valid engine creation: 16kHz, 512 frame, 160 hop, 13 MFCCs
    let handle = sonon_wasm_create(16000.0, 512, 160, 13);
    assert_ne!(handle, 0, "Expected non-zero handle for valid parameters");

    // Invalid parameters check
    assert_eq!(sonon_wasm_create(0.0, 512, 160, 13), 0);
    assert_eq!(sonon_wasm_create(-16000.0, 512, 160, 13), 0);
    assert_eq!(sonon_wasm_create(16000.0, 300, 160, 13), 0); // Non power of 2

    // Reset engine state
    assert_eq!(sonon_wasm_reset(handle), 0);

    // Destroy engine
    assert_eq!(sonon_wasm_destroy(handle), 0);
    // Double destroy should fail with -1
    assert_eq!(sonon_wasm_destroy(handle), -1);
}

#[test]
fn test_wasm_shared_buffers_operations() {
    let _lock = TEST_LOCK.lock().unwrap();
    let capacity = sonon_wasm_get_input_buffer_capacity();
    assert_eq!(capacity, 16384, "Input buffer capacity should be 16384");

    // Test sample set and get
    sonon_wasm_set_input_sample(0, 0.42);
    sonon_wasm_set_input_sample(100, -0.85);
    sonon_wasm_set_input_sample(16383, 0.99);

    assert!((sonon_wasm_get_input_sample(0) - 0.42).abs() < 1e-5);
    assert!((sonon_wasm_get_input_sample(100) - (-0.85)).abs() < 1e-5);
    assert!((sonon_wasm_get_input_sample(16383) - 0.99).abs() < 1e-5);

    // Out of bounds get should return 0.0 safely
    assert_eq!(sonon_wasm_get_input_sample(20000), 0.0);

    // Out of bounds set should be ignored safely without panic
    sonon_wasm_set_input_sample(50000, 1.0);
}

#[test]
fn test_wasm_zero_shot_enrollment_and_klatt_synthesis() {
    let _lock = TEST_LOCK.lock().unwrap();
    let handle = sonon_wasm_create(16000.0, 512, 160, 13);
    assert_ne!(handle, 0);

    let phrase = "take off";
    for (i, byte) in phrase.as_bytes().iter().enumerate() {
        sonon_wasm_set_string_byte(i, *byte);
    }

    // Enroll from text in WASM string buffer
    let enrolled_frames = sonon_wasm_enroll_text(handle, phrase.len(), 0.58);
    assert!(
        enrolled_frames > 0,
        "Zero-shot enrollment should yield frames"
    );

    // Synthesize audio into WASM input buffer
    let synth_samples = sonon_wasm_synthesize_text(handle, phrase.len());
    assert!(
        synth_samples > 0,
        "Klatt speech synthesis should produce samples"
    );

    // Verify waveform written into input buffer has active energy
    let mut energy = 0.0f32;
    for i in 0..synth_samples.min(1000) {
        let s = sonon_wasm_get_input_sample(i);
        energy += s * s;
    }
    assert!(
        energy > 0.01,
        "Synthesized waveform should have non-zero acoustic energy"
    );

    sonon_wasm_destroy(handle);
}

#[test]
fn test_wasm_end_to_end_streaming_and_spotting() {
    let _lock = TEST_LOCK.lock().unwrap();
    let handle = sonon_wasm_create(16000.0, 512, 160, 13);
    assert_ne!(handle, 0);

    let keyword = "land";
    for (i, byte) in keyword.as_bytes().iter().enumerate() {
        sonon_wasm_set_string_byte(i, *byte);
    }

    // Enroll keyword
    let frames = sonon_wasm_enroll_text(handle, keyword.len(), 0.55);
    assert!(frames > 0);

    // Synthesize waveform directly into WASM input buffer
    let synth_samples = sonon_wasm_synthesize_text(handle, keyword.len());
    assert!(synth_samples > 0);

    // Stream the synthesized audio in 256-sample chunks to simulate live browser audio
    let chunk_size = 256;
    let mut detected_events = 0;
    let mut offset = 0;

    while offset < synth_samples {
        let count = chunk_size.min(synth_samples - offset);
        // Copy chunk to front of buffer for ingestion
        for i in 0..count {
            let s = sonon_wasm_get_input_sample(offset + i);
            sonon_wasm_set_input_sample(i, s);
        }

        let events = sonon_wasm_ingest(handle, count);
        detected_events += events;
        offset += count;
    }

    // Also feed a short silence tail to flush DTW path
    for i in 0..512 {
        sonon_wasm_set_input_sample(i, 0.0);
    }
    detected_events += sonon_wasm_ingest(handle, 512);

    assert!(
        detected_events > 0,
        "Expected at least one wake-word detection event"
    );

    // Check last keyword event details
    let name_len = sonon_wasm_get_last_keyword_len();
    assert_eq!(name_len, keyword.len());

    let mut detected_name = String::new();
    for i in 0..name_len {
        detected_name.push(sonon_wasm_get_last_keyword_byte(i) as char);
    }
    assert_eq!(detected_name, keyword);

    let confidence = sonon_wasm_get_last_keyword_confidence();
    assert!(
        confidence >= 0.55,
        "Confidence should meet or exceed threshold: {}",
        confidence
    );

    sonon_wasm_destroy(handle);
}

#[test]
fn test_wasm_telemetry_and_diagnostics() {
    let _lock = TEST_LOCK.lock().unwrap();
    let handle = sonon_wasm_create(16000.0, 512, 160, 13);
    assert_ne!(handle, 0);

    // Health monitoring
    assert_eq!(sonon_wasm_enable_health_monitoring(handle), 0);
    assert_eq!(sonon_wasm_enable_cwt_profiler(handle, 2), 0);
    assert_eq!(sonon_wasm_enable_rotor_notch(handle, 2, 4, 15.0), 0);
    assert_eq!(sonon_wasm_update_rpm(handle, 4800.0), 0);

    // Check baseline telemetry values
    let health = sonon_wasm_get_health_score(handle);
    assert!((0.0..=1.0).contains(&health));

    let fatigue = sonon_wasm_get_cwt_fatigue_index(handle);
    assert!((0.0..=1.0).contains(&fatigue));

    let kurtosis = sonon_wasm_get_cwt_kurtosis(handle);
    assert!(kurtosis >= 0.0);

    // Ingest silence and verify VAD is inactive
    for i in 0..1024 {
        sonon_wasm_set_input_sample(i, 0.0);
    }
    sonon_wasm_ingest(handle, 1024);
    assert_eq!(sonon_wasm_get_vad_active(handle), 0);

    sonon_wasm_destroy(handle);
}

#[test]
fn test_wasm_audio_buffer_enrollment_and_template_management() {
    let _lock = TEST_LOCK.lock().unwrap();
    let handle = sonon_wasm_create(16000.0, 512, 160, 13);
    assert_ne!(handle, 0);

    assert_eq!(sonon_wasm_get_template_count(handle), 0);

    // Synthesize "falcon" audio directly into input buffer
    let keyword = "falcon";
    for (i, b) in keyword.as_bytes().iter().enumerate() {
        sonon_wasm_set_string_byte(i, *b);
    }
    let synth_samples = sonon_wasm_synthesize_text(handle, keyword.len());
    assert!(synth_samples > 0);

    // Now enroll from the audio in the buffer (mimicking microphone voice recording)
    let enrolled_frames =
        sonon_wasm_enroll_audio_buffer(handle, keyword.len(), synth_samples, 0.58);
    assert!(enrolled_frames > 0);
    assert_eq!(sonon_wasm_get_template_count(handle), 1);

    // Clear templates
    assert_eq!(sonon_wasm_clear_templates(handle), 0);
    assert_eq!(sonon_wasm_get_template_count(handle), 0);

    sonon_wasm_destroy(handle);
}
