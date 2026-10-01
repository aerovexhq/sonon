//! Phase 7 Verification Test Suite: C-ABI FFI Layer, Shared Memory Audio Ingestion & C/C++/Python Bindings.

#![deny(unsafe_code)]

use sonon::capi::*;
use sonon::shm::{ShmAudioChannel, ShmHeader, SHM_MAGIC, SHM_VERSION};
use std::time::Instant;

#[test]
fn test_shm_channel_creation_and_header_serialization() {
    let test_path = "/tmp/test_sonon_shm_header.bin";
    let _ = std::fs::remove_file(test_path);

    let channel = ShmAudioChannel::create_or_open(test_path, 16000.0, 2048)
        .expect("Failed to create SHM channel");

    assert_eq!(channel.sample_rate(), 16000.0);
    assert_eq!(channel.capacity(), 2048);

    let header = ShmHeader::new(16000.0, 2048);
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 64);

    let decoded = ShmHeader::from_bytes(&bytes).expect("Failed to decode header");
    assert_eq!(decoded.magic, SHM_MAGIC);
    assert_eq!(decoded.version, SHM_VERSION);
    assert_eq!(decoded.sample_rate, 16000.0);
    assert_eq!(decoded.capacity, 2048);
    assert_eq!(decoded.health_score, 1.0);

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_shm_audio_ring_buffer_write_read_roundtrip() {
    let test_path = "/tmp/test_sonon_shm_roundtrip.bin";
    let _ = std::fs::remove_file(test_path);

    let mut channel = ShmAudioChannel::create_or_open(test_path, 16000.0, 1024)
        .expect("Failed to create SHM channel");

    // 1. Contiguous write and read (500 samples)
    let samples: Vec<f32> = (0..500).map(|i| (i as f32) * 0.001).collect();
    let written = channel.write_samples(&samples).expect("Write failed");
    assert_eq!(written, 500);

    let mut read_buf = Vec::new();
    let read_count = channel
        .read_available_samples(&mut read_buf)
        .expect("Read failed");
    assert_eq!(read_count, 500);
    assert_eq!(read_buf.len(), 500);
    for i in 0..500 {
        assert!((read_buf[i] - samples[i]).abs() < 1e-6);
    }

    // 2. Wrapped write across ring buffer boundary (800 samples into 1024 cap)
    let wrap_samples: Vec<f32> = (0..800).map(|i| -(i as f32) * 0.002).collect();
    let written_wrap = channel
        .write_samples(&wrap_samples)
        .expect("Wrapped write failed");
    assert_eq!(written_wrap, 800);

    let mut read_wrap_buf = Vec::new();
    let read_wrap_count = channel
        .read_available_samples(&mut read_wrap_buf)
        .expect("Wrapped read failed");
    assert_eq!(read_wrap_count, 800);
    assert_eq!(read_wrap_buf.len(), 800);
    for i in 0..800 {
        assert!((read_wrap_buf[i] - wrap_samples[i]).abs() < 1e-6);
    }

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_shm_health_status_and_keyword_event_ipc() {
    let test_path = "/tmp/test_sonon_shm_ipc.bin";
    let _ = std::fs::remove_file(test_path);

    let mut channel = ShmAudioChannel::create_or_open(test_path, 16000.0, 1024)
        .expect("Failed to create SHM channel");

    // Health status
    channel
        .write_health_status(0.825, 2)
        .expect("Failed to write health");
    let (health, sev) = channel.read_health_status().expect("Failed to read health");
    assert!((health - 0.825).abs() < 1e-5);
    assert_eq!(sev, 2);

    // Detection event
    channel
        .write_detection_event("plank", 0.945)
        .expect("Failed to write event");
    let event = channel
        .read_detection_event()
        .expect("Failed to read event");
    assert!(event.is_some());
    let (kw, conf) = event.unwrap();
    assert_eq!(kw, "plank");
    assert!((conf - 0.945).abs() < 1e-5);

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_capi_engine_lifecycle_and_rpm_telemetry() {
    let engine_h = sonon_engine_create(16000.0, 512, 160, 13);
    assert_ne!(engine_h, 0, "sonon_engine_create must return valid handle");

    assert_eq!(sonon_engine_update_motor_rpm(engine_h, 6200.0), 0);
    assert_eq!(sonon_engine_enable_rotor_notch(engine_h, 2, 3, 25.0), 0);
    assert_eq!(sonon_engine_enable_health_monitoring(engine_h), 0);

    let health = sonon_engine_get_health_score(engine_h);
    assert!(
        health >= 0.0 && health <= 1.0,
        "Health score must be normalized"
    );

    let sev = sonon_engine_get_worst_severity(engine_h);
    assert_eq!(sev, 0, "Initial severity must be Normal");

    assert_eq!(
        sonon_engine_destroy(engine_h),
        0,
        "sonon_engine_destroy must succeed"
    );
    assert_eq!(
        sonon_engine_destroy(engine_h),
        -1,
        "Repeated destroy must return -1"
    );
}

#[test]
fn test_capi_shm_audio_pump_integration() {
    let _ = std::fs::remove_file(sonon::DEFAULT_SHM_PATH);
    let engine_h = sonon_engine_create(16000.0, 512, 160, 13);
    assert_ne!(engine_h, 0);

    let shm_h = sonon_shm_create(16000.0, 4096);
    assert_ne!(shm_h, 0);

    // Write 0.1s of test tone (1600 samples)
    assert_eq!(sonon_shm_write_test_tone(shm_h, 800.0, 0.1), 0);

    // Pump samples into engine
    let pumped = sonon_shm_pump(engine_h, shm_h);
    assert_eq!(pumped, 1600, "Should pump exactly 1600 samples");

    // Clean up
    assert_eq!(sonon_engine_destroy(engine_h), 0);
    assert_eq!(sonon_shm_destroy(shm_h), 0);
}

#[test]
fn test_shm_throughput_benchmark() {
    let test_path = "/tmp/test_sonon_shm_perf.bin";
    let _ = std::fs::remove_file(test_path);

    let mut channel = ShmAudioChannel::create_or_open(test_path, 16000.0, 16384)
        .expect("Failed to create SHM channel");

    let total_samples = 160_000; // 10 seconds of audio
    let chunk_size = 512;
    let chunk = vec![0.05f32; chunk_size];

    let mut read_buf = Vec::with_capacity(chunk_size);

    let start = Instant::now();
    for _ in 0..(total_samples / chunk_size) {
        let _ = channel.write_samples(&chunk);
        let _ = channel.read_available_samples(&mut read_buf);
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / 16000.0;

    println!(
        "POSIX SHM Ring Buffer Throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    let min_target = if cfg!(debug_assertions) {
        2_000_000.0
    } else {
        5_000_000.0
    };

    assert!(
        samples_per_sec > min_target,
        "SHM throughput must exceed {min_target:.0} samples/sec, got {samples_per_sec:.0}"
    );

    let _ = std::fs::remove_file(test_path);
}
