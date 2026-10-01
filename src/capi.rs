//! C-ABI Foreign Function Interface (FFI) for Sonon acoustic engine.
//!
//! Provides standard C-callable exports using safe handle tables and shared memory transport
//! in 100% pure safe Rust.

#![deny(unsafe_code)]

use crate::engine::SononEngine;
use crate::health::MotorHealthConfig;
use crate::shm::{ShmAudioChannel, DEFAULT_SHM_PATH};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_ENGINE_HANDLE: AtomicU64 = AtomicU64::new(1);
static NEXT_SHM_HANDLE: AtomicU64 = AtomicU64::new(1);

static ENGINE_REGISTRY: Mutex<Option<HashMap<u64, SononEngine>>> = Mutex::new(None);
static SHM_REGISTRY: Mutex<Option<HashMap<u64, ShmAudioChannel>>> = Mutex::new(None);

fn with_engine_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<u64, SononEngine>) -> R,
{
    let mut guard = ENGINE_REGISTRY.lock().expect("Engine registry lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

fn with_shm_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<u64, ShmAudioChannel>) -> R,
{
    let mut guard = SHM_REGISTRY.lock().expect("SHM registry lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

/// Create a new SononEngine instance.
///
/// Returns non-zero integer handle on success, or 0 on failure.
pub extern "C" fn sonon_engine_create(
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    num_mfcc: usize,
) -> u64 {
    if sample_rate <= 0.0 || frame_size == 0 || (frame_size & (frame_size - 1)) != 0 {
        return 0;
    }

    let engine = SononEngine::new(sample_rate, frame_size, hop_size, num_mfcc);
    let handle = NEXT_ENGINE_HANDLE.fetch_add(1, Ordering::SeqCst);

    with_engine_registry(|map| {
        map.insert(handle, engine);
    });

    handle
}

/// Destroy a SononEngine instance and release all internal resources.
///
/// Returns 0 on success, -1 if handle not found.
pub extern "C" fn sonon_engine_destroy(handle: u64) -> i32 {
    let removed = with_engine_registry(|map| map.remove(&handle).is_some());
    if removed {
        0
    } else {
        -1
    }
}

/// Update live motor RPM telemetry on the specified engine.
///
/// Returns 0 on success, -1 if handle not found.
pub extern "C" fn sonon_engine_update_motor_rpm(handle: u64, rpm: f32) -> i32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.update_motor_rpm(rpm);
            0
        } else {
            -1
        }
    })
}

/// Enable rotor notch filtering tracking motor RPM harmonics.
///
/// Returns 0 on success, -1 if handle not found.
pub extern "C" fn sonon_engine_enable_rotor_notch(
    handle: u64,
    num_blades: usize,
    num_harmonics: usize,
    q_factor: f32,
) -> i32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.enable_rotor_notch(num_blades, num_harmonics, q_factor);
            0
        } else {
            -1
        }
    })
}

/// Enable acoustic health monitoring on the engine.
///
/// Returns 0 on success, -1 if handle not found.
pub extern "C" fn sonon_engine_enable_health_monitoring(handle: u64) -> i32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.enable_health_monitoring(MotorHealthConfig::default());
            0
        } else {
            -1
        }
    })
}

/// Retrieve the latest evaluated airframe health score (1.0 = pristine, 0.0 = failure).
///
/// Returns -1.0 if handle not found or no evaluation yet available.
pub extern "C" fn sonon_engine_get_health_score(handle: u64) -> f32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine
                .latest_health_snapshot()
                .map(|s| s.overall_health_score)
                .unwrap_or(1.0)
        } else {
            -1.0
        }
    })
}

/// Retrieve worst anomaly severity code (0 = Normal, 1 = Advisory, 2 = Warning, 3 = Critical).
///
/// Returns 255 if handle not found.
pub extern "C" fn sonon_engine_get_worst_severity(handle: u64) -> u32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine
                .latest_health_snapshot()
                .map(|s| s.worst_severity as u32)
                .unwrap_or(0)
        } else {
            255
        }
    })
}

/// Enrolls a synthetic tone keyword template into the DTW engine.
///
/// Returns 0 on success, -1 on failure.
pub extern "C" fn sonon_engine_enroll_tone_keyword(
    handle: u64,
    tone_freq_hz: f32,
    duration_sec: f32,
    threshold: f32,
) -> i32 {
    with_engine_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            let sample_rate = 16000.0;
            let n = (duration_sec * sample_rate) as usize;
            let audio: Vec<f32> = (0..n)
                .map(|i| (2.0 * PI * tone_freq_hz * (i as f32) / sample_rate).sin())
                .collect();
            let feat = engine.extract_features(&audio);
            engine.enroll_keyword("tone", feat, threshold);
            0
        } else {
            -1
        }
    })
}

/// Creates a shared memory audio channel for zero-copy IPC.
///
/// Returns non-zero handle on success, 0 on failure.
pub extern "C" fn sonon_shm_create(sample_rate: f32, capacity: usize) -> u64 {
    let channel = match ShmAudioChannel::create_or_open(DEFAULT_SHM_PATH, sample_rate, capacity) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    let handle = NEXT_SHM_HANDLE.fetch_add(1, Ordering::SeqCst);
    with_shm_registry(|map| {
        map.insert(handle, channel);
    });

    handle
}

/// Destroys a shared memory audio channel handle.
///
/// Returns 0 on success, -1 on error.
pub extern "C" fn sonon_shm_destroy(shm_handle: u64) -> i32 {
    let removed = with_shm_registry(|map| map.remove(&shm_handle).is_some());
    if removed {
        0
    } else {
        -1
    }
}

/// Reads all unread samples from the shared memory channel and ingests them into the engine.
///
/// Returns number of ingested samples, or -1 on error.
pub extern "C" fn sonon_shm_pump(engine_handle: u64, shm_handle: u64) -> i32 {
    let mut sample_buf = Vec::new();

    // 1. Read samples from SHM
    let read_result = with_shm_registry(|shm_map| {
        if let Some(shm) = shm_map.get_mut(&shm_handle) {
            shm.read_available_samples(&mut sample_buf)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "SHM handle not found",
            ))
        }
    });

    let count = match read_result {
        Ok(c) => c,
        Err(_) => return -1,
    };

    if count == 0 {
        return 0;
    }

    // 2. Ingest samples into engine
    let (detected_events, health_score, worst_sev) = with_engine_registry(|eng_map| {
        if let Some(engine) = eng_map.get_mut(&engine_handle) {
            let events = engine.ingest_samples(&sample_buf);
            let health = engine
                .latest_health_snapshot()
                .map(|s| (s.overall_health_score, s.worst_severity as u32))
                .unwrap_or((1.0, 0));
            (events, health.0, health.1)
        } else {
            (Vec::new(), 1.0, 0)
        }
    });

    // 3. Write back health and detection status to SHM header
    with_shm_registry(|shm_map| {
        if let Some(shm) = shm_map.get_mut(&shm_handle) {
            let _ = shm.write_health_status(health_score, worst_sev);
            if let Some(first) = detected_events.first() {
                let _ = shm.write_detection_event(&first.keyword, first.confidence);
            }
        }
    });

    count as i32
}

/// Test helper: writes a pure tone slice directly to the shared memory channel.
///
/// Returns 0 on success, -1 on failure.
pub extern "C" fn sonon_shm_write_test_tone(
    shm_handle: u64,
    freq_hz: f32,
    duration_sec: f32,
) -> i32 {
    let sample_rate = 16000.0;
    let n = (duration_sec * sample_rate) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.5 * (2.0 * PI * freq_hz * (i as f32) / sample_rate).sin())
        .collect();

    with_shm_registry(|map| {
        if let Some(shm) = map.get_mut(&shm_handle) {
            match shm.write_samples(&samples) {
                Ok(_) => 0,
                Err(_) => -1,
            }
        } else {
            -1
        }
    })
}
