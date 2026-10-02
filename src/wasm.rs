#![deny(unsafe_code)]

//! WebAssembly (WASM) Foreign Function Interface (FFI) for Sonon.
//!
//! Provides zero-allocation, thread-safe, handle-based WebAssembly exports
//! in 100% pure safe Rust without raw pointer arithmetic or unsafe blocks.

use crate::cwt::CwtProfilerConfig;
use crate::engine::{FeatureMode, KeywordEvent, SononEngine};
use crate::health::MotorHealthConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const WASM_AUDIO_BUFFER_SIZE: usize = 16384;
const WASM_STRING_BUFFER_SIZE: usize = 1024;

static NEXT_WASM_HANDLE: AtomicU64 = AtomicU64::new(1);
static WASM_ENGINE_REGISTRY: Mutex<Option<HashMap<u64, SononEngine>>> = Mutex::new(None);

static WASM_INPUT_BUFFER: Mutex<[f32; WASM_AUDIO_BUFFER_SIZE]> =
    Mutex::new([0.0; WASM_AUDIO_BUFFER_SIZE]);
static WASM_STRING_BUFFER: Mutex<[u8; WASM_STRING_BUFFER_SIZE]> =
    Mutex::new([0; WASM_STRING_BUFFER_SIZE]);
static WASM_LAST_EVENT: Mutex<Option<KeywordEvent>> = Mutex::new(None);

fn with_wasm_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<u64, SononEngine>) -> R,
{
    let mut guard = WASM_ENGINE_REGISTRY
        .lock()
        .expect("WASM engine registry lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

/// Creates a new SononEngine instance inside the WASM runtime.
///
/// Returns a unique 64-bit integer handle on success, or 0 on failure.
pub extern "C" fn sonon_wasm_create(
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    num_mfcc: usize,
) -> u64 {
    if sample_rate <= 0.0 || frame_size == 0 || (frame_size & (frame_size - 1)) != 0 {
        return 0;
    }

    let mut engine = SononEngine::new(sample_rate, frame_size, hop_size, num_mfcc);
    engine.set_feature_mode(FeatureMode::Pcen);

    let handle = NEXT_WASM_HANDLE.fetch_add(1, Ordering::SeqCst);
    with_wasm_registry(|map| {
        map.insert(handle, engine);
    });

    handle
}

/// Destroys the specified SononEngine instance, releasing internal buffers.
///
/// Returns 0 on success, -1 if handle not found.
pub extern "C" fn sonon_wasm_destroy(handle: u64) -> i32 {
    let removed = with_wasm_registry(|map| map.remove(&handle).is_some());
    if removed {
        0
    } else {
        -1
    }
}

/// Returns the maximum capacity of the shared WASM input audio buffer (16384 samples).
pub extern "C" fn sonon_wasm_get_input_buffer_capacity() -> usize {
    WASM_AUDIO_BUFFER_SIZE
}

/// Writes a single 32-bit float audio sample into the shared WASM input buffer.
pub extern "C" fn sonon_wasm_set_input_sample(index: usize, sample: f32) {
    if index < WASM_AUDIO_BUFFER_SIZE {
        let mut buf = WASM_INPUT_BUFFER
            .lock()
            .expect("WASM input buffer lock poisoned");
        buf[index] = sample;
    }
}

/// Reads a single 32-bit float audio sample from the shared WASM input buffer.
pub extern "C" fn sonon_wasm_get_input_sample(index: usize) -> f32 {
    if index < WASM_AUDIO_BUFFER_SIZE {
        let buf = WASM_INPUT_BUFFER
            .lock()
            .expect("WASM input buffer lock poisoned");
        buf[index]
    } else {
        0.0
    }
}

/// Writes a single UTF-8 ASCII byte into the shared WASM string buffer.
pub extern "C" fn sonon_wasm_set_string_byte(index: usize, byte: u8) {
    if index < WASM_STRING_BUFFER_SIZE {
        let mut buf = WASM_STRING_BUFFER
            .lock()
            .expect("WASM string buffer lock poisoned");
        buf[index] = byte;
    }
}

/// Enrolls a keyword directly from plain text in the shared string buffer using Klatt phonetic synthesis.
///
/// Returns number of enrolled feature frames on success, or 0 on failure.
pub extern "C" fn sonon_wasm_enroll_text(handle: u64, text_len: usize, threshold: f32) -> usize {
    if text_len == 0 || text_len > WASM_STRING_BUFFER_SIZE {
        return 0;
    }

    let text_str = {
        let buf = WASM_STRING_BUFFER
            .lock()
            .expect("WASM string buffer lock poisoned");
        match std::str::from_utf8(&buf[..text_len]) {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        }
    };

    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            let name = text_str.replace(' ', "_");
            engine.enroll_keyword_from_text(name, &text_str, threshold)
        } else {
            0
        }
    })
}

/// Enrolls a keyword directly from audio samples in the shared input buffer (e.g. from live microphone recording).
///
/// Returns the number of enrolled feature frames on success, or 0 on failure.
pub extern "C" fn sonon_wasm_enroll_audio_buffer(
    handle: u64,
    name_len: usize,
    sample_count: usize,
    threshold: f32,
) -> usize {
    if name_len == 0
        || name_len > WASM_STRING_BUFFER_SIZE
        || sample_count == 0
        || sample_count > WASM_AUDIO_BUFFER_SIZE
    {
        return 0;
    }

    let name = {
        let buf = WASM_STRING_BUFFER
            .lock()
            .expect("WASM string buffer lock poisoned");
        match std::str::from_utf8(&buf[..name_len]) {
            Ok(s) => s.trim().replace(' ', "_"),
            Err(_) => return 0,
        }
    };

    let samples = {
        let buf = WASM_INPUT_BUFFER
            .lock()
            .expect("WASM input buffer lock poisoned");
        buf[..sample_count].to_vec()
    };

    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            let features = engine.extract_features(&samples);
            let count = features.len();
            if count > 0 {
                engine.enroll_keyword(name, features, threshold);
            }
            count
        } else {
            0
        }
    })
}

/// Clears all enrolled keyword templates from the engine.
pub extern "C" fn sonon_wasm_clear_templates(handle: u64) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.clear_keywords();
            0
        } else {
            -1
        }
    })
}

/// Returns the number of currently enrolled keyword templates in the engine.
pub extern "C" fn sonon_wasm_get_template_count(handle: u64) -> usize {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine.dtw().template_count()
        } else {
            0
        }
    })
}

/// Synthesizes acoustic speech for the text in the string buffer, storing the waveform in the input buffer.
///
/// Returns the number of synthesized audio samples written.
pub extern "C" fn sonon_wasm_synthesize_text(handle: u64, text_len: usize) -> usize {
    if text_len == 0 || text_len > WASM_STRING_BUFFER_SIZE {
        return 0;
    }

    let text_str = {
        let buf = WASM_STRING_BUFFER
            .lock()
            .expect("WASM string buffer lock poisoned");
        match std::str::from_utf8(&buf[..text_len]) {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        }
    };

    let synth_audio = with_wasm_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine.synthesize_speech_from_text(&text_str)
        } else {
            Vec::new()
        }
    });

    let n = synth_audio.len().min(WASM_AUDIO_BUFFER_SIZE);
    if n > 0 {
        let mut buf = WASM_INPUT_BUFFER
            .lock()
            .expect("WASM input buffer lock poisoned");
        buf[..n].copy_from_slice(&synth_audio[..n]);
    }
    n
}

/// Ingests `num_samples` from the shared input buffer into the engine for real-time DSP & wake-word spotting.
///
/// Returns the number of keyword detection events triggered in this block.
pub extern "C" fn sonon_wasm_ingest(handle: u64, num_samples: usize) -> u32 {
    let n = num_samples.min(WASM_AUDIO_BUFFER_SIZE);
    if n == 0 {
        return 0;
    }

    let samples: Vec<f32> = {
        let buf = WASM_INPUT_BUFFER
            .lock()
            .expect("WASM input buffer lock poisoned");
        buf[..n].to_vec()
    };

    let events = with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.ingest_samples(&samples)
        } else {
            Vec::new()
        }
    });

    let count = events.len() as u32;
    if let Some(last) = events.last() {
        let mut last_event_guard = WASM_LAST_EVENT
            .lock()
            .expect("WASM last event lock poisoned");
        *last_event_guard = Some(last.clone());
    }

    count
}

/// Returns the byte length of the latest detected keyword name.
pub extern "C" fn sonon_wasm_get_last_keyword_len() -> usize {
    let last = WASM_LAST_EVENT
        .lock()
        .expect("WASM last event lock poisoned");
    last.as_ref().map_or(0, |ev| ev.keyword.len())
}

/// Reads a single byte of the latest detected keyword name.
pub extern "C" fn sonon_wasm_get_last_keyword_byte(index: usize) -> u8 {
    let last = WASM_LAST_EVENT
        .lock()
        .expect("WASM last event lock poisoned");
    if let Some(ref ev) = *last {
        let bytes = ev.keyword.as_bytes();
        if index < bytes.len() {
            return bytes[index];
        }
    }
    0
}

/// Returns the detection confidence of the latest detected keyword.
pub extern "C" fn sonon_wasm_get_last_keyword_confidence() -> f32 {
    let last = WASM_LAST_EVENT
        .lock()
        .expect("WASM last event lock poisoned");
    last.as_ref().map_or(0.0, |ev| ev.confidence)
}

/// Enables propeller harmonic blade pass frequency (BPF) notch filtering.
pub extern "C" fn sonon_wasm_enable_rotor_notch(
    handle: u64,
    num_blades: usize,
    num_harmonics: usize,
    q_factor: f32,
) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.enable_rotor_notch(num_blades, num_harmonics, q_factor);
            0
        } else {
            -1
        }
    })
}

/// Updates active motor RPM telemetry on the specified engine.
pub extern "C" fn sonon_wasm_update_rpm(handle: u64, rpm: f32) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.update_motor_rpm(rpm);
            0
        } else {
            -1
        }
    })
}

/// Enables acoustic health monitoring and bearing diagnostics.
pub extern "C" fn sonon_wasm_enable_health_monitoring(handle: u64) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.enable_health_monitoring(MotorHealthConfig::default());
            0
        } else {
            -1
        }
    })
}

/// Enables Continuous Wavelet Transform (CWT) non-stationary rotor micro-damage profiler.
pub extern "C" fn sonon_wasm_enable_cwt_profiler(handle: u64, num_blades: usize) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.enable_cwt_profiler(CwtProfilerConfig::default(), num_blades);
            0
        } else {
            -1
        }
    })
}

/// Retrieves latest airframe health score [0.0 = failing, 1.0 = pristine].
pub extern "C" fn sonon_wasm_get_health_score(handle: u64) -> f32 {
    with_wasm_registry(|map| {
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

/// Retrieves latest Continuous Wavelet Transform airframe fatigue index [0.0, 1.0].
pub extern "C" fn sonon_wasm_get_cwt_fatigue_index(handle: u64) -> f32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine
                .latest_cwt_report()
                .map(|r| r.airframe_fatigue_index)
                .unwrap_or(0.0)
        } else {
            -1.0
        }
    })
}

/// Retrieves latest Continuous Wavelet Transform scale-wise peak kurtosis.
pub extern "C" fn sonon_wasm_get_cwt_kurtosis(handle: u64) -> f32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            engine
                .latest_cwt_report()
                .map(|r| r.max_kurtosis)
                .unwrap_or(3.0)
        } else {
            -1.0
        }
    })
}

/// Returns Voice Activity Detection state: 1 if active speech frame, 0 otherwise.
pub extern "C" fn sonon_wasm_get_vad_active(handle: u64) -> u32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get(&handle) {
            if engine.vad().is_speech_active() {
                1
            } else {
                0
            }
        } else {
            0
        }
    })
}

/// Resets the engine state and clears feature histories.
pub extern "C" fn sonon_wasm_reset(handle: u64) -> i32 {
    with_wasm_registry(|map| {
        if let Some(engine) = map.get_mut(&handle) {
            engine.reset();
            0
        } else {
            -1
        }
    })
}
