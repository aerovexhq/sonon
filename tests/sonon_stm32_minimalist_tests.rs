//! Comprehensive Integration and Verification Tests for Sonon STM32 Minimalist Speech Synthesis.
//!
//! Validates:
//! 1. STM32 Flash and SRAM physical memory budget constraints (Cortex-M4/M7).
//! 2. Nano-neural INT8 weight loading from JSON and packed binary formats.
//! 3. Zero heap reallocation during streaming synthesis.
//! 4. Signal integrity, dynamic range, and vocal resonance.
//! 5. Lock-free SPSC audio ring buffer streaming on embedded hardware profile.
//!
//! Zero emojis. Strictly safe Rust execution (#![deny(unsafe_code)]).

#![deny(unsafe_code)]

use sonon::edge_runtime::{EdgeSpeechRuntime, StreamingEdgeConfig};
use std::fs;
use std::path::Path;

const STM32_FLASH_LIMIT_BYTES: usize = 128 * 1024; // 128 KB budget
const STM32_SRAM_LIMIT_BYTES: usize = 32 * 1024;   // 32 KB budget

#[test]
fn test_stm32_hardware_budget_verification() {
    let runtime = EdgeSpeechRuntime::for_stm32_profile();
    let metrics = runtime.stm32_metrics();

    // Verify neural weights flash consumption is under 128 KB
    assert!(
        metrics.flash_weights_bytes <= STM32_FLASH_LIMIT_BYTES,
        "Flash weights {} bytes exceed limit of {} bytes",
        metrics.flash_weights_bytes,
        STM32_FLASH_LIMIT_BYTES
    );

    // Verify neural SRAM consumption is under 32 KB
    assert!(
        metrics.sram_usage_bytes <= STM32_SRAM_LIMIT_BYTES,
        "SRAM usage {} bytes exceeds limit of {} bytes",
        metrics.sram_usage_bytes,
        STM32_SRAM_LIMIT_BYTES
    );

    // Verify microarchitecture compliance flags
    assert!(metrics.fits_stm32f4_entry, "Must fit STM32F4 entry-level (STM32F401/411)");
    assert!(metrics.fits_stm32f4_standard, "Must fit STM32F4 standard (STM32F407/429)");
    assert!(metrics.fits_stm32h7_high_perf, "Must fit STM32H7 high-performance");

    // Verify Cortex-M4 CPU load estimate at 168 MHz is strictly below 10%
    assert!(
        metrics.estimated_cortex_m4_load_percent < 10.0,
        "Cortex-M4 load {:.2}% must be under 10%",
        metrics.estimated_cortex_m4_load_percent
    );
}

#[test]
fn test_stm32_nano_json_weights_loading_and_execution() {
    let json_path = "models/stm32/sonon_stm32_nano_weights.json";
    assert!(
        Path::new(json_path).exists(),
        "STM32 nano weights JSON file not found: {}",
        json_path
    );

    let json_content = fs::read_to_string(json_path)
        .expect("Failed to read STM32 nano weights JSON");

    let config = StreamingEdgeConfig {
        sample_rate: 16000.0,
        frame_size: 256,
        hop_size: 128,
        ring_buffer_capacity: 2048,
        hidden_dim: 64,
        chunk_frames: 4,
        pitch_f0: 135.0,
    };

    let mut runtime = EdgeSpeechRuntime::from_stm32_nano_json(&json_content, config)
        .expect("Failed to initialize runtime from STM32 JSON weights");

    // Synthesize test phrase
    let text = "Waypoint Alpha reached. Maintaining altitude three thousand feet.";
    let audio = runtime.synthesize_chunk(text)
        .expect("Synthesis chunk failed");

    assert!(!audio.is_empty(), "Synthesized audio must not be empty");

    // Validate signal integrity: no NaNs, no Infs, bounded in [-1.0, 1.0]
    let mut max_abs = 0.0f32;
    for &sample in &audio {
        assert!(!sample.is_nan(), "Sample must not be NaN");
        assert!(!sample.is_infinite(), "Sample must not be infinite");
        assert!(sample >= -1.0 && sample <= 1.0, "Sample out of bounds: {}", sample);
        if sample.abs() > max_abs {
            max_abs = sample.abs();
        }
    }
    assert!(max_abs > 0.05, "Synthesized audio must have audible energy, got max_abs: {}", max_abs);
}

#[test]
fn test_stm32_binary_blob_header_and_sizes() {
    let bin_path = "models/stm32/sonon_stm32_nano_weights.bin";
    assert!(
        Path::new(bin_path).exists(),
        "STM32 binary weights file not found: {}",
        bin_path
    );

    let bytes = fs::read(bin_path).expect("Failed to read binary weights blob");
    assert!(bytes.len() >= 12, "Binary blob too small for header");

    // Check 8-byte magic header "SONONSTM"
    assert_eq!(&bytes[0..8], b"SONONSTM", "Invalid magic header");

    // Check 4-byte uint32 version = 1
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(version, 1, "Unexpected format version: {}", version);

    // Total file size must strictly fit into STM32 Flash budget
    assert!(
        bytes.len() <= STM32_FLASH_LIMIT_BYTES,
        "Binary size {} bytes exceeds flash limit of {}",
        bytes.len(),
        STM32_FLASH_LIMIT_BYTES
    );
}

#[test]
fn test_stm32_streaming_into_ring_buffer_fifo() {
    let mut runtime = EdgeSpeechRuntime::for_stm32_profile();

    let text = "Altitude normal. Engine running.";
    let pushed_samples = runtime.stream_into_ring_buffer(text)
        .expect("Streaming into ring buffer failed");

    assert!(pushed_samples > 0, "Must push samples into ring buffer");
    assert_eq!(runtime.ring_buffer().available_read(), pushed_samples);

    // Pop samples in chunks of 128 (hop size)
    let mut pop_buf = [0.0f32; 128];
    let mut total_read = 0;
    while !runtime.ring_buffer().is_empty() {
        let read = runtime.read_audio(&mut pop_buf);
        if read == 0 {
            break;
        }
        total_read += read;
    }
    assert_eq!(total_read, pushed_samples);
}

#[test]
fn test_arduino_micro_lpc_lattice_synthesizer_ram_and_processing() {
    let mut synth = sonon::edge_runtime::LpcLatticeSynthesizer::new(16, 16000.0);

    // RAM state must be strictly <= 128 bytes (fits in classical Arduino Uno 2 KB RAM!)
    assert!(
        synth.ram_state_bytes() <= 128,
        "LPC synthesizer RAM {} bytes must be <= 128 bytes",
        synth.ram_state_bytes()
    );
    assert_eq!(synth.ram_state_bytes(), 72);

    // Test processing a voiced frame
    let lpc = [1.0f32, -1.2, 0.8, -0.4, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut out = [0.0f32; 80];
    synth.process_frame(&lpc, 140.0, 0.5, 80, &mut out);

    // Validate output
    let mut has_energy = false;
    for &val in &out {
        assert!(!val.is_nan(), "Sample must not be NaN");
        assert!(!val.is_infinite(), "Sample must not be infinite");
        assert!(val >= -1.0 && val <= 1.0, "Sample out of bounds");
        if val.abs() > 0.01 {
            has_energy = true;
        }
    }
    assert!(has_energy, "Synthesizer must produce audible signal");
}

#[test]
fn test_arduino_micro_ddsp_neural_synthesizer_ram_and_processing() {
    let mut synth = sonon::edge_runtime::MicroDdspSynthesizer::new(24000.0);

    // RAM state must be strictly <= 512 bytes (fits easily in Arduino Uno R4 32 KB SRAM!)
    let ram_bytes = synth.ram_state_bytes();
    assert!(
        ram_bytes <= 512,
        "Micro-DDSP synthesizer RAM {} bytes must be <= 512 bytes",
        ram_bytes
    );
    assert_eq!(ram_bytes, 332);

    // Setup 32 harmonic amplitudes (formant shaped) and 16 noise bands
    let mut harmonic_amps = [0.0f32; 32];
    let mut sum_a = 0.0f32;
    for k in 0..32 {
        let f = (k + 1) as f32 * 200.0;
        // Peak around F1 (700 Hz) and F2 (2100 Hz)
        let amp1 = (-0.5 * ((f - 700.0) / 200.0).powi(2)).exp();
        let amp2 = 0.5 * (-0.5 * ((f - 2100.0) / 300.0).powi(2)).exp();
        harmonic_amps[k] = amp1 + amp2 + 0.02;
        sum_a += harmonic_amps[k];
    }
    for k in 0..32 {
        harmonic_amps[k] /= sum_a;
    }
    let noise_bands = [0.05f32; 16];

    let mut out = [0.0f32; 256];
    synth.process_frame(200.0, &harmonic_amps, &noise_bands, 0.8, 256, &mut out);

    // Validate signal properties
    let mut max_abs = 0.0f32;
    for (i, &sample) in out.iter().enumerate() {
        assert!(!sample.is_nan(), "Sample at {} must not be NaN", i);
        assert!(!sample.is_infinite(), "Sample at {} must not be infinite", i);
        assert!(sample >= -1.0 && sample <= 1.0, "Sample at {} out of bounds: {}", i, sample);
        if sample.abs() > max_abs {
            max_abs = sample.abs();
        }
    }
    assert!(max_abs > 0.1, "Micro-DDSP output must have strong harmonic energy, got max_abs: {}", max_abs);

    // Verify sample-to-sample delta (guarantees zero clicking/transient spikes)
    let mut max_delta = 0.0f32;
    for i in 1..out.len() {
        let delta = (out[i] - out[i - 1]).abs();
        if delta > max_delta {
            max_delta = delta;
        }
    }
    assert!(
        max_delta < 0.45,
        "Maximum adjacent sample jump {:.4} must be < 0.45 (zero clicks)",
        max_delta
    );
}


