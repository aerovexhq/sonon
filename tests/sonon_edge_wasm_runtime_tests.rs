//! Comprehensive integration and unit tests for Subphase 5:
//! Ultra-Fast Deterministic On-Device Edge & WebAssembly Speech Synthesis Runtime.
//!
//! Validates:
//! - StaticTensorArena pre-allocated memory bounds, double buffering, and zero reallocation.
//! - QuantizedLinear symmetric INT8 quantization reconstruction error bound (< 2.0%).
//! - AudioSpscRingBuffer lock-free FIFO ordering, atomic memory orderings, and wraparound indexing.
//! - EdgeSpeechRuntime chunked streaming synthesis and sub-15ms throughput.
//! - SononEngine edge runtime integration and deterministic reproducibility.

#![deny(unsafe_code)]

use sonon::edge_runtime::{
    AudioSpscRingBuffer, EdgeSpeechRuntime, QuantizedLinear, StaticTensorArena,
    StreamingEdgeConfig,
};
use sonon::SononEngine;
use std::time::Instant;

#[test]
fn test_static_tensor_arena_memory_bounds_and_ping_pong() {
    let partition_size = 128usize;
    let mut arena = StaticTensorArena::new(partition_size);

    assert_eq!(arena.partition_size(), 128);
    assert_eq!(arena.total_capacity(), 256);
    assert!(!arena.is_ping_pong_swapped());

    // Record base pointer address to verify zero reallocations
    let ptr_before = arena.scratch_a().as_ptr();

    // Populate scratch_a and scratch_b with distinct patterns
    for (i, val) in arena.scratch_a().iter_mut().enumerate() {
        *val = (i + 1) as f32;
    }
    for (i, val) in arena.scratch_b().iter_mut().enumerate() {
        *val = (i + 201) as f32;
    }

    // Verify values via immutable references
    for i in 0..partition_size {
        assert_eq!(arena.scratch_a_ref()[i], (i + 1) as f32);
        assert_eq!(arena.scratch_b_ref()[i], (i + 201) as f32);
        assert_eq!(arena.buffer_a()[i], (i + 1) as f32);
        assert_eq!(arena.buffer_b()[i], (i + 201) as f32);
    }

    // Test simultaneous mutable partition access
    {
        let (buf_a, buf_b) = arena.scratch_pair_mut();
        buf_a[0] = 999.0;
        buf_b[0] = 888.0;
    }
    assert_eq!(arena.scratch_a_ref()[0], 999.0);
    assert_eq!(arena.scratch_b_ref()[0], 888.0);

    // Test ping-pong swapping logic
    assert_eq!(arena.current_input()[0], 999.0);
    arena.swap_ping_pong();
    assert!(arena.is_ping_pong_swapped());
    assert_eq!(arena.current_input()[0], 888.0);

    arena.swap_ping_pong();
    assert!(!arena.is_ping_pong_swapped());
    assert_eq!(arena.current_input()[0], 999.0);

    // Test reset zeroing
    arena.reset();
    assert!(!arena.is_ping_pong_swapped());
    for &val in arena.scratch_a_ref() {
        assert_eq!(val, 0.0);
    }
    for &val in arena.scratch_b_ref() {
        assert_eq!(val, 0.0);
    }

    // Verify pointer address remains identical (guarantee zero reallocation)
    let ptr_after = arena.scratch_a().as_ptr();
    assert_eq!(ptr_before, ptr_after, "Arena buffer must not reallocate on heap");
}

#[test]
fn test_quantized_linear_int8_quantization_error_bound_vs_fp32() {
    let in_features = 64usize;
    let out_features = 32usize;

    // Construct deterministic Xavier weight matrix
    let q_linear = QuantizedLinear::new_deterministic(in_features, out_features, 42);
    assert_eq!(q_linear.in_features, in_features);
    assert_eq!(q_linear.out_features, out_features);
    assert_eq!(q_linear.weight.weights.len(), out_features * in_features);
    assert_eq!(q_linear.weight.scales.len(), out_features);

    // Reconstruct FP32 reference weights from deterministic generation
    let n_elements = in_features * out_features;
    let scale = (2.0 / (in_features + out_features) as f32).sqrt();
    let mut fp32_weights = Vec::with_capacity(n_elements);
    for idx in 0..n_elements {
        let angle = (idx + 42 * 31337 + 1) as f32 * 0.1731;
        let val = angle.sin() * (angle * 1.6180339).cos();
        fp32_weights.push(val * scale);
    }

    // 1. Verify weight matrix quantization reconstruction error (< 2.0%)
    let mut weight_diff_sq = 0.0f32;
    let mut weight_ref_sq = 0.0f32;
    for j in 0..out_features {
        let s_w = q_linear.weight.scales[j];
        for i in 0..in_features {
            let orig = fp32_weights[j * in_features + i];
            let recon = (q_linear.weight.weights[j * in_features + i] as f32) * s_w;
            let diff = recon - orig;
            weight_diff_sq += diff * diff;
            weight_ref_sq += orig * orig;
        }
    }
    let weight_recon_error = (weight_diff_sq / weight_ref_sq).sqrt();
    assert!(
        weight_recon_error < 0.020,
        "Weight reconstruction error must be < 2.0%, obtained: {:.4}%",
        weight_recon_error * 100.0
    );

    // 2. Construct test activation input vector
    let mut input = Vec::with_capacity(in_features);
    for i in 0..in_features {
        let x = (i as f32 * 0.231 + 1.0).sin() * (i as f32 * 0.117).cos();
        input.push(x);
    }

    // Compute reference FP32 matrix-vector dot product
    let mut ref_output = vec![0.0f32; out_features];
    for j in 0..out_features {
        let row_start = j * in_features;
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += fp32_weights[row_start + i] * input[i];
        }
        ref_output[j] = sum;
    }

    // Compute INT8 quantized forward pass
    let mut quant_output = vec![0.0f32; out_features];
    q_linear
        .forward(&input, &mut quant_output)
        .expect("Quantized forward should succeed");

    // Calculate relative L2 reconstruction error (< 2.0%)
    let mut sum_diff_sq = 0.0f32;
    let mut sum_ref_sq = 0.0f32;
    for j in 0..out_features {
        let diff = quant_output[j] - ref_output[j];
        sum_diff_sq += diff * diff;
        sum_ref_sq += ref_output[j] * ref_output[j];
    }

    let rel_l2_error = (sum_diff_sq / sum_ref_sq).sqrt();
    assert!(
        rel_l2_error < 0.020,
        "INT8 reconstruction error must be < 2.0%, obtained: {:.4}%",
        rel_l2_error * 100.0
    );

    // Dimension mismatch checks
    let invalid_input = vec![0.0f32; in_features - 1];
    let mut valid_out = vec![0.0f32; out_features];
    assert!(q_linear.forward(&invalid_input, &mut valid_out).is_err());

    let mut invalid_out = vec![0.0f32; out_features + 1];
    assert!(q_linear.forward(&input, &mut invalid_out).is_err());
}

#[test]
fn test_audio_spsc_ring_buffer_fifo_ordering_wraparound_and_saturation() {
    let capacity_req = 100usize;
    let mut ring = AudioSpscRingBuffer::new(capacity_req);

    // Verify power-of-two rounding (100 -> 128)
    assert_eq!(ring.capacity(), 128);
    assert_eq!(ring.available_read(), 0);
    assert_eq!(ring.available_write(), 128);
    assert!(ring.is_empty());
    assert!(!ring.is_full());

    // 1. Partial push and pop
    let samples_1: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    let pushed_1 = ring.push(&samples_1);
    assert_eq!(pushed_1, 50);
    assert_eq!(ring.available_read(), 50);
    assert_eq!(ring.available_write(), 78);

    let mut read_buf_1 = vec![0.0f32; 30];
    let popped_1 = ring.pop(&mut read_buf_1);
    assert_eq!(popped_1, 30);
    assert_eq!(ring.available_read(), 20);
    assert_eq!(ring.available_write(), 108);

    for i in 0..30 {
        assert!((read_buf_1[i] - samples_1[i]).abs() < 1e-6);
    }

    // 2. Capacity saturation: attempt pushing more than available slots
    let large_batch: Vec<f32> = (0..150).map(|i| (i + 100) as f32).collect();
    let pushed_large = ring.push(&large_batch);
    assert_eq!(pushed_large, 108, "Should saturate exactly at remaining capacity");
    assert_eq!(ring.available_write(), 0);
    assert_eq!(ring.available_read(), 128);
    assert!(ring.is_full());

    // Non-blocking write when full returns 0
    let empty_push = ring.push(&[1.0, 2.0, 3.0]);
    assert_eq!(empty_push, 0);

    // 3. Continuous FIFO drain and wraparound verification
    let mut drain_buf = vec![0.0f32; 128];
    let drained = ring.pop(&mut drain_buf);
    assert_eq!(drained, 128);
    assert_eq!(ring.available_read(), 0);
    assert!(ring.is_empty());

    // Check first 20 drained samples match remainder of samples_1
    for i in 0..20 {
        assert!((drain_buf[i] - samples_1[30 + i]).abs() < 1e-6);
    }
    // Check remaining 108 drained samples match large_batch
    for i in 0..108 {
        assert!((drain_buf[20 + i] - large_batch[i]).abs() < 1e-6);
    }

    // 4. Repeated wraparound cycling over 10,000 samples
    for cycle in 0..100 {
        let block: Vec<f32> = (0..64).map(|i| (cycle * 64 + i) as f32).collect();
        let p = ring.push(&block);
        assert_eq!(p, 64);

        let mut out_block = vec![0.0f32; 64];
        let r = ring.pop(&mut out_block);
        assert_eq!(r, 64);

        for i in 0..64 {
            assert_eq!(out_block[i], block[i]);
        }
    }
}

#[test]
fn test_edge_speech_runtime_streaming_chunk_generation_and_throughput() {
    let config = StreamingEdgeConfig {
        sample_rate: 16000.0,
        frame_size: 256,
        hop_size: 128,
        ring_buffer_capacity: 8192,
        hidden_dim: 64,
        chunk_frames: 4,
        pitch_f0: 130.0,
    };

    let mut runtime = EdgeSpeechRuntime::new(config);

    // Empty text handling
    let empty_res = runtime.synthesize_chunk("   ").expect("Empty text should succeed");
    assert!(empty_res.is_empty());

    // Measure time-to-first-audio (TTFA) execution
    let start_time = Instant::now();
    let audio = runtime
        .synthesize_chunk("STATUS")
        .expect("Synthesis chunk should succeed");
    let elapsed = start_time.elapsed();

    // Verify sub-15ms execution time (allow headroom for unoptimized debug builds)
    let max_allowed_ms = if cfg!(debug_assertions) { 150 } else { 15 };
    assert!(
        elapsed.as_millis() < max_allowed_ms,
        "Time-to-first-audio execution must be within threshold ({}ms), took: {:?}",
        max_allowed_ms,
        elapsed
    );

    assert!(!audio.is_empty(), "Generated audio chunk must not be empty");
    for &sample in &audio {
        assert!(sample.is_finite(), "Audio samples must be finite");
        assert!(
            sample >= -1.0 && sample <= 1.0,
            "Audio samples must be bounded within [-1.0, 1.0]"
        );
    }

    // Stream into lock-free audio ring buffer
    let pushed = runtime
        .stream_into_ring_buffer("ABORT")
        .expect("Streaming into ring buffer should succeed");
    assert!(pushed > 0);
    assert_eq!(runtime.ring_buffer().available_read(), pushed);

    let mut read_audio = vec![0.0f32; pushed];
    let popped = runtime.read_audio(&mut read_audio);
    assert_eq!(popped, pushed);
}

#[test]
fn test_sonon_engine_edge_speech_runtime_integration() {
    let mut engine = SononEngine::new(16000.0, 256, 128, 13);

    // Initially disabled
    assert!(engine.edge_speech_runtime().is_none());
    assert!(engine.synthesize_edge_speech_streaming("TAKEOFF").is_err());

    // Enable edge runtime
    let edge_cfg = StreamingEdgeConfig {
        sample_rate: 16000.0,
        frame_size: 256,
        hop_size: 128,
        ring_buffer_capacity: 4096,
        hidden_dim: 64,
        chunk_frames: 4,
        pitch_f0: 140.0,
    };
    engine.enable_edge_speech_runtime(edge_cfg);

    assert!(engine.edge_speech_runtime().is_some());
    assert!(engine.edge_speech_runtime_mut().is_some());

    // Synthesize streaming speech via SononEngine
    let audio = engine
        .synthesize_edge_speech_streaming("LAND")
        .expect("Engine streaming speech synthesis should succeed");

    assert!(!audio.is_empty());
    for &sample in &audio {
        assert!(sample.is_finite());
        assert!(sample >= -1.0 && sample <= 1.0);
    }

    // Disable edge runtime
    engine.disable_edge_speech_runtime();
    assert!(engine.edge_speech_runtime().is_none());
    assert!(engine.synthesize_edge_speech_streaming("LAND").is_err());
}

#[test]
fn test_deterministic_reproducibility_across_repeated_synthesis() {
    let config = StreamingEdgeConfig::default();
    let mut runtime = EdgeSpeechRuntime::new(config);

    let phrase = "EMERGENCY HOLD";

    // Run 1
    let audio_1 = runtime
        .synthesize_chunk(phrase)
        .expect("Run 1 synthesis should succeed");

    // Reset runtime
    runtime.reset();

    // Run 2
    let audio_2 = runtime
        .synthesize_chunk(phrase)
        .expect("Run 2 synthesis should succeed");

    assert_eq!(
        audio_1.len(),
        audio_2.len(),
        "Audio output length must be identical"
    );

    // Bit-for-bit exact reproducibility
    for (i, (&s1, &s2)) in audio_1.iter().zip(audio_2.iter()).enumerate() {
        assert_eq!(
            s1.to_bits(),
            s2.to_bits(),
            "Sample mismatch at index {}: {} vs {}",
            i,
            s1,
            s2
        );
    }
}
