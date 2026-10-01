//! Phase 8 Verification Test Suite: Fixed-Point Q15/Q31 DSP Math, Heapless Buffers & Microcontroller Portability.

#![deny(unsafe_code)]

use sonon::fixed::{FixedDtwMatcher, StaticAudioBuffer, Q15, Q31};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_q15_conversions_and_arithmetic() {
    let test_floats = [-1.0f32, -0.75, -0.5, -0.1, 0.0, 0.1, 0.5, 0.75, 0.9999];
    for &f in &test_floats {
        let q = Q15::from_f32(f);
        let recon = q.to_f32();
        assert!(
            (f - recon).abs() < 1e-4,
            "Q15 conversion mismatch for {f}: got {recon}"
        );
    }

    // Multiplication
    let a = Q15::from_f32(0.5);
    let b = Q15::from_f32(0.5);
    let prod = a.mul(b);
    assert!((prod.to_f32() - 0.25).abs() < 1e-4);

    // Negative multiplication
    let c = Q15::from_f32(-0.5);
    let prod_neg = a.mul(c);
    assert!((prod_neg.to_f32() - (-0.25)).abs() < 1e-4);

    // Saturating addition
    let max = Q15::ONE;
    let add_sat = max.saturating_add(Q15::from_f32(0.1));
    assert_eq!(add_sat, Q15::ONE);

    // Saturating subtraction
    let min = Q15::MIN;
    let sub_sat = min.saturating_sub(Q15::from_f32(0.1));
    assert_eq!(sub_sat, Q15::MIN);
}

#[test]
fn test_q31_conversions_and_arithmetic() {
    let test_floats = [-0.999f32, -0.5, 0.0, 0.25, 0.75, 0.999];
    for &f in &test_floats {
        let q = Q31::from_f32(f);
        let recon = q.to_f32();
        assert!(
            (f - recon).abs() < 1e-7,
            "Q31 conversion mismatch for {f}: got {recon}"
        );
    }

    let a = Q31::from_f32(0.6);
    let b = Q31::from_f32(0.4);
    let prod = a.mul(b);
    assert!((prod.to_f32() - 0.24).abs() < 1e-5);
}

#[test]
fn test_q15_lookup_trig_functions() {
    // Test across 360 degrees (0 to 65535 in 16-bit phase)
    let step = 1024u16;
    for p in (0..=65535).step_by(step as usize) {
        let phase_u16 = p as u16;
        let angle_rad = (phase_u16 as f32 / 65536.0) * 2.0 * PI;

        let sin_lut = Q15::lut_sin(phase_u16).to_f32();
        let sin_exact = angle_rad.sin();
        assert!(
            (sin_lut - sin_exact).abs() < 0.02,
            "Sine LUT error too high at {angle_rad} rad: LUT={sin_lut}, Exact={sin_exact}"
        );

        let cos_lut = Q15::lut_cos(phase_u16).to_f32();
        let cos_exact = angle_rad.cos();
        assert!(
            (cos_lut - cos_exact).abs() < 0.02,
            "Cosine LUT error too high at {angle_rad} rad: LUT={cos_lut}, Exact={cos_exact}"
        );
    }
}

#[test]
fn test_q15_dot_product() {
    let n = 128;
    let a_f32: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32) * 0.5).collect();
    let b_f32: Vec<f32> = (0..n)
        .map(|i| ((n - i) as f32) / (n as f32) * 0.5)
        .collect();

    let a_q15: Vec<Q15> = a_f32.iter().map(|&x| Q15::from_f32(x)).collect();
    let b_q15: Vec<Q15> = b_f32.iter().map(|&x| Q15::from_f32(x)).collect();

    let dot_raw = Q15::dot_product(&a_q15, &b_q15);
    let dot_q15 = (dot_raw as f32) / 32768.0;

    let mut dot_exact = 0.0f32;
    for i in 0..n {
        dot_exact += a_f32[i] * b_f32[i];
    }

    assert!(
        (dot_q15 - dot_exact).abs() < 0.02,
        "Q15 dot product mismatch: got {dot_q15}, expected {dot_exact}"
    );

    let dot_norm = Q15::dot_product_normalized(&a_q15, &b_q15).to_f32();
    assert!(
        (dot_norm - (dot_exact / (n as f32))).abs() < 1e-3,
        "Q15 normalized dot product mismatch"
    );
}

#[test]
fn test_static_audio_buffer_heapless_circular_fifo() {
    let mut buf = StaticAudioBuffer::<256>::new();
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    assert_eq!(buf.capacity(), 256);

    // Push 100 samples
    for i in 0..100 {
        buf.push(Q15(i as i16));
    }
    assert_eq!(buf.len(), 100);

    // Peek 50 samples
    let mut peek_out = [Q15::ZERO; 50];
    let peeked = buf.peek_slice(&mut peek_out);
    assert_eq!(peeked, 50);
    for i in 0..50 {
        assert_eq!(peek_out[i], Q15(i as i16));
    }
    assert_eq!(buf.len(), 100, "Peek must not alter length");

    // Pop 50 samples
    for i in 0..50 {
        let val = buf.pop().expect("Pop must succeed");
        assert_eq!(val, Q15(i as i16));
    }
    assert_eq!(buf.len(), 50);

    // Push 300 samples to test circular overwriting
    for i in 100..400 {
        buf.push(Q15(i as i16));
    }
    assert_eq!(buf.len(), 256, "Buffer must saturate at capacity 256");

    // Oldest surviving sample should be (400 - 256) = 144
    let first = buf.pop().expect("Pop should succeed");
    assert_eq!(first, Q15(144));
}

#[test]
fn test_fixed_point_dtw_matcher_accuracy() {
    // 10 frames of 13 coefficients each
    let frame_1 = [Q15::from_f32(0.2); 13];
    let frame_2 = [Q15::from_f32(0.4); 13];
    let frame_3 = [Q15::from_f32(0.8); 13];

    let seq_a = [
        &frame_1[..],
        &frame_1[..],
        &frame_2[..],
        &frame_2[..],
        &frame_3[..],
    ];

    // Identical sequence should have zero distance
    let dist_self = FixedDtwMatcher::compute_distance_q15(&seq_a, &seq_a, 3);
    assert_eq!(dist_self, 0, "Identical sequences must have zero DTW distance");

    // Time-dilated sequence: frame repeated
    let seq_warped = [
        &frame_1[..],
        &frame_1[..],
        &frame_1[..],
        &frame_2[..],
        &frame_2[..],
        &frame_3[..],
    ];
    let dist_warped = FixedDtwMatcher::compute_distance_q15(&seq_a, &seq_warped, 3);
    assert!(
        dist_warped < 50,
        "Time-warped alignment distance must be minimal, got {dist_warped}"
    );

    // Divergent sequence: different coefficients
    let frame_div = [Q15::from_f32(-0.8); 13];
    let seq_div = [
        &frame_div[..],
        &frame_div[..],
        &frame_div[..],
        &frame_div[..],
        &frame_div[..],
    ];
    let dist_div = FixedDtwMatcher::compute_distance_q15(&seq_a, &seq_div, 3);
    assert!(
        dist_div > 5000,
        "Divergent sequence distance must be high, got {dist_div}"
    );
}

#[test]
fn test_fixed_point_dsp_throughput() {
    let total_samples = 320_000; // 20 seconds at 16 kHz
    let mut buf = StaticAudioBuffer::<512>::new();

    let a = [Q15::from_f32(0.35); 32];
    let b = [Q15::from_f32(0.42); 32];

    let start = Instant::now();
    for i in 0..total_samples {
        buf.push(Q15((i % 1000) as i16));
        if i % 32 == 0 {
            let _ = Q15::dot_product(&a, &b);
        }
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / 16000.0;

    println!(
        "Q15 Fixed-Point Microcontroller DSP Throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    let min_target = if cfg!(debug_assertions) {
        5_000_000.0
    } else {
        20_000_000.0
    };

    assert!(
        samples_per_sec > min_target,
        "Q15 fixed-point throughput must exceed {min_target:.0} samples/sec, got {samples_per_sec:.0}"
    );
}
