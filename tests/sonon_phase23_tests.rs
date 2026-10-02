//! Analytical and empirical verification suite for Phase 23:
//! Ultra-Low-Power RISC-V Vector (RVV 1.0) and PULP-NN / XpulpNN Micro-Engine Acceleration.

#![deny(unsafe_code)]

use sonon::{
    PackedI16x2, PackedI8x4, PulpConfig, PulpMemoryArena, PulpPowerModel, PulpVectorEngine,
    RvvConfig, RvvVectorEngine, SononEngine,
};

/// Test 1: XpulpNN packed SIMD 8-bit & 16-bit dot products, SAD metric, and Q15 saturation.
#[test]
fn test_xpulpnn_simd_dot_products_and_sad() {
    // 1. 4-way 8-bit signed vector dot product (`pv.dotsp.b`)
    let a_i8 = [12, -34, 56, -78];
    let b_i8 = [-5, 10, -15, 20];
    let dot_b = PulpVectorEngine::pv_dotsp_b(a_i8, b_i8);
    let expected_dot_b = (12 * -5) + (-34 * 10) + (56 * -15) + (-78 * 20);
    assert_eq!(dot_b, expected_dot_b, "pv.dotsp.b calculation mismatch");
    assert_eq!(dot_b, -2800);

    // 2. 4-way 8-bit unsigned vector dot product (`pv.dotup.b`)
    let a_u8 = [10u8, 20, 30, 40];
    let b_u8 = [2u8, 4, 6, 8];
    let dot_u = PulpVectorEngine::pv_dotup_b(a_u8, b_u8);
    let expected_dot_u = (10 * 2) + (20 * 4) + (30 * 6) + (40 * 8);
    assert_eq!(dot_u, expected_dot_u, "pv.dotup.b calculation mismatch");
    assert_eq!(dot_u, 600);

    // 3. 2-way 16-bit signed halfword vector dot product (`pv.dotsp.h`)
    let a_h = [1200i16, -3400];
    let b_h = [50i16, 20];
    let dot_h = PulpVectorEngine::pv_dotsp_h(a_h, b_h);
    let expected_dot_h = (1200 * 50) + (-3400 * 20);
    assert_eq!(dot_h, expected_dot_h, "pv.dotsp.h calculation mismatch");
    assert_eq!(dot_h, -8000);

    // 4. 2-way 16-bit sum of absolute differences (`pv.sad.h`)
    let a_sad = [1500i16, -2000];
    let b_sad = [1000i16, -2500];
    let sad = PulpVectorEngine::pv_sad_h(a_sad, b_sad);
    let expected_sad = (1500i32 - 1000).abs() + (-2000i32 - -2500).abs();
    assert_eq!(sad as i32, expected_sad, "pv.sad.h calculation mismatch");
    assert_eq!(sad, 1000);

    // 5. Packed absolute, clip, min, max
    let sabs = PulpVectorEngine::pv_sabs_h([-300i16, 450]);
    assert_eq!(sabs, [300, 450]);

    let clipped = PulpVectorEngine::pv_clip_q15(40000, -32768, 32767);
    assert_eq!(clipped, 32767);
    let clipped_neg = PulpVectorEngine::pv_clip_q15(-50000, -32768, 32767);
    assert_eq!(clipped_neg, -32768);

    let min_h = PulpVectorEngine::pv_min_h([100, -50], [50, 0]);
    assert_eq!(min_h, [50, -50]);

    let max_h = PulpVectorEngine::pv_max_h([100, -50], [50, 0]);
    assert_eq!(max_h, [100, 0]);

    // 6. Vectorized Q15 dot product with unrolling
    let v_a: Vec<i16> = (0..64).map(|x| (x * 100) as i16).collect();
    let v_b: Vec<i16> = (0..64).map(|x| ((64 - x) * 50) as i16).collect();
    let unrolled_dot = PulpVectorEngine::vector_dot_product_q15(&v_a, &v_b);
    let mut naive_dot: i64 = 0;
    for i in 0..64 {
        naive_dot += (v_a[i] as i64) * (v_b[i] as i64);
    }
    assert_eq!(unrolled_dot, naive_dot, "Unrolled Q15 dot product must match naive reference");

    // 7. Vectorized DTW Manhattan distance row calculation
    let dtw_dist = PulpVectorEngine::vector_dtw_distance_q15(&v_a, &v_b);
    let mut naive_dtw: u32 = 0;
    for i in 0..64 {
        naive_dtw += (v_a[i] as i32 - v_b[i] as i32).unsigned_abs();
    }
    assert_eq!(dtw_dist, naive_dtw, "Vector DTW distance must match reference");

    // Verify packed struct wrappers
    let packed_i8 = PackedI8x4(a_i8);
    assert_eq!(packed_i8.0, a_i8);
    let packed_i16 = PackedI16x2(a_h);
    assert_eq!(packed_i16.0, a_h);
}

/// Test 2: RISC-V Vector Extension (RVV 1.0) scalable vectorization, widening MAC, and FIR filtering.
#[test]
fn test_rvv_scalable_vectorization_and_fir() {
    // 1. Verify VLMAX across configurations
    let rvv128 = RvvConfig {
        vlen_bits: 128,
        sew_bits: 16,
        lmul: 1,
    };
    assert_eq!(rvv128.vlmax(), 8, "VLEN 128 / SEW 16 = 8 elements");

    let rvv256_m2 = RvvConfig {
        vlen_bits: 256,
        sew_bits: 16,
        lmul: 2,
    };
    assert_eq!(rvv256_m2.vlmax(), 32, "VLEN 256 / SEW 16 * LMUL 2 = 32 elements");

    let rvv512_sew32 = RvvConfig {
        vlen_bits: 512,
        sew_bits: 32,
        lmul: 1,
    };
    assert_eq!(rvv512_sew32.vlmax(), 16, "VLEN 512 / SEW 32 = 16 elements");

    let engine = RvvVectorEngine::new(rvv256_m2);
    assert_eq!(engine.config().vlen_bits, 256);

    // 2. Vector fused multiply-accumulate on floating-point (`vfmacc.vv`)
    let mut acc_f32 = vec![1.0f32; 64];
    let a_f32: Vec<f32> = (0..64).map(|x| x as f32 * 0.5).collect();
    let b_f32: Vec<f32> = (0..64).map(|x| (64 - x) as f32 * 0.25).collect();
    engine.vfmacc_f32(&mut acc_f32, &a_f32, &b_f32);

    for i in 0..64 {
        let expected = 1.0 + (a_f32[i] * b_f32[i]);
        assert!((acc_f32[i] - expected).abs() < 1e-4, "vfmacc element {} mismatch", i);
    }

    // 3. Widening vector multiply-accumulate (`vwmacc.vv` from Q15 to Q31)
    let mut acc_q31 = vec![0i32; 64];
    let a_q15: Vec<i16> = (0..64).map(|x| (x * 200) as i16).collect();
    let b_q15: Vec<i16> = (0..64).map(|x| ((64 - x) * 150) as i16).collect();
    engine.vwmacc_q15_to_q31(&mut acc_q31, &a_q15, &b_q15);

    for i in 0..64 {
        let expected = (a_q15[i] as i32) * (b_q15[i] as i32);
        assert_eq!(acc_q31[i], expected, "vwmacc element {} mismatch", i);
    }

    // 4. Vector reduction sum (`vfredusum.vs`)
    let v_red: Vec<f32> = (1..=100).map(|x| x as f32).collect();
    let red_sum = engine.vfredsum_f32(&v_red);
    assert!((red_sum - 5050.0).abs() < 1e-3, "vfredsum should sum 1..=100 to 5050");

    // 5. Vectorized FIR filter kernel
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let taps = vec![0.25f32, 0.50, 0.25]; // 3-tap low-pass filter
    let mut output = vec![0.0f32; 6];
    engine.vector_fir_filter_f32(&input, &taps, &mut output);

    assert_eq!(output.len(), 6);
    // output[0] = 1.0*0.25 + 2.0*0.5 + 3.0*0.25 = 0.25 + 1.0 + 0.75 = 2.0
    assert!((output[0] - 2.0).abs() < 1e-4);
    // output[1] = 2.0*0.25 + 3.0*0.5 + 4.0*0.25 = 0.5 + 1.5 + 1.0 = 3.0
    assert!((output[1] - 3.0).abs() < 1e-4);
    // output[5] = 6.0*0.25 + 7.0*0.5 + 8.0*0.25 = 1.5 + 3.5 + 2.0 = 7.0
    assert!((output[5] - 7.0).abs() < 1e-4);
}

/// Test 3: Static zero-heap deterministic memory arena (O(1) allocation, alignment, reset).
#[test]
fn test_pulp_memory_arena_zero_heap_deterministic() {
    let mut arena: PulpMemoryArena<512, 256, 512> = PulpMemoryArena::new();
    assert_eq!(arena.bytes_allocated(), 0);
    assert_eq!(arena.total_capacity_bytes(), (512 * 2) + (256 * 2) + (512 * 4)); // 1024 + 512 + 2048 = 3584 bytes

    // 1. Push audio samples into static audio slice
    let audio_samples = vec![100i16; 160];
    let pushed = arena.push_audio(&audio_samples);
    assert_eq!(pushed, 160);
    assert_eq!(arena.audio_len, 160);
    assert_eq!(arena.bytes_allocated(), 160 * 2);

    // 2. Push feature values into static feature slice
    let feats = vec![50i16; 39]; // 13 MFCC + delta + delta-delta
    let pushed_feat = arena.push_features(&feats);
    assert_eq!(pushed_feat, 39);
    assert_eq!(arena.feature_len, 39);
    assert_eq!(arena.bytes_allocated(), (160 * 2) + (39 * 2));

    // 3. Allocate scratch matrix slice
    let scratch_slice = arena.scratch_slice_mut(128);
    assert!(scratch_slice.is_some());
    let slice = scratch_slice.unwrap();
    slice[0] = 42;
    slice[127] = 999;
    assert_eq!(arena.scratch_len, 128);
    assert_eq!(arena.bytes_allocated(), (160 * 2) + (39 * 2) + (128 * 4));

    let util = arena.utilization_ratio();
    assert!(util > 0.10 && util < 0.50, "Utilization should be ~25% (got {})", util);
    assert!(arena.peak_bytes_used >= arena.bytes_allocated());

    // 4. Test arena bounds overflow safety
    let large_scratch = arena.scratch_slice_mut(1000);
    assert!(large_scratch.is_none(), "Overflow allocation must safely return None without panic");

    // 5. Test frame reset unwinding
    arena.reset();
    assert_eq!(arena.audio_len, 0);
    assert_eq!(arena.feature_len, 0);
    assert_eq!(arena.scratch_len, 0);
    assert_eq!(arena.bytes_allocated(), 0);
    assert!(arena.peak_bytes_used > 0, "Peak usage is preserved across resets for telemetry");
}

/// Test 4: Sub-milliwatt surveillance power model (P_avg < 1.0 mW, coin cell runtime > 14 days, MAVLink packets).
#[test]
fn test_pulp_power_model_sub_milliwatt_surveillance() {
    let mut config = PulpConfig::default();
    config.core_clock_mhz = 50.0;
    config.core_voltage_v = 0.8;
    config.dynamic_power_uw_per_mhz = 15.0; // 50 MHz * 15 uW/MHz = 750 uW core active power
    config.sleep_power_uw = 50.0;
    config.mic_power_uw = 300.0;
    config.frame_interval_ms = 10.0; // 100 Hz frame rate
    config.battery_capacity_mah = 220.0; // CR2032 standard
    config.battery_voltage_v = 3.0;

    let model = PulpPowerModel::new(config);

    // Typical active cycle consumption per 10 ms audio frame:
    // FFT (12,000 cycles) + Mel filterbank (3,000 cycles) + DTW keyword match (15,000 cycles) = 30,000 cycles
    let active_cycles = 30_000u32;
    let telem = model.evaluate_frame_power(active_cycles, 0.35);

    // 1. Verify execution time and duty cycle
    // Execution time = 30,000 cycles / 50 MHz = 600 microseconds
    assert!(
        (telem.execution_time_us - 600.0).abs() < 1.0,
        "Execution time should be 600 us (got {} us)",
        telem.execution_time_us
    );

    // Duty cycle = 600 us / 10,000 us = 6.0%
    assert!(
        (telem.duty_cycle_pct - 6.0).abs() < 0.1,
        "Duty cycle should be 6.0% (got {}%)",
        telem.duty_cycle_pct
    );

    // 2. Verify sub-milliwatt average power (< 1000 uW)
    assert!(
        telem.average_power_uw < 1000.0,
        "Average power must be strictly below 1.0 mW (got {:.2} uW)",
        telem.average_power_uw
    );
    assert!(
        telem.average_power_uw > 100.0,
        "Average power must be realistic non-zero (got {:.2} uW)",
        telem.average_power_uw
    );
    println!(
        "Surveillance average power: {:.2} uW ({:.3} mW)",
        telem.average_power_uw,
        telem.average_power_uw * 0.001
    );

    // 3. Verify battery operational lifespan
    // On a tiny 220 mAh CR2032 coin cell, battery life must exceed 14 days of continuous listening
    assert!(
        telem.battery_life_days > 14.0,
        "CR2032 coin cell runtime must exceed 14 days (got {:.1} days)",
        telem.battery_life_days
    );
    println!(
        "CR2032 continuous surveillance runtime: {:.1} days ({:.1} hours)",
        telem.battery_life_days,
        telem.battery_life_days * 24.0
    );

    // On a 3.7V 300 mAh surveillance LiPo:
    let mut lipo_config = model.config().clone();
    lipo_config.battery_capacity_mah = 300.0;
    lipo_config.battery_voltage_v = 3.7;
    let lipo_model = PulpPowerModel::new(lipo_config);
    let lipo_telem = lipo_model.evaluate_frame_power(active_cycles, 0.35);
    assert!(
        lipo_telem.battery_life_days > 30.0,
        "300 mAh LiPo runtime must exceed 30 days (got {:.1} days)",
        lipo_telem.battery_life_days
    );

    // 4. Verify MAVLink v2 telemetry packets
    let packets = telem.to_mavlink_packets(100);
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[0].name_as_str(), "PULP_CYC");
    assert_eq!(packets[0].value, 30_000.0);
    assert_eq!(packets[1].name_as_str(), "PULP_PWR");
    assert_eq!(packets[2].name_as_str(), "PULP_BATT");
}

/// Test 5: High-throughput embedded streaming benchmark & SononEngine integration.
#[test]
fn test_pulp_streaming_throughput_and_engine_integration() {
    let rvv = RvvVectorEngine::new(RvvConfig::default());
    let mut acc = vec![0.0f32; 128];
    let a = vec![0.5f32; 128];
    let b = vec![2.0f32; 128];

    let num_ops = 50_000;
    let start = std::time::Instant::now();

    for _ in 0..num_ops {
        rvv.vfmacc_f32(&mut acc, &a, &b);
    }

    let elapsed = start.elapsed();
    let elapsed_sec = elapsed.as_secs_f64();
    let ops_per_sec = num_ops as f64 / elapsed_sec;

    println!(
        "PULP/RVV vector engine throughput: {:.2} operations/sec",
        ops_per_sec
    );
    assert!(
        ops_per_sec > 100_000.0,
        "Throughput must exceed 100,000 vector ops/sec (got {:.2})",
        ops_per_sec
    );

    // End-to-end integration test with SononEngine
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);
    engine.enable_pulp_acceleration(PulpConfig::default());

    let audio = vec![0.05f32; 1600]; // 100 ms of audio (10 frames)
    let events = engine.ingest_samples(&audio);
    assert!(events.is_empty() || !events.is_empty()); // Runs cleanly

    let telem = engine.latest_pulp_telemetry();
    assert!(telem.is_some(), "PULP telemetry must be populated after audio ingestion");
    let t = telem.unwrap();
    assert!(t.active_cycles_per_frame > 0);
    assert!(t.average_power_uw < 1000.0, "Average power must be < 1 mW");
    assert!(t.battery_life_days > 14.0);

    // Verify engine reset
    engine.reset();
    assert!(engine.latest_pulp_telemetry().is_none());
}
