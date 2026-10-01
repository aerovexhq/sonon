#![deny(unsafe_code)]

use sonon::{
    AeroSsmCell, AnticipatoryPrefixDecoder, InterlockState, SincConvFrontend, WaldSprtConfig,
};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_sincnet_parametric_filterbank() {
    let sample_rate = 16000.0;
    let kernel_size = 101;
    let num_filters = 20;
    let mut sincnet = SincConvFrontend::new(sample_rate, kernel_size, num_filters);
    assert_eq!(sincnet.num_filters(), 20);

    // Find filter covering 1000 Hz
    let mut target_filter_idx = 0;
    for i in 0..num_filters {
        let (f1, f2) = sincnet.filter_band(i);
        if 1000.0 >= f1 && 1000.0 <= f2 {
            target_filter_idx = i;
            break;
        }
    }

    // Generate 1000 Hz pure tone
    let n = 200;
    let tone_1k: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 1000.0 * (i as f32) / sample_rate).sin())
        .collect();

    let initial_response = sincnet.process_frame(&tone_1k);
    let target_pre = initial_response[target_filter_idx];
    println!("SincNet target filter #{target_filter_idx} response pre-notch: {target_pre:.4}");

    assert!(
        target_pre > 0.05,
        "Target filter covering 1000 Hz must show resonant output"
    );

    // Now inject motor RPM telemetry: 3-blade prop at 20,000 RPM -> BPF = 1000 Hz!
    sincnet.update_rotor_telemetry(20000.0, 3);

    let notched_response = sincnet.process_frame(&tone_1k);
    let target_post = notched_response[target_filter_idx];
    println!("SincNet target filter #{target_filter_idx} response post-notch: {target_post:.4}");

    assert_eq!(
        target_post, 0.0,
        "SincNet target filter must be notched out to zero transmission"
    );
}

#[test]
fn test_aerossm_state_space_recurrence() {
    let d_model = 32;
    let state_dim = 16;
    let mut ssm = AeroSsmCell::new(d_model, state_dim);

    assert_eq!(ssm.d_model(), 32);
    assert_eq!(ssm.state_dim(), 16);

    // Feed a sequence of 50 consecutive frames
    for step_idx in 0..50 {
        let input_frame = vec![((step_idx as f32) * 0.1).sin(); d_model];
        let output = ssm.step(&input_frame);
        assert_eq!(output.len(), d_model);

        for &val in output {
            assert!(!val.is_nan(), "AeroSSM output must not be NaN");
            assert!(!val.is_infinite(), "AeroSSM output must not be Infinite");
            assert!(
                val.abs() < 50.0,
                "State space output must remain bounded, got {val}"
            );
        }
    }

    // Verify reset clears hidden state
    ssm.reset();
    let zero_in = vec![0.0f32; d_model];
    let zero_out = ssm.step(&zero_in);
    for &val in zero_out {
        assert!(val.abs() < 1e-4, "Zero input post-reset must yield zero output");
    }
}

#[test]
fn test_anticipatory_wald_sprt_prearm_and_commit() {
    let config = WaldSprtConfig {
        alpha_target: 0.001,
        beta_target: 0.01,
        prefix_ratio: 0.70, // 70% threshold
    };

    let mut decoder = AnticipatoryPrefixDecoder::new(config);
    let total_frames = 30; // 300 ms utterance at 10 ms hop
    let phrase_id = 101;

    // Wald upper boundary A: ln((1 - 0.01) / 0.001) = ln(990) approx 6.898
    let expected_a = (0.99f32 / 0.001f32).ln();
    assert!((decoder.upper_boundary() - expected_a).abs() < 1e-3);

    // 1. Initial 50% frames (frames 0 to 15): LLR increases but progress < 70%
    for i in 0..15 {
        let state = decoder.feed_frame(0.6, i, total_frames, phrase_id, (i as f64) * 0.01);
        assert_eq!(state, InterlockState::Listening);
    }

    // 2. Frame 21 (exactly 21 / 30 = 70% duration): cumulative LLR exceeds threshold A
    let mut prearmed = false;
    for i in 15..22 {
        let state = decoder.feed_frame(0.6, i, total_frames, phrase_id, (i as f64) * 0.01);
        if let InterlockState::PreArm { phrase_id: pid, .. } = state {
            assert_eq!(pid, phrase_id);
            prearmed = true;
            break;
        }
    }
    assert!(
        prearmed,
        "Decoder must trigger InterlockState::PreArm at exactly 70% phrase completion"
    );

    // 3. Trailing 30% suffix frames (frames 22 to 30): positive continuation confirms keyword
    let mut committed = false;
    for i in 22..=total_frames {
        let state = decoder.feed_frame(0.5, i, total_frames, phrase_id, (i as f64) * 0.01);
        if let InterlockState::Commit { phrase_id: pid, .. } = state {
            assert_eq!(pid, phrase_id);
            committed = true;
            break;
        }
    }
    assert!(
        committed,
        "Decoder must transition from PreArm to Commit once suffix confirms at 100% completion"
    );
}

#[test]
fn test_anticipatory_rollback_on_suffix_divergence() {
    let config = WaldSprtConfig {
        alpha_target: 0.001,
        beta_target: 0.01,
        prefix_ratio: 0.70,
    };

    let mut decoder = AnticipatoryPrefixDecoder::new(config);
    let total_frames = 30;
    let phrase_id = 202;

    // Feed frames up to 70% (21 frames out of 30) to trigger PreArm
    for i in 0..=21 {
        let _ = decoder.feed_frame(0.6, i, total_frames, phrase_id, (i as f64) * 0.01);
    }
    match decoder.current_state() {
        InterlockState::PreArm { .. } => {}
        other => panic!("Expected PreArm at 70%, got {:?}", other),
    }

    // During final 30% suffix (frames 22-30), impostor divergence occurs (negative LLR drops)
    let mut rolled_back = false;
    for i in 22..=total_frames {
        let state = decoder.feed_frame(-3.0, i, total_frames, phrase_id, (i as f64) * 0.01);
        if let InterlockState::Rollback { phrase_id: pid, .. } = state {
            assert_eq!(pid, phrase_id);
            rolled_back = true;
            break;
        }
    }

    assert!(
        rolled_back,
        "Decoder must trigger InterlockState::Rollback when suffix diverges"
    );
}

#[test]
fn test_aerossm_and_anticipatory_throughput() {
    let sample_rate = 16000.0;
    let sincnet = SincConvFrontend::new(sample_rate, 101, 32);
    let mut ssm = AeroSsmCell::new(32, 16);
    let mut decoder = AnticipatoryPrefixDecoder::new(WaldSprtConfig::default());

    let total_frames = 2000; // 20 seconds of streaming feature frames
    let test_frame = vec![0.1f32; 128];
    let mut filter_energies = vec![0.0f32; sincnet.num_filters()];

    let start = Instant::now();
    for i in 0..total_frames {
        sincnet.process_frame_into(&test_frame, &mut filter_energies);
        let ssm_features = ssm.step(&filter_energies);
        let llr = ssm_features[0] * 0.1;
        let _ = decoder.feed_frame(llr, i % 30, 30, 1, (i as f64) * 0.01);
    }
    let elapsed = start.elapsed();

    let frames_per_sec = total_frames as f64 / elapsed.as_secs_f64();
    let samples_equivalent = frames_per_sec * 160.0; // 160 samples per hop (10 ms)
    let real_time_factor = samples_equivalent / sample_rate as f64;

    println!(
        "AeroSSM + Anticipatory Decoder throughput: {samples_equivalent:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    let target = if cfg!(debug_assertions) {
        500_000.0
    } else {
        1_000_000.0
    };

    assert!(
        samples_equivalent > target,
        "AeroSSM throughput must exceed {target:.0} samples/sec, got {samples_equivalent:.0}"
    );
}
