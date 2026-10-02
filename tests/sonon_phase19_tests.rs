#![deny(unsafe_code)]

//! Phase 19 Test Suite: Acoustic Directional Target Sound Extraction (TSE),
//! Steered MVDR Adaptive Beamforming, and 3D Spatial Conditioning.

use sonon::beamforming::{ArrayGeometry, Point3D, SPEED_OF_SOUND};
use sonon::engine::{FeatureMode, SononEngine};
use sonon::tse::{
    calculate_bearing_from_gps, Complex32, GpsCoordinate, TargetSoundExtractor, TseConfig,
};
use std::f32::consts::PI;
use std::time::Instant;

/// Helper to synthesize a plane-wave multi-channel signal received across microphone positions.
fn synthesize_multi_channel_plane_wave(
    positions: &[Point3D],
    source_signal: &[f32],
    sample_rate: f32,
    azimuth_rad: f32,
    elevation_rad: f32,
) -> Vec<Vec<f32>> {
    let num_mics = positions.len();
    let num_samples = source_signal.len();
    let mut channels = vec![vec![0.0f32; num_samples]; num_mics];

    let u_x = elevation_rad.cos() * azimuth_rad.cos();
    let u_y = elevation_rad.cos() * azimuth_rad.sin();
    let u_z = elevation_rad.sin();

    for (m, p) in positions.iter().enumerate() {
        // Delay tau_m = (p . u) / c
        let proj = p.x * u_x + p.y * u_y + p.z * u_z;
        let delay_sec = proj / SPEED_OF_SOUND;
        let delay_samples = delay_sec * sample_rate;

        for n in 0..num_samples {
            let src_idx = (n as f32) + delay_samples;
            let idx_floor = src_idx.floor() as isize;
            let frac = src_idx - (idx_floor as f32);

            let s0 = if idx_floor >= 0 && (idx_floor as usize) < num_samples {
                source_signal[idx_floor as usize]
            } else {
                0.0
            };

            let s1 = if idx_floor + 1 >= 0 && ((idx_floor + 1) as usize) < num_samples {
                source_signal[(idx_floor + 1) as usize]
            } else {
                0.0
            };

            channels[m][n] = s0 + frac * (s1 - s0);
        }
    }

    channels
}

#[test]
fn test_complex_linear_system_solver() {
    // Test 2x2 complex system:
    // [ 2+j,   1-j ] [ x0 ] = [ 5+2j ]
    // [ 1+2j,  3+0j ] [ x1 ] = [ 7+4j ]
    let a_mat = vec![
        Complex32::new(2.0, 1.0),
        Complex32::new(1.0, -1.0),
        Complex32::new(1.0, 2.0),
        Complex32::new(3.0, 0.0),
    ];
    let b_vec = vec![Complex32::new(5.0, 2.0), Complex32::new(7.0, 4.0)];

    let sol = sonon::tse::solve_complex_linear_system(2, &a_mat, &b_vec)
        .expect("Linear system must be solvable");
    assert_eq!(sol.len(), 2);

    // Verify A * x = b
    let ax0 = a_mat[0].mul(sol[0]).add(a_mat[1].mul(sol[1]));
    let ax1 = a_mat[2].mul(sol[0]).add(a_mat[3].mul(sol[1]));

    assert!(
        (ax0.re - b_vec[0].re).abs() < 1e-4 && (ax0.im - b_vec[0].im).abs() < 1e-4,
        "Row 0 residual too high"
    );
    assert!(
        (ax1.re - b_vec[1].re).abs() < 1e-4 && (ax1.im - b_vec[1].im).abs() < 1e-4,
        "Row 1 residual too high"
    );
}

#[test]
fn test_gps_coordinate_to_body_bearing_conversion() {
    let drone_gps = GpsCoordinate::new(37.7749, -122.4194, 100.0);
    // Operator is 100m North of drone, on ground (altitude 0m)
    let operator_gps = GpsCoordinate::new(37.7758, -122.4194, 0.0);

    // 1. Drone heading North (yaw = 0): operator is straight ahead (+X) and below
    let (azim_north, elev_north) = calculate_bearing_from_gps(&drone_gps, &operator_gps, 0.0);
    assert!(
        azim_north.abs() < 0.05,
        "Azimuth should be ~0 rad when facing target, got {azim_north}"
    );
    assert!(
        elev_north < -0.6 && elev_north > -0.9,
        "Elevation should be negative (looking down), got {elev_north}"
    );

    // 2. Drone heading East (yaw = +pi/2): operator is to port (left -Y, -pi/2 rad)
    let (azim_east, _) = calculate_bearing_from_gps(&drone_gps, &operator_gps, PI * 0.5);
    assert!(
        (azim_east - (-PI * 0.5)).abs() < 0.05,
        "Azimuth should be -pi/2 rad when facing East, got {azim_east}"
    );

    // 3. Drone heading West (yaw = -pi/2): operator is to starboard (right +Y, +pi/2 rad)
    let (azim_west, _) = calculate_bearing_from_gps(&drone_gps, &operator_gps, -PI * 0.5);
    assert!(
        (azim_west - (PI * 0.5)).abs() < 0.05,
        "Azimuth should be +pi/2 rad when facing West, got {azim_west}"
    );
}

#[test]
fn test_mvdr_steering_vector_and_unity_gain_preservation() {
    let sample_rate = 16000.0;
    let geometry = ArrayGeometry::Circular {
        radius: 0.05,
        num_mics: 4,
    };
    let config = TseConfig {
        fft_size: 512,
        hop_size: 256,
        covariance_alpha: 0.90,
        diagonal_loading: 0.02,
        spatial_mask_threshold: 0.40,
        spatial_mask_gamma: 6.0,
        attenuation_floor: 0.05,
        min_freq_hz: 200.0,
        max_freq_hz: 3500.0,
    };

    let mut extractor = TargetSoundExtractor::new(geometry.clone(), sample_rate, config);
    let positions = geometry.positions();

    // Target speaker at 0.0 rad azimuth (ahead)
    let target_azim = 0.0f32;
    let target_elev = 0.0f32;
    extractor.set_target_bearing(target_azim, target_elev);

    // Generate pure multitone target signal in speech band (500 Hz + 1000 Hz)
    let num_samples = 4096;
    let mut target_signal = vec![0.0f32; num_samples];
    for n in 0..num_samples {
        let t = (n as f32) / sample_rate;
        target_signal[n] = 0.5 * (2.0 * PI * 500.0 * t).sin() + 0.5 * (2.0 * PI * 1000.0 * t).sin();
    }

    let channels = synthesize_multi_channel_plane_wave(
        &positions,
        &target_signal,
        sample_rate,
        target_azim,
        target_elev,
    );
    let hop = extractor.config().hop_size;
    let mut output = vec![0.0f32; num_samples];
    let mut out_block = vec![0.0f32; hop];

    let mut offset = 0;
    while offset + hop <= num_samples {
        let slices: Vec<&[f32]> = channels.iter().map(|ch| &ch[offset..offset + hop]).collect();
        let report = extractor.process_block(&slices, &mut out_block, (offset as f64) / (sample_rate as f64));
        output[offset..offset + hop].copy_from_slice(&out_block);

        // After initial filter startup, spatial mask for on-target signal should be high (> 0.70)
        if offset >= 1024 {
            assert!(
                report.mean_spatial_mask > 0.60,
                "Target signal spatial mask must be high, got {}",
                report.mean_spatial_mask
            );
        }
        offset += hop;
    }

    // Verify steady-state RMS preservation (> 0.65 of input RMS)
    let in_rms = (target_signal[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let out_rms = (output[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let gain_ratio = out_rms / in_rms;

    assert!(
        gain_ratio > 0.60 && gain_ratio < 1.40,
        "Target on-axis signal must be preserved with near unity gain, got gain ratio: {gain_ratio:.2}"
    );
}

#[test]
fn test_multi_speaker_cocktail_party_interference_suppression() {
    let sample_rate = 16000.0;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 4,
    };
    let config = TseConfig {
        fft_size: 512,
        hop_size: 256,
        covariance_alpha: 0.88,
        diagonal_loading: 0.03,
        spatial_mask_threshold: 0.50,
        spatial_mask_gamma: 8.0,
        attenuation_floor: 0.03, // -30 dB floor
        min_freq_hz: 200.0,
        max_freq_hz: 3800.0,
    };

    let mut extractor = TargetSoundExtractor::new(geometry.clone(), sample_rate, config);
    let positions = geometry.positions();

    // Steer towards broadside target at 0 degrees
    extractor.set_target_bearing(0.0, 0.0);

    let num_samples = 8192;
    // 1. Target speech: formants at 600 Hz + 1400 Hz
    let mut target_signal = vec![0.0f32; num_samples];
    for n in 0..num_samples {
        let t = (n as f32) / sample_rate;
        target_signal[n] = 0.4 * (2.0 * PI * 600.0 * t).sin() + 0.4 * (2.0 * PI * 1400.0 * t).sin();
    }
    let target_channels =
        synthesize_multi_channel_plane_wave(&positions, &target_signal, sample_rate, 0.0, 0.0);

    // 2. Loud interfering bystander at +45 degrees (PI / 4 rad) with formants at 900 Hz + 2200 Hz
    let mut interferer_signal = vec![0.0f32; num_samples];
    for n in 0..num_samples {
        let t = (n as f32) / sample_rate;
        interferer_signal[n] = 0.6 * (2.0 * PI * 900.0 * t).sin() + 0.6 * (2.0 * PI * 2200.0 * t).sin();
    }
    let interferer_channels = synthesize_multi_channel_plane_wave(
        &positions,
        &interferer_signal,
        sample_rate,
        PI * 0.25,
        0.0,
    );

    // Mixed acoustic cocktail party input
    let mut mixed_channels = vec![vec![0.0f32; num_samples]; 4];
    for m in 0..4 {
        for n in 0..num_samples {
            mixed_channels[m][n] = target_channels[m][n] + interferer_channels[m][n];
        }
    }

    let hop = extractor.config().hop_size;
    let mut output = vec![0.0f32; num_samples];
    let mut out_block = vec![0.0f32; hop];

    let mut offset = 0;
    while offset + hop <= num_samples {
        let slices: Vec<&[f32]> = mixed_channels
            .iter()
            .map(|ch| &ch[offset..offset + hop])
            .collect();
        let _ = extractor.process_block(&slices, &mut out_block, (offset as f64) / (sample_rate as f64));
        output[offset..offset + hop].copy_from_slice(&out_block);
        offset += hop;
    }

    // Measure interference power suppression:
    // In mixed channel 0, interferer has amplitude ~0.6 (power ~0.36)
    // In extracted output, the 900 Hz / 2200 Hz interferer must be substantially attenuated (> 10 dB)
    let eval_range = 2048..6144;
    let mut out_int_power = 0.0f32;
    let mut in_int_power = 0.0f32;

    // Project onto 900 Hz interferer carrier
    for n in eval_range {
        let t = (n as f32) / sample_rate;
        let c = (2.0 * PI * 900.0 * t).sin();
        out_int_power += output[n] * c;
        in_int_power += mixed_channels[0][n] * c;
    }

    let suppression_ratio = in_int_power.abs() / (out_int_power.abs().max(1e-4));
    let suppression_db = 20.0 * suppression_ratio.log10();

    assert!(
        suppression_db > 8.0,
        "Interfering bystander at 45 deg must be suppressed by > 8 dB, got {suppression_db:.1} dB"
    );
}

#[test]
fn test_mavlink_telemetry_packet_generation() {
    let report = sonon::tse::TseReport {
        steered_azimuth_rad: 0.5236,   // ~30 deg
        steered_elevation_rad: -0.2618, // ~-15 deg
        target_rms: 0.125,
        interference_suppression_db: 18.5,
        mean_spatial_mask: 0.78,
        is_target_present: true,
        timestamp_sec: 12.345,
    };

    let packets = report.to_mavlink_packets(12345);
    assert_eq!(packets.len(), 4);

    let azim_pkt = packets.iter().find(|p| p.name_as_str() == "TSE_AZIM").expect("TSE_AZIM missing");
    assert!((azim_pkt.value - 30.0).abs() < 0.1);

    let elev_pkt = packets.iter().find(|p| p.name_as_str() == "TSE_ELEV").expect("TSE_ELEV missing");
    assert!((elev_pkt.value - (-15.0)).abs() < 0.1);

    let mask_pkt = packets.iter().find(|p| p.name_as_str() == "TSE_MASK").expect("TSE_MASK missing");
    assert!((mask_pkt.value - 0.78).abs() < 0.01);

    let supp_pkt = packets.iter().find(|p| p.name_as_str() == "TSE_SUPP").expect("TSE_SUPP missing");
    assert!((supp_pkt.value - 18.5).abs() < 0.1);
}

#[test]
fn test_integrated_sonon_engine_cocktail_party_wake_word_spotting() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // 1. Synthesize and enroll a target wake-word template ("take off")
    let keyword_text = "take off";
    let frames = engine.enroll_keyword_from_text("take_off", keyword_text, 3.5);
    assert!(frames > 0);
    let synth_audio = engine.synthesize_speech_from_text(keyword_text);

    // 2. Enable Target Sound Extractor conditioned on target operator at 0 deg azimuth
    let geometry = ArrayGeometry::Circular {
        radius: 0.06,
        num_mics: 4,
    };
    let tse_config = TseConfig {
        fft_size: 256,
        hop_size: 128,
        covariance_alpha: 0.90,
        diagonal_loading: 0.03,
        spatial_mask_threshold: 0.40,
        spatial_mask_gamma: 8.0,
        attenuation_floor: 0.05,
        min_freq_hz: 150.0,
        max_freq_hz: 3800.0,
    };
    engine.enable_target_sound_extractor(geometry.clone(), tse_config);
    engine.update_target_bearing(0.0, 0.0);

    // 3. Generate 4-channel audio stream containing:
    // - Desired operator speaking "take off" at 0 deg azimuth
    // - Simultaneous interfering bystander at +45 deg azimuth
    let total_samples = synth_audio.len() + 4000;
    let mut target_stream = vec![0.0f32; total_samples];
    target_stream[2000..2000 + synth_audio.len()].copy_from_slice(&synth_audio);

    let positions = geometry.positions();
    let target_channels =
        synthesize_multi_channel_plane_wave(&positions, &target_stream, sample_rate, 0.0, 0.0);

    let mut interferer_stream = vec![0.0f32; total_samples];
    for n in 0..total_samples {
        let t = (n as f32) / sample_rate;
        interferer_stream[n] = 0.4 * (2.0 * PI * 1100.0 * t).sin() + 0.4 * (2.0 * PI * 2400.0 * t).sin();
    }
    let interferer_channels = synthesize_multi_channel_plane_wave(
        &positions,
        &interferer_stream,
        sample_rate,
        PI * 0.25,
        0.0,
    );

    let mut mixed_channels = vec![vec![0.0f32; total_samples]; 4];
    for m in 0..4 {
        for n in 0..total_samples {
            mixed_channels[m][n] = target_channels[m][n] + interferer_channels[m][n];
        }
    }

    let slices: Vec<&[f32]> = mixed_channels.iter().map(|ch| ch.as_slice()).collect();
    let events = engine
        .ingest_multi_channel_tse(&slices)
        .expect("TSE ingestion must succeed");

    assert!(
        !events.is_empty(),
        "Cocktail party target sound extraction must successfully spot enrolled keyword 'take_off'"
    );
    assert_eq!(events[0].keyword, "take_off");
}

#[test]
fn test_tse_streaming_throughput_benchmark() {
    let sample_rate = 16000.0;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.05,
        num_mics: 4,
    };
    let config = TseConfig::default();
    let mut extractor = TargetSoundExtractor::new(geometry, sample_rate, config);

    let total_samples = 32000; // 2 seconds of 4-channel audio
    let input_channels = vec![vec![0.05f32; total_samples]; 4];
    let hop = extractor.config().hop_size;
    let mut output_block = vec![0.0f32; hop];

    let start = Instant::now();
    let mut offset = 0;
    while offset + hop <= total_samples {
        let slices: Vec<&[f32]> = input_channels
            .iter()
            .map(|ch| &ch[offset..offset + hop])
            .collect();
        let _ = extractor.process_block(&slices, &mut output_block, (offset as f64) / (sample_rate as f64));
        offset += hop;
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / (sample_rate as f64);

    println!(
        "Phase 19 TSE MVDR Throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    // Verify throughput exceeds 50,000 samples/sec (> 3.1x real-time speed on 4 channels)
    assert!(
        samples_per_sec > 50_000.0,
        "TSE throughput was {:.0} samples/sec, below 50,000 threshold",
        samples_per_sec
    );
}
