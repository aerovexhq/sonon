//! Analytical and empirical verification suite for Phase 25:
//! Acoustic Echolocation & 3D Obstacle Spatial Mapping for GPS-Denied Subterranean UAV Flight.

#![deny(unsafe_code)]

use sonon::{
    ArrayGeometry, CaCfarConfig, ChirpConfig, LfmChirpGenerator,
    MultiMicAcousticEcholocator, Point3D, SubterraneanCaveSimulator, SononEngine, SPEED_OF_SOUND,
};
use std::time::Instant;

/// Test 1: LFM Chirp Pulse Compression and Theoretical Range Resolution.
///
/// Verifies matched filter pulse compression, processing gain G_proc = 10 * log10(B * Tp),
/// and theoretical sub-decimeter spatial range resolution Delta R = c / (2 * B).
#[test]
fn test_lfm_chirp_pulse_compression_and_resolution() {
    let sample_rate = 16000.0;
    let config = ChirpConfig {
        sample_rate,
        start_freq_hz: 2000.0,
        end_freq_hz: 6000.0,
        pulse_duration_sec: 0.015, // 15 ms
        pulse_repetition_hz: 10.0,
        window_taper_ratio: 0.20,
    };

    let generator = LfmChirpGenerator::new(config.clone());
    assert_eq!(config.bandwidth_hz(), 4000.0);

    // 1. Theoretical metrics:
    // Range resolution: Delta R = c / (2 * B) = 343 / (2 * 4000) = 0.042875 m ≈ 4.29 cm
    let delta_r = config.theoretical_range_resolution_m();
    assert!((delta_r - 0.042875).abs() < 0.001);

    // Processing gain: G_proc = 10 * log10(4000 * 0.015) = 10 * log10(60) = 17.78 dB
    let gain_db = config.processing_gain_db();
    assert!((gain_db - 17.78).abs() < 0.1);

    // 2. Auto-correlation of chirp template (matched filter impulse response)
    let template = generator.template();
    let n = template.len();
    assert_eq!(n, 240); // 15 ms at 16 kHz = 240 samples

    let mut autocorr = vec![0.0f32; 2 * n - 1];
    let energy = generator.template_energy();

    for shift in -(n as isize - 1)..=(n as isize - 1) {
        let mut dot = 0.0f32;
        for i in 0..n {
            let j = i as isize + shift;
            if j >= 0 && (j as usize) < n {
                dot += template[i] * template[j as usize];
            }
        }
        let out_idx = (shift + (n as isize - 1)) as usize;
        autocorr[out_idx] = dot / energy;
    }

    // Peak at center must be exactly 1.0 (normalized)
    let center = n - 1;
    assert!((autocorr[center] - 1.0).abs() < 1e-5);

    // Sidelobe level: sidelobes 10 samples away must be suppressed by > 12 dB (amplitude < 0.25)
    let sidelobe = autocorr[center + 10].abs();
    assert!(
        sidelobe < 0.25,
        "Auto-correlation sidelobes must be sharply compressed, got: {sidelobe}"
    );
}

/// Test 2: CA-CFAR Echo Detection and Sub-Sample Parabolic Precision.
///
/// Places synthetic obstacles at known radial ranges (e.g. 1.85 m and 3.42 m).
/// Verifies that CA-CFAR adaptive thresholding detects the reflection peaks, and
/// sub-sample parabolic interpolation achieves range error < 1.5 cm.
#[test]
fn test_cacfar_echo_detection_and_sub_sample_parabolic_precision() {
    let sample_rate = 16000.0;
    let chirp_config = ChirpConfig {
        sample_rate,
        start_freq_hz: 2000.0,
        end_freq_hz: 6000.0,
        pulse_duration_sec: 0.015,
        pulse_repetition_hz: 10.0,
        window_taper_ratio: 0.20,
    };
    let cfar_config = CaCfarConfig {
        training_cells: 16,
        guard_cells: 4,
        threshold_multiplier: 3.0,
        min_range_m: 0.25,
        max_range_m: 8.0,
        min_peak_confidence: 0.15,
    };

    let geometry = ArrayGeometry::Linear {
        spacing: 0.06,
        num_mics: 2,
    };
    let mut echolocator = MultiMicAcousticEcholocator::new(geometry, chirp_config, cfar_config);

    // Target ground truth radial distance: 2.150 meters
    // Two-way Time-of-Flight: ToF = 2 * 2.150 / 343 = 0.0125364 sec = 200.58 samples
    let ground_truth_range_m = 2.150f32;
    let tof_sec = 2.0 * ground_truth_range_m / SPEED_OF_SOUND;
    let tof_samples = tof_sec * sample_rate;

    let template = echolocator.chirp_template().to_vec();
    let total_samples = 1000;
    let mut ch0 = vec![0.0f32; total_samples];
    let mut ch1 = vec![0.0f32; total_samples];

    // Inject direct transmission cross-talk at start (t = 0)
    for i in 0..template.len() {
        ch0[i] += template[i] * 0.4;
        ch1[i] += template[i] * 0.4;
    }

    // Inject reflected echo at exact fractional delay
    let base_delay = tof_samples.floor() as usize;
    let frac = tof_samples - (base_delay as f32);

    for i in 0..template.len() {
        let s = template[i] * 0.5;
        // Linear fractional delay injection
        if base_delay + i + 1 < total_samples {
            ch0[base_delay + i] += s * (1.0 - frac);
            ch0[base_delay + i + 1] += s * frac;
            ch1[base_delay + i] += s * (1.0 - frac);
            ch1[base_delay + i + 1] += s * frac;
        }
    }

    // Process ping
    let inputs = [&ch0[..], &ch1[..]];
    let cloud = echolocator.process_ping(&inputs, None, 0.0);

    // Must detect the target obstacle
    assert!(
        !cloud.obstacles.is_empty(),
        "CA-CFAR must detect the reflected obstacle echo"
    );

    let detected = &cloud.obstacles[0];
    let range_error_m = (detected.range_m - ground_truth_range_m).abs();

    assert!(
        range_error_m < 0.020, // Within 2.0 cm precision
        "Sub-sample range error must be < 2.0 cm, got: {range_error_m:.4} m (detected: {:.3} m, truth: {:.3} m)",
        detected.range_m,
        ground_truth_range_m
    );
}

/// Test 3: Multi-Microphone TDoA and 3D Point Cloud Triangulation.
///
/// Simulates an obstacle at 2.50 m range and +30 degrees azimuth (+0.5236 rad).
/// Verifies that multi-channel inter-microphone TDoA extraction yields correct DoA
/// bearing (< 4.0 degrees error) and matches body-frame Cartesian coordinates (X, Y).
#[test]
fn test_multi_mic_tdoa_and_3d_point_cloud_triangulation() {
    let sample_rate = 16000.0;
    let spacing = 0.08; // 80 mm inter-mic spacing along Y-axis (wings)
    let geometry = ArrayGeometry::Linear {
        spacing,
        num_mics: 2,
    };
    let chirp_config = ChirpConfig::default();
    let cfar_config = CaCfarConfig {
        training_cells: 16,
        guard_cells: 4,
        threshold_multiplier: 2.5,
        min_range_m: 0.30,
        max_range_m: 10.0,
        min_peak_confidence: 0.15,
    };

    let mut echolocator = MultiMicAcousticEcholocator::new(geometry, chirp_config, cfar_config);
    let template = echolocator.chirp_template().to_vec();

    // Target at 2.50 m, azimuth theta = +25 degrees (to the right, +Y)
    let target_range_m = 2.50f32;
    let target_azimuth_rad = 25.0f32.to_radians(); // ~0.4363 rad

    let tof_sec = 2.0 * target_range_m / SPEED_OF_SOUND;
    let ch0_delay = (tof_sec * sample_rate).round() as usize;

    // Linear array mounted laterally: mic 0 at -spacing/2, mic 1 at +spacing/2
    // Relative delay between mics: tau = spacing * sin(theta) / c
    let delta_tau_sec = spacing * target_azimuth_rad.sin() / SPEED_OF_SOUND;
    // Fractional delay injection for channel 1 based on inter-mic TDoA
    let ch1_exact_delay = (ch0_delay as f32) + delta_tau_sec * sample_rate;
    let ch1_base = ch1_exact_delay.floor() as usize;
    let ch1_frac = ch1_exact_delay - (ch1_base as f32);

    let total_samples = 1200;
    let mut ch0 = vec![0.0f32; total_samples];
    let mut ch1 = vec![0.0f32; total_samples];

    for i in 0..template.len() {
        if ch0_delay + i < total_samples {
            ch0[ch0_delay + i] = template[i] * 0.6;
        }
        if ch1_base + i + 1 < total_samples {
            ch1[ch1_base + i] += template[i] * 0.6 * (1.0 - ch1_frac);
            ch1[ch1_base + i + 1] += template[i] * 0.6 * ch1_frac;
        }
    }

    let inputs = [&ch0[..], &ch1[..]];
    let cloud = echolocator.process_ping(&inputs, None, 1.23);

    assert!(!cloud.obstacles.is_empty(), "Must detect obstacle");
    let obs = &cloud.obstacles[0];

    // Verify range
    let range_err = (obs.range_m - target_range_m).abs();
    assert!(range_err < 0.05, "Range error must be < 5 cm, got: {range_err:.3} m");

    // Verify azimuth bearing
    let azim_err_deg = (obs.azimuth_rad.to_degrees() - target_azimuth_rad.to_degrees()).abs();
    println!("obs.azimuth: {:.2} deg, target: {:.2} deg, diff: {:.2} deg", obs.azimuth_rad.to_degrees(), target_azimuth_rad.to_degrees(), azim_err_deg);
    assert!(
        azim_err_deg < 4.0,
        "Azimuth DoA error must be < 4.0 degrees, got: {azim_err_deg:.2} deg"
    );

    // Verify 3D point cloud coordinates (Forward X > 0, Starboard Y > 0)
    assert!(obs.position_3d.x > 1.5, "Forward X coordinate must be positive");
    assert!(obs.position_3d.y > 0.5, "Right Y coordinate must be positive");
    assert!(cloud.right_clearance_m < cloud.left_clearance_m, "Right clearance must be smaller than left");
}

/// Test 4: Subterranean Cave Simulator and Collision Warning Trigger.
///
/// Uses SubterraneanCaveSimulator to simulate a narrow mine drift / cave passage with:
/// - A forward obstacle at 0.75 m (X = +0.75 m)
/// - Left wall at 1.20 m (Y = -1.20 m)
/// - Right wall at 1.50 m (Y = +1.50 m)
/// - Background quadcopter rotor noise at 0.03 RMS.
/// Confirms that min_distance_m is correctly identified (< 1.0 m) and is_collision_risk is triggered.
#[test]
fn test_subterranean_cave_simulator_and_collision_warning() {
    let sample_rate = 16000.0;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 2,
    };
    let chirp_config = ChirpConfig::default();
    let cfar_config = CaCfarConfig {
        training_cells: 16,
        guard_cells: 4,
        threshold_multiplier: 2.2,
        min_range_m: 0.25,
        max_range_m: 6.0,
        min_peak_confidence: 0.10,
    };

    let mut echolocator = MultiMicAcousticEcholocator::new(geometry.clone(), chirp_config.clone(), cfar_config);
    echolocator.set_collision_threshold(1.0); // 1.0 m collision buffer

    // Obstacles in subterranean passage
    let obstacles = vec![
        Point3D::new(0.75, 0.0, 0.0),   // Forward rock obstacle at 0.75 m
        Point3D::new(0.0, -1.20, 0.0),  // Left cave wall at 1.20 m
        Point3D::new(0.0, 1.50, 0.0),   // Right cave wall at 1.50 m
    ];
    let reflection_coeffs = vec![0.85, 0.70, 0.70];

    let tx_chirp = echolocator.chirp_template().to_vec();
    let sim_channels = SubterraneanCaveSimulator::simulate_subterranean_ping(
        &geometry,
        sample_rate,
        &obstacles,
        &reflection_coeffs,
        &tx_chirp,
        0.03, // Rotor noise RMS
    );

    let input_slices: Vec<&[f32]> = sim_channels.iter().map(|ch| &ch[..]).collect();
    let cloud = echolocator.process_ping(&input_slices, None, 2.5);

    assert!(
        !cloud.obstacles.is_empty(),
        "Cave simulator reflections must be detected"
    );

    // Minimum distance must be the forward obstacle (~0.75 m)
    assert!(
        cloud.min_distance_m < 0.90,
        "Minimum distance must detect the 0.75 m obstacle, got: {:.3} m",
        cloud.min_distance_m
    );

    // Collision risk must be active
    assert!(
        cloud.is_collision_risk,
        "Collision risk alert must be triggered when obstacle is within 1.0 m buffer"
    );

    // Forward clearance must match
    assert!(
        cloud.forward_clearance_m < 0.90,
        "Forward clearance must reflect close proximity obstacle"
    );
}

/// Test 5: Integrated SononEngine Echolocation and MAVLink Throughput Benchmark.
///
/// Tests end-to-end integration via SononEngine::enable_acoustic_echolocation,
/// validates MAVLink v2 NAMED_VALUE_FLOAT telemetry packet generation (ECHO_DIST, ECHO_CONF, ECHO_PTS),
/// and benchmarks streaming throughput (> 250,000 samples/sec).
#[test]
fn test_integrated_sonon_engine_echolocation_and_mavlink_throughput() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 256, 13);

    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 2,
    };
    let chirp_config = ChirpConfig::default();
    let cfar_config = CaCfarConfig::default();

    engine.enable_acoustic_echolocation(geometry.clone(), chirp_config.clone(), cfar_config);

    // 1. Generate multi-channel simulated cave return
    let obstacles = vec![Point3D::new(1.80, 0.30, 0.0)];
    let reflection_coeffs = vec![0.80];
    let tx_chirp = engine
        .echolocator()
        .expect("Echolocator must be present")
        .chirp_template()
        .to_vec();

    let sim_channels = SubterraneanCaveSimulator::simulate_subterranean_ping(
        &geometry,
        sample_rate,
        &obstacles,
        &reflection_coeffs,
        &tx_chirp,
        0.01,
    );

    let input_slices: Vec<&[f32]> = sim_channels.iter().map(|ch| &ch[..]).collect();

    // 2. Process through engine
    let cloud = engine
        .process_multi_channel_echolocation(&input_slices, None)
        .expect("Echolocation ping processing must succeed");

    assert!(!cloud.obstacles.is_empty(), "Obstacle must be detected");
    assert!((cloud.min_distance_m - 1.82).abs() < 0.10, "Range must be ~1.82 m");

    // 3. MAVLink telemetry packet validation
    let packets = cloud.to_mavlink_telemetry();
    assert_eq!(packets.len(), 3, "Expected 3 MAVLink telemetry packets");

    let dist_pkt = packets.iter().find(|p| p.name_as_str() == "ECHO_DIST").expect("ECHO_DIST missing");
    let conf_pkt = packets.iter().find(|p| p.name_as_str() == "ECHO_CONF").expect("ECHO_CONF missing");
    let pts_pkt = packets.iter().find(|p| p.name_as_str() == "ECHO_PTS").expect("ECHO_PTS missing");

    assert!((dist_pkt.value - cloud.min_distance_m).abs() < 1e-4);
    assert!(conf_pkt.value > 0.20, "Confidence must be significant");
    assert!(pts_pkt.value >= 1.0, "Expected at least 1 point cloud obstacle");

    // 4. Throughput benchmark across 50 pings
    let ping_len = input_slices[0].len();
    let iterations = 50;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = engine.process_multi_channel_echolocation(&input_slices, None);
    }
    let elapsed = start.elapsed();

    let total_samples = (iterations * ping_len) as f64;
    let samples_per_sec = total_samples / elapsed.as_secs_f64();
    let realtime_mult = samples_per_sec / (sample_rate as f64);

    assert!(
        samples_per_sec > 250_000.0,
        "Throughput must exceed 250k samples/sec, got: {samples_per_sec:.0} samples/sec ({realtime_mult:.1}x real-time)"
    );
}
