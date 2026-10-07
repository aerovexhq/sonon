#![deny(unsafe_code)]

//! Comprehensive test suite for multi-channel spatial null-steering beamforming,
//! motor BPF harmonic cancellation, and moving UAV platform flight kinematics co-simulation for KWS.

use sonon::beamforming::{ArrayGeometry, Point3D, SPEED_OF_SOUND};
use sonon::engine::{FeatureMode, SononEngine};
use sonon::tse::{FlightDynamicsSimulator, TargetSoundExtractor, TseConfig};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_lcmv_spatial_null_steering_attenuation() {
    let sample_rate = 16000.0f32;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 4,
    };

    let mut config = TseConfig::default();
    config.fft_size = 512;
    config.hop_size = 256;
    config.motor_nulls_enabled = true;
    config.min_freq_hz = 150.0;
    config.max_freq_hz = 3500.0;

    let mut extractor = TargetSoundExtractor::new(geometry.clone(), sample_rate, config);

    // Steer target mainlobe toward broadside (azimuth 0 rad)
    extractor.set_target_bearing(0.0, 0.0);

    // Place explicit spatial null at +45 degrees (azimuth +pi/4 rad)
    let null_azimuth = PI / 4.0;
    extractor.set_spatial_null_directions(&[(null_azimuth, 0.0)]);

    assert_eq!(extractor.null_directions().len(), 1);
    assert!((extractor.null_directions()[0].0 - null_azimuth).abs() < 1e-4);

    let positions = geometry.positions();
    let num_mics = positions.len();
    let test_len = 4096;

    // Generate multi-channel tone arriving from target direction (azimuth 0 rad, broadside)
    let f_target = 800.0f32;
    let mut target_channels = vec![vec![0.0f32; test_len]; num_mics];
    for m in 0..num_mics {
        for t in 0..test_len {
            let time_s = t as f32 / sample_rate;
            // At broadside, inter-mic delay is 0
            target_channels[m][t] = (2.0 * PI * f_target * time_s).sin() * 0.5;
        }
    }

    // Generate multi-channel tone arriving from null direction (azimuth +45 degrees)
    let f_null = 800.0f32;
    let u_x = null_azimuth.cos();
    let u_y = null_azimuth.sin();
    let mut null_channels = vec![vec![0.0f32; test_len]; num_mics];
    for m in 0..num_mics {
        let pos = &positions[m];
        let tau = (pos.x * u_x + pos.y * u_y) / SPEED_OF_SOUND;
        for t in 0..test_len {
            let time_s = t as f32 / sample_rate + tau;
            null_channels[m][t] = (2.0 * PI * f_null * time_s).sin() * 0.5;
        }
    }

    // Process target signal through null-steering extractor
    let hop = 256;
    let mut target_out = vec![0.0f32; hop];
    let mut total_target_energy = 0.0f32;
    let mut offset = 0;
    while offset + hop <= test_len {
        let slices: Vec<&[f32]> = (0..num_mics)
            .map(|m| &target_channels[m][offset..offset + hop])
            .collect();
        let _ = extractor.process_block(&slices, &mut target_out, offset as f64 / sample_rate as f64);
        if offset > 1024 {
            // Steady state
            total_target_energy += target_out.iter().map(|&x| x * x).sum::<f32>();
        }
        offset += hop;
    }

    // Reset extractor and process null interference signal
    extractor.reset();
    let mut null_out = vec![0.0f32; hop];
    let mut total_null_energy = 0.0f32;
    offset = 0;
    while offset + hop <= test_len {
        let slices: Vec<&[f32]> = (0..num_mics)
            .map(|m| &null_channels[m][offset..offset + hop])
            .collect();
        let _ = extractor.process_block(&slices, &mut null_out, offset as f64 / sample_rate as f64);
        if offset > 1024 {
            // Steady state
            total_null_energy += null_out.iter().map(|&x| x * x).sum::<f32>();
        }
        offset += hop;
    }

    let suppression_ratio = total_target_energy / (total_null_energy.max(1e-9));
    let suppression_db = 10.0 * suppression_ratio.log10();

    assert!(
        suppression_db > 20.0,
        "Spatial null suppression {:.1} dB is below required 20 dB threshold",
        suppression_db
    );
}

#[test]
fn test_bpf_harmonic_selective_null_steering() {
    let sample_rate = 16000.0f32;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 4,
    };

    let mut config = TseConfig::default();
    config.motor_nulls_enabled = true;
    config.selective_bpf_only = true;
    config.num_blades = 2;
    config.bpf_harmonics = 3;
    config.bpf_guard_bandwidth_hz = 70.0;

    let mut extractor = TargetSoundExtractor::new(geometry.clone(), sample_rate, config);

    // 4500 RPM -> BPF fundamental = 150 Hz, 2nd harmonic = 300 Hz, 3rd = 450 Hz
    let motor_rpms = vec![4500.0];
    extractor.set_motor_rpms(&motor_rpms, 2);
    extractor.set_target_bearing(0.0, 0.0);
    extractor.set_spatial_null_directions(&[(PI / 3.0, 0.0)]); // Null at 60 degrees

    assert!(extractor.config().selective_bpf_only);
    assert_eq!(extractor.motor_rpms(), &[4500.0]);

    // Test harmonic at 300 Hz arriving from null direction (60 degrees) vs target direction
    let test_len = 3072;
    let positions = geometry.positions();
    let num_mics = positions.len();

    let u_x = (PI / 3.0).cos();
    let u_y = (PI / 3.0).sin();
    let mut bpf_channels = vec![vec![0.0f32; test_len]; num_mics];
    for m in 0..num_mics {
        let pos = &positions[m];
        let tau = (pos.x * u_x + pos.y * u_y) / SPEED_OF_SOUND;
        for t in 0..test_len {
            let time_s = t as f32 / sample_rate + tau;
            bpf_channels[m][t] = (2.0 * PI * 300.0 * time_s).sin() * 0.4;
        }
    }

    let hop = 256;
    let mut out = vec![0.0f32; hop];
    let mut bpf_out_energy = 0.0f32;
    let mut offset = 0;
    while offset + hop <= test_len {
        let slices: Vec<&[f32]> = (0..num_mics)
            .map(|m| &bpf_channels[m][offset..offset + hop])
            .collect();
        let _ = extractor.process_block(&slices, &mut out, offset as f64 / sample_rate as f64);
        if offset > 1024 {
            bpf_out_energy += out.iter().map(|&x| x * x).sum::<f32>();
        }
        offset += hop;
    }

    // Input energy on single microphone
    let input_energy: f32 = bpf_channels[0][1024..test_len]
        .iter()
        .map(|&x| x * x)
        .sum();
    let notch_suppression_db = 10.0 * (input_energy / bpf_out_energy.max(1e-9)).log10();

    assert!(
        notch_suppression_db > 15.0,
        "Motor BPF harmonic suppression {:.1} dB below 15 dB target",
        notch_suppression_db
    );
}

#[test]
fn test_flight_dynamics_simulator_attitude_and_bearing() {
    let drone_pos = Point3D::new(0.0, 0.0, -10.0); // 10m altitude
    let operator_pos = Point3D::new(20.0, 0.0, 0.0); // 20m forward

    let mut sim = FlightDynamicsSimulator::new(drone_pos, operator_pos);

    // Initial distance: sqrt(20^2 + 10^2) = sqrt(500) ≈ 22.36m
    let dist0 = sim.distance_to_operator();
    assert!((dist0 - 22.3606).abs() < 0.01);

    // Initial bearing: operator is straight ahead (+X), below drone (-Z in body)
    let (az0, el0) = sim.relative_bearing_to_operator();
    assert!(az0.abs() < 1e-4, "Initial azimuth should be 0 (straight ahead)");
    assert!(
        el0 < 0.0,
        "Initial elevation should be negative (pointing down to ground operator)"
    );
    let expected_el = (-10.0f32).atan2(20.0);
    assert!((el0 - expected_el).abs() < 1e-3);

    // Apply yaw rotation rate of +pi/2 rad/s (90 deg/s) for 1.0s
    sim = sim.with_rates(0.0, 0.0, PI / 2.0);
    sim.step(1.0);

    assert!((sim.yaw_rad - PI / 2.0).abs() < 1e-4);

    // With drone yawed 90 deg clockwise (facing East/+Y), stationary operator (at North/+X) is now to the left (port, -90 deg)
    let (az1, el1) = sim.relative_bearing_to_operator();
    assert!(
        (az1 - (-PI / 2.0)).abs() < 1e-3,
        "Yawed azimuth {} should be -pi/2 rad",
        az1
    );
    assert!((el1 - expected_el).abs() < 1e-3);

    // Translational velocity step forward in world (+X at 5 m/s)
    sim = sim.with_velocity(5.0, 0.0, 0.0).with_rates(0.0, 0.0, 0.0);
    sim.step(2.0); // Moved 10m closer

    let dist2 = sim.distance_to_operator();
    // New distance: sqrt(10^2 + 10^2) ≈ 14.14m
    assert!((dist2 - 14.1421).abs() < 0.05);
}

#[test]
fn test_moving_platform_kws_co_simulation() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    // Enroll keyword phrase "take off" with calibrated noise tolerance threshold
    let keyword = "take off";
    let ref_audio = engine.synthesize_speech_from_text(keyword);
    let ref_feats = engine.extract_features(&ref_audio);
    engine.enroll_keyword_banded(keyword, ref_feats, 11.5, 12);
    assert_eq!(engine.dtw().template_count(), 1);

    // Configure 4-microphone tetrahedral array
    let array = ArrayGeometry::Tetrahedral { radius: 0.06 };
    let mut tse_config = TseConfig::default();
    tse_config.motor_nulls_enabled = true;
    tse_config.min_freq_hz = 120.0;
    tse_config.max_freq_hz = 3800.0;
    tse_config.spatial_mask_threshold = 0.10;
    tse_config.attenuation_floor = 0.25;
    engine.enable_target_sound_extractor(array.clone(), tse_config.clone());

    // Quadcopter motor positions on airframe (X-configuration, arm length 0.22m)
    let motor_positions = vec![
        Point3D::new(0.155, 0.155, 0.0),   // Motor 1 (Front Right)
        Point3D::new(-0.155, 0.155, 0.0),  // Motor 2 (Rear Right)
        Point3D::new(-0.155, -0.155, 0.0), // Motor 3 (Rear Left)
        Point3D::new(0.155, -0.155, 0.0),  // Motor 4 (Front Left)
    ];
    let motor_rpms = vec![4800.0, 4820.0, 4790.0, 4810.0];
    engine.update_multi_motor_rpm(&motor_rpms);
    engine.set_tse_motor_null_positions(&motor_positions);

    // Synthesize target voice "take off"
    let target_audio = engine.synthesize_speech_from_text(keyword);
    assert!(!target_audio.is_empty());

    // Moving flight scenario: Drone at (5, 2, -8) flying forward at 2.5 m/s with yaw rate
    let drone_pos = Point3D::new(5.0, 2.0, -8.0);
    let operator_pos = Point3D::new(0.0, 0.0, 0.0);
    let mut sim = FlightDynamicsSimulator::new(drone_pos, operator_pos)
        .with_velocity(2.5, 0.0, 0.0)
        .with_rates(0.0, 0.0, 0.10);

    // Continuous audio stream: 0.4s background lead -> speech -> 0.6s trail
    let lead_samples = (0.4 * sample_rate) as usize;
    let trail_samples = (0.6 * sample_rate) as usize;
    let mut padded_speech = Vec::new();
    padded_speech.extend(vec![0.0f32; lead_samples]);
    padded_speech.extend_from_slice(&target_audio);
    padded_speech.extend(vec![0.0f32; trail_samples]);

    // Step simulation and update steered bearing
    sim.step(0.5);
    let (steered_az, steered_el) = sim.relative_bearing_to_operator();
    engine.update_target_bearing(steered_az, steered_el);

    // Synthesize multi-channel audio with target speech + 4 quadcopter motors
    let multi_channel_inputs = sim.synthesize_multi_channel_co_simulation(
        &array,
        &padded_speech,
        &motor_positions,
        &motor_rpms,
        sample_rate,
    );

    let channel_refs: Vec<&[f32]> = multi_channel_inputs.iter().map(|ch| &ch[..]).collect();

    // Ingest multi-channel audio through steered null-constrained MVDR into KWS
    let events = engine
        .ingest_multi_channel_tse(&channel_refs)
        .expect("TSE multi-channel ingestion failed");

    // Verify keyword detection
    let matched = events.iter().any(|e| e.keyword == keyword);
    assert!(
        matched,
        "Keyword '{}' must be spotted under moving platform co-simulation with rotor noise",
        keyword
    );

    let best_event = events.iter().find(|e| e.keyword == keyword).unwrap();
    assert!(
        best_event.confidence > 0.05,
        "Event confidence {} below expected 0.05 threshold",
        best_event.confidence
    );

    // Test rejection: pure multi-rotor noise without speech must trigger 0 false alarms
    engine.reset();
    let silence = vec![0.0f32; target_audio.len()];
    let noise_only_inputs = sim.synthesize_multi_channel_co_simulation(
        &array,
        &silence,
        &motor_positions,
        &motor_rpms,
        sample_rate,
    );
    let noise_refs: Vec<&[f32]> = noise_only_inputs.iter().map(|ch| &ch[..]).collect();
    let noise_events = engine
        .ingest_multi_channel_tse(&noise_refs)
        .expect("TSE noise ingestion failed");
    assert_eq!(
        noise_events.len(),
        0,
        "Pure quadcopter noise must produce 0 false alarms"
    );
}

#[test]
fn test_spatial_null_steering_streaming_throughput() {
    let sample_rate = 16000.0f32;
    let geometry = ArrayGeometry::Linear {
        spacing: 0.08,
        num_mics: 4,
    };
    let mut config = TseConfig::default();
    config.motor_nulls_enabled = true;
    let mut extractor = TargetSoundExtractor::new(geometry, sample_rate, config);
    extractor.set_spatial_null_directions(&[(PI / 4.0, 0.0), (-PI / 4.0, 0.0)]);

    let hop = 256;
    let block: Vec<f32> = (0..hop)
        .map(|i| (2.0 * PI * 440.0 * (i as f32 / sample_rate)).sin())
        .collect();
    let multi_inputs: Vec<&[f32]> = vec![&block; 4];
    let mut output = vec![0.0f32; hop];

    // Warmup
    for _ in 0..10 {
        let _ = extractor.process_block(&multi_inputs, &mut output, 0.0);
    }

    let iterations = 1000;
    let start = Instant::now();
    for i in 0..iterations {
        let timestamp = i as f64 * (hop as f64 / sample_rate as f64);
        let _ = extractor.process_block(&multi_inputs, &mut output, timestamp);
    }
    let elapsed = start.elapsed();

    let total_samples = iterations * hop;
    let throughput = (total_samples as f64) / elapsed.as_secs_f64();
    let rtf = throughput / (sample_rate as f64);

    // Must exceed 50,000 samples/sec (> 3x real-time in unoptimized debug test profile, > 6x in release)
    let min_target = if cfg!(debug_assertions) { 50_000.0 } else { 100_000.0 };
    assert!(
        throughput > min_target,
        "Throughput {:.0} samples/sec ({:.1}x real-time) is below target",
        throughput,
        rtf
    );
}
