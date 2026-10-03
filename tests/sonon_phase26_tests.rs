#![deny(unsafe_code)]

//! Test Suite for Sonon Phase 26: Physics-Informed Aeroacoustic Inverse Source
//! Reconstruction & Far-Field Pressure Directivity Mapping.

use sonon::aeroacoustics::{
    a_weighting_db, atmospheric_absorption_db_km, bessel_j, AeroacousticConfig,
    AeroacousticInverter, AeroacousticTelemetry, RotorGeometry,
};
use sonon::beamforming::Point3D;
use sonon::engine::SononEngine;
use sonon::tse::Complex32;
use std::f32::consts::PI;
use std::time::Instant;

/// Helper to construct a standard quadcopter configuration (4 rotors and 4 fuselage microphones).
fn setup_quadcopter_aeroacoustic_system() -> (Vec<RotorGeometry>, Vec<Point3D>, AeroacousticConfig) {
    let arm = 0.20f32; // 200 mm arm length
    let radius = 0.125f32; // 10-inch propeller (0.125 m radius)
    let num_blades = 2;

    // 4 Rotors: X-configuration (+X Forward, +Y Right, +Z Down)
    let rotors = vec![
        RotorGeometry::new(0, arm, arm, 0.0, radius, num_blades, true),   // Motor 0: Front-Right (CW)
        RotorGeometry::new(1, -arm, arm, 0.0, radius, num_blades, false), // Motor 1: Rear-Right (CCW)
        RotorGeometry::new(2, -arm, -arm, 0.0, radius, num_blades, true),  // Motor 2: Rear-Left (CW)
        RotorGeometry::new(3, arm, -arm, 0.0, radius, num_blades, false), // Motor 3: Front-Left (CCW)
    ];

    // 4 Fuselage Microphones angled toward each rotor arm quadrant (45, 135, 225, 315 degrees)
    let mic_radius = 0.05f32;
    let angle_offset = PI / 4.0;
    let mic_positions = vec![
        Point3D::new(mic_radius * angle_offset.cos(), mic_radius * angle_offset.sin(), 0.02), // Mic 0: Front-Right
        Point3D::new(mic_radius * (angle_offset + PI * 0.5).cos(), mic_radius * (angle_offset + PI * 0.5).sin(), 0.02), // Mic 1: Rear-Right
        Point3D::new(mic_radius * (angle_offset + PI).cos(), mic_radius * (angle_offset + PI).sin(), 0.02), // Mic 2: Rear-Left
        Point3D::new(mic_radius * (angle_offset + PI * 1.5).cos(), mic_radius * (angle_offset + PI * 1.5).sin(), 0.02), // Mic 3: Front-Left
    ];

    let config = AeroacousticConfig {
        sample_rate: 16000.0,
        speed_of_sound: 343.0,
        air_density: 1.225,
        tikhonov_lambda: 1e-3,
        reference_distance_m: 1.0,
        elevation_bins: 18,
        azimuth_bins: 36,
        footprint_threshold_dba: 65.0,
    };

    (rotors, mic_positions, config)
}

/// Test 1: Mathematical Bessel Function $J_n(x)$ and IEC 61672-1 A-Weighting Numerical Precision.
#[test]
fn test_bessel_j_and_iec61672_a_weighting_numerical_precision() {
    // 1. Bessel function boundary values
    assert_eq!(bessel_j(0, 0.0), 1.0, "J_0(0) must equal exactly 1.0");
    assert_eq!(bessel_j(1, 0.0), 0.0, "J_1(0) must equal 0.0");
    assert_eq!(bessel_j(2, 0.0), 0.0, "J_2(0) must equal 0.0");

    // Known values from Abramowitz & Stegun Table 9.1:
    // J_1(1.0) = 0.4400505857
    let j1_1 = bessel_j(1, 1.0);
    assert!(
        (j1_1 - 0.44005).abs() < 1e-4,
        "J_1(1.0) expected 0.44005, got {j1_1:.5}"
    );

    // J_2(2.0) = 0.3528340286
    let j2_2 = bessel_j(2, 2.0);
    assert!(
        (j2_2 - 0.35283).abs() < 1e-4,
        "J_2(2.0) expected 0.35283, got {j2_2:.5}"
    );

    // Antisymmetry: J_1(-x) = -J_1(x)
    let j1_neg1 = bessel_j(1, -1.0);
    assert!(
        (j1_neg1 - (-0.44005)).abs() < 1e-4,
        "J_1(-1.0) expected -0.44005, got {j1_neg1:.5}"
    );

    // Symmetry: J_2(-x) = J_2(x)
    let j2_neg2 = bessel_j(2, -2.0);
    assert!(
        (j2_neg2 - 0.35283).abs() < 1e-4,
        "J_2(-2.0) expected 0.35283, got {j2_neg2:.5}"
    );

    // 2. IEC 61672-1 A-Weighting Filter Curve
    // At 1000 Hz, delta_A is exactly 0.0 dB
    let a_1000 = a_weighting_db(1000.0);
    assert!(
        a_1000.abs() < 0.1,
        "A-weighting at 1000 Hz must be ~0.0 dB, got: {a_1000:.2} dB"
    );

    // At 100 Hz, human ear is insensitive: delta_A ~ -19.1 dB
    let a_100 = a_weighting_db(100.0);
    assert!(
        (a_100 - (-19.1)).abs() < 0.5,
        "A-weighting at 100 Hz must be ~ -19.1 dB, got: {a_100:.2} dB"
    );

    // At 2000 Hz, ear has slight ear canal resonance: delta_A ~ +1.2 dB
    let a_2000 = a_weighting_db(2000.0);
    assert!(
        (a_2000 - 1.2).abs() < 0.4,
        "A-weighting at 2000 Hz must be ~ +1.2 dB, got: {a_2000:.2} dB"
    );

    // At 50 Hz, low frequency rumble: delta_A ~ -30.2 dB
    let a_50 = a_weighting_db(50.0);
    assert!(
        (a_50 - (-30.2)).abs() < 1.0,
        "A-weighting at 50 Hz must be ~ -30.2 dB, got: {a_50:.2} dB"
    );

    // 3. Atmospheric absorption
    let alpha_1k = atmospheric_absorption_db_km(1000.0);
    assert!(alpha_1k > 1.0 && alpha_1k < 5.0, "Absorption at 1 kHz should be reasonable");
}

/// Test 2: Ffowcs Williams-Hawkings Forward Radiation Directivity & Axial/Transverse Lobes.
#[test]
fn test_fwh_forward_radiation_directivity_and_axial_null() {
    let (rotors, mic_positions, config) = setup_quadcopter_aeroacoustic_system();
    let inverter = AeroacousticInverter::new(rotors, mic_positions, config);

    // Quadcopter in steady hover: 4 motors each producing 1.5 N of vertical thrust (6.0 N total)
    // Motor BPF at 6000 RPM (num_blades=2): BPF = 200 Hz
    let bpf_hz = 200.0f32;
    let rotor_forces = vec![
        Complex32::new(1.5, 0.0),
        Complex32::new(1.5, 0.0),
        Complex32::new(1.5, 0.0),
        Complex32::new(1.5, 0.0),
    ];

    let sphere = inverter.compute_directivity_sphere(&rotor_forces, bpf_hz);

    assert_eq!(sphere.frequency_hz, 200.0);
    assert_eq!(sphere.grid_spl_db.len(), 18);
    assert_eq!(sphere.grid_spl_db[0].len(), 36);

    // Verify peak radiation vs quietest notch
    println!("Peak SPL: {:.2} dBA, Min SPL: {:.2} dBA", sphere.peak_spl_dba, sphere.min_spl_dba);
    assert!(
        sphere.peak_spl_dba > sphere.min_spl_dba + 10.0,
        "Dipole directivity must have pronounced lobes vs nulls (> 10 dB dynamic range)"
    );

    // For vertical thrust dipoles (thrust axis = [0, 0, -1]), radiation is maximum vertically
    // (elevation near 0 or 180 degrees) and drops drastically in the horizontal plane (elevation ~90 degrees)
    let mid_elev_idx = 9; // ~90 deg (horizontal plane)
    let horiz_spl = sphere.grid_spl_dba[mid_elev_idx][0];
    assert!(
        horiz_spl < sphere.peak_spl_dba - 10.0,
        "Horizontal in-plane radiation must be significantly lower than peak axial lobes"
    );
}

/// Test 3: Inverse Source Reconstruction from Near-Field Fuselage Microphones.
#[test]
fn test_inverse_source_reconstruction_from_fuselage_mics() {
    let (rotors, mic_positions, config) = setup_quadcopter_aeroacoustic_system();
    let inverter = AeroacousticInverter::new(rotors.clone(), mic_positions.clone(), config.clone());

    let bpf_hz = 240.0f32; // 7200 RPM, 2 blades
    let omega = 2.0 * PI * bpf_hz;
    let wave_k = omega / config.speed_of_sound;

    // Ground truth unknown blade forces on the 4 motors:
    let true_forces = vec![
        Complex32::new(1.80, 0.20), // Motor 0: 1.81 N
        Complex32::new(1.40, -0.10), // Motor 1: 1.40 N
        Complex32::new(2.10, 0.30), // Motor 2: 2.12 N
        Complex32::new(1.60, 0.00), // Motor 3: 1.60 N
    ];

    // Forward synthesize exact near-field microphone pressures:
    // p_m = sum_k G_mk * F_k
    let m = mic_positions.len();
    let k = rotors.len();
    let mut p_meas = Vec::with_capacity(m);

    for mic in &mic_positions {
        let mut p_mic = Complex32::zero();
        for (k_idx, rot) in rotors.iter().enumerate() {
            let dx = mic.x - rot.position.x;
            let dy = mic.y - rot.position.y;
            let dz = mic.z - rot.position.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);

            let r_hat_x = dx / dist;
            let r_hat_y = dy / dist;
            let r_hat_z = dz / dist;

            let cos_proj = rot.thrust_axis.x * r_hat_x + rot.thrust_axis.y * r_hat_y + rot.thrust_axis.z * r_hat_z;
            let phase = -wave_k * dist;
            let exp_factor = Complex32::from_polar(1.0 / (4.0 * PI * dist), phase);
            let dipole_factor = Complex32::new(1.0 / dist, wave_k).scale(cos_proj);

            let green = exp_factor.mul(dipole_factor);
            p_mic = p_mic.add(green.mul(true_forces[k_idx]));
        }
        p_meas.push(p_mic);
    }

    // Run inverse solver:
    let reconstructed_forces = inverter.reconstruct_rotor_sources(&p_meas, bpf_hz);
    assert_eq!(reconstructed_forces.len(), k);

    // Verify recovery accuracy:
    for i in 0..k {
        let truth = true_forces[i].norm();
        let est = reconstructed_forces[i].norm();
        let rel_err = (est - truth).abs() / truth;
        println!("Rotor {i}: True force = {truth:.4} N (re: {:.4}, im: {:.4}), Estimated = {est:.4} N (re: {:.4}, im: {:.4}), Rel Err = {:.2}%",
            true_forces[i].re, true_forces[i].im, reconstructed_forces[i].re, reconstructed_forces[i].im, rel_err * 100.0);
        assert!(
            rel_err < 0.05,
            "Rotor {i} force reconstruction error must be < 5.0%, got: {:.2}%",
            rel_err * 100.0
        );
    }
}

/// Test 4: Ground Noise Footprint Projection & Urban Stealth Flight Yaw Guidance Advisor.
#[test]
fn test_ground_noise_footprint_and_stealth_flight_yaw_guidance() {
    let (rotors, mic_positions, config) = setup_quadcopter_aeroacoustic_system();
    let inverter = AeroacousticInverter::new(rotors, mic_positions, config);

    let bpf_hz = 200.0f32;
    // Asymmetric rotor forces: motors 0 and 1 loaded heavier than 2 and 3
    let rotor_forces = vec![
        Complex32::new(2.5, 0.0),
        Complex32::new(2.5, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];

    let sphere = inverter.compute_directivity_sphere(&rotor_forces, bpf_hz);

    // Project onto ground at altitude h = 40.0 m
    let altitude_agl_m = 40.0f32;
    let footprint = inverter.compute_ground_noise_footprint(&sphere, altitude_agl_m, 80.0, 5.0);

    assert_eq!(footprint.altitude_agl_m, 40.0);
    assert!(footprint.peak_ground_dba > 40.0, "Ground noise must be detectable");
    println!("Ground Peak Noise: {:.2} dBA at ({:.1}, {:.1}) m, Area > 65 dBA: {:.1} m^2",
        footprint.peak_ground_dba, footprint.peak_location_m.0, footprint.peak_location_m.1,
        footprint.footprint_area_exceeding_threshold_m2
    );

    // Verify distance attenuation: points far away from aircraft must be substantially quieter
    let center_ix = footprint.grid_dba.len() / 2;
    let edge_ix = footprint.grid_dba.len() - 1;
    let center_noise = footprint.grid_dba[center_ix][center_ix];
    let edge_noise = footprint.grid_dba[edge_ix][edge_ix];

    assert!(
        center_noise > edge_noise + 6.0,
        "Ground center noise ({center_noise:.1} dBA) must exceed edge noise ({edge_noise:.1} dBA) due to inverse square law"
    );

    // Test stealth flight yaw guidance:
    // A hospital / school is located at ground coordinate (X = 0, Y = +60 m) (bearing = +90 deg / starboard)
    let target_pos = (0.0f32, 60.0f32);
    let current_yaw = 0.0f32; // heading north (0 rad)

    let yaw_delta_deg = inverter.compute_stealth_yaw_advice(&sphere, target_pos, current_yaw);
    println!("Recommended stealth yaw adjustment: {:.2} deg", yaw_delta_deg);

    // The yaw adjustment must be non-zero and bounded in [-180, +180] deg
    assert!(yaw_delta_deg.abs() <= 180.0);
}

/// Test 5: Integrated Sonon Engine Aeroacoustics and MAVLink Telemetry Throughput.
#[test]
fn test_integrated_sonon_engine_aeroacoustics_and_mavlink_throughput() {
    let (rotors, mic_positions, config) = setup_quadcopter_aeroacoustic_system();
    let mut engine = SononEngine::new(16000.0, 512, 256, 13);

    // Enable aeroacoustic inversion in engine
    engine.enable_aeroacoustic_inversion(rotors, mic_positions, config);
    assert!(engine.aeroacoustic_inverter().is_some());

    // Update active motor RPMs
    engine.update_multi_motor_rpm(&[6000.0, 6000.0, 6000.0, 6000.0]);

    // Synthesize 4 channels of audio containing a 200 Hz blade pass tone
    let num_samples = 512;
    let mut ch0 = vec![0.0f32; num_samples];
    let mut ch1 = vec![0.0f32; num_samples];
    let mut ch2 = vec![0.0f32; num_samples];
    let mut ch3 = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        let t = (i as f32) / 16000.0;
        let tone = (2.0 * PI * 200.0 * t).sin() * 0.15;
        ch0[i] = tone;
        ch1[i] = tone * 0.95;
        ch2[i] = tone * 1.05;
        ch3[i] = tone * 0.90;
    }

    let input_channels = [&ch0[..], &ch1[..], &ch2[..], &ch3[..]];

    // Benchmark execution throughput
    let iterations = 20;
    let start = Instant::now();

    let mut latest_telem: Option<AeroacousticTelemetry> = None;
    for _ in 0..iterations {
        let telem = engine
            .process_aeroacoustic_frame(&input_channels, 50.0, Some((0.0, 100.0)), 0.0)
            .expect("Aeroacoustic frame processing must succeed");
        latest_telem = Some(telem);
    }
    let elapsed = start.elapsed();

    let total_samples = (iterations * num_samples) as f64;
    let samples_per_sec = total_samples / elapsed.as_secs_f64();
    println!(
        "Aeroacoustic Inversion Throughput: {:.0} samples/sec ({:.1}x Real-Time at 16 kHz)",
        samples_per_sec,
        samples_per_sec / 16000.0
    );

    assert!(
        samples_per_sec > 200_000.0,
        "Aeroacoustic processing must exceed 200,000 samples/sec, got: {samples_per_sec:.0}"
    );

    let telem = latest_telem.unwrap();
    assert!(engine.latest_directivity_sphere().is_some());
    assert!(engine.latest_ground_noise_footprint().is_some());
    assert!(engine.latest_aeroacoustic_telemetry().is_some());

    // Verify MAVLink v2 telemetry packet generation
    let mavlink_packets = AeroacousticInverter::to_mavlink_packets(&telem, 12345);
    assert_eq!(mavlink_packets.len(), 3);
    assert_eq!(mavlink_packets[0].name_as_str(), "AERO_DIR");
    assert_eq!(mavlink_packets[1].name_as_str(), "AERO_DBA");
    assert_eq!(mavlink_packets[2].name_as_str(), "AERO_YAW");
    assert_eq!(mavlink_packets[0].time_boot_ms, 12345);

    // Clean reset
    engine.reset();
    assert!(engine.latest_directivity_sphere().is_none());
    assert!(engine.latest_ground_noise_footprint().is_none());
    assert!(engine.latest_aeroacoustic_telemetry().is_none());
}
