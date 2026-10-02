//! Phase 6 Verification Test Suite: Drone Acoustic Health Monitoring & Propeller Anomaly Diagnostics.

#![deny(unsafe_code)]

use sonon::engine::SononEngine;
use sonon::health::{
    AcousticHealthMonitor, AnomalySeverity, MavlinkNamedValueFloat, MotorHealthConfig,
};
use std::f32::consts::PI;
use std::time::Instant;

/// Generate synthetic acoustic signal for a 2-blade rotor at specified RPM.
///
/// If `damaged_blade` is true, an asymmetric aerodynamic impulse is injected at
/// the shaft rotational frequency (RPM / 60) and its sidebands.
fn generate_rotor_signal(
    sample_rate: f32,
    num_samples: usize,
    rpm: f32,
    bpf_amplitude: f32,
    rot_subharmonic_amplitude: f32,
    bearing_noise_amplitude: f32,
) -> Vec<f32> {
    let rot_freq = rpm / 60.0;
    let bpf_freq = 2.0 * rot_freq;

    (0..num_samples)
        .map(|i| {
            let t = (i as f32) / sample_rate;
            // Primary BPF tone and second harmonic
            let bpf = bpf_amplitude * (2.0 * PI * bpf_freq * t).sin()
                + 0.3 * bpf_amplitude * (2.0 * PI * 2.0 * bpf_freq * t).sin();

            // Rotational subharmonic (asymmetric blade impulse)
            let rot = rot_subharmonic_amplitude * (2.0 * PI * rot_freq * t).sin()
                + 0.5 * rot_subharmonic_amplitude * (2.0 * PI * (bpf_freq - rot_freq) * t).sin();

            // Bearing friction / high-frequency vibration (5 kHz + 6 kHz tones + random-like phase)
            let bearing = bearing_noise_amplitude
                * ((2.0 * PI * 5200.0 * t).sin() + 0.8 * (2.0 * PI * 6400.0 * t).sin());

            bpf + rot + bearing
        })
        .collect()
}

#[test]
fn test_pristine_rotor_health_evaluation() {
    let sample_rate = 16000.0;
    let fft_size = 1024;
    let config = MotorHealthConfig::default();
    let mut monitor = AcousticHealthMonitor::new(sample_rate, fft_size, config);

    // Motor running at 6000 RPM (Rot = 100 Hz, BPF = 200 Hz).
    // Pristine condition: High BPF (0.8), zero subharmonic (0.001), zero bearing noise (0.001).
    let rpm = 6000.0;
    monitor.update_motor_rpm(0, rpm);

    let signal = generate_rotor_signal(sample_rate, fft_size, rpm, 0.8, 0.001, 0.001);
    let snapshot = monitor.analyze_frame(&signal, 0.0);

    assert_eq!(
        snapshot.motor_reports.len(),
        1,
        "Should generate 1 motor report"
    );
    let report = &snapshot.motor_reports[0];

    assert_eq!(report.motor_id, 0);
    assert!((report.bpf_hz - 200.0).abs() < 1e-3);
    assert!((report.rot_hz - 100.0).abs() < 1e-3);
    assert!(
        report.imbalance_ratio < 0.10,
        "Pristine rotor imbalance ratio must be low, got {}",
        report.imbalance_ratio
    );
    assert_eq!(
        report.severity,
        AnomalySeverity::Normal,
        "Pristine rotor must evaluate to Normal severity"
    );
    assert!(
        snapshot.overall_health_score > 0.90,
        "Overall health score should be > 0.90, got {}",
        snapshot.overall_health_score
    );
}

#[test]
fn test_chipped_blade_imbalance_anomaly_detection() {
    let sample_rate = 16000.0;
    let fft_size = 1024;
    let config = MotorHealthConfig::default();
    let mut monitor = AcousticHealthMonitor::new(sample_rate, fft_size, config);

    let rpm = 6000.0;
    monitor.update_motor_rpm(0, rpm);

    // Chipped / damaged blade: High subharmonic at 100 Hz (0.7) relative to BPF (0.8).
    let damaged_signal = generate_rotor_signal(sample_rate, fft_size, rpm, 0.8, 0.7, 0.01);
    let snapshot = monitor.analyze_frame(&damaged_signal, 1.25);

    let report = &snapshot.motor_reports[0];
    println!(
        "Chipped blade diagnosis - Imbalance Ratio: {:.3}, Severity: {:?}, Health Score: {:.3}",
        report.imbalance_ratio, report.severity, snapshot.overall_health_score
    );

    assert!(
        report.imbalance_ratio >= 0.45,
        "Damaged blade must produce imbalance ratio >= 0.45, got {}",
        report.imbalance_ratio
    );
    assert!(
        report.severity >= AnomalySeverity::Warning,
        "Damaged blade must trigger Warning or Critical severity"
    );
    assert!(
        snapshot.overall_health_score < 0.60,
        "Health score must decline under blade damage, got {}",
        snapshot.overall_health_score
    );
}

#[test]
fn test_motor_bearing_wear_and_kurtosis_detection() {
    let sample_rate = 16000.0;
    let fft_size = 1024;
    let config = MotorHealthConfig::default();
    let mut monitor = AcousticHealthMonitor::new(sample_rate, fft_size, config);

    let rpm = 6000.0;
    monitor.update_motor_rpm(0, rpm);

    // Severe bearing wear: dominant high-frequency friction acoustic emission at 5.2 kHz
    let bearing_signal = generate_rotor_signal(sample_rate, fft_size, rpm, 0.3, 0.01, 0.9);
    let snapshot = monitor.analyze_frame(&bearing_signal, 2.5);

    let report = &snapshot.motor_reports[0];
    println!(
        "Bearing wear diagnosis - Friction Ratio: {:.3}, Kurtosis: {:.3}, Severity: {:?}",
        report.bearing_friction_ratio, report.spectral_kurtosis, report.severity
    );

    assert!(
        report.bearing_friction_ratio >= 0.35,
        "Bearing wear must produce friction ratio >= 0.35, got {}",
        report.bearing_friction_ratio
    );
    assert!(
        report.severity >= AnomalySeverity::Warning,
        "Bearing wear must trigger Warning or Critical severity"
    );
}

#[test]
fn test_quadcopter_multi_motor_differentiation() {
    let sample_rate = 16000.0;
    let fft_size = 2048;
    let config = MotorHealthConfig::default();
    let mut monitor = AcousticHealthMonitor::new(sample_rate, fft_size, config);

    // 4 Motors on a quadcopter with realistic differential flight speeds:
    // Motor 0: 4500 RPM (Rot = 75 Hz, BPF = 150 Hz)
    // Motor 1: 5400 RPM (Rot = 90 Hz, BPF = 180 Hz)
    // Motor 2: 6300 RPM (Rot = 105 Hz, BPF = 210 Hz) - DAMAGED BLADE
    // Motor 3: 7200 RPM (Rot = 120 Hz, BPF = 240 Hz)
    let rpms = vec![4500.0, 5400.0, 6300.0, 7200.0];
    monitor.update_motor_rpms(&rpms);

    // Synthesize composite acoustic signal: Motor 0, 1, 3 are pristine, Motor 2 has chipped blade
    let mut composite_signal = vec![0.0f32; fft_size];
    for (id, &rpm) in rpms.iter().enumerate() {
        let (sub_amp, bpf_amp) = if id == 2 {
            (0.8f32, 0.7f32) // Motor 2 has blade defect
        } else {
            (0.01f32, 0.7f32) // Pristine
        };

        let motor_sig = generate_rotor_signal(sample_rate, fft_size, rpm, bpf_amp, sub_amp, 0.01);
        for i in 0..fft_size {
            composite_signal[i] += motor_sig[i];
        }
    }

    let snapshot = monitor.analyze_frame(&composite_signal, 3.0);
    assert_eq!(snapshot.motor_reports.len(), 4);

    // Motors 0, 1, 3 should be Normal
    assert_eq!(
        snapshot.motor_reports[0].severity,
        AnomalySeverity::Normal,
        "Motor 0 must be Normal"
    );
    assert_eq!(
        snapshot.motor_reports[1].severity,
        AnomalySeverity::Normal,
        "Motor 1 must be Normal"
    );
    assert_eq!(
        snapshot.motor_reports[3].severity,
        AnomalySeverity::Normal,
        "Motor 3 must be Normal"
    );

    // Motor 2 must be flagged with anomaly
    assert!(
        snapshot.motor_reports[2].severity >= AnomalySeverity::Warning,
        "Motor 2 must be flagged with Warning or Critical, got {:?}",
        snapshot.motor_reports[2].severity
    );
    assert_eq!(snapshot.worst_severity, snapshot.motor_reports[2].severity);
}

#[test]
fn test_mavlink_telemetry_packet_generation() {
    let sample_rate = 16000.0;
    let fft_size = 1024;
    let config = MotorHealthConfig::default();
    let mut monitor = AcousticHealthMonitor::new(sample_rate, fft_size, config);

    monitor.update_motor_rpms(&[5500.0, 5500.0]);
    let signal = generate_rotor_signal(sample_rate, fft_size, 5500.0, 0.8, 0.05, 0.05);
    let snapshot = monitor.analyze_frame(&signal, 0.5);

    let time_boot_ms = 45200;
    let packets = monitor.generate_mavlink_telemetry(&snapshot, time_boot_ms);

    // Should generate: SONON_HLTH, SONON_STAT, SON_IMB1, SON_BRG1, SON_IMB2, SON_BRG2
    assert_eq!(packets.len(), 6, "Expected 6 MAVLink packets for 2 motors");

    assert_eq!(packets[0].time_boot_ms, time_boot_ms);
    assert_eq!(packets[0].name_as_str(), "SONON_HLTH");
    assert!((packets[0].value - snapshot.overall_health_score).abs() < 1e-4);

    assert_eq!(packets[1].name_as_str(), "SONON_STAT");
    assert_eq!(packets[1].value, AnomalySeverity::Normal.as_code());

    assert_eq!(packets[2].name_as_str(), "SON_IMB1");
    assert_eq!(packets[3].name_as_str(), "SON_BRG1");
    assert_eq!(packets[4].name_as_str(), "SON_IMB2");
    assert_eq!(packets[5].name_as_str(), "SON_BRG2");

    // Verify string roundtrip
    let custom = MavlinkNamedValueFloat::new(100, "VEX_DIAG", 42.5);
    assert_eq!(custom.name_as_str(), "VEX_DIAG");
    assert_eq!(custom.value, 42.5);
}

#[test]
fn test_integrated_sonon_engine_health_throughput() {
    let sample_rate = 16000.0;
    let frame_size = 512;
    let hop_size = 160;
    let num_mfcc = 13;

    let mut engine = SononEngine::new(sample_rate, frame_size, hop_size, num_mfcc);
    engine.enable_rotor_notch(2, 3, 30.0);
    engine.update_motor_rpm(6000.0);
    engine.enable_health_monitoring(MotorHealthConfig::default());

    // Generate 10 seconds of streaming audio (160,000 samples)
    let total_samples = 160_000;
    let stream_audio = generate_rotor_signal(sample_rate, total_samples, 6000.0, 0.6, 0.05, 0.02);

    let start = Instant::now();
    let _ = engine.ingest_samples(&stream_audio);
    let elapsed = start.elapsed();

    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / sample_rate as f64;

    println!(
        "SononEngine with Rotor Notch + Acoustic Health Monitoring Throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    // Verify health snapshot was evaluated
    let latest_snapshot = engine.latest_health_snapshot();
    assert!(
        latest_snapshot.is_some(),
        "SononEngine must evaluate latest health snapshot"
    );
    let snapshot = latest_snapshot.unwrap();
    assert_eq!(snapshot.worst_severity, AnomalySeverity::Normal);
    assert!(snapshot.overall_health_score > 0.85);

    assert!(
        samples_per_sec > 500_000.0,
        "Engine throughput must exceed 500,000 samples/sec with health monitoring, got {samples_per_sec:.0}"
    );
}
