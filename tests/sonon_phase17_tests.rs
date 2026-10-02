//! Phase 17 analytical test suite: Continuous Wavelet Transform (CWT)
//! non-stationary rotor micro-damage profiler, multi-resolution vibration decomposition,
//! and autonomous MAVLink telemetry emission.

#![deny(unsafe_code)]

use sonon::cwt::{
    ContinuousWaveletFilterbank, CwtProfilerConfig, RotorDamageProfiler, WaveletType,
};
use sonon::engine::SononEngine;
use sonon::health::AnomalySeverity;
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_morlet_wavelet_filterbank_and_frequency_mapping() {
    let sample_rate = 16000.0f32;
    let num_scales = 16;
    let min_freq = 100.0;
    let max_freq = 7500.0;

    let fb = ContinuousWaveletFilterbank::new(
        WaveletType::ComplexMorlet { omega_0: 6.0 },
        num_scales,
        min_freq,
        max_freq,
        sample_rate,
    );

    assert_eq!(fb.frequencies().len(), num_scales);
    assert_eq!(fb.scales().len(), num_scales);
    assert_eq!(fb.wavelet_type(), WaveletType::ComplexMorlet { omega_0: 6.0 });

    // Frequencies should be monotonically increasing from min_freq to max_freq
    let freqs = fb.frequencies();
    assert!((freqs[0] - min_freq).abs() < 5.0, "Lowest frequency must match ~100 Hz, got {}", freqs[0]);
    assert!((freqs[num_scales - 1] - max_freq).abs() < 5.0, "Highest frequency must match ~7500 Hz, got {}", freqs[num_scales - 1]);

    for i in 1..num_scales {
        assert!(
            freqs[i] > freqs[i - 1],
            "Frequencies must monotonically increase with index"
        );
    }

    // Scales must be inversely proportional to frequencies: a_0 > a_1 > ...
    let scales = fb.scales();
    for i in 1..num_scales {
        assert!(
            scales[i] < scales[i - 1],
            "Scales must monotonically decrease as frequency increases"
        );
    }
}

#[test]
fn test_mexican_hat_wavelet_zero_dc_bias() {
    let sample_rate = 16000.0f32;
    let fb = ContinuousWaveletFilterbank::new(
        WaveletType::MexicanHat,
        8,
        200.0,
        6000.0,
        sample_rate,
    );

    assert_eq!(fb.wavelet_type(), WaveletType::MexicanHat);

    // Compute CWT of a pure DC signal (constant 1.0)
    let dc_signal = vec![1.0f32; 512];
    let scalogram = fb.transform(&dc_signal);

    // Mexican Hat has zero DC mean (integral = 0), so response to DC must be near zero in interior
    for s in 0..8 {
        let interior_mags = &scalogram.magnitudes[s][150..350];
        let max_dc_response = interior_mags.iter().fold(0.0f32, |acc, &x| acc.max(x));
        assert!(
            max_dc_response < 1e-4,
            "Mexican Hat response to pure DC must be near zero, got {} at scale {}",
            max_dc_response,
            s
        );
    }
}

#[test]
fn test_cwt_scalogram_synthetic_transient_impulse_detection() {
    let sample_rate = 16000.0f32;
    let n_samples = 256;

    let fb = ContinuousWaveletFilterbank::new(
        WaveletType::MexicanHat,
        16,
        150.0,
        7000.0,
        sample_rate,
    );

    // Baseline: low-amplitude Gaussian-like noise (sine sum)
    let mut signal = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = (i as f32) / sample_rate;
        let val = 0.05 * (2.0 * PI * 300.0 * t).sin() + 0.03 * (2.0 * PI * 800.0 * t).sin();
        signal.push(val);
    }

    let baseline_scalogram = fb.transform(&signal);
    let (_idx, baseline_kurt, _freq) = baseline_scalogram.peak_kurtosis();
    assert!(
        baseline_kurt < 4.5,
        "Stationary signal must have low kurtosis, got {}",
        baseline_kurt
    );

    // Inject sharp micro-crack transient impulse at sample 128 (impact strike)
    signal[128] += 1.5;
    signal[129] -= 1.0;

    let transient_scalogram = fb.transform(&signal);
    let (best_idx, transient_kurt, peak_freq) = transient_scalogram.peak_kurtosis();

    assert!(
        transient_kurt > 6.0,
        "Impulsive micro-crack shock must drive peak kurtosis > 6.0, got {}",
        transient_kurt
    );
    assert!(
        peak_freq > 1000.0,
        "Micro-crack transient should peak in high-frequency wavelet bands, got {} Hz at scale {}",
        peak_freq,
        best_idx
    );
}

#[test]
fn test_rotor_damage_profiler_blade_crack_and_flutter_detection() {
    let sample_rate = 16000.0f32;
    let config = CwtProfilerConfig::default();
    let mut profiler = RotorDamageProfiler::new(sample_rate, config, 2);

    // Set motor RPM to 3600 (BPF = 2 * 3600 / 60 = 120 Hz)
    profiler.update_rpm(3600.0);

    // 1. Pristine nominal rotor acoustic frame (nominal BPF tone with gentle harmonics)
    let n_samples = 512;
    let mut pristine_frame = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = (i as f32) / sample_rate;
        let s = 0.2 * (2.0 * PI * 120.0 * t).sin() + 0.05 * (2.0 * PI * 240.0 * t).sin();
        pristine_frame.push(s);
    }

    let pristine_report = profiler.analyze_frame(&pristine_frame, 0.0);
    assert_eq!(
        pristine_report.severity,
        AnomalySeverity::Normal,
        "Pristine rotor must evaluate to Normal severity"
    );
    assert!(
        pristine_report.airframe_fatigue_index < 0.25,
        "Pristine airframe fatigue index must be < 0.25, got {}",
        pristine_report.airframe_fatigue_index
    );
    assert!(
        pristine_report.max_kurtosis < 4.5,
        "Pristine rotor kurtosis must be low, got {}",
        pristine_report.max_kurtosis
    );

    // 2. Damaged rotor frame: simulate periodic blade micro-crack acoustic clicks and high-frequency spalling
    let mut damaged_frame = pristine_frame.clone();
    // Inject periodic micro-crack transient spikes corresponding to blade rotation (every 133 samples ~ 120 Hz)
    for k in (50..n_samples).step_by(133) {
        damaged_frame[k] += 1.8;
        damaged_frame[k + 1] -= 1.2;
    }
    // Inject ultrasonic/high-frequency crack emission (> 3500 Hz)
    for i in 0..n_samples {
        let t = (i as f32) / sample_rate;
        damaged_frame[i] += 0.4 * (2.0 * PI * 4200.0 * t).sin();
    }

    let damaged_report = profiler.analyze_frame(&damaged_frame, 1.0);
    assert!(
        damaged_report.severity >= AnomalySeverity::Warning,
        "Damaged rotor must escalate to Warning or Critical severity, got {:?}",
        damaged_report.severity
    );
    assert!(
        damaged_report.airframe_fatigue_index > 0.45,
        "Damaged airframe fatigue index must exceed 0.45, got {}",
        damaged_report.airframe_fatigue_index
    );
    assert!(
        damaged_report.max_kurtosis > 5.0,
        "Damaged rotor peak kurtosis must exceed 5.0, got {}",
        damaged_report.max_kurtosis
    );
    assert!(
        damaged_report.high_freq_crack_energy_ratio > 0.10,
        "High-frequency crack energy ratio must be elevated, got {}",
        damaged_report.high_freq_crack_energy_ratio
    );
}

#[test]
fn test_mavlink_telemetry_packet_generation() {
    let sample_rate = 16000.0f32;
    let profiler = RotorDamageProfiler::new(sample_rate, CwtProfilerConfig::default(), 2);

    let dummy_frame = vec![0.1f32; 256];
    let report = profiler.analyze_frame(&dummy_frame, 2.5);

    let packets = report.generate_mavlink_telemetry(998877);
    assert_eq!(packets.len(), 4, "Must generate exactly 4 MAVLink telemetry packets");

    let names: Vec<&str> = packets.iter().map(|p| p.name_as_str()).collect();
    assert!(names.contains(&"FATIGUE"));
    assert!(names.contains(&"FLUTTER"));
    assert!(names.contains(&"CWT_KURT"));
    assert!(names.contains(&"CRACK_ENG"));

    for pkt in &packets {
        assert_eq!(pkt.time_boot_ms, 998877);
    }

    let fatigue_pkt = packets.iter().find(|p| p.name_as_str() == "FATIGUE").unwrap();
    assert_eq!(fatigue_pkt.value, report.airframe_fatigue_index);
}

#[test]
fn test_cwt_engine_streaming_throughput_benchmark() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Enable CWT profiler on engine
    let config = CwtProfilerConfig {
        wavelet_type: WaveletType::MexicanHat,
        num_scales: 12,
        min_freq_hz: 200.0,
        max_freq_hz: 7000.0,
        ..Default::default()
    };
    engine.enable_cwt_profiler(config, 2);
    engine.update_motor_rpm(4200.0);

    assert!(engine.cwt_profiler().is_some());
    assert!(engine.latest_cwt_report().is_none());

    // Generate 1 second (16000 samples) of audio
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate).sin())
        .collect();

    let start = Instant::now();
    let events = engine.ingest_samples(&samples);
    let elapsed = start.elapsed();

    assert!(events.is_empty());
    assert!(
        engine.latest_cwt_report().is_some(),
        "CWT report must be populated after ingesting frames"
    );

    let report = engine.latest_cwt_report().unwrap();
    assert_eq!(report.rpm, 4200.0);

    let samples_per_sec = 16000.0 / elapsed.as_secs_f64();
    // Verify throughput exceeds 40,000 samples/sec (> 2.5x real-time speed with continuous CWT)
    assert!(
        samples_per_sec > 40_000.0,
        "CWT engine throughput was {:.0} samples/sec, below 40,000 threshold",
        samples_per_sec
    );
}
