//! Phase 16 analytical test suite: Doppler shift compensation, high-speed in-flight
//! kinematic velocity tracking, and relativistic acoustic frequency warping.

#![deny(unsafe_code)]

use sonon::doppler::{speed_of_sound_at_temp, DopplerCompensator, DopplerConfig};
use sonon::engine::SononEngine;
use sonon::stft::FftProcessor;
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_doppler_temperature_speed_of_sound() {
    let c_subzero = speed_of_sound_at_temp(-20.0);
    let c_0c = speed_of_sound_at_temp(0.0);
    let c_20c = speed_of_sound_at_temp(20.0);
    let c_desert = speed_of_sound_at_temp(45.0);

    // Theoretical values: c(T) = 331.3 * sqrt(T_K / 273.15)
    // -20 C (253.15 K) ~ 318.9 m/s
    //   0 C (273.15 K) = 331.3 m/s
    //  20 C (293.15 K) ~ 343.2 m/s
    //  45 C (318.15 K) ~ 357.5 m/s
    assert!((c_0c - 331.3).abs() < 0.2, "Speed at 0 C should be ~331.3 m/s");
    assert!((c_20c - 343.2).abs() < 0.5, "Speed at 20 C should be ~343.2 m/s");
    assert!((c_subzero - 318.9).abs() < 0.5, "Speed at -20 C should be ~318.9 m/s");
    assert!((c_desert - 357.5).abs() < 0.5, "Speed at 45 C should be ~357.5 m/s");

    // Monotonic temperature dependence
    assert!(c_subzero < c_0c);
    assert!(c_0c < c_20c);
    assert!(c_20c < c_desert);
}

#[test]
fn test_doppler_kinematic_3d_projection_and_scaling() {
    let config = DopplerConfig {
        temperature_celsius: 20.0,
        max_velocity_mps: 60.0,
        smoothing_alpha: 0.0, // Instant update for deterministic unit testing
        filterbank_quantization_step: 0.001,
    };

    let mut compensator = DopplerCompensator::new(config, 26, 256, 16000.0, 80.0, 8000.0);

    // Target speaker at forward boresight (1, 0, 0)
    compensator.update_target_ray(1.0, 0.0, 0.0);

    // Closing at +34.32 m/s (~0.1 Mach at 20 C)
    compensator.update_velocity_3d(34.32, 0.0, 0.0);
    let factor_closing = compensator.doppler_factor();
    assert!(
        (factor_closing - 1.10).abs() < 0.01,
        "Closing at 0.1 Mach should scale frequencies by ~1.10, got {}",
        factor_closing
    );

    // Receding at -34.32 m/s (-0.1 Mach)
    compensator.update_velocity_3d(-34.32, 0.0, 0.0);
    let factor_receding = compensator.doppler_factor();
    assert!(
        (factor_receding - 0.90).abs() < 0.01,
        "Receding at 0.1 Mach should scale frequencies by ~0.90, got {}",
        factor_receding
    );

    // Transverse flight (crosswind or orbiting): velocity perpendicular to line of sight (0, 34.32, 0)
    compensator.update_velocity_3d(0.0, 34.32, 0.0);
    let factor_transverse = compensator.doppler_factor();
    assert!(
        (factor_transverse - 1.00).abs() < 0.001,
        "Transverse flight should have zero line-of-sight Doppler shift, got {}",
        factor_transverse
    );

    // Oblique line of sight: target at 45 degrees azimuth
    compensator.update_target_bearing(PI / 4.0, 0.0);
    // Moving forward at 34.32 m/s -> line-of-sight component is 34.32 * cos(45 deg) = 24.27 m/s (~0.0707 Mach)
    compensator.update_velocity_3d(34.32, 0.0, 0.0);
    let factor_oblique = compensator.doppler_factor();
    assert!(
        (factor_oblique - 1.0707).abs() < 0.01,
        "Oblique Doppler factor should reflect cos(theta) projection, got {}",
        factor_oblique
    );
}

#[test]
fn test_doppler_active_filterbank_warping() {
    let config = DopplerConfig {
        temperature_celsius: 20.0,
        max_velocity_mps: 50.0,
        smoothing_alpha: 0.0,
        filterbank_quantization_step: 0.01,
    };

    let mut compensator = DopplerCompensator::new(config, 26, 256, 16000.0, 100.0, 8000.0);

    // Baseline stationary
    let baseline_factor = compensator.doppler_factor();
    assert_eq!(baseline_factor, 1.0);

    // Accelerate drone forward towards speaker to 35 m/s (~10% Doppler expansion)
    compensator.update_velocity_3d(35.0, 0.0, 0.0);
    let active_fb = compensator.active_filterbank();

    // Verify filterbank is cached and functional
    let dummy_power = vec![1.0f32; 129];
    let energies = active_fb.compute_energies(&dummy_power);
    assert_eq!(energies.len(), 26);
    assert!(energies.iter().all(|&e| e > 0.0));

    // Reset back to baseline
    compensator.reset();
    assert_eq!(compensator.doppler_factor(), 1.0);
}

#[test]
fn test_doppler_fractional_resampler_pitch_restoration() {
    let sample_rate = 16000.0;
    let n_samples = 4096;
    let original_freq = 440.0; // A4 tone
    let alpha = 1.10; // 10% Doppler shift up (closing vehicle)

    // Synthesize Doppler-shifted received tone: 440 * 1.10 = 484 Hz
    let shifted_freq = original_freq * alpha;
    let mut received_tone = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = (i as f32) / sample_rate;
        received_tone.push((2.0 * PI * shifted_freq * t).sin());
    }

    // FFT of received tone to confirm shifted frequency peak
    let fft = FftProcessor::new(n_samples);
    let shifted_power = fft.power_spectrum(&received_tone);
    let bin_width = sample_rate / (n_samples as f32); // 3.906 Hz/bin
    let max_bin_shifted = shifted_power
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(bin, _)| bin)
        .unwrap();
    let peak_shifted_hz = (max_bin_shifted as f32) * bin_width;
    assert!(
        (peak_shifted_hz - shifted_freq).abs() < bin_width * 1.5,
        "Shifted peak should be ~484 Hz, got {} Hz",
        peak_shifted_hz
    );

    // Physically undo Doppler shift via cubic Hermite fractional resampler
    let restored_tone = DopplerCompensator::resample_audio(&received_tone, alpha);
    assert!(
        restored_tone.len() > 0,
        "Resampled buffer should not be empty"
    );

    // Take 4096-sample window of restored audio and find peak
    let mut restored_window = vec![0.0f32; n_samples];
    let copy_len = restored_tone.len().min(n_samples);
    restored_window[..copy_len].copy_from_slice(&restored_tone[..copy_len]);

    let restored_power = fft.power_spectrum(&restored_window);
    let max_bin_restored = restored_power
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(bin, _)| bin)
        .unwrap();
    let peak_restored_hz = (max_bin_restored as f32) * bin_width;

    assert!(
        (peak_restored_hz - original_freq).abs() < bin_width * 1.5,
        "Restored peak should match original 440 Hz, got {} Hz",
        peak_restored_hz
    );
}

#[test]
fn test_engine_doppler_wake_word_spotting_high_speed_flyby() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    engine.set_feature_mode(sonon::FeatureMode::Pcen);

    // Enroll zero-shot synthesized keyword "take off"
    let keyword_text = "take off";
    let frames = engine.enroll_keyword_from_text("take_off", keyword_text, 3.5);
    assert!(frames > 0);
    let synth_audio = engine.synthesize_speech_from_text(keyword_text);

    // Simulate high-speed closing flyby at 34.3 m/s (Doppler factor ~ 1.10)
    let doppler_factor = 1.10f32;
    // When sound arrives at a closing drone, the waveform is compressed in time and shifted up in pitch:
    // advance through source samples by doppler_factor per receiver sample
    let mut received_audio = Vec::new();
    let mut step = 0.0f32;
    while (step as usize) + 1 < synth_audio.len() {
        let i = step.floor() as usize;
        let frac = step - (i as f32);
        let sample = synth_audio[i] * (1.0 - frac) + synth_audio[i + 1] * frac;
        received_audio.push(sample);
        step += doppler_factor;
    }

    // Enable Doppler compensation on the engine
    let doppler_config = DopplerConfig {
        temperature_celsius: 20.0,
        max_velocity_mps: 50.0,
        smoothing_alpha: 0.0, // Instant response
        filterbank_quantization_step: 0.005,
    };
    engine.enable_doppler_compensation(doppler_config);
    // Feed MAVLink/autopilot velocity: closing along line of sight at 34.3 m/s
    engine.update_kinematic_velocity(34.3, 0.0, 0.0);
    assert!((engine.doppler_scale_factor() - 1.10).abs() < 0.02);

    // Ingest the Doppler-shifted audio into the engine with Doppler filterbank compensation active
    let events = engine.ingest_samples(&received_audio);
    assert!(
        !events.is_empty(),
        "Engine with Doppler compensation should successfully detect 'take_off' during high-speed flyby"
    );
    assert_eq!(events[0].keyword, "take_off");
}

#[test]
fn test_doppler_compensation_throughput_benchmark() {
    let sample_rate = 16000.0;
    let n_samples = 16000; // 1 second
    let samples: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate).sin())
        .collect();

    let start = Instant::now();
    let iterations = 100;
    let mut total_output_samples = 0;

    for i in 0..iterations {
        let factor = 1.0 + 0.001 * (i as f32);
        let resampled = DopplerCompensator::resample_audio(&samples, factor);
        total_output_samples += resampled.len();
    }
    let elapsed = start.elapsed();

    let total_input_samples = n_samples * iterations;
    let samples_per_sec = (total_input_samples as f64) / elapsed.as_secs_f64();

    // Verify throughput exceeds 2,000,000 samples/sec (> 125x real-time)
    assert!(
        samples_per_sec > 2_000_000.0,
        "Doppler resampler throughput was {:.0} samples/sec, below 2,000,000 threshold",
        samples_per_sec
    );
    assert!(total_output_samples > 0);
}
