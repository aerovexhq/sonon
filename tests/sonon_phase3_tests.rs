#![deny(unsafe_code)]

use sonon::{
    BiquadNotchFilter, FeatureMode, RotorHarmonicNotchBank,
    SpectralSubtractionConfig, SpectralSubtractionSuppressor, SononEngine,
};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_biquad_notch_attenuation_depth() {
    let sample_rate = 16000.0;
    let notch_freq = 300.0;
    let mut notch = BiquadNotchFilter::new(sample_rate, notch_freq, 10.0);

    // Generate pure 300 Hz test tone (0.5s)
    let n = (0.5 * sample_rate) as usize;
    let mut tone: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * notch_freq * (i as f32) / sample_rate).sin())
        .collect();

    let input_power: f32 = tone[n / 2..].iter().map(|&x| x * x).sum::<f32>() / (n as f32 * 0.5);

    // Filter the tone
    notch.process_block(&mut tone);

    // Measure residual power in steady state (second half of block)
    let output_power: f32 = tone[n / 2..].iter().map(|&x| x * x).sum::<f32>() / (n as f32 * 0.5);

    let attenuation_db = 10.0 * (input_power / output_power.max(1e-12)).log10();
    println!("Notch attenuation at 300 Hz: {attenuation_db:.2} dB");

    assert!(
        attenuation_db > 25.0,
        "Notch filter must attenuate target frequency by at least 25 dB, got {attenuation_db:.2} dB"
    );
}

#[test]
fn test_dynamic_rpm_harmonic_tracking() {
    let sample_rate = 16000.0;
    let mut bank = RotorHarmonicNotchBank::new(sample_rate, 3, 3, 12.0);

    // Set 6,000 RPM on 3-blade prop -> BPF = (3 * 6000) / 60 = 300 Hz
    bank.update_rpm(6000.0);
    let freqs1 = bank.active_frequencies();
    assert_eq!(freqs1.len(), 3);
    assert!((freqs1[0] - 300.0).abs() < 1e-3);
    assert!((freqs1[1] - 600.0).abs() < 1e-3);
    assert!((freqs1[2] - 900.0).abs() < 1e-3);

    // Dynamic shift during vehicle climb: 8,400 RPM -> BPF = 420 Hz
    bank.update_rpm(8400.0);
    let freqs2 = bank.active_frequencies();
    assert_eq!(freqs2.len(), 3);
    assert!((freqs2[0] - 420.0).abs() < 1e-3);
    assert!((freqs2[1] - 840.0).abs() < 1e-3);
    assert!((freqs2[2] - 1260.0).abs() < 1e-3);

    // Filter two-tone synthetic rotor whine (420 Hz + 840 Hz)
    let n = (0.5 * sample_rate) as usize;
    let mut whine: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.5 * (2.0 * PI * 420.0 * t).sin() + 0.3 * (2.0 * PI * 840.0 * t).sin()
        })
        .collect();

    let pre_power: f32 = whine[n / 2..].iter().map(|&x| x * x).sum::<f32>();
    bank.process_block(&mut whine);
    let post_power: f32 = whine[n / 2..].iter().map(|&x| x * x).sum::<f32>();

    let atten_db = 10.0 * (pre_power / post_power.max(1e-12)).log10();
    println!("Harmonic bank dual-tone attenuation: {atten_db:.2} dB");
    assert!(
        atten_db > 20.0,
        "Harmonic notch bank must attenuate dual harmonic tones by > 20 dB, got {atten_db:.2} dB"
    );
}

#[test]
fn test_multi_motor_rpm_averaging() {
    let sample_rate = 16000.0;
    let mut bank = RotorHarmonicNotchBank::new(sample_rate, 2, 2, 10.0);

    // Quadcopter with 4 motors around 9,000 RPM (mean: 9000 RPM, 2-blade prop -> BPF = 300 Hz)
    let motor_telemetry = [8950.0, 9050.0, 8980.0, 9020.0];
    bank.update_multi_motor_rpm(&motor_telemetry);

    assert!((bank.current_rpm() - 9000.0).abs() < 1.0);
    let freqs = bank.active_frequencies();
    assert_eq!(freqs.len(), 2);
    assert!((freqs[0] - 300.0).abs() < 1.0);
    assert!((freqs[1] - 600.0).abs() < 1.0);
}

#[test]
fn test_spectral_subtraction_suppression() {
    let num_bins = 257; // 512-point FFT has 257 unique bins
    let config = SpectralSubtractionConfig {
        alpha: 2.0,
        spectral_floor: 0.05,
        noise_adapt_rate: 0.1,
    };
    let mut suppressor = SpectralSubtractionSuppressor::new(num_bins, config);

    // Adapt to synthetic stationary drone white noise spectrum
    let noise_spectrum = vec![1.0f32; num_bins];
    for _ in 0..20 {
        let mut frame = noise_spectrum.clone();
        suppressor.process_spectrum(&mut frame, false);
    }

    // Now process a noisy speech frame (signal + noise)
    let mut noisy_speech_frame: Vec<f32> = noise_spectrum
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            // Add speech formant peak at bin 30
            if (25..=35).contains(&k) {
                n + 5.0
            } else {
                n
            }
        })
        .collect();

    suppressor.process_spectrum(&mut noisy_speech_frame, true);

    // Non-speech bins should be clamped to spectral floor (approx 0.05 * 1.0)
    assert!(
        noisy_speech_frame[5] < 0.2,
        "Non-speech bins must be suppressed by spectral subtraction, got {}",
        noisy_speech_frame[5]
    );

    // Speech formant bin should retain strong energy
    assert!(
        noisy_speech_frame[30] > 3.0,
        "Speech formant bin must be preserved, got {}",
        noisy_speech_frame[30]
    );
}

#[test]
fn test_rotor_noise_rejection_wake_word_spotting() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 26);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Synthetic clean keyword: dual-tone chirp (600 Hz -> 900 Hz over 0.3s)
    let n = (0.3 * sample_rate) as usize;
    let clean_keyword: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / (n as f32);
            let freq = 600.0 + 300.0 * t;
            0.5 * (2.0 * PI * freq * (i as f32) / sample_rate).sin()
        })
        .collect();

    let clean_features = engine.extract_features(&clean_keyword);
    engine.enroll_keyword("land", clean_features, 3.5);

    // Enable telemetry rotor notch bank targeting 350 Hz blade pass frequency
    // (e.g., 3-blade prop at 7,000 RPM -> BPF = 350 Hz)
    engine.enable_rotor_notch(3, 2, 12.0);
    engine.update_motor_rpm(7000.0);

    // Enable spectral subtraction
    engine.enable_spectral_subtraction(SpectralSubtractionConfig::default());

    // Inject heavy rotor whine at 350 Hz into incoming keyword test signal
    let test_noisy_signal: Vec<f32> = clean_keyword
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let t = i as f32 / sample_rate;
            let rotor_tone = 0.8 * (2.0 * PI * 350.0 * t).sin();
            s + rotor_tone
        })
        .collect();

    let events = engine.ingest_samples(&test_noisy_signal);
    assert!(
        !events.is_empty(),
        "Keyword 'land' must be detected even under heavy 350 Hz rotor whine when notch filtering is active"
    );
    assert_eq!(events[0].keyword, "land");
    assert!(events[0].confidence > 0.2);
}

#[test]
fn test_throughput_phase3_full_pipeline() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 26);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Enable Phase 3 DSP pipeline
    engine.enable_rotor_notch(3, 3, 10.0);
    engine.update_motor_rpm(6500.0);
    engine.enable_spectral_subtraction(SpectralSubtractionConfig::default());

    // Enroll template
    let template_audio: Vec<f32> = (0..(0.25 * sample_rate) as usize)
        .map(|i| (2.0 * PI * 800.0 * (i as f32) / sample_rate).sin())
        .collect();
    let template_feat = engine.extract_features(&template_audio);
    engine.enroll_keyword("abort", template_feat, 3.0);

    // Generate 10 seconds of synthetic audio stream (160,000 samples)
    let total_samples = 160_000;
    let test_stream: Vec<f32> = (0..total_samples)
        .map(|i| 0.1 * (2.0 * PI * 325.0 * (i as f32) / sample_rate).sin())
        .collect();

    let start = Instant::now();
    let _ = engine.ingest_samples(&test_stream);
    let elapsed = start.elapsed();

    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / sample_rate as f64;

    println!(
        "Phase 3 full pipeline throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    let min_threshold = if cfg!(debug_assertions) { 200_000.0 } else { 300_000.0 };

    assert!(
        samples_per_sec > min_threshold,
        "Full Phase 3 pipeline throughput must exceed {min_threshold:.0} samples/sec, got {samples_per_sec:.0}"
    );
}
