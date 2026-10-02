//! Phase 14 Verification: Adaptive Acoustic Echo Cancellation (AEC),
//! Normalized Least Mean Squares (NLMS) filtering, Geigel Double-Talk Detection (DTD),
//! and full SononEngine barge-in keyword spotting.

#![deny(unsafe_code)]

use sonon::aec::{AcousticEchoCanceller, AecConfig, DtdState};
use sonon::engine::SononEngine;
use std::f32::consts::PI;
use std::time::Instant;

/// Helper to generate deterministic pseudo-random white noise sequence.
fn generate_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((state >> 33) as f32) / (u32::MAX as f32);
            (val - 0.5) * 0.8
        })
        .collect()
}

#[test]
fn test_aec_synthetic_impulse_response_convergence() {
    let config = AecConfig {
        filter_length: 64,
        step_size: 0.35,
        regularization: 1e-4,
        leakage_factor: 0.99999,
        dtd_threshold: 0.75,
        dtd_hangover_samples: 80,
        min_reference_energy: 1e-4,
    };
    let mut aec = AcousticEchoCanceller::new(config);

    // True acoustic impulse response (loudspeaker to mic path)
    let mut true_h = vec![0.0f32; 32];
    true_h[4] = 0.65;
    true_h[5] = -0.35;
    true_h[6] = 0.20;
    true_h[7] = -0.10;
    true_h[8] = 0.05;

    let n = 5000;
    let ref_signal = generate_noise(n, 12345);

    // Desired microphone signal = true_h * ref_signal
    let mut mic_signal = vec![0.0f32; n];
    for i in 0..n {
        let mut echo = 0.0f32;
        for (k, &hk) in true_h.iter().enumerate() {
            if i >= k {
                echo += hk * ref_signal[i - k];
            }
        }
        mic_signal[i] = echo;
    }

    let mut clean_output = vec![0.0f32; n];
    aec.process_block(&mic_signal, &ref_signal, &mut clean_output);

    // Measure echo power in initial 500 samples vs converged last 1000 samples
    let init_mic_pwr: f32 = mic_signal[0..500].iter().map(|&x| x * x).sum::<f32>() / 500.0;
    let final_err_pwr: f32 = clean_output[4000..5000].iter().map(|&x| x * x).sum::<f32>() / 1000.0;
    let final_mic_pwr: f32 = mic_signal[4000..5000].iter().map(|&x| x * x).sum::<f32>() / 1000.0;

    let erle_db = 10.0 * (final_mic_pwr / final_err_pwr.max(1e-12)).log10();

    assert!(
        init_mic_pwr > 0.01,
        "Microphone input signal must have non-zero power"
    );
    assert!(
        erle_db > 25.0,
        "AEC must achieve > 25.0 dB ERLE echo cancellation, got {:.2} dB",
        erle_db
    );
    assert!(
        final_err_pwr < 0.001,
        "Residual error power must be attenuated below 0.001, got {:.6}",
        final_err_pwr
    );
}

#[test]
fn test_aec_double_talk_detection_preserves_near_end_speech() {
    let config = AecConfig {
        filter_length: 64,
        step_size: 0.35,
        regularization: 1e-4,
        leakage_factor: 0.99999,
        dtd_threshold: 0.75,
        dtd_hangover_samples: 120,
        min_reference_energy: 1e-4,
    };
    let mut aec = AcousticEchoCanceller::new(config);

    let mut true_h = vec![0.0f32; 24];
    true_h[3] = 0.50;
    true_h[4] = -0.25;

    let n = 6000;
    let ref_signal = generate_noise(n, 54321);

    // Near-end user speech (16 kHz sine wave burst from sample 3000 to 4500)
    let sample_rate = 16000.0f32;
    let speech_freq = 500.0f32;
    let mut near_end_speech = vec![0.0f32; n];
    for i in 3000..4500 {
        let t = (i as f32) / sample_rate;
        near_end_speech[i] = 0.55 * (2.0 * PI * speech_freq * t).sin();
    }

    let mut mic_signal = vec![0.0f32; n];
    for i in 0..n {
        let mut echo = 0.0f32;
        for (k, &hk) in true_h.iter().enumerate() {
            if i >= k {
                echo += hk * ref_signal[i - k];
            }
        }
        mic_signal[i] = echo + near_end_speech[i];
    }

    let mut clean_output = vec![0.0f32; n];
    let mut double_talk_events = 0;

    for i in 0..n {
        clean_output[i] = aec.process_sample(mic_signal[i], ref_signal[i]);
        if aec.dtd_state() == DtdState::DoubleTalk {
            double_talk_events += 1;
        }
    }

    // Verify DTD engaged during user speech
    assert!(
        double_talk_events > 500,
        "Double-Talk Detector must trigger and freeze filter during near-end speech, triggered {}",
        double_talk_events
    );

    // Verify user speech is preserved in the output
    let near_end_input_pwr: f32 = near_end_speech[3200..4200].iter().map(|&x| x * x).sum::<f32>() / 1000.0;
    let clean_output_pwr: f32 = clean_output[3200..4200].iter().map(|&x| x * x).sum::<f32>() / 1000.0;

    assert!(
        clean_output_pwr > 0.5 * near_end_input_pwr,
        "Near-end speech must be preserved during double-talk: in={:.4}, out={:.4}",
        near_end_input_pwr,
        clean_output_pwr
    );
}

#[test]
fn test_aec_silence_stability_no_drift() {
    let mut aec = AcousticEchoCanceller::new(AecConfig::default());

    // Feed 2000 samples of pure silence
    for _ in 0..2000 {
        let out = aec.process_sample(0.0, 0.0);
        assert_eq!(out, 0.0, "Silence input must yield silence output");
    }

    assert_eq!(aec.dtd_state(), DtdState::Silence);
    for &w in aec.weights() {
        assert!(w.abs() < 1e-6, "Weights must not drift during silence");
    }
}

#[test]
fn test_sonon_engine_aec_integration_barge_in_wake_word_spotting() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Enable AEC with responsive learning rate
    engine.enable_aec(AecConfig {
        filter_length: 64,
        step_size: 0.40,
        regularization: 1e-4,
        leakage_factor: 0.99999,
        dtd_threshold: 0.75,
        dtd_hangover_samples: 80,
        min_reference_energy: 1e-4,
    });

    assert!(engine.aec().is_some(), "AEC must be enabled on engine");
    engine.set_feature_mode(sonon::FeatureMode::Pcen);

    // Synthesize clean keyword audio ("abort")
    let keyword_len = 4000; // 250 ms
    let mut keyword_audio = vec![0.0f32; keyword_len];
    for i in 0..keyword_len {
        let t = (i as f32) / sample_rate;
        // Two formant tones
        keyword_audio[i] = 0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin();
    }

    let features = engine.extract_features(&keyword_audio);
    assert!(!features.is_empty(), "Keyword features must not be empty");
    engine.enroll_keyword("abort", features, 5.0);

    // Simulate loud far-end loudspeaker playback (reference signal)
    let stream_len = 8000;
    let ref_music = generate_noise(stream_len, 98765);

    // Simulated acoustic echo path: h = [0.0, 0.0, 0.6, -0.3]
    let true_h = [0.0f32, 0.0, 0.6, -0.3];
    let mut mic_audio = vec![0.0f32; stream_len];

    for i in 0..stream_len {
        let mut echo = 0.0f32;
        for (k, &hk) in true_h.iter().enumerate() {
            if i >= k {
                echo += hk * ref_music[i - k];
            }
        }
        mic_audio[i] = echo;
    }

    // Embed the keyword in the microphone audio between sample 3000 and 7000 (barge-in over music!)
    for i in 0..keyword_len {
        mic_audio[3000 + i] += keyword_audio[i];
    }

    // Ingest with reference audio stream for active echo cancellation
    let events = engine.ingest_samples_with_reference(&mic_audio, &ref_music);

    assert!(
        events.iter().any(|e| e.keyword == "abort"),
        "Engine with AEC must recognize keyword even while loud speaker playback is active"
    );

    let final_erle = engine.aec().unwrap().erle_db();
    assert!(
        final_erle >= 0.0,
        "AEC ERLE metric must be non-negative, got {:.2}",
        final_erle
    );
}

#[test]
fn test_aec_disable_and_reset() {
    let mut engine = SononEngine::new(16000.0, 256, 128, 13);
    engine.enable_aec(AecConfig::default());
    assert!(engine.aec().is_some());

    engine.disable_aec();
    assert!(engine.aec().is_none());
}

#[test]
fn test_aec_throughput_benchmark() {
    let n = 16000; // 1 second of audio
    let ref_signal = generate_noise(n, 11223);
    let mic_signal = generate_noise(n, 33445);
    let mut clean_out = vec![0.0f32; n];

    let mut aec = AcousticEchoCanceller::new(AecConfig {
        filter_length: 64,
        ..Default::default()
    });

    let start = Instant::now();
    for _ in 0..10 {
        aec.process_block(&mic_signal, &ref_signal, &mut clean_out);
    }
    let elapsed = start.elapsed();

    let total_samples = n * 10;
    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();

    // Verify throughput exceeds 400,000 samples/sec (> 25x real time)
    assert!(
        samples_per_sec > 400_000.0,
        "AEC throughput must exceed 400,000 samples/sec (> 25x real time), achieved {:.2}",
        samples_per_sec
    );
}
