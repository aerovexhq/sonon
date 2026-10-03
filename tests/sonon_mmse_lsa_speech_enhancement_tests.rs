#![deny(unsafe_code)]

use sonon::engine::SononEngine;
use sonon::mmse_lsa::{
    compute_lsa_gain, exponential_integral_e1, ImcraConfig, ImcraNoiseEstimator, LsaConfig,
    MmseLsaFilter,
};
use sonon::phonetic::{G2pEngine, KlattSynthesizer};

#[test]
fn test_exponential_integral_and_lsa_gain_analytical_properties() {
    // 1. Verify Abramowitz & Stegun mathematical reference points for E_1(x)
    // E_1(1.0) = 0.219383934...
    let e1_at_1 = exponential_integral_e1(1.0);
    assert!(
        (e1_at_1 - 0.2193839).abs() < 1e-5,
        "E_1(1.0) was {}, expected ~0.219384",
        e1_at_1
    );

    // E_1(0.5) = 0.55977359...
    let e1_at_half = exponential_integral_e1(0.5);
    assert!(
        (e1_at_half - 0.5597736).abs() < 1e-4,
        "E_1(0.5) was {}, expected ~0.559774",
        e1_at_half
    );

    // E_1(2.0) = 0.0489005...
    let e1_at_2 = exponential_integral_e1(2.0);
    assert!(
        (e1_at_2 - 0.0489005).abs() < 1e-5,
        "E_1(2.0) was {}, expected ~0.048901",
        e1_at_2
    );

    // Monotonic strictly decreasing property: x1 < x2 => E_1(x1) > E_1(x2)
    let test_points = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    for w in test_points.windows(2) {
        let x1 = w[0];
        let x2 = w[1];
        let y1 = exponential_integral_e1(x1);
        let y2 = exponential_integral_e1(x2);
        assert!(
            y1 > y2,
            "E_1 must be strictly decreasing: E_1({}) = {} <= E_1({}) = {}",
            x1, y1, x2, y2
        );
    }

    // 2. Analytical properties of Bayesian Log-Spectral Amplitude gain G_LSA
    // Low SNR (-20 dB): When prior SNR is tiny and post SNR indicates noise floor, gain clamps to min_gain
    let min_gain = 0.08; // -22 dB
    let g_low = compute_lsa_gain(0.01, 1.0, min_gain);
    assert!(
        (g_low - min_gain).abs() < 0.01,
        "Gain at deep negative SNR must clamp near min_gain {}, got {}",
        min_gain, g_low
    );

    // High SNR (+25 dB): Gain must approach 1.0
    let g_high = compute_lsa_gain(300.0, 300.0, min_gain);
    assert!(
        g_high > 0.98 && g_high <= 1.0,
        "Gain at high SNR must approach 1.0, got {}",
        g_high
    );

    // Monotonicity: Higher a priori SNR yields strictly higher spectral gain
    let g_mid1 = compute_lsa_gain(1.0, 1.5, min_gain);
    let g_mid2 = compute_lsa_gain(5.0, 5.5, min_gain);
    assert!(
        g_mid2 > g_mid1,
        "Higher SNR must produce higher LSA gain: g(5)={} <= g(1)={}",
        g_mid2, g_mid1
    );
}

#[test]
fn test_imcra_non_stationary_noise_psd_tracking_under_drone_rpm_ramp() {
    let num_bins = 257; // 512-point FFT
    let config = ImcraConfig {
        sub_window_len: 8,
        num_sub_windows: 4,
        alpha_s: 0.85,
        alpha_d: 0.80,
        ..ImcraConfig::default()
    };
    let mut imcra = ImcraNoiseEstimator::new(config, num_bins);

    // Phase 1: Drone hovering at low RPM (low noise floor, power = 0.01)
    for _ in 0..64 {
        let low_noise = vec![0.01f32; num_bins];
        imcra.update(&low_noise);
    }
    let low_noise_est = imcra.average_noise_power();
    assert!(
        (low_noise_est - 0.01).abs() < 0.003,
        "IMCRA initial noise estimate {:.4} should track hover floor 0.01",
        low_noise_est
    );

    // Phase 2: Throttle ramp to high RPM (noise power quadruples to 0.04)
    for _ in 0..128 {
        let high_noise = vec![0.04f32; num_bins];
        imcra.update(&high_noise);
    }
    let high_noise_est = imcra.average_noise_power();
    assert!(
        high_noise_est > 0.030,
        "IMCRA must track non-stationary RPM throttle increase (got {:.4}, expected > 0.030)",
        high_noise_est
    );

    // Speech presence probability on pure noise must remain close to 0
    let mean_spp: f32 = imcra.speech_presence_prob().iter().sum::<f32>() / (num_bins as f32);
    assert!(
        mean_spp < 0.05,
        "Speech presence probability on pure ambient drone noise must remain < 0.05, got {:.4}",
        mean_spp
    );
}

#[test]
fn test_imcra_speech_presence_discrimination_during_active_utterance() {
    let num_bins = 257;
    let config = ImcraConfig {
        sub_window_len: 8,
        num_sub_windows: 4,
        gamma_0: 3.0,
        gamma_1: 1.5,
        ..ImcraConfig::default()
    };
    let mut imcra = ImcraNoiseEstimator::new(config, num_bins);

    // Prime with background drone hum (spectral peaks at BPF bins 20 and 40)
    for _ in 0..50 {
        let mut noise_frame = vec![0.005f32; num_bins];
        noise_frame[20] = 0.04;
        noise_frame[40] = 0.03;
        imcra.update(&noise_frame);
    }

    // Inject strong speech formant peak at bin 75 (1200 Hz formant)
    let mut speech_frame = vec![0.005f32; num_bins];
    speech_frame[20] = 0.04;
    speech_frame[40] = 0.03;
    speech_frame[75] = 0.50; // +20 dB speech burst

    // Update with speech frame
    for _ in 0..5 {
        imcra.update(&speech_frame);
    }

    let spp = imcra.speech_presence_prob();
    assert!(
        spp[75] > 0.80,
        "Speech presence probability at formant bin 75 must be high, got {:.3}",
        spp[75]
    );
    assert!(
        spp[150] < 0.10,
        "Speech presence probability in inactive bin 150 must remain low, got {:.3}",
        spp[150]
    );

    // Verify noise PSD at bin 75 was frozen and did not absorb speech burst
    let noise_psd = imcra.noise_psd();
    assert!(
        noise_psd[75] < 0.015,
        "Noise PSD at speech formant bin 75 must not absorb speech energy, got {:.4}",
        noise_psd[75]
    );
}

#[test]
fn test_mmse_lsa_snr_improvement_and_musical_noise_elimination() {
    let num_bins = 129; // 256-point STFT
    let config = LsaConfig {
        alpha_dd: 0.96,
        min_gain_db: -20.0,
        imcra: ImcraConfig {
            sub_window_len: 8,
            num_sub_windows: 4,
            gamma_0: 3.0,
            gamma_1: 1.5,
            ..ImcraConfig::default()
        },
        ..LsaConfig::default()
    };
    let mut filter = MmseLsaFilter::new(num_bins, config);

    // 1. Prime background noise
    for _ in 0..40 {
        let mut noise_spectrum = vec![0.02f32; num_bins];
        filter.process_spectrum(&mut noise_spectrum);
    }

    // 2. Measure attenuation on pure noise frame
    let mut test_noise = vec![0.02f32; num_bins];
    filter.process_spectrum(&mut test_noise);
    let suppressed_noise_power: f32 = test_noise.iter().sum::<f32>() / (num_bins as f32);
    let attenuation_db = 10.0 * (0.02 / (suppressed_noise_power + 1e-9)).log10();

    assert!(
        attenuation_db > 14.0,
        "MMSE-LSA attenuation on background noise must be > 14.0 dB, achieved {:.1} dB",
        attenuation_db
    );

    // 3. Measure speech preservation
    let mut speech_with_noise = vec![0.02f32; num_bins];
    // Strong speech formants at bins 30 and 60
    speech_with_noise[30] = 0.80;
    speech_with_noise[60] = 0.50;

    let mut clean_speech_frame = speech_with_noise.clone();
    filter.process_spectrum(&mut clean_speech_frame);

    // Formant bins must retain majority of power
    assert!(
        clean_speech_frame[30] > 0.20,
        "Speech formant at bin 30 must be preserved (got {:.3}, input 0.80)",
        clean_speech_frame[30]
    );
    assert!(
        clean_speech_frame[60] > 0.10,
        "Speech formant at bin 60 must be preserved (got {:.3}, input 0.50)",
        clean_speech_frame[60]
    );

    // Inactive noise bins must remain deeply attenuated
    assert!(
        clean_speech_frame[10] < 0.005,
        "Noise bin 10 must be attenuated (got {:.5}, input 0.02)",
        clean_speech_frame[10]
    );

    // Telemetry verification
    let telem = filter.telemetry();
    assert!(telem.mean_gain_db < -5.0);
}

#[test]
fn test_sonon_engine_streaming_kws_recall_boost_under_heavy_drone_noise() {
    let sample_rate = 16000.0;
    let mut engine_clean = SononEngine::new(sample_rate, 512, 160, 13);
    let mut engine_enhanced = SononEngine::new(sample_rate, 512, 160, 13);

    // 1. Synthesize keyword "emergency"
    let synth = KlattSynthesizer::new(sample_rate);
    let seg = G2pEngine::text_to_phonemes("emergency");
    let audio_kw = synth.synthesize(&seg);
    assert!(!audio_kw.is_empty());

    // 2. Enable Bayesian MMSE-LSA speech enhancement on second engine
    let lsa_config = LsaConfig {
        alpha_dd: 0.96,
        min_gain_db: -20.0,
        min_prior_snr_db: -18.0,
        imcra: ImcraConfig {
            sub_window_len: 12,
            num_sub_windows: 4,
            gamma_0: 3.0,
            gamma_1: 1.5,
            ..ImcraConfig::default()
        },
    };
    engine_enhanced.enable_mmse_lsa_enhancement(lsa_config);
    assert!(engine_enhanced.mmse_lsa().is_some());

    // Extract reference features on the enhanced engine manifold
    let ref_feats = engine_enhanced.extract_features(&audio_kw);
    let threshold = 9.5;
    engine_clean.enroll_keyword_banded("emergency", ref_feats.clone(), threshold, 16);
    engine_enhanced.enroll_keyword_banded("emergency", ref_feats.clone(), threshold, 16);

    // 3. Synthesize heavy non-stationary drone rotor noise:
    // Propeller BPF harmonics (150 Hz, 300 Hz, 450 Hz) + aerodynamic turbulent rumble
    let kw_len = audio_kw.len();
    let leading_noise_len = 16000;
    let trailing_len = 3200;
    let total_len = leading_noise_len + kw_len + trailing_len;

    let mut noisy_stream = Vec::with_capacity(total_len);
    for i in 0..total_len {
        let t = i as f32 / sample_rate;
        // Motor BPF tonal hum + harmonics
        let drone_tonal = 0.08 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            + 0.03 * (2.0 * std::f32::consts::PI * 450.0 * t).sin();
        // Turbulent broadband pseudo-noise
        let pseudo_rand = ((i * 1103515245 + 12345) & 0x7FFFFFFF) as f32 / (0x7FFFFFFF as f32) - 0.5;
        let drone_rumble = 0.04 * pseudo_rand;

        let noise_sample = drone_tonal + drone_rumble;
        let speech_sample = if i >= leading_noise_len && (i - leading_noise_len) < kw_len {
            audio_kw[i - leading_noise_len]
        } else {
            0.0
        };

        noisy_stream.push(speech_sample + noise_sample);
    }

    // Test Case A: Pure drone noise before keyword must NOT trigger false alarm on either engine
    let leading_noise_slice = &noisy_stream[0..leading_noise_len];
    let events_noise_clean = engine_clean.ingest_samples(leading_noise_slice);
    let events_noise_enh = engine_enhanced.ingest_samples(leading_noise_slice);

    assert!(
        !events_noise_clean.iter().any(|e| e.keyword == "emergency"),
        "Raw engine falsely spotted keyword on pure drone rotor noise"
    );
    assert!(
        !events_noise_enh.iter().any(|e| e.keyword == "emergency"),
        "Enhanced engine falsely spotted keyword on pure drone rotor noise"
    );

    // Test Case B: Ingest full stream containing keyword submerged under drone noise
    // Engine with MMSE-LSA enhancement must cleanly spot "emergency"
    engine_clean.reset();
    engine_enhanced.reset();

    let noisy_kw_slice = &noisy_stream[leading_noise_len..leading_noise_len + kw_len];
    let feats_raw_kw = engine_clean.extract_features(noisy_kw_slice);
    let feats_enh_kw = engine_enhanced.extract_features(noisy_kw_slice);

    let dist_raw = sonon::dtw::DtwMatcher::compute_distance(&feats_raw_kw, &ref_feats);
    let dist_enh = sonon::dtw::DtwMatcher::compute_distance(&feats_enh_kw, &ref_feats);

    assert!(
        dist_enh < dist_raw,
        "MMSE-LSA must substantially reduce DTW distance under noise (enhanced: {:.3}, raw: {:.3})",
        dist_enh, dist_raw
    );
    assert!(
        dist_enh <= threshold,
        "Enhanced keyword distance {:.3} must fall within threshold {:.3}",
        dist_enh, threshold
    );

    let events_raw = engine_clean.ingest_samples(&noisy_stream);
    let events_enhanced = engine_enhanced.ingest_samples(&noisy_stream);

    assert!(
        !events_raw.iter().any(|e| e.keyword == "emergency"),
        "Raw un-enhanced engine falsely/accidentally triggered without enhancement under heavy noise"
    );
    assert!(
        events_enhanced.iter().any(|e| e.keyword == "emergency"),
        "Enhanced engine with MMSE-LSA failed to spot 'emergency' under severe drone noise"
    );

    let telem = engine_enhanced.latest_mmse_lsa_telemetry().expect("Telemetry must exist");
    assert!(
        telem.mean_gain_db < -3.0,
        "Telemetry must reflect active noise attenuation (got {:.1} dB)",
        telem.mean_gain_db
    );
}

#[test]
fn test_mmse_lsa_realtime_throughput_benchmark() {
    let sample_rate = 16000.0;
    let n_samples = 32000; // 2.0 seconds of audio

    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.enable_mmse_lsa_enhancement(LsaConfig::default());

    // Enroll a dummy phrase
    let seg = G2pEngine::text_to_phonemes("hold");
    let synth = KlattSynthesizer::new(sample_rate);
    let wave = synth.synthesize(&seg);
    let feats = engine.extract_features(&wave);
    engine.enroll_keyword_banded("hold", feats, 5.0, 12);

    // Synthetic audio buffer
    let audio: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let start = std::time::Instant::now();
    let events = engine.ingest_samples(&audio);
    let elapsed = start.elapsed();

    let throughput_sps = (n_samples as f64) / elapsed.as_secs_f64();
    println!(
        "MMSE-LSA Streaming Engine Throughput: {:.0} samples/sec ({:.1}x real-time), events: {}",
        throughput_sps,
        throughput_sps / (sample_rate as f64),
        events.len()
    );

    assert!(
        throughput_sps > 30000.0,
        "Throughput {:.0} samples/sec was below 30,000 samples/sec threshold in unoptimized test mode",
        throughput_sps
    );
}
