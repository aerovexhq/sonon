//! Comprehensive test suite for Partitioned-Block Frequency-Domain Adaptive Filtering (PBFDAF),
//! Magnitude Squared Coherence (MSC) Double-Talk Detection, Residual Echo Suppression (RES),
//! and full-duplex Wake-Word Spotting (KWS) during simultaneous synthetic speech playback.

#![deny(unsafe_code)]

use sonon::aec::{
    DtdConfig, DtdState, PbfdafConfig, ResConfig,
    SubbandAec, SubbandAecConfig,
};
use sonon::phonetic::{G2pEngine, KlattSynthesizer};
use sonon::SononEngine;
use std::time::Instant;

/// Generate synthetic acoustic Room Impulse Response (RIR).
fn generate_synthetic_rir(len: usize, decay: f32) -> Vec<f32> {
    let mut rir = vec![0.0f32; len];
    if len == 0 {
        return rir;
    }
    // Direct path
    rir[0] = 0.70;
    // Early reflections
    if len > 8 {
        rir[8] = -0.35;
    }
    if len > 19 {
        rir[19] = 0.22;
    }
    if len > 35 {
        rir[35] = -0.15;
    }
    if len > 52 {
        rir[52] = 0.10;
    }
    // Late diffuse exponential tail
    for i in 60..len {
        let env = (-decay * (i as f32)).exp();
        let noise = ((i * 1103515245 + 12345) % 65536) as f32 / 32768.0 - 1.0;
        rir[i] = 0.08 * env * noise;
    }
    rir
}

/// Convolve signal with impulse response.
fn convolve(signal: &[f32], ir: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; signal.len()];
    for i in 0..signal.len() {
        let mut sum = 0.0f32;
        for j in 0..ir.len() {
            if i >= j {
                sum += signal[i - j] * ir[j];
            }
        }
        out[i] = sum;
    }
    out
}

#[test]
fn test_pbfdaf_linear_echo_cancellation_and_convergence() {
    let sample_rate = 16000.0;
    let n_samples = 16000; // 1.0 second

    // Reference playback: multi-tone synthetic speech-like signal
    let reference: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.4 * (2.0 * std::f32::consts::PI * 250.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1800.0 * t).sin()
        })
        .collect();

    // 120-tap synthetic acoustic room impulse response (7.5 ms reflection tail)
    let rir = generate_synthetic_rir(120, 0.02);
    let mic_echo = convolve(&reference, &rir);

    let config = SubbandAecConfig {
        pbfdaf: PbfdafConfig {
            block_size: 64,
            num_partitions: 8, // 8 * 64 = 512 taps
            step_size: 0.40,
            regularization: 1e-4,
            leakage_factor: 0.99999,
            power_smoothing: 0.85,
            enable_gradient_constraint: true,
        },
        dtd: DtdConfig::default(),
        res: ResConfig {
            enabled: false, // Evaluate pure linear adaptive filter convergence first
            ..ResConfig::default()
        },
        sample_rate,
    };

    let mut aec = SubbandAec::new(config);
    let mut clean = vec![0.0f32; n_samples];
    aec.process_block(&mic_echo, &reference, &mut clean);

    // Initial block energy vs steady-state (last 4000 samples)
    let early_mic: f32 = mic_echo[0..2000].iter().map(|s| s * s).sum::<f32>() / 2000.0;
    let steady_err: f32 = clean[12000..n_samples].iter().map(|s| s * s).sum::<f32>() / 4000.0;
    let steady_mic: f32 = mic_echo[12000..n_samples].iter().map(|s| s * s).sum::<f32>() / 4000.0;

    let erle = 10.0 * ((steady_mic + 1e-9) / (steady_err + 1e-9)).log10();

    assert!(
        early_mic > 0.01,
        "Microphone echo energy must be active initially"
    );
    assert!(
        erle > 20.0,
        "Linear PBFDAF steady-state ERLE {:.1} dB was below 20.0 dB threshold",
        erle
    );
    assert!(
        aec.erle_db() > 18.0,
        "Internal smoothed ERLE telemetry {:.1} dB was below 18.0 dB",
        aec.erle_db()
    );
}

#[test]
fn test_residual_echo_suppression_post_filter_attenuation() {
    let sample_rate = 16000.0;
    let n_samples = 16000;

    let reference: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.5 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
        })
        .collect();

    let rir = generate_synthetic_rir(96, 0.03);
    let mic_echo = convolve(&reference, &rir);

    // 1. Without RES
    let cfg_no_res = SubbandAecConfig {
        pbfdaf: PbfdafConfig {
            block_size: 64,
            num_partitions: 8,
            step_size: 0.35,
            ..PbfdafConfig::default()
        },
        dtd: DtdConfig::default(),
        res: ResConfig {
            enabled: false,
            ..ResConfig::default()
        },
        sample_rate,
    };
    let mut aec_no_res = SubbandAec::new(cfg_no_res);
    let mut clean_no_res = vec![0.0f32; n_samples];
    aec_no_res.process_block(&mic_echo, &reference, &mut clean_no_res);

    // 2. With RES
    let cfg_with_res = SubbandAecConfig {
        pbfdaf: PbfdafConfig {
            block_size: 64,
            num_partitions: 8,
            step_size: 0.35,
            ..PbfdafConfig::default()
        },
        dtd: DtdConfig::default(),
        res: ResConfig {
            enabled: true,
            suppression_factor: 1.8,
            min_gain: 0.02,
            echo_leakage_ratio: 0.25,
        },
        sample_rate,
    };
    let mut aec_with_res = SubbandAec::new(cfg_with_res);
    let mut clean_with_res = vec![0.0f32; n_samples];
    aec_with_res.process_block(&mic_echo, &reference, &mut clean_with_res);

    let err_no_res: f32 = clean_no_res[12000..n_samples].iter().map(|s| s * s).sum::<f32>();
    let err_with_res: f32 = clean_with_res[12000..n_samples].iter().map(|s| s * s).sum::<f32>();

    let res_attenuation_db = 10.0 * ((err_no_res + 1e-12) / (err_with_res + 1e-12)).log10();

    assert!(
        res_attenuation_db > 8.0,
        "RES post-filter must provide at least 8.0 dB additional attenuation, got {:.1} dB",
        res_attenuation_db
    );
}

#[test]
fn test_coherence_double_talk_detection_under_simultaneous_speech() {
    let sample_rate = 16000.0;
    let n_samples = 24000; // 1.5 seconds

    // 1. Reference loudspeaker playback: speech-like audio
    let reference: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
        })
        .collect();

    let rir = generate_synthetic_rir(64, 0.05);
    let mut mic = convolve(&reference, &rir);

    // 2. Near-end speaker talks during the middle section (samples 8000..16000)
    for i in 8000..16000 {
        let t = (i - 8000) as f32 / sample_rate;
        let near_speech = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.4 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin();
        mic[i] += near_speech;
    }

    let config = SubbandAecConfig {
        pbfdaf: PbfdafConfig {
            block_size: 64,
            num_partitions: 6,
            step_size: 0.30,
            ..PbfdafConfig::default()
        },
        dtd: DtdConfig {
            coherence_threshold: 0.45,
            hangover_blocks: 6,
            ..DtdConfig::default()
        },
        res: ResConfig::default(),
        sample_rate,
    };

    let mut aec = SubbandAec::new(config);
    let mut clean = vec![0.0f32; n_samples];

    // Process in 128-sample chunks and monitor DTD states
    let mut double_talk_detected = false;
    let mut echo_only_detected = false;

    let chunk_size = 128;
    for i in (0..n_samples).step_by(chunk_size) {
        let end = (i + chunk_size).min(n_samples);
        let m_chunk = &mic[i..end];
        let r_chunk = &reference[i..end];
        let mut c_chunk = vec![0.0f32; end - i];

        aec.process_block(m_chunk, r_chunk, &mut c_chunk);
        clean[i..end].copy_from_slice(&c_chunk);

        if i < 7000 && aec.dtd_state() == DtdState::EchoOnly {
            echo_only_detected = true;
        }
        if i >= 9000 && i <= 15000 && aec.is_double_talk() {
            double_talk_detected = true;
        }
    }

    assert!(
        echo_only_detected,
        "AEC must identify EchoOnly state during single-talk far-end playback"
    );
    assert!(
        double_talk_detected,
        "Coherence DTD failed to identify DoubleTalk state during concurrent near-end speech"
    );
}

#[test]
fn test_subband_aec_arbitrary_streaming_block_sizes() {
    let _sample_rate = 16000.0;
    let n_samples = 4800; // 300 ms

    let reference: Vec<f32> = (0..n_samples)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let rir = generate_synthetic_rir(32, 0.08);
    let mic = convolve(&reference, &rir);

    let config = SubbandAecConfig::default();
    let mut aec = SubbandAec::new(config);

    // Test non-power-of-two chunk sizes (e.g. 160 samples = 10ms frame at 16kHz)
    let chunk_size = 160;
    let mut clean = vec![0.0f32; n_samples];

    for i in (0..n_samples).step_by(chunk_size) {
        let end = (i + chunk_size).min(n_samples);
        let m_chunk = &mic[i..end];
        let r_chunk = &reference[i..end];
        let mut c_chunk = vec![0.0f32; end - i];

        aec.process_block(m_chunk, r_chunk, &mut c_chunk);
        clean[i..end].copy_from_slice(&c_chunk);
    }

    // Verify output buffer is finite and populated
    for (idx, &s) in clean.iter().enumerate() {
        if !s.is_finite() {
            panic!("Processed sample was non-finite at index {}: {}", idx, s);
        }
    }

    // Single sample streaming test
    let mut aec_single = SubbandAec::new(SubbandAecConfig::default());
    for i in 0..128 {
        let out = aec_single.process_sample(mic[i], reference[i]);
        assert!(out.is_finite());
    }
}

#[test]
fn test_sonon_engine_full_duplex_kws_during_loud_synthetic_playback() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);

    // 1. Enroll keyword "falcon"
    let synth = KlattSynthesizer::new(sample_rate);
    let seg_kw = G2pEngine::text_to_phonemes("falcon");
    let audio_kw = synth.synthesize(&seg_kw);
    let feats_ref = engine.extract_features(&audio_kw);

    let threshold = 6.0;
    engine.enroll_keyword_banded("falcon", feats_ref.clone(), threshold, 16);

    // 2. Enable Subband AEC on engine
    let aec_config = SubbandAecConfig {
        pbfdaf: PbfdafConfig {
            block_size: 64,
            num_partitions: 8,
            step_size: 0.35,
            ..PbfdafConfig::default()
        },
        dtd: DtdConfig::default(),
        res: ResConfig {
            enabled: true,
            suppression_factor: 1.5,
            min_gain: 0.05,
            echo_leakage_ratio: 0.20,
        },
        sample_rate,
    };
    engine.enable_subband_aec(aec_config);
    assert!(engine.subband_aec().is_some());

    // 3. Synthesize a loud distractor phrase that plays out of the robot loudspeaker
    let seg_playback = G2pEngine::text_to_phonemes("initiating system diagnostic check");
    let audio_playback = synth.synthesize(&seg_playback);
    assert!(audio_playback.len() > 8000);

    // Generate room acoustic echo from loudspeaker to microphone
    let rir = generate_synthetic_rir(64, 0.04);
    let echo_only = convolve(&audio_playback, &rir);

    // Test Case A: Loudspeaker playback alone MUST NOT trigger "falcon"
    engine.reset();
    let events_echo = engine.ingest_samples_full_duplex(&echo_only, &audio_playback);
    assert!(
        !events_echo.iter().any(|e| e.keyword == "falcon"),
        "Engine falsely spotted 'falcon' during loud synthetic playback echo alone"
    );

    // Test Case B: Full-duplex double-talk.
    // Near-end operator speaks "falcon" concurrently during ongoing loudspeaker playback!
    let mut double_talk_mic = echo_only.clone();
    let start_offset = 9600; // Speak at 600 ms after filter has reached steady-state convergence
    for (i, &s) in audio_kw.iter().enumerate() {
        if start_offset + i < double_talk_mic.len() {
            double_talk_mic[start_offset + i] += s;
        }
    }

    let mut clean_dt = vec![0.0f32; double_talk_mic.len()];
    engine.subband_aec_mut().unwrap().process_block(&double_talk_mic, &audio_playback, &mut clean_dt);

    let kw_len = audio_kw.len();
    let extracted_kw = &clean_dt[start_offset..start_offset + kw_len];
    let feats_ext = engine.extract_features(extracted_kw);
    let feats_raw = engine.extract_features(&double_talk_mic[start_offset..start_offset + kw_len]);
    let dist_raw = sonon::dtw::DtwMatcher::compute_distance(&feats_raw, &feats_ref);
    let dist = sonon::dtw::DtwMatcher::compute_distance(&feats_ext, &feats_ref);

    // Verify acoustic echo cancellation significantly improved phrase match fidelity
    assert!(
        dist < dist_raw,
        "Subband AEC must reduce DTW distance compared to raw contaminated microphone signal (cleaned: {:.3}, raw: {:.3})",
        dist, dist_raw
    );
    assert!(
        dist <= threshold,
        "Extracted keyword DTW distance {:.3} must be within calibrated threshold {:.3}",
        dist, threshold
    );

    let events_dt = engine.ingest_samples(&clean_dt);
    assert!(
        events_dt.iter().any(|e| e.keyword == "falcon"),
        "Engine failed to spot 'falcon' in full-duplex double-talk with concurrent loudspeaker playback"
    );
}

#[test]
fn test_subband_aec_realtime_throughput_benchmark() {
    let sample_rate = 16000.0;
    let n_samples = 32000; // 2.0 seconds

    let reference: Vec<f32> = (0..n_samples)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let rir = generate_synthetic_rir(64, 0.05);
    let mic = convolve(&reference, &rir);

    let config = SubbandAecConfig::default();
    let mut aec = SubbandAec::new(config);
    let mut clean = vec![0.0f32; n_samples];

    let start = Instant::now();
    let chunk_size = 128;
    for i in (0..n_samples).step_by(chunk_size) {
        let end = (i + chunk_size).min(n_samples);
        aec.process_block(&mic[i..end], &reference[i..end], &mut clean[i..end]);
    }
    let elapsed = start.elapsed();

    let throughput = (n_samples as f64) / elapsed.as_secs_f64();
    let rt_ratio = throughput / (sample_rate as f64);

    let min_throughput = if cfg!(debug_assertions) { 40000.0 } else { 150000.0 };
    assert!(
        throughput > min_throughput,
        "Throughput {:.0} samples/sec was below {:.0} bound ({:.1}x real-time)",
        throughput,
        min_throughput,
        rt_ratio
    );
}
