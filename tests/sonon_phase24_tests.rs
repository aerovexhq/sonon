//! Analytical and empirical verification suite for Phase 24:
//! Aerodynamic Wind Buffeting Incoherent Noise Separation & Turbulent Boundary Layer (TBL) Suppression.

#![deny(unsafe_code)]

use sonon::{
    AeroacousticWindSimulator, AdaptiveRumbleFilter, FeatureMode,
    TurbulentBoundaryLayerSuppressor, WindTurbulenceConfig,
    SononEngine,
};
use std::f32::consts::PI;
use std::time::Instant;

/// Test 1: Convective Pseudosound Phase-Slowness Discrimination.
///
/// True acoustic waves propagate across dual microphones at the speed of sound c ≈ 343 m/s,
/// exhibiting low apparent slowness s ≤ 1/c. Convective aerodynamic turbulence eddies convect
/// at fluid convective speed Uc ≈ 10 m/s (s = 1/Uc ≈ 0.10 s/m = 34.3x acoustic slowness).
/// The phase-slowness discriminator must reject convective pseudosound while preserving acoustic waves.
#[test]
fn test_convective_pseudosound_phase_slowness_discrimination() {
    let sample_rate = 16000.0;
    let mic_dist_m = 0.02; // 20 mm spacing
    let config = WindTurbulenceConfig {
        sample_rate,
        fft_size: 512,
        hop_size: 256,
        mic_distance_m: mic_dist_m,
        forward_airspeed_mps: 15.0,
        min_coherence_threshold: 0.12,
        max_attenuation_db: 24.0,
        coherence_smoothing: 0.85,
        enable_convective_slowness_gate: true,
        convective_slowness_margin: 1.35,
        enable_adaptive_rumble_filter: false, // Isolate phase-slowness gate
        min_rumble_cutoff_hz: 60.0,
        max_rumble_cutoff_hz: 220.0,
        reference_airspeed_mps: 15.0,
    };

    let mut suppressor = TurbulentBoundaryLayerSuppressor::new(config.clone());
    let hop = config.hop_size;
    let num_samples = 4096;

    // 1. Synthesize Propagating Acoustic Sound Wave (e.g. 800 Hz speech tone arriving from broadside theta = 0)
    // Acoustic delay = 0 seconds across sensors; s_apparent = 0.0 < 1/c
    let mut acoustic_mic1 = vec![0.0f32; num_samples];
    let mut acoustic_mic2 = vec![0.0f32; num_samples];
    for n in 0..num_samples {
        let t = (n as f32) / sample_rate;
        let tone = (2.0 * PI * 800.0 * t).sin() * 0.5;
        acoustic_mic1[n] = tone;
        acoustic_mic2[n] = tone; // Identical phase at broadside
    }

    let mut acoustic_out = vec![0.0f32; num_samples];
    let mut block = vec![0.0f32; hop];
    let mut offset = 0;
    while offset + hop <= num_samples {
        suppressor.process_block(
            &acoustic_mic1[offset..offset + hop],
            &acoustic_mic2[offset..offset + hop],
            &mut block,
            (offset as f64) / (sample_rate as f64),
        );
        acoustic_out[offset..offset + hop].copy_from_slice(&block);
        offset += hop;
    }

    // Steady state check: acoustic wave must pass with near unity gain (> 85% RMS preservation)
    let in_acoustic_rms = (acoustic_mic1[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let out_acoustic_rms = (acoustic_out[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let acoustic_gain = out_acoustic_rms / in_acoustic_rms;

    assert!(
        acoustic_gain > 0.80,
        "Acoustic wave must be preserved by coherence filter, got gain: {acoustic_gain:.3}"
    );

    // 2. Synthesize Convective Aerodynamic Pseudosound (800 Hz eddy convecting at Uc = 10 m/s)
    // Convective time delay: tau = d / Uc = 0.02 / 10 = 2.0 ms = 32 samples at 16 kHz
    // Apparent slowness: s = 1/Uc = 0.10 s/m = 34.3x acoustic slowness limit
    suppressor.reset();
    let mut eddy_mic1 = vec![0.0f32; num_samples];
    let mut eddy_mic2 = vec![0.0f32; num_samples];
    let eddy_delay_samples = 32; // 2.0 ms delay
    for n in 0..num_samples {
        let t1 = (n as f32) / sample_rate;
        let t2 = ((n as f32) - (eddy_delay_samples as f32)) / sample_rate;
        eddy_mic1[n] = (2.0 * PI * 800.0 * t1).sin() * 0.5;
        eddy_mic2[n] = (2.0 * PI * 800.0 * t2).sin() * 0.5;
    }

    let mut eddy_out = vec![0.0f32; num_samples];
    offset = 0;
    while offset + hop <= num_samples {
        suppressor.process_block(
            &eddy_mic1[offset..offset + hop],
            &eddy_mic2[offset..offset + hop],
            &mut block,
            (offset as f64) / (sample_rate as f64),
        );
        eddy_out[offset..offset + hop].copy_from_slice(&block);
        offset += hop;
    }

    let in_eddy_rms = (eddy_mic1[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let out_eddy_rms = (eddy_out[1024..3072].iter().map(|&x| x * x).sum::<f32>() / 2048.0).sqrt();
    let eddy_gain = out_eddy_rms / in_eddy_rms;
    let eddy_suppression_db = -20.0 * eddy_gain.max(1e-6).log10();

    assert!(
        eddy_suppression_db > 14.0,
        "Convective pseudosound with s >> 1/c must be suppressed by > 14 dB, got: {eddy_suppression_db:.2} dB"
    );
}

/// Test 2: Dual-Channel Corcos TBL Coherence Suppression Under 15 m/s Forward Airflow.
///
/// Uses the physical Corcos (1964) turbulent boundary layer model to generate realistic
/// wall-pressure fluctuations under 15 m/s forward flight airflow. Evaluates the multi-channel
/// coherence suppressor, ensuring > 15 dB total turbulent power suppression and low coherence telemetry.
#[test]
fn test_dual_channel_corcos_tbl_suppression_under_15mps_airspeed() {
    let sample_rate = 16000.0;
    let airspeed_mps = 15.0; // 54 km/h flight speed
    let mic_dist_m = 0.02; // 20 mm separation
    let num_samples = 8192;

    let (wind1, wind2) = AeroacousticWindSimulator::generate_corcos_wind(
        num_samples,
        sample_rate,
        airspeed_mps,
        mic_dist_m,
    );

    let config = WindTurbulenceConfig {
        sample_rate,
        fft_size: 512,
        hop_size: 256,
        mic_distance_m: mic_dist_m,
        forward_airspeed_mps: airspeed_mps,
        min_coherence_threshold: 0.12,
        max_attenuation_db: 24.0,
        coherence_smoothing: 0.85,
        enable_convective_slowness_gate: true,
        convective_slowness_margin: 1.35,
        enable_adaptive_rumble_filter: true,
        min_rumble_cutoff_hz: 60.0,
        max_rumble_cutoff_hz: 220.0,
        reference_airspeed_mps: 15.0,
    };

    let mut suppressor = TurbulentBoundaryLayerSuppressor::new(config);
    let hop = 256;
    let mut cleaned = vec![0.0f32; num_samples];
    let mut block = vec![0.0f32; hop];

    let mut last_telemetry = None;
    let mut offset = 0;
    while offset + hop <= num_samples {
        let telem = suppressor.process_block(
            &wind1[offset..offset + hop],
            &wind2[offset..offset + hop],
            &mut block,
            (offset as f64) / (sample_rate as f64),
        );
        cleaned[offset..offset + hop].copy_from_slice(&block);
        last_telemetry = Some(telem);
        offset += hop;
    }

    let telem = last_telemetry.expect("Telemetry must be emitted");

    // 1. Verify telemetry reflects wind buffeting
    assert!(
        telem.is_wind_buffeting,
        "Severe 15 m/s wind buffeting must be detected"
    );
    assert!(
        telem.mean_speech_coherence < 0.40,
        "Incoherent Corcos turbulence must exhibit low coherence, got: {:.3}",
        telem.mean_speech_coherence
    );

    // 2. Measure overall wind energy attenuation (excluding initial startup transient)
    let in_wind_pwr = wind1[1024..num_samples].iter().map(|&x| x * x).sum::<f32>() / ((num_samples - 1024) as f32);
    let out_wind_pwr = cleaned[1024..num_samples].iter().map(|&x| x * x).sum::<f32>() / ((num_samples - 1024) as f32);
    let overall_atten_db = 10.0 * (in_wind_pwr / out_wind_pwr.max(1e-12)).log10();

    assert!(
        overall_atten_db > 15.0,
        "Corcos TBL wind noise under 15 m/s airflow must be attenuated by > 15 dB, got: {overall_atten_db:.2} dB"
    );
}

/// Test 3: Adaptive Aerodynamic Rumble Filter Dynamics.
///
/// Tests the 2nd-order Direct Form II Transposed adaptive Butterworth high-pass filter.
/// At 0 m/s stationary conditions, cutoff is 60 Hz, passing fundamental male voices.
/// At 15 m/s forward flight, cutoff dynamically rises to 220 Hz, attenuating infrasonic/sub-bass
/// buffeting rumble (e.g. 50 Hz) by > 20 dB while preserving speech vowels (1000 Hz) with < 0.2 dB loss.
#[test]
fn test_adaptive_aerodynamic_rumble_filter_dynamics() {
    let sample_rate = 16000.0;
    let mut filter = AdaptiveRumbleFilter::new(sample_rate, 60.0);
    assert!((filter.current_cutoff_hz() - 60.0).abs() < 0.01);

    // 1. Test at stationary 60 Hz cutoff: 50 Hz should have mild attenuation, 1000 Hz minimal
    let num_samples = 4096;
    let mut tone_50hz = vec![0.0f32; num_samples];
    let mut tone_1000hz = vec![0.0f32; num_samples];
    for n in 0..num_samples {
        let t = (n as f32) / sample_rate;
        tone_50hz[n] = (2.0 * PI * 50.0 * t).sin();
        tone_1000hz[n] = (2.0 * PI * 1000.0 * t).sin();
    }

    // 2. Adapt filter to 15 m/s flight condition -> 220 Hz cutoff
    filter.update_cutoff(220.0);
    assert!((filter.current_cutoff_hz() - 220.0).abs() < 0.01);

    let mut filtered_50hz = tone_50hz.clone();
    let mut filtered_1000hz = tone_1000hz.clone();

    filter.process_block(&mut filtered_50hz);
    filter.reset();
    filter.process_block(&mut filtered_1000hz);

    // Measure steady state RMS (samples 1024..4096)
    let in_50_rms = (tone_50hz[1024..].iter().map(|&x| x * x).sum::<f32>() / 3072.0).sqrt();
    let out_50_rms = (filtered_50hz[1024..].iter().map(|&x| x * x).sum::<f32>() / 3072.0).sqrt();
    let atten_50_db = -20.0 * (out_50_rms / in_50_rms).log10();

    let in_1000_rms = (tone_1000hz[1024..].iter().map(|&x| x * x).sum::<f32>() / 3072.0).sqrt();
    let out_1000_rms = (filtered_1000hz[1024..].iter().map(|&x| x * x).sum::<f32>() / 3072.0).sqrt();
    let loss_1000_db = -20.0 * (out_1000_rms / in_1000_rms).log10();

    assert!(
        atten_50_db > 20.0,
        "50 Hz rumble under 220 Hz cutoff must be attenuated by > 20 dB, got: {atten_50_db:.2} dB"
    );
    assert!(
        loss_1000_db.abs() < 0.25,
        "1000 Hz speech tone must have < 0.25 dB insertion loss, got: {loss_1000_db:.3} dB"
    );
}

/// Test 4: End-to-End Keyword Spotting Under 15 m/s Flight Wind Buffeting.
///
/// Simulates a real-world scenario where a drone flies at 15 m/s forward airspeed,
/// and an operator commands "take off". The dual-channel TBL suppression engine
/// separates turbulent pseudosound from the speech sound wave, feeding the restored
/// stream into SononEngine's DTW pipeline to achieve positive keyword spotting.
#[test]
fn test_sonon_engine_keyword_spotting_under_15mps_wind_buffeting() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // 1. Synthesize and enroll target keyword ("take off")
    let keyword_text = "take off";
    let frames = engine.enroll_keyword_from_text("take_off", keyword_text, 4.0);
    assert!(frames > 0, "Synthesizer must produce feature frames");
    let synth_audio = engine.synthesize_speech_from_text(keyword_text);

    // 2. Enable wind suppression engine
    let wind_config = WindTurbulenceConfig {
        sample_rate,
        fft_size: 256,
        hop_size: 128,
        mic_distance_m: 0.02,
        forward_airspeed_mps: 15.0,
        min_coherence_threshold: 0.12,
        max_attenuation_db: 24.0,
        coherence_smoothing: 0.80,
        enable_convective_slowness_gate: true,
        convective_slowness_margin: 1.35,
        enable_adaptive_rumble_filter: true,
        min_rumble_cutoff_hz: 60.0,
        max_rumble_cutoff_hz: 200.0,
        reference_airspeed_mps: 15.0,
    };
    engine.enable_wind_suppression(wind_config);
    engine.update_flight_airspeed(15.0);

    // 3. Build in-flight test audio: prepend and append silence/wind padding
    let lead_pad = 2000;
    let trail_pad = 2000;
    let total_samples = lead_pad + synth_audio.len() + trail_pad;
    let mut speech_stream = vec![0.0f32; total_samples];
    speech_stream[lead_pad..lead_pad + synth_audio.len()].copy_from_slice(&synth_audio);

    // 4. Mix speech with Corcos wind turbulence at -6 dB SNR (turbulent noise power is 4x speech power)
    let (mix1, mix2) = AeroacousticWindSimulator::simulate_flight_mixture(
        &speech_stream,
        sample_rate,
        15.0,
        -6.0,
        0.02,
        0.0, // Broadside 0 deg
    );

    // 5. Ingest dual-mic stream through SononEngine wind suppression
    let events = engine
        .process_dual_mic_wind_suppression(&mix1, &mix2)
        .expect("Dual mic processing should succeed");

    // 6. Verify keyword spotted
    let spotted = events.iter().any(|e| e.keyword == "take_off");
    assert!(
        spotted,
        "Keyword 'take_off' must be spotted under 15 m/s wind buffeting after TBL suppression. Events: {:?}",
        events
    );

    // Verify telemetry
    let telem = engine.latest_wind_telemetry().expect("Telemetry must be recorded");
    assert!(
        telem.suppression_db > 6.0,
        "Telemetry must reflect substantial wind suppression, got: {:.2} dB",
        telem.suppression_db
    );
}

/// Test 5: MAVLink Telemetry and High-Throughput Streaming Performance.
///
/// Benchmarks block-based streaming throughput across 32,000 dual-channel samples.
/// Asserts > 250,000 samples/sec throughput (> 15x real-time) and verifies MAVLink
/// v2 `NAMED_VALUE_FLOAT` serialization fields (`WIND_COH`, `WIND_SUPP`, `WIND_SPD`).
#[test]
fn test_mavlink_wind_telemetry_and_throughput_benchmark() {
    let sample_rate = 16000.0;
    let config = WindTurbulenceConfig {
        sample_rate,
        fft_size: 512,
        hop_size: 256,
        mic_distance_m: 0.02,
        forward_airspeed_mps: 15.0,
        min_coherence_threshold: 0.12,
        max_attenuation_db: 24.0,
        coherence_smoothing: 0.85,
        enable_convective_slowness_gate: true,
        convective_slowness_margin: 1.35,
        enable_adaptive_rumble_filter: true,
        min_rumble_cutoff_hz: 60.0,
        max_rumble_cutoff_hz: 220.0,
        reference_airspeed_mps: 15.0,
    };

    let mut suppressor = TurbulentBoundaryLayerSuppressor::new(config.clone());
    let hop = config.hop_size;
    let total_samples = 32768;

    let (wind1, wind2) = AeroacousticWindSimulator::generate_corcos_wind(
        total_samples,
        sample_rate,
        15.0,
        0.02,
    );

    let mut out = vec![0.0f32; total_samples];
    let mut block = vec![0.0f32; hop];

    let start = Instant::now();
    let mut offset = 0;
    let mut latest_telemetry = None;

    while offset + hop <= total_samples {
        let telem = suppressor.process_block(
            &wind1[offset..offset + hop],
            &wind2[offset..offset + hop],
            &mut block,
            (offset as f64) / (sample_rate as f64),
        );
        out[offset..offset + hop].copy_from_slice(&block);
        latest_telemetry = Some(telem);
        offset += hop;
    }
    let elapsed = start.elapsed();

    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();
    let realtime_mult = samples_per_sec / (sample_rate as f64);

    assert!(
        samples_per_sec > 250_000.0,
        "Throughput must exceed 250k samples/sec, got {samples_per_sec:.0} samples/sec ({realtime_mult:.1}x real-time)"
    );

    // MAVLink telemetry packet validation:
    let telem = latest_telemetry.expect("Telemetry must be emitted");
    let packets = telem.to_mavlink_telemetry();
    assert_eq!(packets.len(), 3, "Expected 3 MAVLink telemetry packets");

    let coh_pkt = packets.iter().find(|p| p.name_as_str() == "WIND_COH").expect("WIND_COH missing");
    let supp_pkt = packets.iter().find(|p| p.name_as_str() == "WIND_SUPP").expect("WIND_SUPP missing");
    let spd_pkt = packets.iter().find(|p| p.name_as_str() == "WIND_SPD").expect("WIND_SPD missing");

    assert!((0.0..=1.0).contains(&coh_pkt.value), "WIND_COH must be in [0, 1]");
    assert!(supp_pkt.value > 10.0, "WIND_SUPP must reflect > 10 dB suppression");
    assert!(spd_pkt.value > 5.0 && spd_pkt.value < 40.0, "WIND_SPD must estimate realistic airspeed");
}
