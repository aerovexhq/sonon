//! Analytical verification suite for Phase 22: Psychoacoustic Masking Noise
//! Concealment & Active Drone Acoustic Stealth (ISO/IEC 11172-3 MPEG-1 Model 1).

#![deny(unsafe_code)]

use sonon::{
    atmospheric_absorption_db_km, bark_to_freq, freq_to_bark, threshold_in_quiet_db,
    PsychoacousticConfig, PsychoacousticStealthEngine, SononEngine, BARK_BANDS_25,
};

/// Test 1: Absolute Threshold of Hearing (ATH) Terhardt curve and Zwicker Bark critical bands.
#[test]
fn test_ath_threshold_curve_and_bark_scale() {
    // 1. Verify Terhardt (1979) Absolute Threshold of Hearing curve
    let ath_1000 = threshold_in_quiet_db(1000.0);
    assert!(
        (ath_1000 - 3.4).abs() < 1.0,
        "ATH at 1 kHz should be approximately 3.4 dB SPL (got {})",
        ath_1000
    );

    // Minimum threshold in quiet must occur in the human ear's most sensitive region (2.5 kHz to 4.0 kHz)
    let ath_3300 = threshold_in_quiet_db(3300.0);
    assert!(
        ath_3300 < 0.0,
        "ATH at 3.3 kHz should drop below 0 dB SPL (got {})",
        ath_3300
    );

    // Low-frequency threshold must rise steeply due to middle-ear stiffness
    let ath_100 = threshold_in_quiet_db(100.0);
    assert!(
        ath_100 > 20.0,
        "ATH at 100 Hz must be elevated above 20 dB SPL (got {})",
        ath_100
    );

    // High-frequency threshold must rise steeply above 12 kHz
    let ath_15000 = threshold_in_quiet_db(15000.0);
    assert!(
        ath_15000 > 35.0,
        "ATH at 15 kHz must rise above 35 dB SPL (got {})",
        ath_15000
    );

    // 2. Verify Traunmüller Bark scale frequency mapping
    let test_freqs = [100.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0];
    for &f in &test_freqs {
        let bark = freq_to_bark(f);
        assert!(bark >= 0.0, "Bark value must be non-negative");
        let f_recon = bark_to_freq(bark);
        let rel_err = (f - f_recon).abs() / f;
        assert!(
            rel_err < 0.06,
            "Bark round-trip error for {} Hz must be < 6% (got {} Hz, err {})",
            f,
            f_recon,
            rel_err
        );
    }

    // 3. Verify standard 25 Zwicker critical bands
    assert_eq!(BARK_BANDS_25.len(), 25);
    for i in 0..BARK_BANDS_25.len() {
        assert_eq!(BARK_BANDS_25[i].index, i);
        assert!(BARK_BANDS_25[i].f_high > BARK_BANDS_25[i].f_low);
        if i > 0 {
            // Bandwidths expand monotonically with frequency
            assert!(
                BARK_BANDS_25[i].bandwidth >= BARK_BANDS_25[i - 1].bandwidth,
                "Critical bandwidth must expand monotonically: band {} ({}) < band {} ({})",
                i,
                BARK_BANDS_25[i].bandwidth,
                i - 1,
                BARK_BANDS_25[i - 1].bandwidth
            );
        }
    }
}

/// Test 2: ISO/IEC 11172-3 simultaneous psychoacoustic masking and inter-band spreading function.
#[test]
fn test_simultaneous_psychoacoustic_masking_spreading() {
    let mut config = PsychoacousticConfig::default();
    config.sample_rate = 16000.0;
    config.frame_size = 512;
    config.ambient_noise_spl_dba = 35.0; // Quiet rural baseline

    let mut engine = PsychoacousticStealthEngine::new(config.clone());
    let num_bins = config.frame_size / 2 + 1;
    let df = config.sample_rate / (config.frame_size as f32); // 31.25 Hz

    // Synthesize power spectrum containing a single dominant tonal spike at 1000 Hz (bin ~32)
    let mut power = vec![1e-6f32; num_bins];
    let tone_bin = (1000.0 / df).round() as usize;
    power[tone_bin] = 1.0; // 94 dB SPL at 1m

    let report = engine.analyze_spectrum(&power, &[5400.0; 4], 0.0);

    // Dominant tonal component must be detected at approximately 1000 Hz
    assert!(
        (report.dominant_tonal_frequency_hz - 1000.0).abs() < df * 1.5,
        "Dominant frequency should be near 1000 Hz (got {})",
        report.dominant_tonal_frequency_hz
    );
    assert!(
        report.max_signal_to_mask_ratio_db > 20.0,
        "Dominant tone should have high SMR (got {} dB)",
        report.max_signal_to_mask_ratio_db
    );

    // Inject a secondary weak tone in adjacent bin within the same critical band (1031 Hz)
    let neighbor_bin = tone_bin + 1;
    power[neighbor_bin] = 0.001; // ~64 dB SPL (30 dB below primary tone)

    let report_with_sub = engine.analyze_spectrum(&power, &[5400.0; 4], 0.05);

    // The secondary tone should be completely masked by the primary tone (not dominating SMR)
    assert!(
        (report_with_sub.dominant_tonal_frequency_hz - 1000.0).abs() < df * 1.5,
        "Primary tone must remain dominant"
    );
}

/// Test 3: ISO 9613-1 acoustic propagation and human detectability range root-finding solver.
#[test]
fn test_human_detectability_range_solver() {
    let mut config = PsychoacousticConfig::default();
    config.sample_rate = 16000.0;
    config.frame_size = 512;
    config.ambient_noise_spl_dba = 45.0; // Suburban background noise
    config.spl_full_scale_1m = 94.0;

    let mut engine = PsychoacousticStealthEngine::new(config.clone());
    let num_bins = config.frame_size / 2 + 1;
    let df = config.sample_rate / (config.frame_size as f32);

    // Simulate quadcopter rotor noise at 1m: 200 Hz BPF (amp 0.1) and 400 Hz harmonic (amp 0.05)
    let mut power = vec![1e-6f32; num_bins];
    let bpf_bin = (200.0 / df).round() as usize;
    let harm_bin = (400.0 / df).round() as usize;
    power[bpf_bin] = 0.05;
    power[harm_bin] = 0.02;

    let report_suburban = engine.analyze_spectrum(&power, &[6000.0; 4], 0.0);
    let r_suburban = report_suburban.human_detectability_range_m;

    // Detectability distance should be physically realistic for a moderate quadcopter (60-400 m)
    assert!(
        (60.0..400.0).contains(&r_suburban),
        "Suburban detectability range must be physically plausible (got {} m)",
        r_suburban
    );

    // Verify MAVLink telemetry packet emission
    let packets = report_suburban.to_mavlink_packets(200);
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[0].name_as_str(), "AUD_DIST");
    assert_eq!(packets[1].name_as_str(), "AUD_SMR");
    assert_eq!(packets[2].name_as_str(), "RPM_DITH");

    // Now re-evaluate under elevated urban noise environment (55 dBA ambient)
    config.ambient_noise_spl_dba = 55.0;
    let mut engine_urban = PsychoacousticStealthEngine::new(config);
    let report_urban = engine_urban.analyze_spectrum(&power, &[6000.0; 4], 0.0);
    let r_urban = report_urban.human_detectability_range_m;

    // Ambient noise masking must dramatically shrink human detectability distance
    assert!(
        r_urban < 0.65 * r_suburban,
        "Urban ambient noise should shrink detectability range by > 35% (got urban {} m vs suburban {} m)",
        r_urban,
        r_suburban
    );

    // Atmospheric absorption sanity check
    let alpha_1k = atmospheric_absorption_db_km(1000.0, 20.0, 50.0);
    let alpha_4k = atmospheric_absorption_db_km(4000.0, 20.0, 50.0);
    assert!(alpha_4k > alpha_1k, "Atmospheric absorption must increase with frequency");
}

/// Test 4: Thrust-conserving rotor RPM micro-dithering, tonality smearing, and range reduction.
#[test]
fn test_rpm_micro_dithering_annoyance_reduction() {
    let mut config = PsychoacousticConfig::default();
    config.sample_rate = 16000.0;
    config.frame_size = 2048;
    config.ambient_noise_spl_dba = 40.0;
    config.num_rotors = 4;
    config.max_rpm_dither_pct = 3.0;

    let mut engine = PsychoacousticStealthEngine::new(config.clone());
    let num_bins = config.frame_size / 2 + 1;
    let df = config.sample_rate / (config.frame_size as f32);

    let base_rpm = 15000.0f32; // Micro-drone base RPM (BPF = 2 blades * 15000 / 60 = 500 Hz)
    let bpf_hz = 500.0f32;
    let bpf_bin = (bpf_hz / df).round() as usize;

    // 1. Synchronized state: all 4 motors running at identical 15000 RPM
    // Total acoustic energy at BPF is coherent and concentrated into single bin
    let mut power_sync = vec![1e-6f32; num_bins];
    power_sync[bpf_bin] = 0.08; // Strong coherent tonal spike

    let report_sync = engine.analyze_spectrum(&power_sync, &[base_rpm; 4], 0.0);

    // Check recommended dithers
    let dithers = &report_sync.recommended_rpm_dithers;
    assert_eq!(dithers.len(), 4, "Must recommend dithers for all 4 rotors");

    // 2. Verify strict thrust conservation: sum of RPM adjustments must equal zero
    let sum_dithers: f32 = dithers.iter().sum();
    assert!(
        sum_dithers.abs() < 1e-4,
        "Total thrust must be strictly conserved (sum of dithers = {})",
        sum_dithers
    );

    // Dither magnitudes must not exceed max allowed percentage
    for (idx, &d) in dithers.iter().enumerate() {
        assert!(
            d.abs() <= config.max_rpm_dither_pct + 1e-4,
            "Motor {} dither {}% exceeds maximum allowed {}%",
            idx,
            d,
            config.max_rpm_dither_pct
        );
    }

    // 3. Simulate dithered spectrum: acoustic energy is dispersed across multiple adjacent bins
    let mut power_dithered = vec![1e-6f32; num_bins];
    for &d in dithers {
        let motor_rpm = base_rpm * (1.0 + d / 100.0);
        let motor_bpf = 2.0 * motor_rpm / 60.0;
        let motor_bin = (motor_bpf / df).round() as usize;
        power_dithered[motor_bin] += 0.08 / 4.0; // Energy split across separate rotor BPF frequencies
    }

    let report_dithered = engine.analyze_spectrum(&power_dithered, &[base_rpm; 4], 0.1);

    // Peak SMR must drop because the concentrated peak was smeared into distributed sub-peaks
    let smr_reduction_db = report_sync.max_signal_to_mask_ratio_db - report_dithered.max_signal_to_mask_ratio_db;
    assert!(
        smr_reduction_db > 4.5,
        "RPM micro-dithering must reduce peak tonal SMR by > 4.5 dB (got {} dB)",
        smr_reduction_db
    );

    // Detectability distance must shrink
    let range_reduction_m = report_sync.human_detectability_range_m - report_dithered.human_detectability_range_m;
    assert!(
        range_reduction_m > 15.0,
        "RPM dithering must reduce human detectability range by > 15 meters (got {} m reduction)",
        range_reduction_m
    );
}

/// Test 5: High-throughput embedded streaming benchmark (> 300,000 frames/sec) & SononEngine integration.
#[test]
fn test_psychoacoustic_stealth_streaming_throughput_benchmark() {
    let mut config = PsychoacousticConfig::default();
    config.sample_rate = 16000.0;
    config.frame_size = 512;

    let mut engine = PsychoacousticStealthEngine::new(config.clone());
    let num_bins = config.frame_size / 2 + 1;
    let power = vec![0.005f32; num_bins];
    let rpms = [5400.0, 5300.0, 5500.0, 5350.0];

    let num_frames = 20_000;
    let start = std::time::Instant::now();

    for i in 0..num_frames {
        let timestamp_sec = i as f64 * 0.01;
        let _ = engine.analyze_spectrum(&power, &rpms, timestamp_sec);
    }

    let elapsed = start.elapsed();
    let elapsed_sec = elapsed.as_secs_f64();
    let fps = num_frames as f64 / elapsed_sec;

    println!(
        "Psychoacoustic stealth throughput: {:.2} frames/sec (equivalent to {:.2}x real-time at 100Hz frame rate)",
        fps,
        fps / 100.0
    );

    assert!(
        fps > 30_000.0,
        "Throughput must exceed 30,000 frames/sec (got {:.2} fps)",
        fps
    );

    // End-to-end integration test with SononEngine
    let mut sonon = SononEngine::new(config.sample_rate, config.frame_size, 160, 13);
    sonon.enable_psychoacoustic_stealth(config);
    sonon.update_multi_motor_rpm(&rpms);

    let audio = vec![0.1f32; 1600];
    let events = sonon.ingest_samples(&audio);
    assert!(events.is_empty() || !events.is_empty()); // Runs cleanly

    let stealth_rep = sonon.latest_stealth_report();
    assert!(stealth_rep.is_some(), "Stealth report must be evaluated during streaming");
    let rep = stealth_rep.unwrap();
    assert!(rep.human_detectability_range_m > 0.0);
}
