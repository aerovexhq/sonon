//! Analytical verification suite for Phase 21: Bio-Inspired Micro-Tympanum
//! Differential Microphone Emulation (*Ormia Ochracea* Mechanics).

#![deny(unsafe_code)]

use sonon::{
    OrmiaBiquad, OrmiaBridgeFilter, OrmiaConfig, OrmiaDirectionEstimator,
    SononEngine,
};
use std::f32::consts::PI;

/// Test 1: Biquad pole stability, coefficient validity, and modal resonance.
#[test]
fn test_biquad_pole_stability_and_resonance() {
    let sample_rate = 16000.0;
    let f_bending = 2200.0;
    let q_bending = 2.2;

    let mut biquad = OrmiaBiquad::new(f_bending, q_bending, sample_rate, 1.0);

    // 1. Verify Schur-Cohn / Jury stability criteria for 2nd order discrete filter:
    //    a2 < 1.0, a2 > -1.0, and |a1| < 1.0 + a2.
    let a = biquad.a_coeffs();
    let a1 = a[0];
    let a2 = a[1];

    assert!(a2 < 1.0, "a2 must be strictly less than 1.0 for stability");
    assert!(a2 > -1.0, "a2 must be strictly greater than -1.0 for stability");
    assert!(
        a1.abs() < 1.0 + a2,
        "|a1| must be less than 1 + a2 for stable poles inside unit circle"
    );

    // 2. Impulse response stability test: feed delta impulse
    let mut impulse_response = Vec::with_capacity(500);
    impulse_response.push(biquad.process(1.0));
    for _ in 1..500 {
        impulse_response.push(biquad.process(0.0));
    }

    // Verify impulse response is non-trivial and decays asymptotically to zero
    assert!(impulse_response[0] > 0.0, "Impulse onset must be non-zero");
    let tail_max = impulse_response[400..]
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(
        tail_max < 1e-4,
        "Impulse response must decay to zero (got tail max {})",
        tail_max
    );

    for (idx, &val) in impulse_response.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Impulse response sample {} must be finite",
            idx
        );
    }

    // 3. Frequency response verification
    let (dc_mag, _) = biquad.frequency_response(0.1, sample_rate);
    assert!(
        (dc_mag - 1.0).abs() < 0.05,
        "DC gain should be approximately 1.0 (got {})",
        dc_mag
    );

    let (res_mag, _) = biquad.frequency_response(f_bending, sample_rate);
    assert!(
        (res_mag - q_bending).abs() < 0.25,
        "Gain at resonance should be approximately Q = {} (got {})",
        q_bending,
        res_mag
    );

    let (hf_mag, _) = biquad.frequency_response(6500.0, sample_rate);
    assert!(
        hf_mag < 0.35,
        "High frequency magnitude should be strongly attenuated (got {})",
        hf_mag
    );

    // 4. Repeat check for anti-symmetric rocking biquad
    let f_rocking = 3100.0;
    let q_rocking = 3.5;
    let biquad_rock = OrmiaBiquad::new(f_rocking, q_rocking, sample_rate, 0.5);
    let a_rock = biquad_rock.a_coeffs();
    assert!(a_rock[1] < 1.0 && a_rock[1] > -1.0);
    assert!(a_rock[0].abs() < 1.0 + a_rock[1]);
}

/// Test 2: Sub-2mm dual MEMS acoustic delay and spatial IID amplification (> 20 dB).
#[test]
fn test_sub_2mm_itd_and_iid_spatial_amplification() {
    let mut config = OrmiaConfig::default();
    config.sample_rate = 16000.0;
    config.mic_distance_m = 0.0012; // 1.2 mm
    config.speed_of_sound = 343.0;
    config.f_bending = 2200.0;
    config.q_bending = 2.2;
    config.f_rocking = 3100.0;
    config.q_rocking = 3.5;

    let mut filter = OrmiaBridgeFilter::new(config.clone());

    // 1. Off-axis incidence at theta = +45 degrees
    let theta_rad = 45.0f32.to_radians();
    let delay_sec = (config.mic_distance_m * theta_rad.sin()) / config.speed_of_sound;
    let f_test = 2600.0f32; // In between bending and rocking modes
    let phase_delay_rad = 2.0 * PI * f_test * delay_sec;

    let num_samples = 3000;
    let mut s1 = vec![0.0f32; num_samples];
    let mut s2 = vec![0.0f32; num_samples];

    for n in 0..num_samples {
        let t = n as f32 / config.sample_rate;
        s1[n] = (2.0 * PI * f_test * t).sin();
        s2[n] = (2.0 * PI * f_test * t - phase_delay_rad).sin();
    }

    // Verify raw acoustic input has 0 dB IID (identical amplitude)
    let raw_p1: f32 = s1[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let raw_p2: f32 = s2[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let raw_iid_db = 10.0 * (raw_p1 / raw_p2).log10();
    assert!(
        raw_iid_db.abs() < 0.05,
        "Raw input signals must have 0 dB IID (got {})",
        raw_iid_db
    );

    // Pass through mechanical bridge filter
    let mut x1 = vec![0.0f32; num_samples];
    let mut x2 = vec![0.0f32; num_samples];
    filter.process_block(&s1, &s2, &mut x1, &mut x2);

    // Measure mechanical outputs over steady-state tail
    let mech_p1: f32 = x1[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let mech_p2: f32 = x2[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let mech_iid_db = 10.0 * (mech_p1 / mech_p2).log10();

    // Verify mechanical bridge delivers > 20 dB of effective IID amplification!
    assert!(
        mech_iid_db > 20.0,
        "Mechanical IID amplification must exceed 20 dB (got {} dB)",
        mech_iid_db
    );
    assert!(
        mech_p1 > 100.0 * mech_p2,
        "Ipsilateral power must exceed contralateral by > 100x"
    );

    // 2. Broadside symmetry test (theta = 0 deg)
    filter.reset();
    let mut x1_zero = vec![0.0f32; num_samples];
    let mut x2_zero = vec![0.0f32; num_samples];
    filter.process_block(&s1, &s1, &mut x1_zero, &mut x2_zero);

    let zero_p1: f32 = x1_zero[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let zero_p2: f32 = x2_zero[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let zero_iid_db = 10.0 * (zero_p1 / zero_p2).log10();
    assert!(
        zero_iid_db.abs() < 0.05,
        "Broadside sound must produce 0 dB IID (got {})",
        zero_iid_db
    );

    // 3. Contralateral inversion test (theta = -45 deg)
    filter.reset();
    let mut x1_neg = vec![0.0f32; num_samples];
    let mut x2_neg = vec![0.0f32; num_samples];
    filter.process_block(&s2, &s1, &mut x1_neg, &mut x2_neg); // s2 leads s1

    let neg_p1: f32 = x1_neg[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let neg_p2: f32 = x2_neg[1500..].iter().map(|&x| x * x).sum::<f32>() / 1500.0;
    let neg_iid_db = 10.0 * (neg_p1 / neg_p2).log10();
    assert!(
        neg_iid_db < -20.0,
        "Contralateral incidence must produce < -20 dB IID (got {} dB)",
        neg_iid_db
    );
}

/// Test 3: Direction-of-Arrival (DoA) azimuth estimation linearity, monotonicity, and RMSE.
#[test]
fn test_azimuth_estimation_linearity_and_rmse() {
    let mut config = OrmiaConfig::default();
    config.sample_rate = 16000.0;
    config.smoothing_alpha = 0.08;
    config.iid_calibration_db = 26.0;

    let angles_deg = [-60.0f32, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0];
    let mut estimated_angles = Vec::new();
    let f_test = 2600.0f32;
    let num_samples = 2000;

    for &theta_deg in &angles_deg {
        let mut estimator = OrmiaDirectionEstimator::new(config.clone());
        let theta_rad = theta_deg.to_radians();
        let delay_sec = (config.mic_distance_m * theta_rad.sin()) / config.speed_of_sound;
        let phase_delay_rad = 2.0 * PI * f_test * delay_sec;

        let mut mic1 = vec![0.0f32; num_samples];
        let mut mic2 = vec![0.0f32; num_samples];
        for n in 0..num_samples {
            let t = n as f32 / config.sample_rate;
            mic1[n] = (2.0 * PI * f_test * t).sin();
            mic2[n] = (2.0 * PI * f_test * t - phase_delay_rad).sin();
        }

        let mut out1 = vec![0.0f32; num_samples];
        let mut out2 = vec![0.0f32; num_samples];
        let telem = estimator.process_block(&mic1, &mic2, &mut out1, &mut out2);
        estimated_angles.push(telem.azimuth_deg);

        // Verify MAVLink telemetry packet generation
        let packets = telem.to_mavlink_packets(100);
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].name_as_str(), "ORM_AZIM");
        assert_eq!(packets[1].name_as_str(), "ORM_IID");
        assert_eq!(packets[2].name_as_str(), "ORM_GAIN");
        assert!(telem.confidence > 0.5);
    }

    // 1. Verify strict monotonicity
    for i in 0..estimated_angles.len() - 1 {
        assert!(
            estimated_angles[i] < estimated_angles[i + 1],
            "Azimuth estimates must be strictly monotonic: {} >= {}",
            estimated_angles[i],
            estimated_angles[i + 1]
        );
    }

    // 2. Verify broadside center is accurate
    let center_idx = angles_deg.iter().position(|&x| x == 0.0).unwrap();
    assert!(
        estimated_angles[center_idx].abs() < 1.0,
        "Broadside estimate must be within 1.0 degree of zero (got {})",
        estimated_angles[center_idx]
    );

    // 3. Compute Root-Mean-Square Error (RMSE) across angles
    let mut sum_sq_err = 0.0f32;
    for (i, &true_angle) in angles_deg.iter().enumerate() {
        let err = estimated_angles[i] - true_angle;
        sum_sq_err += err * err;
    }
    let rmse = (sum_sq_err / angles_deg.len() as f32).sqrt();
    assert!(
        rmse < 6.5,
        "Azimuth estimation RMSE must be below 6.5 degrees (got {})",
        rmse
    );
}

/// Test 4: Drone common-mode rotor noise rejection and voice SNR improvement.
#[test]
fn test_drone_noise_rejection_with_ormia_bridge() {
    let sample_rate = 16000.0;
    let mut config = OrmiaConfig::default();
    config.sample_rate = sample_rate;
    config.f_bending = 2200.0;
    config.f_rocking = 3100.0;

    let num_samples = 3200;
    let mut mic1 = vec![0.0f32; num_samples];
    let mut mic2 = vec![0.0f32; num_samples];

    // Rotor noise: symmetric/diffuse low-frequency harmonics (BPF = 200 Hz, 400 Hz, 600 Hz)
    // Common-mode to both microphones (tau_noise = 0)
    for n in 0..num_samples {
        let t = n as f32 / sample_rate;
        let rotor_noise = 0.5 * (2.0 * PI * 200.0 * t).sin()
            + 0.3 * (2.0 * PI * 400.0 * t).sin()
            + 0.2 * (2.0 * PI * 600.0 * t).sin();
        mic1[n] += rotor_noise;
        mic2[n] += rotor_noise;
    }

    // Target voice formant at 2500 Hz arriving from theta = +35 degrees
    let theta_rad = 35.0f32.to_radians();
    let delay_sec = (config.mic_distance_m * theta_rad.sin()) / config.speed_of_sound;
    let f_voice = 2500.0f32;
    let phase_delay_rad = 2.0 * PI * f_voice * delay_sec;
    let voice_amp = 0.15f32;

    for n in 0..num_samples {
        let t = n as f32 / sample_rate;
        mic1[n] += voice_amp * (2.0 * PI * f_voice * t).sin();
        mic2[n] += voice_amp * (2.0 * PI * f_voice * t - phase_delay_rad).sin();
    }

    // Calculate raw input voice-to-noise power ratio:
    // Rotor noise power = 0.5^2 / 2 + 0.3^2 / 2 + 0.2^2 / 2 = 0.125 + 0.045 + 0.02 = 0.19
    // Voice power = 0.15^2 / 2 = 0.01125
    let raw_voice_power = voice_amp * voice_amp * 0.5;
    let raw_noise_power = (0.5f32.powi(2) + 0.3f32.powi(2) + 0.2f32.powi(2)) * 0.5;
    let raw_snr_db = 10.0 * (raw_voice_power / raw_noise_power).log10();
    assert!(
        raw_snr_db < -10.0,
        "Input SNR must be deeply negative (got {} dB)",
        raw_snr_db
    );

    // Process through Ormia bridge
    let mut bridge = OrmiaBridgeFilter::new(config.clone());
    let mut out1 = vec![0.0f32; num_samples];
    let mut out2 = vec![0.0f32; num_samples];
    bridge.process_block(&mic1, &mic2, &mut out1, &mut out2);

    // Evaluate frequency component amplitudes using DFT bin at 200 Hz (noise) vs 2500 Hz (voice)
    let steady_slice = &out1[1600..3200];
    let n_pts = steady_slice.len();

    let mut voice_re = 0.0f32;
    let mut voice_im = 0.0f32;
    let mut noise_re = 0.0f32;
    let mut noise_im = 0.0f32;

    for (i, &s) in steady_slice.iter().enumerate() {
        let t = (1600 + i) as f32 / sample_rate;
        voice_re += s * (2.0 * PI * f_voice * t).cos();
        voice_im += s * (2.0 * PI * f_voice * t).sin();

        noise_re += s * (2.0 * PI * 200.0 * t).cos();
        noise_im += s * (2.0 * PI * 200.0 * t).sin();
    }

    let out_voice_power = (voice_re * voice_re + voice_im * voice_im) / (n_pts as f32).powi(2);
    let out_noise_power = (noise_re * noise_re + noise_im * noise_im) / (n_pts as f32).powi(2);
    let out_snr_db = 10.0 * (out_voice_power / out_noise_power).log10();

    let snr_improvement_db = out_snr_db - raw_snr_db;
    assert!(
        snr_improvement_db > 9.0,
        "Ormia bridge must improve voice SNR by > 9.0 dB against drone rotor noise (got {} dB improvement)",
        snr_improvement_db
    );

    // Integrate with SononEngine
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.enable_ormia_bridge(config);
    let events = engine.process_dual_mic_ormia(&mic1, &mic2).unwrap();
    assert!(events.is_empty() || !events.is_empty()); // Runs cleanly without error

    let telem = engine.latest_ormia_telemetry().unwrap();
    assert!(
        telem.azimuth_deg > 0.5,
        "Estimated azimuth should indicate voice incident from right side (got {})",
        telem.azimuth_deg
    );
    assert!(
        telem.amplification_gain_db > 0.5,
        "Amplification gain should be positive under noisy mixture (got {})",
        telem.amplification_gain_db
    );
}

/// Test 5: High-throughput embedded streaming benchmark (> 1,500,000 samples/sec).
#[test]
fn test_ormia_streaming_throughput_benchmark() {
    let mut config = OrmiaConfig::default();
    config.sample_rate = 16000.0;
    let mut estimator = OrmiaDirectionEstimator::new(config);

    let num_samples = 250_000;
    let mut s1 = vec![0.0f32; num_samples];
    let mut s2 = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        let t = i as f32 / 16000.0;
        s1[i] = (2.0 * PI * 2400.0 * t).sin();
        s2[i] = (2.0 * PI * 2400.0 * t + 0.05).sin();
    }

    let mut out1 = vec![0.0f32; num_samples];
    let mut out2 = vec![0.0f32; num_samples];

    let start = std::time::Instant::now();
    let telem = estimator.process_block(&s1, &s2, &mut out1, &mut out2);
    let elapsed = start.elapsed();

    let elapsed_sec = elapsed.as_secs_f64();
    let throughput = num_samples as f64 / elapsed_sec;

    println!(
        "Ormia bridge streaming throughput: {:.2} samples/sec ({:.2}x real-time at 16kHz)",
        throughput,
        throughput / 16000.0
    );

    assert!(
        throughput > 1_500_000.0,
        "Throughput must exceed 1,500,000 samples/sec (got {:.2} samples/sec)",
        throughput
    );
    assert!(telem.confidence > 0.0);
}
