#![deny(unsafe_code)]

use sonon::{
    ArrayGeometry, DelayAndSumBeamformer, DoaEstimator, GccPhatEstimator, SPEED_OF_SOUND,
};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_array_geometry_coordinate_generation() {
    // 1. Linear Array (lateral wing mount along Y-axis)
    let linear = ArrayGeometry::Linear {
        spacing: 0.04,
        num_mics: 4,
    };
    let pts_lin = linear.positions();
    assert_eq!(pts_lin.len(), 4);
    assert!((pts_lin[1].y - pts_lin[0].y - 0.04).abs() < 1e-4);
    assert!((pts_lin[3].y - pts_lin[0].y - 0.12).abs() < 1e-4);

    // 2. Circular Array
    let circular = ArrayGeometry::Circular {
        radius: 0.05,
        num_mics: 4,
    };
    let pts_circ = circular.positions();
    assert_eq!(pts_circ.len(), 4);
    for p in &pts_circ {
        let dist = (p.x * p.x + p.y * p.y).sqrt();
        assert!((dist - 0.05).abs() < 1e-4, "Mic must lie on circle of radius 0.05");
    }

    // 3. Tetrahedral Array
    let tetra = ArrayGeometry::Tetrahedral { radius: 0.06 };
    let pts_tet = tetra.positions();
    assert_eq!(pts_tet.len(), 4);
    for p in &pts_tet {
        let dist = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        assert!((dist - 0.06).abs() < 1e-3, "Mic must lie on sphere of radius 0.06");
    }
}

#[test]
fn test_gcc_phat_tdoa_estimation() {
    let sample_rate = 16000.0;
    let fft_size = 512;
    let estimator = GccPhatEstimator::new(fft_size, sample_rate);

    // Generate band-limited speech-like formant wave (1200 Hz to 2400 Hz)
    let n = 256;
    let base_sig: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.6 * (2.0 * PI * 1200.0 * t).sin() + 0.4 * (2.0 * PI * 2400.0 * t).sin()
        })
        .collect();

    // Delay channel 2 by 4 samples (4 / 16000 = 0.00025 seconds)
    let delay_samples = 4;
    let expected_delay_sec = delay_samples as f32 / sample_rate;

    let mut ch1 = vec![0.0f32; n];
    let mut ch2 = vec![0.0f32; n];

    for i in 0..n {
        ch1[i] = base_sig[i];
        if i >= delay_samples {
            ch2[i] = base_sig[i - delay_samples];
        }
    }

    let (estimated_delay, peak) = estimator.estimate_tdoa(&ch1, &ch2, 0.002);
    println!(
        "GCC-PHAT estimated delay: {estimated_delay:.6} s, expected: {expected_delay_sec:.6} s (peak: {peak:.3})"
    );

    assert!(peak > 0.01, "Correlation peak must be positive");
    assert!(
        (estimated_delay - expected_delay_sec).abs() < 0.0001,
        "Estimated delay must match ground truth within sub-sample tolerance, got {estimated_delay} vs {expected_delay_sec}"
    );
}

#[test]
fn test_linear_array_azimuth_estimation() {
    let sample_rate = 16000.0;
    let spacing = 0.10; // 10 cm drone wing baseline
    let doa = DoaEstimator::new(
        ArrayGeometry::Linear {
            spacing,
            num_mics: 2,
        },
        sample_rate,
        512,
    );

    // Physically simulate delay of 2 samples (0.000125 seconds)
    let delay_samples = 2;
    let tau_sec = delay_samples as f32 / sample_rate;
    // Expected physical azimuth: sin(theta) = c * tau / d
    let expected_theta_deg = (SPEED_OF_SOUND * tau_sec / spacing).asin() * 180.0 / PI;

    let n = 256;
    let base_sig: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.5 * (2.0 * PI * 1000.0 * t).sin() + 0.5 * (2.0 * PI * 1800.0 * t).sin()
        })
        .collect();

    let mut ch0 = vec![0.0f32; n];
    let mut ch1 = vec![0.0f32; n];

    for i in 0..n {
        ch0[i] = base_sig[i];
        if i >= delay_samples {
            ch1[i] = base_sig[i - delay_samples];
        }
    }

    let frames: [&[f32]; 2] = [&ch0, &ch1];
    let est_azimuth_rad = doa.estimate_azimuth(&frames);
    let est_azimuth_deg = est_azimuth_rad * 180.0 / PI;

    println!(
        "DoA estimated azimuth: {est_azimuth_deg:.2} deg (expected: {expected_theta_deg:.2} deg)"
    );

    assert!(
        (est_azimuth_deg - expected_theta_deg).abs() < 1.0,
        "Estimated azimuth must be within 1 degree of expected physical angle"
    );
}

#[test]
fn test_delay_and_sum_spatial_beamforming_gain() {
    let sample_rate = 16000.0;
    let spacing = 0.04;
    let num_mics = 4;
    let mut beamformer = DelayAndSumBeamformer::new(
        ArrayGeometry::Linear { spacing, num_mics },
        sample_rate,
    );

    // Steer forward broadside (azimuth = 0 rad, directly in front of drone along +X)
    beamformer.steer(0.0, 0.0);

    let block_len = 512;
    // On-axis target: acoustic wave from +X arrives at all lateral Y mics at identical time
    let target_tone: Vec<f32> = (0..block_len)
        .map(|i| (2.0 * PI * 800.0 * (i as f32) / sample_rate).sin())
        .collect();

    let multi_channel = vec![target_tone.as_slice(); num_mics];
    let mut beamformed_out = vec![0.0f32; block_len];

    // Prime the delay buffer
    beamformer.process_block(&multi_channel, &mut beamformed_out);
    beamformer.process_block(&multi_channel, &mut beamformed_out);

    let in_power: f32 = target_tone.iter().map(|&x| x * x).sum::<f32>() / block_len as f32;
    let out_power: f32 = beamformed_out.iter().map(|&x| x * x).sum::<f32>() / block_len as f32;

    let gain_db = 10.0 * (out_power / in_power.max(1e-12)).log10();
    println!("On-axis broadside beamformer gain: {gain_db:.2} dB");
    assert!(
        gain_db.abs() < 0.5,
        "On-axis target must experience 0 dB gain in beamformer passband, got {gain_db:.2} dB"
    );

    // Off-axis interference (e.g. drone motor noise from the side)
    // Each successive mic receives signal delayed by 3 samples
    let mut off_axis_ch = vec![vec![0.0f32; block_len]; num_mics];
    for m in 0..num_mics {
        let delay = m * 3;
        for i in delay..block_len {
            off_axis_ch[m][i] = target_tone[i - delay];
        }
    }

    let off_axis_slices: Vec<&[f32]> = off_axis_ch.iter().map(|v| v.as_slice()).collect();
    let mut off_axis_out = vec![0.0f32; block_len];

    beamformer.process_block(&off_axis_slices, &mut off_axis_out);
    let off_out_power: f32 = off_axis_out.iter().map(|&x| x * x).sum::<f32>() / block_len as f32;

    let spatial_rejection_db = 10.0 * (out_power / off_out_power.max(1e-12)).log10();
    println!("Off-axis spatial noise rejection: {spatial_rejection_db:.2} dB");
    assert!(
        spatial_rejection_db > 2.5,
        "Beamformer must attenuate off-axis acoustic interference"
    );
}

#[test]
fn test_circular_array_360_degree_beamforming() {
    let sample_rate = 16000.0;
    let radius = 0.05;
    let num_mics = 4;
    let mut beamformer = DelayAndSumBeamformer::new(
        ArrayGeometry::Circular { radius, num_mics },
        sample_rate,
    );

    // Steer to 90 degrees (along positive Y-axis)
    beamformer.steer(PI * 0.5, 0.0);
    assert!((beamformer.steered_azimuth() - PI * 0.5).abs() < 1e-4);

    let block_len = 256;
    let test_frame = vec![0.5f32; block_len];
    let multi_ch = vec![test_frame.as_slice(); num_mics];
    let mut out = vec![0.0f32; block_len];

    beamformer.process_block(&multi_ch, &mut out);
    assert_eq!(out.len(), block_len);
}

#[test]
fn test_multi_channel_beamforming_throughput() {
    let sample_rate = 16000.0;
    let num_mics = 4;
    let mut beamformer = DelayAndSumBeamformer::new(
        ArrayGeometry::Circular {
            radius: 0.06,
            num_mics,
        },
        sample_rate,
    );
    beamformer.steer(0.0, 0.0);

    let total_samples = 320_000; // 20 seconds of 4-channel audio
    let block_size = 512;

    let ch_data = vec![0.1f32; block_size];
    let multi_ch = vec![ch_data.as_slice(); num_mics];
    let mut beamformed_block = vec![0.0f32; block_size];

    let start = Instant::now();
    let mut processed = 0;
    while processed < total_samples {
        beamformer.process_block(&multi_ch, &mut beamformed_block);
        processed += block_size;
    }
    let elapsed = start.elapsed();

    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let real_time_factor = samples_per_sec / sample_rate as f64;

    println!(
        "Multi-channel 4-mic beamformer standalone throughput: {samples_per_sec:.0} samples/sec ({real_time_factor:.1}x real-time)"
    );

    let target = if cfg!(debug_assertions) {
        400_000.0
    } else {
        500_000.0
    };

    assert!(
        samples_per_sec > target,
        "Beamformer standalone throughput must exceed {target:.0} samples/sec, got {samples_per_sec:.0}"
    );
}
