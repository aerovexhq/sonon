//! Unit tests for Universal Neural Vocoder & Waveform Synthesizer (BigVGAN-v2 + Anti-Aliased SnakeBeta)
//! in pure safe Rust for modules/sonon.

#![deny(unsafe_code)]

use sonon::flow_matching::{CfmConfig, FlowSolverScheme};
use sonon::vocoder::{
    bessel_i0, kaiser_window, snake_beta_scalar, AntiAliasedAmpBlock, BigVganVocoder,
    KaiserLowPassFilter, MultiPeriodDiscriminator, MultiResolutionStftDiscriminator, SnakeBeta,
    VocoderConfig, VocoderLossEvaluator,
};
use sonon::SononEngine;
use std::f32::consts::PI;

#[test]
fn test_snake_beta_periodic_harmonic_generation_and_parameter_decoupling() {
    let channels = 3;
    let alpha = vec![1.0, 2.5, 1.0];
    let beta = vec![1.0, 1.0, 0.2];
    let snake = SnakeBeta::new_with_params(alpha.clone(), beta.clone());

    assert_eq!(snake.channels, 3);
    assert_eq!(snake.alpha, alpha);
    assert_eq!(snake.beta, beta);

    let num_samples = 100;
    let mut channel_inputs = vec![vec![0.0f32; num_samples]; channels];
    for c in 0..channels {
        for t in 0..num_samples {
            let phase = 2.0 * PI * (t as f32) / 25.0;
            channel_inputs[c][t] = phase.sin();
        }
    }

    let output = snake.forward(&channel_inputs);
    assert_eq!(output.len(), 3);
    assert_eq!(output[0].len(), num_samples);

    // Verify non-linear harmonic generation and absence of NaN / Inf
    for c in 0..channels {
        for t in 0..num_samples {
            let val = output[c][t];
            assert!(val.is_finite(), "Output must be finite at [{}][{}]", c, t);
        }
    }

    // Verify parameter decoupling:
    // Channel 0 (alpha=1.0, beta=1.0) vs Channel 1 (alpha=2.5, beta=1.0)
    // Different alpha must produce different frequency oscillation values
    let mut diff_alpha_sum = 0.0f32;
    for t in 0..num_samples {
        diff_alpha_sum += (output[0][t] - output[1][t]).abs();
    }
    assert!(
        diff_alpha_sum > 1.0,
        "Different alpha parameters must produce distinct harmonic frequencies"
    );

    // Channel 0 (alpha=1.0, beta=1.0) vs Channel 2 (alpha=1.0, beta=0.2)
    // Different beta must modulate amplitude resonance scaling
    let mut diff_beta_sum = 0.0f32;
    for t in 0..num_samples {
        diff_beta_sum += (output[0][t] - output[2][t]).abs();
    }
    assert!(
        diff_beta_sum > 1.0,
        "Different beta parameters must produce distinct formant resonance scaling"
    );

    // Verify numerical stability with large input magnitudes
    let large_x = 50.0f32;
    let s_out = snake_beta_scalar(large_x, 1.0, 1.0);
    assert!(
        s_out.is_finite(),
        "Scalar output must not overflow or collapse on large inputs"
    );

    // Verify linear identity preservation as beta -> 0 and small x
    let small_x = 0.001f32;
    let identity_approx = snake_beta_scalar(small_x, 1.0, 0.0);
    assert!(
        (identity_approx - small_x).abs() < 1e-4,
        "Near beta=0 with small x, SnakeBeta smoothly preserves identity: got {} vs {}",
        identity_approx,
        small_x
    );
}

#[test]
fn test_kaiser_window_low_pass_fir_filter_normalization_and_stopband_attenuation() {
    // 1. Bessel I_0 verification
    assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
    assert!(bessel_i0(1.0) > bessel_i0(0.0));
    assert!(bessel_i0(6.0) > bessel_i0(3.0));

    // 2. Kaiser window symmetry and shape
    let size = 15;
    let beta = 6.0;
    let w = kaiser_window(size, beta);
    assert_eq!(w.len(), size);
    let center = size / 2;
    assert!(
        (w[center] - 1.0).abs() < 1e-5,
        "Kaiser window peak must equal unity at center"
    );
    for i in 0..center {
        assert!(
            (w[i] - w[size - 1 - i]).abs() < 1e-5,
            "Kaiser window must be perfectly symmetric"
        );
        assert!(w[i] >= 0.0, "Window values must be non-negative");
    }

    // 3. Kaiser Low-Pass Filter unity sum normalization
    let filter = KaiserLowPassFilter::new(15, 0.25, 6.0);
    let kernel_sum: f32 = filter.kernel.iter().sum();
    assert!(
        (kernel_sum - 1.0).abs() < 1e-5,
        "Filter kernel sum must be normalized to unity: got {}",
        kernel_sum
    );

    // 4. DC pass-through property
    let dc_signal = vec![2.5f32; 64];
    let dc_filtered = filter.filter_1d(&dc_signal);
    assert_eq!(dc_filtered.len(), dc_signal.len());
    for (idx, &val) in dc_filtered.iter().enumerate().skip(5).take(50) {
        assert!(
            (val - 2.5).abs() < 1e-4,
            "DC signal must be preserved at index {}: got {}",
            idx,
            val
        );
    }

    // 5. Stopband attenuation verification:
    // Compare attenuation of passband signal (f = 0.05) vs stopband signal (f = 0.45)
    let n = 256;
    let mut low_freq = vec![0.0f32; n];
    let mut high_freq = vec![0.0f32; n];
    for i in 0..n {
        low_freq[i] = (2.0 * PI * 0.05 * (i as f32)).sin();
        high_freq[i] = (2.0 * PI * 0.45 * (i as f32)).sin();
    }

    let low_filtered = filter.filter_1d(&low_freq);
    let high_filtered = filter.filter_1d(&high_freq);

    // Compute RMS energy in central region away from edges
    let slice_range = 30..220;
    let low_rms: f32 = (slice_range
        .clone()
        .map(|i| low_filtered[i] * low_filtered[i])
        .sum::<f32>()
        / slice_range.len() as f32)
        .sqrt();
    let high_rms: f32 = (slice_range
        .clone()
        .map(|i| high_filtered[i] * high_filtered[i])
        .sum::<f32>()
        / slice_range.len() as f32)
        .sqrt();

    assert!(
        low_rms > 0.6,
        "Passband signal energy must be preserved: got {}",
        low_rms
    );
    assert!(
        high_rms < 0.15,
        "Stopband signal must be strongly attenuated: got {}",
        high_rms
    );
    assert!(
        low_rms / high_rms > 4.0,
        "Stopband attenuation ratio must be significant"
    );

    // 6. Zero-phase preservation: peak of a symmetric delta impulse at center
    let mut impulse = vec![0.0f32; 65];
    impulse[32] = 1.0;
    let impulse_filtered = filter.filter_1d(&impulse);
    let mut max_idx = 0;
    let mut max_val = -1.0;
    for (i, &v) in impulse_filtered.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    assert_eq!(
        max_idx, 32,
        "Symmetric FIR convolution must maintain zero-phase delay"
    );
}

#[test]
fn test_anti_aliased_amp_block_residual_computation_and_dilation_behavior() {
    let channels = 8;
    let kernel_size = 3;
    let dilation = 2;
    let lpf_size = 7;
    let lpf_cutoff = 0.45;
    let lpf_beta = 6.0;
    let salt = 42;

    let amp_block = AntiAliasedAmpBlock::new(
        channels,
        kernel_size,
        dilation,
        lpf_size,
        lpf_cutoff,
        lpf_beta,
        salt,
    );

    let seq_len = 32;
    let mut input = vec![vec![0.0f32; seq_len]; channels];
    for c in 0..channels {
        for t in 0..seq_len {
            input[c][t] = (c as f32 * 0.3 + t as f32 * 0.1).sin();
        }
    }

    let output = amp_block.forward(&input);

    // Dimensions must be strictly preserved
    assert_eq!(output.len(), channels);
    for c in 0..channels {
        assert_eq!(output[c].len(), seq_len);
    }

    // Residual verification: output must contain residual + non-linear transform
    let mut diff_sum = 0.0f32;
    for c in 0..channels {
        for t in 0..seq_len {
            diff_sum += (output[c][t] - input[c][t]).abs();
            assert!(output[c][t].is_finite());
        }
    }
    assert!(
        diff_sum > 0.01,
        "AMP block non-linear residual branch must contribute non-zero modification"
    );

    // Verify with larger dilation
    let amp_block_dilated = AntiAliasedAmpBlock::new(
        channels,
        kernel_size,
        5,
        lpf_size,
        lpf_cutoff,
        lpf_beta,
        salt + 10,
    );
    let dilated_output = amp_block_dilated.forward(&input);
    assert_eq!(dilated_output.len(), channels);
    assert_eq!(dilated_output[0].len(), seq_len);
}

#[test]
fn test_bigvgan_vocoder_upsampling_dimension_scaling() {
    let config = VocoderConfig::fast_test_config();
    let hop = config.total_hop_size();
    assert_eq!(hop, 8, "Fast test config hop size must be 4 * 2 = 8");

    let vocoder = BigVganVocoder::new(config);

    // Test with T = 12 mel frames
    let num_frames = 12;
    let mel_frames = vec![vec![0.1f32; 80]; num_frames];
    let waveform = vocoder.synthesize_waveform(&mel_frames);

    let expected_samples = num_frames * hop;
    assert_eq!(
        waveform.len(),
        expected_samples,
        "Synthesized waveform length must scale exactly by total hop size"
    );

    // Test with T = 24 frames
    let num_frames_2 = 24;
    let mel_frames_2 = vec![vec![0.2f32; 80]; num_frames_2];
    let waveform_2 = vocoder.synthesize_waveform(&mel_frames_2);
    assert_eq!(waveform_2.len(), num_frames_2 * hop);

    // Empty input returns empty output
    let empty_wav = vocoder.synthesize_waveform(&[]);
    assert!(empty_wav.is_empty());
}

#[test]
fn test_audio_output_boundedness_strictly_within_minus_one_to_plus_one() {
    let config = VocoderConfig::fast_test_config();
    let vocoder = BigVganVocoder::new(config);

    // Create extreme mel values to challenge saturation bounds
    let num_frames = 16;
    let mut extreme_mel = vec![vec![0.0f32; 80]; num_frames];
    for t in 0..num_frames {
        for c in 0..80 {
            extreme_mel[t][c] = if (t + c) % 2 == 0 { 50.0 } else { -50.0 };
        }
    }

    let waveform = vocoder.synthesize_waveform(&extreme_mel);
    assert_eq!(waveform.len(), num_frames * 8);

    for (idx, &sample) in waveform.iter().enumerate() {
        assert!(sample.is_finite(), "Sample {} must be finite", idx);
        assert!(
            sample >= -1.0 && sample <= 1.0,
            "Sample {} at value {} strictly violates [-1.0, 1.0] bounds",
            idx,
            sample
        );
    }
}

#[test]
fn test_multi_period_discriminator_2d_period_extraction_across_prime_periods() {
    let mpd = MultiPeriodDiscriminator::default();
    assert_eq!(mpd.discriminators.len(), 5);

    let waveform_len = 128;
    let mut waveform = vec![0.0f32; waveform_len];
    for i in 0..waveform_len {
        waveform[i] = (2.0 * PI * 0.08 * (i as f32)).sin() * 0.5;
    }

    let output = mpd.forward(&waveform);

    assert_eq!(output.period_scores.len(), 5);
    assert_eq!(output.period_logits.len(), 5);
    assert_eq!(output.features.len(), 5);

    for p_idx in 0..5 {
        assert!(
            output.period_scores[p_idx].is_finite(),
            "Period score must be finite"
        );
        assert!(
            !output.period_logits[p_idx].is_empty(),
            "Logits must not be empty"
        );
        // Each discriminator has 4 convolutional layer features
        assert_eq!(output.features[p_idx].len(), 4);
    }

    assert!(output.mean_score.is_finite());
}

#[test]
fn test_multi_resolution_stft_discriminator_multiscale_fft_evaluation() {
    let mrsd = MultiResolutionStftDiscriminator::new(&[(128, 32), (256, 64)]);
    assert_eq!(mrsd.resolutions.len(), 2);

    let signal_len = 512;
    let mut waveform = vec![0.0f32; signal_len];
    for i in 0..signal_len {
        waveform[i] = (2.0 * PI * 0.1 * i as f32).sin() + 0.3 * (2.0 * PI * 0.25 * i as f32).cos();
    }

    let output = mrsd.forward(&waveform);

    assert_eq!(output.resolution_scores.len(), 2);
    assert_eq!(output.resolution_logits.len(), 2);
    assert_eq!(output.features.len(), 2);
    assert_eq!(output.phase_coherences.len(), 2);

    for idx in 0..2 {
        assert!(output.resolution_scores[idx].is_finite());
        assert!(!output.resolution_logits[idx].is_empty());
        let pc = output.phase_coherences[idx];
        assert!(
            pc >= -1.0 && pc <= 1.0,
            "Phase coherence must lie in [-1.0, 1.0]: got {}",
            pc
        );
    }
}

#[test]
fn test_sonon_engine_synthesize_waveform_from_mel() {
    let mut engine = SononEngine::new(16000.0, 512, 128, 13);

    let mel_frames = vec![vec![0.05f32; 80]; 10];

    // When neural vocoder is disabled, must return error
    let err_res = engine.synthesize_waveform_from_mel(&mel_frames);
    assert!(err_res.is_err());
    assert_eq!(err_res.unwrap_err(), "Neural vocoder is not enabled");

    // Enable neural vocoder with fast test configuration
    engine.enable_neural_vocoder(VocoderConfig::fast_test_config());
    assert!(engine.neural_vocoder().is_some());

    let wav_res = engine.synthesize_waveform_from_mel(&mel_frames);
    assert!(wav_res.is_ok());
    let wav = wav_res.unwrap();
    assert_eq!(wav.len(), 10 * 8);

    for s in &wav {
        assert!(s.is_finite());
        assert!(*s >= -1.0 && *s <= 1.0);
    }

    // Disable neural vocoder
    engine.disable_neural_vocoder();
    assert!(engine.neural_vocoder().is_none());
    assert!(engine.synthesize_waveform_from_mel(&mel_frames).is_err());
}

#[test]
fn test_end_to_end_sonon_engine_synthesize_speech_audio_pipeline() {
    let mut engine = SononEngine::new(16000.0, 512, 128, 13);

    // Configure CFM Speech Synthesizer
    let cfm_config = CfmConfig {
        hidden_dim: 32,
        latent_dim: 80,
        num_heads: 2,
        num_layers: 2,
        context_dim: 32,
        seed: 12345,
        frames_per_char: 2,
    };
    engine.enable_cfm_synthesizer(cfm_config);

    // Configure BigVGAN neural vocoder
    engine.enable_neural_vocoder(VocoderConfig::fast_test_config());

    let text = "Clear air patrol route";
    let audio_res = engine.synthesize_speech_audio(text, None, 2, FlowSolverScheme::Euler);
    assert!(audio_res.is_ok());

    let waveform = audio_res.unwrap();
    assert!(
        !waveform.is_empty(),
        "End-to-end waveform must not be empty"
    );

    // Verify samples are strictly bounded and finite
    for &sample in &waveform {
        assert!(sample.is_finite());
        assert!(sample >= -1.0 && sample <= 1.0);
    }

    // Test with speaker acoustic prompt reference
    let speaker_slice = vec![0.1f32; 160];
    let audio_speaker_res = engine.synthesize_speech_audio(
        "Ascent to altitude 200",
        Some(&speaker_slice),
        2,
        FlowSolverScheme::Midpoint,
    );
    assert!(audio_speaker_res.is_ok());
    let speaker_waveform = audio_speaker_res.unwrap();
    assert!(!speaker_waveform.is_empty());
    for &sample in &speaker_waveform {
        assert!(sample.is_finite());
        assert!(sample >= -1.0 && sample <= 1.0);
    }
}

#[test]
fn test_deterministic_reproducibility_across_repeated_calls() {
    let config1 = VocoderConfig::fast_test_config();
    let vocoder1 = BigVganVocoder::new(config1);

    let config2 = VocoderConfig::fast_test_config();
    let vocoder2 = BigVganVocoder::new(config2);

    let mel_frames = vec![vec![0.15f32; 80]; 8];

    let wav1 = vocoder1.synthesize_waveform(&mel_frames);
    let wav2 = vocoder2.synthesize_waveform(&mel_frames);

    assert_eq!(wav1.len(), wav2.len());
    for i in 0..wav1.len() {
        assert_eq!(
            wav1[i], wav2[i],
            "Deterministic synthesis must produce bit-for-bit identical waveforms at index {}",
            i
        );
    }
}

#[test]
fn test_vocoder_loss_evaluator_adversarial_feature_matching_and_mel_loss() {
    let evaluator = VocoderLossEvaluator::new(80, 256, 16000.0, 0.0, 8000.0);

    let audio_len = 512;
    let mut real_audio = vec![0.0f32; audio_len];
    let mut fake_audio = vec![0.0f32; audio_len];

    for i in 0..audio_len {
        real_audio[i] = (2.0 * PI * 0.05 * (i as f32)).sin() * 0.7;
        fake_audio[i] = (2.0 * PI * 0.05 * (i as f32)).sin() * 0.5
            + 0.1 * (2.0 * PI * 0.2 * (i as f32)).cos();
    }

    let report = evaluator.evaluate_losses(&real_audio, &fake_audio, 64);

    assert!(report.generator_adversarial_loss >= 0.0);
    assert!(report.discriminator_adversarial_loss >= 0.0);
    assert!(report.feature_matching_loss >= 0.0);
    assert!(report.mel_spectral_loss >= 0.0);
    assert!(report.total_generator_loss >= 0.0);
    assert!(report.total_discriminator_loss >= 0.0);

    assert!(report.generator_adversarial_loss.is_finite());
    assert!(report.discriminator_adversarial_loss.is_finite());
    assert!(report.feature_matching_loss.is_finite());
    assert!(report.mel_spectral_loss.is_finite());
    assert!(report.total_generator_loss.is_finite());
}
