//! Integration and unit tests for Continuous Dyadic Morlet Wavelet Filterbank,
//! Calderon Analytical Inversion, Port-Hamiltonian Two-Mass Vocal Fold Oscillator,
//! and Hybrid Physical-Neural Wavelet Flow Matching Speech Synthesizer.

#![deny(unsafe_code)]

use sonon::flow_matching::FlowSolverScheme;
use sonon::wavelet_synthesis::{
    CalderonWaveletInverter, DyadicMorletCwt, GlottalFlowGenerator, PortHamiltonianVocalFold,
    WaveletPhysicalFlowSynthesizer, WaveletSynthesizerConfig, MORLET_C_DELTA,
};
use sonon::SononEngine;
use std::f32::consts::PI;

#[test]
fn test_morlet_wavelet_filterbank_construction_and_dyadic_progression() {
    let octaves = 7usize;
    let voices_per_octave = 12usize;
    let sample_rate = 16000.0f32;
    let cwt = DyadicMorletCwt::new(octaves, voices_per_octave, sample_rate);

    assert_eq!(cwt.octaves(), 7);
    assert_eq!(cwt.voices_per_octave(), 12);
    assert_eq!(cwt.total_scales(), 84);
    assert_eq!(cwt.sample_rate(), 16000.0);

    let scales = cwt.scales();
    let freqs = cwt.frequencies();
    assert_eq!(scales.len(), 84);
    assert_eq!(freqs.len(), 84);

    // Verify dyadic scale progression: a_j = a_0 * 2^(j / M)
    let a0 = cwt.a0();
    assert_eq!(a0, 2.0);
    assert!((scales[0] - 2.0).abs() < 1e-6);

    for j in 0..cwt.total_scales() {
        let expected_a = a0 * 2.0f32.powf(j as f32 / voices_per_octave as f32);
        let diff = (scales[j] - expected_a).abs();
        assert!(
            diff < 1e-4,
            "Scale mismatch at index {}: got {}, expected {}",
            j,
            scales[j],
            expected_a
        );

        if j > 0 {
            assert!(
                scales[j] > scales[j - 1],
                "Scales must be strictly monotonically increasing"
            );
            let ratio = scales[j] / scales[j - 1];
            let expected_ratio = 2.0f32.powf(1.0 / voices_per_octave as f32);
            assert!(
                (ratio - expected_ratio).abs() < 1e-4,
                "Dyadic scale ratio mismatch between index {} and {}",
                j - 1,
                j
            );
        }
    }

    // Verify frequency coverage spans from sub-100 Hz up to near Nyquist
    let f_highest = freqs[0];
    let f_lowest = freqs[cwt.total_scales() - 1];
    assert!(
        f_highest >= 7000.0,
        "Highest frequency must reach near-Nyquist, got {}",
        f_highest
    );
    assert!(
        f_lowest <= 80.0,
        "Lowest frequency must cover human fundamental pitch, got {}",
        f_lowest
    );
}

#[test]
fn test_forward_wavelet_scalogram_extraction_on_acoustic_signals() {
    let sample_rate = 16000.0f32;
    let cwt = DyadicMorletCwt::new(6, 12, sample_rate);
    let num_samples = 640; // 40 ms

    // Synthesize two acoustic tones: 250 Hz and 1500 Hz
    let f1 = 250.0f32;
    let f2 = 1500.0f32;
    let mut signal = Vec::with_capacity(num_samples);
    for n in 0..num_samples {
        let t = n as f32 / sample_rate;
        let s = (2.0 * PI * f1 * t).sin() * 0.6 + (2.0 * PI * f2 * t).sin() * 0.4;
        signal.push(s);
    }

    let scalogram = cwt.forward(&signal);
    assert_eq!(scalogram.num_scales(), 72);
    assert_eq!(scalogram.num_samples(), num_samples);
    assert!(scalogram.total_energy() > 0.0);

    // Verify all magnitudes are strictly non-negative
    for s in 0..scalogram.num_scales() {
        for n in 0..scalogram.num_samples() {
            let m = scalogram.magnitude[s][n];
            assert!(m >= 0.0, "Magnitude must be non-negative at scale {}, sample {}", s, n);
            assert!(m.is_finite(), "Magnitude must be finite");
        }
    }

    // Find scale closest to f1 (250 Hz) and f2 (1500 Hz)
    let mut idx_f1 = 0;
    let mut diff_f1 = f32::MAX;
    let mut idx_f2 = 0;
    let mut diff_f2 = f32::MAX;

    for (i, &f) in scalogram.frequencies.iter().enumerate() {
        if (f - f1).abs() < diff_f1 {
            diff_f1 = (f - f1).abs();
            idx_f1 = i;
        }
        if (f - f2).abs() < diff_f2 {
            diff_f2 = (f - f2).abs();
            idx_f2 = i;
        }
    }

    let eng_f1 = scalogram.scale_energy(idx_f1);
    let eng_f2 = scalogram.scale_energy(idx_f2);
    let eng_low = scalogram.scale_energy(scalogram.num_scales() - 1); // ~60 Hz
    let eng_high = scalogram.scale_energy(0); // ~7500 Hz

    // Energies at resonant frequencies must significantly dominate out-of-band scales
    assert!(
        eng_f1 > eng_low * 5.0,
        "Energy at 250 Hz ({}) should exceed out-of-band low energy ({})",
        eng_f1,
        eng_low
    );
    assert!(
        eng_f2 > eng_high * 5.0,
        "Energy at 1500 Hz ({}) should exceed out-of-band high energy ({})",
        eng_f2,
        eng_high
    );
}

#[test]
fn test_calderon_wavelet_inversion_reconstruction_roundtrip() {
    let sample_rate = 16000.0f32;
    let octaves = 7;
    let voices_per_octave = 12;
    let cwt = DyadicMorletCwt::new(octaves, voices_per_octave, sample_rate);
    let inverter = CalderonWaveletInverter::new(voices_per_octave, 1.0);

    assert_eq!(inverter.voices_per_octave, 12);
    assert!((inverter.c_delta - MORLET_C_DELTA).abs() < 1e-6);

    // Multi-harmonic acoustic waveform: 180 Hz, 360 Hz, 540 Hz
    let num_samples = 800;
    let mut signal = Vec::with_capacity(num_samples);
    for n in 0..num_samples {
        let t = n as f32 / sample_rate;
        let s = (2.0 * PI * 180.0 * t).sin() * 0.5
            + (2.0 * PI * 360.0 * t).sin() * 0.3
            + (2.0 * PI * 540.0 * t).sin() * 0.2;
        signal.push(s);
    }

    // 1. Forward continuous dyadic Morlet CWT
    let scalogram = cwt.forward(&signal);

    // 2. Analytical Calderon inversion
    let reconstructed = inverter.reconstruct(&scalogram);
    assert_eq!(reconstructed.len(), num_samples);

    // 3. Evaluate Pearson correlation coefficient in the interior (excluding filter margin)
    let margin = 120;
    let mut dot_prod = 0.0f32;
    let mut norm_sig = 0.0f32;
    let mut norm_rec = 0.0f32;

    for i in margin..num_samples - margin {
        let x = signal[i];
        let y = reconstructed[i];
        dot_prod += x * y;
        norm_sig += x * x;
        norm_rec += y * y;
    }

    let pearson_r = dot_prod / (norm_sig.sqrt() * norm_rec.sqrt());
    assert!(
        pearson_r > 0.985,
        "Calderon reconstruction Pearson correlation must exceed 0.985, got {}",
        pearson_r
    );
}

#[test]
fn test_port_hamiltonian_vocal_fold_symplectic_energy_conservation_and_non_negative_flow() {
    let sample_rate = 16000.0f32;
    let f0_target = 160.0f32;

    // 1. Symplectic Stormer-Verlet energy conservation test
    let mut vf = PortHamiltonianVocalFold::new(f0_target, sample_rate);
    let initial_energy = vf.hamiltonian_energy();
    assert!(initial_energy > 0.0, "Initial Hamiltonian energy must be positive");

    let num_steps = 6000;
    let mut max_rel_energy_drift = 0.0f32;

    for _ in 0..num_steps {
        vf.step_undamped_free();
        let e = vf.hamiltonian_energy();
        let rel_drift = (e - initial_energy).abs() / initial_energy;
        max_rel_energy_drift = max_rel_energy_drift.max(rel_drift);
    }

    // Symplectic integrators guarantee bounded energy oscillation without secular drift
    assert!(
        max_rel_energy_drift < 0.002,
        "Symplectic energy conservation failed: max relative drift was {}",
        max_rel_energy_drift
    );

    // 2. Physical glottal volume flow velocity non-negativity test
    let mut generator = GlottalFlowGenerator::new(sample_rate);
    let flow_samples = 3200;
    let flow = generator.generate_flow(flow_samples, f0_target);
    assert_eq!(flow.len(), flow_samples);

    let mut min_flow = f32::MAX;
    let mut max_flow = f32::MIN;

    for (n, &u) in flow.iter().enumerate() {
        assert!(
            u >= 0.0,
            "Glottal volume flow must be strictly non-negative at sample {}, got {}",
            n,
            u
        );
        assert!(u.is_finite(), "Glottal volume flow must be finite at sample {}", n);
        min_flow = min_flow.min(u);
        max_flow = max_flow.max(u);
    }

    // Must oscillate with positive peak flow
    assert!(max_flow > 1e-5, "Glottal peak flow must be non-zero, got {}", max_flow);
    assert_eq!(min_flow, 0.0, "Glottal flow must touch zero during fold closure");
}

#[test]
fn test_physical_glottal_flow_injection_and_pitch_stability() {
    let sample_rate = 16000.0f32;
    let mut generator = GlottalFlowGenerator::new(sample_rate);
    let f0_target = 200.0f32;
    let flow = generator.generate_flow(3200, f0_target);

    let cwt = DyadicMorletCwt::new(7, 12, sample_rate);
    let scalogram = cwt.forward(&flow);
    let dom_freq = scalogram.dominant_frequency();

    // Verify dominant resonance frequency closely matches F0 target (zero octave doubling)
    let rel_freq_error = (dom_freq - f0_target).abs() / f0_target;
    assert!(
        rel_freq_error < 0.12,
        "Dominant glottal flow frequency ({}) must match target F0 ({}) within 12%, rel error = {}",
        dom_freq,
        f0_target,
        rel_freq_error
    );

    // Verify synthesizer incorporates glottal flow injection
    let config = WaveletSynthesizerConfig {
        sample_rate,
        glottal_weight: 0.40,
        ..Default::default()
    };
    let synth = WaveletPhysicalFlowSynthesizer::new(config);
    let audio = synth.synthesize("ROTOR HARMONICS", 180.0, 4, FlowSolverScheme::Midpoint);
    assert!(!audio.is_empty());
}

#[test]
fn test_sonon_engine_synthesize_physical_wavelet_speech_pipeline() {
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);

    // Synthesizer initially disabled
    assert!(engine.wavelet_physical_synthesizer().is_none());
    let err_result = engine.synthesize_physical_wavelet_speech("TAKEOFF", 160.0, 5);
    assert!(err_result.is_err());
    assert_eq!(
        err_result.err().unwrap(),
        "Wavelet physical flow synthesizer is not enabled"
    );

    // Enable synthesizer
    let config = WaveletSynthesizerConfig {
        sample_rate: 16000.0,
        octaves: 6,
        voices_per_octave: 10,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        default_solver_scheme: FlowSolverScheme::Midpoint,
        default_num_steps: 4,
        glottal_weight: 0.35,
        seed: 999,
        hop_size: 160,
        samples_per_char: 800,
        ..Default::default()
    };
    engine.enable_wavelet_physical_synthesizer(config);
    assert!(engine.wavelet_physical_synthesizer().is_some());
    assert!(engine.wavelet_physical_synthesizer_mut().is_some());

    // Synthesize speech audio
    let audio = engine
        .synthesize_physical_wavelet_speech("WAYPOINT REACHED", 150.0, 4)
        .expect("Synthesis must succeed when enabled");

    assert!(!audio.is_empty());
    assert!(audio.len() >= 16000 / 4);

    // Disable synthesizer
    engine.disable_wavelet_physical_synthesizer();
    assert!(engine.wavelet_physical_synthesizer().is_none());
}

#[test]
fn test_output_samples_strictly_bounded_within_unit_interval() {
    let config = WaveletSynthesizerConfig {
        sample_rate: 16000.0,
        octaves: 6,
        voices_per_octave: 8,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        default_num_steps: 3,
        hop_size: 160,
        samples_per_char: 600,
        ..Default::default()
    };
    let synth = WaveletPhysicalFlowSynthesizer::new(config);

    let test_phrases = ["ALTITUDE HOLD", "BRAVO"];
    let test_pitches = [140.0f32, 280.0f32];
    let schemes = [
        FlowSolverScheme::Euler,
        FlowSolverScheme::Midpoint,
        FlowSolverScheme::RungeKutta4,
        FlowSolverScheme::ConsistencyFlow,
    ];

    for &phrase in &test_phrases {
        for &f0 in &test_pitches {
            for &scheme in &schemes {
                let audio = synth.synthesize(phrase, f0, 2, scheme);
                assert!(!audio.is_empty());

                for (idx, &sample) in audio.iter().enumerate() {
                    assert!(
                        sample.is_finite(),
                        "Sample at index {} is not finite: {}",
                        idx,
                        sample
                    );
                    assert!(
                        sample >= -1.0 && sample <= 1.0,
                        "Sample at index {} outside [-1.0, 1.0]: {}",
                        idx,
                        sample
                    );
                }
            }
        }
    }
}

#[test]
fn test_deterministic_reproducibility_across_repeated_calls() {
    let config = WaveletSynthesizerConfig {
        sample_rate: 16000.0,
        octaves: 6,
        voices_per_octave: 8,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        default_num_steps: 4,
        seed: 4242,
        hop_size: 160,
        samples_per_char: 600,
        ..Default::default()
    };

    let synth1 = WaveletPhysicalFlowSynthesizer::new(config.clone());
    let synth2 = WaveletPhysicalFlowSynthesizer::new(config);

    let audio1 = synth1.synthesize("CONFIRM APPROACH", 165.0, 4, FlowSolverScheme::Midpoint);
    let audio2 = synth2.synthesize("CONFIRM APPROACH", 165.0, 4, FlowSolverScheme::Midpoint);

    assert_eq!(audio1.len(), audio2.len());
    for i in 0..audio1.len() {
        let diff = (audio1[i] - audio2[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at sample {}: audio1 = {}, audio2 = {}",
            i,
            audio1[i],
            audio2[i]
        );
    }
}
