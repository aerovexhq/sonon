//! Verification Test Suite for Hardware-Accelerated Streaming Spiking Neural VAD with Neuromorphic Latency.

#![deny(unsafe_code)]

use sonon::engine::{FeatureMode, SononEngine};
use sonon::phonetic::{SyntheticExemplarGenerator, VocalAccent};
use sonon::spiking_vad::{
    BiquadBandpassFilter, LifNeuron, LifNeuronConfig, SpikingNeuralVad,
    SpikingVadConfig,
};
use sonon::zero_shot::{MultiLingualG2p, SupportedLanguage};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_lif_neuron_leak_integration_and_reset_dynamics() {
    let sample_rate = 16000.0f32;
    let config = LifNeuronConfig {
        tau_mem_ms: 10.0,
        v_threshold: 1.0,
        v_rest: 0.0,
        v_reset: 0.0,
        refractory_period_samples: 5,
        soft_reset: true,
    };
    let mut neuron = LifNeuron::new(sample_rate, config);

    assert_eq!(neuron.potential(), 0.0);
    assert_eq!(neuron.spikes_emitted(), 0);
    assert!(neuron.leak_factor() > 0.99 && neuron.leak_factor() < 1.0);

    // 1. Sub-threshold integration and exponential decay
    neuron.step(0.5);
    let p_initial = neuron.potential();
    assert!(
        (p_initial - 0.5).abs() < 1e-4,
        "Initial potential must equal input current"
    );

    // Step with 0 current for 50 samples
    for _ in 0..50 {
        let spiked = neuron.step(0.0);
        assert!(!spiked, "Sub-threshold potential must not spike");
    }
    let p_decayed = neuron.potential();
    assert!(
        p_decayed < p_initial * 0.75,
        "Potential must decay exponentially without input current"
    );

    // 2. Threshold crossing and soft reset
    neuron.reset();
    assert_eq!(neuron.potential(), 0.0);

    // Inject currents to cross threshold
    let spiked1 = neuron.step(0.6);
    assert!(!spiked1);
    let spiked2 = neuron.step(0.6); // 0.6 * lambda + 0.6 > 1.0
    assert!(spiked2, "Crossing threshold must emit a spike");
    assert_eq!(neuron.spikes_emitted(), 1);

    // Soft reset residual: (potential - 1.0)
    let p_after_spike = neuron.potential();
    assert!(
        (0.0..0.3).contains(&p_after_spike),
        "Soft reset must leave residual charge, got {p_after_spike}"
    );

    // 3. Refractory period: next 5 steps must be locked out even with high current
    for _ in 0..5 {
        let locked_spike = neuron.step(2.0);
        assert!(
            !locked_spike,
            "Neuron must not fire during refractory lockout"
        );
    }

    // 4. Post-refractory re-enable
    let rearmed_spike = neuron.step(1.5);
    assert!(
        rearmed_spike,
        "Neuron must fire again after refractory period expires"
    );
    assert_eq!(neuron.spikes_emitted(), 2);
}

#[test]
fn test_sub_millisecond_activation_latency() {
    let sample_rate = 16000.0f32;
    let config = SpikingVadConfig {
        sample_rate,
        ..Default::default()
    };
    let mut vad = SpikingNeuralVad::new(config);

    // 1. Pre-warm with 1600 samples of quiet background (0.1s)
    let background = vec![0.0001f32; 1600];
    let (any_active, _) = vad.process_buffer(&background);
    assert!(!any_active, "Background must be completely inactive");
    assert!(!vad.is_speech_active());

    // 2. Inject an abrupt acoustic speech transient (1200 Hz vowel formant resonance at 0.5 amplitude)
    let onset_sample_idx = 1600;
    let mut activation_sample_idx: Option<usize> = None;

    for i in 0..160 {
        let t = i as f32 / sample_rate;
        // Speech formant carrier
        let sample = 0.5 * (2.0 * PI * 1200.0 * t).sin();
        let is_active = vad.step_sample(sample);

        if is_active && activation_sample_idx.is_none() {
            activation_sample_idx = Some(onset_sample_idx + i);
        }
    }

    assert!(
        activation_sample_idx.is_some(),
        "Speech transient must trigger Spiking Neural VAD activation"
    );

    let activation_idx = activation_sample_idx.unwrap();
    let latency_samples = activation_idx - onset_sample_idx;
    let latency_us = (latency_samples as f32 / sample_rate) * 1_000_000.0;

    // Sub-millisecond latency requirement: < 16 samples (< 1000 µs at 16 kHz)
    assert!(
        latency_samples < 16,
        "Activation latency ({latency_samples} samples) must be strictly sub-millisecond (< 16 samples at 16 kHz)"
    );
    assert!(
        latency_us < 1000.0,
        "Activation latency ({latency_us:.1} µs) must be strictly < 1000 µs (1.0 ms)"
    );

    let telem = vad.telemetry();
    assert!(telem.is_speech_active);
    assert!(telem.total_spikes_emitted > 0);
    assert!(telem.last_activation_latency_us < 1000.0);
}

#[test]
fn test_quiescence_under_silence_and_drone_motor_bpf_noise() {
    let sample_rate = 16000.0f32;
    let config = SpikingVadConfig {
        sample_rate,
        ..Default::default()
    };
    let mut vad = SpikingNeuralVad::new(config);

    // 1. Pure silence: 16000 samples (1.0 second)
    let silence = vec![0.0f32; 16000];
    let (silence_active, silence_telem) = vad.process_buffer(&silence);

    assert!(
        !silence_active,
        "Pure silence must produce zero speech activations"
    );
    assert_eq!(
        silence_telem.total_spikes_emitted, 0,
        "Silence must produce 0 master VAD spikes"
    );
    assert_eq!(
        silence_telem.synaptic_accumulator, 0.0,
        "Synaptic accumulator must remain at rest (0.0) in silence"
    );
    assert_eq!(
        silence_telem.quiescent_ratio, 1.0,
        "Quiescent ratio must be exactly 1.0 (100%) during silence"
    );

    // 2. Drone motor rotor acoustic noise:
    // Blade pass frequency (BPF) fundamental at 150 Hz + harmonics at 300 Hz, 450 Hz
    let mut motor_noise = vec![0.0f32; 16000];
    for (i, sample) in motor_noise.iter_mut().enumerate() {
        let t = i as f32 / sample_rate;
        let ramp = (i as f32 / 160.0).min(1.0); // 10 ms acoustic spool-up ramp
        *sample = ramp * (0.30 * (2.0 * PI * 150.0 * t).sin()
            + 0.15 * (2.0 * PI * 300.0 * t).sin()
            + 0.08 * (2.0 * PI * 450.0 * t).sin());
    }

    let (motor_active, motor_telem) = vad.process_buffer(&motor_noise);

    assert!(
        !motor_active,
        "Drone motor rotor hum must be rejected by inhibitory synaptic weights"
    );
    assert_eq!(
        motor_telem.total_spikes_emitted, 0,
        "Zero master VAD spikes must be emitted in steady drone motor noise"
    );
    assert!(
        !motor_telem.is_speech_active,
        "Speech state must remain inactive in drone motor wash"
    );
    assert!(
        motor_telem.synaptic_accumulator < 0.1,
        "Synaptic accumulator must remain far below activation threshold in drone noise"
    );
}

#[test]
fn test_multi_band_frequency_selectivity() {
    let sample_rate = 16000.0f32;
    let biquad_500 = BiquadBandpassFilter::new(sample_rate, 500.0, 1.5);
    let mut biquad_1200 = BiquadBandpassFilter::new(sample_rate, 1200.0, 1.5);

    assert_eq!(biquad_500.center_freq(), 500.0);
    assert_eq!(biquad_1200.center_freq(), 1200.0);
    assert_eq!(biquad_1200.sample_rate(), sample_rate);

    // Test resonance of 1200 Hz filter at center vs off-center
    let num_samples = 1000;
    let mut peak_on = 0.0f32;
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let s = (2.0 * PI * 1200.0 * t).sin();
        let y = biquad_1200.step(s);
        if i > 200 {
            peak_on = peak_on.max(y.abs());
        }
    }
    assert!(
        peak_on > 0.85,
        "Resonant peak response must be near unity (0 dB), got {peak_on}"
    );

    biquad_1200.reset();
    let mut peak_off = 0.0f32;
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let s = (2.0 * PI * 3500.0 * t).sin();
        let y = biquad_1200.step(s);
        if i > 200 {
            peak_off = peak_off.max(y.abs());
        }
    }
    assert!(
        peak_off < 0.25,
        "Off-frequency response must be strongly attenuated, got {peak_off}"
    );
    assert!(peak_on / peak_off > 3.8, "Selectivity must exceed 3.8x");

    // Test Spiking VAD band-specific excitation
    let config = SpikingVadConfig {
        sample_rate,
        ..Default::default()
    };
    let mut vad = SpikingNeuralVad::new(config);

    // Stream 1200 Hz burst (Band 2)
    let mut speech_burst = vec![0.0f32; 1600];
    for (i, sample) in speech_burst.iter_mut().enumerate() {
        let t = i as f32 / sample_rate;
        *sample = 0.5 * (2.0 * PI * 1200.0 * t).sin();
    }
    vad.process_buffer(&speech_burst);

    let telem = vad.telemetry();
    assert!(
        telem.band_spikes[2] > 0,
        "Band 2 (1200 Hz) must record spikes during 1200 Hz stimulation, got {}",
        telem.band_spikes[2]
    );
}

#[test]
fn test_sonon_engine_spiking_vad_gating_and_throughput() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    // 1. Enable Spiking Neural VAD
    let vad_config = SpikingVadConfig {
        sample_rate,
        ..Default::default()
    };
    engine.enable_spiking_vad(vad_config);

    assert!(engine.spiking_vad().is_some());
    assert!(engine.is_spiking_vad_gating_enabled());

    // 2. Enroll keyword "Plank" via zero-shot
    let rep = engine
        .enroll_keyword_zero_shot(
            "Plank",
            "plank",
            SupportedLanguage::English,
            VocalAccent::GeneralAmerican,
        )
        .expect("Enrollment failed");
    assert!(rep.keyword.eq_ignore_ascii_case("Plank"));

    // 3. Stream silence and drone motor noise: DTW matching should be gated
    let mut noise = vec![0.0f32; 8000];
    for (i, sample) in noise.iter_mut().enumerate() {
        let t = i as f32 / sample_rate;
        *sample = 0.25 * (2.0 * PI * 150.0 * t).sin();
    }

    let noise_events = engine.ingest_samples(&noise);
    assert!(
        noise_events.is_empty(),
        "No keyword events must be detected in drone motor noise"
    );

    let telem = engine
        .latest_spiking_vad_telemetry()
        .expect("Telemetry should be available");
    assert!(!telem.is_speech_active);

    // 4. Synthesize and stream actual speech of "Plank"
    let synth = SyntheticExemplarGenerator::new(sample_rate);
    let target_segments = MultiLingualG2p::text_to_phonemes("plank", SupportedLanguage::English);
    let target_audio = synth.generate_exemplars_from_segments(&target_segments, 1);
    assert!(!target_audio.is_empty());

    // Ingest actual speech: Spiking VAD activates and DTW matching spots "Plank"
    let speech_events = engine.ingest_samples(&target_audio[0]);
    assert!(
        !speech_events.is_empty(),
        "Speech input 'Plank' must be detected by SononEngine gated by Spiking Neural VAD"
    );
    assert!(speech_events[0].confidence > 0.0);

    // 5. High-throughput execution benchmark
    let benchmark_samples = vec![0.05f32; 80_000]; // 5.0 seconds of audio
    let start = Instant::now();
    let _ = engine.ingest_samples(&benchmark_samples);
    let elapsed = start.elapsed();

    let samples_per_sec = (benchmark_samples.len() as f64) / elapsed.as_secs_f64();
    assert!(
        samples_per_sec > 500_000.0,
        "SononEngine throughput ({samples_per_sec:.0} samples/sec) must exceed 500,000 samples/sec (> 30x real-time)"
    );

    // 6. Test disable and reset
    engine.disable_spiking_vad();
    assert!(engine.spiking_vad().is_none());
    assert!(engine.latest_spiking_vad_telemetry().is_none());
}
