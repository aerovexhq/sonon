//! Phase 9 Verification Test Suite: Neuromorphic Silicon Cochlea & Spiking Wake-Word Spotting.

#![deny(unsafe_code)]

use sonon::engine::KeywordEvent;
use sonon::neuromorphic::{GammatoneFilter, NeuromorphicCochlea, SpikeEvent, SpikingKwsCell};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_spike_event_binary_serialization() {
    let ev1 = SpikeEvent::new(1234567, 14, 1);
    assert!(ev1.is_on());
    assert!(!ev1.is_off());
    assert_eq!(ev1.timestamp_us, 1234567);
    assert_eq!(ev1.channel, 14);
    assert_eq!(ev1.polarity, 1);

    let bytes = ev1.to_bytes();
    assert_eq!(bytes.len(), 8);

    let ev1_recon = SpikeEvent::from_bytes(bytes);
    assert_eq!(ev1, ev1_recon);

    let ev2 = SpikeEvent::new(9999999, 31, -1);
    assert!(!ev2.is_on());
    assert!(ev2.is_off());
    let ev2_recon = SpikeEvent::from_bytes(ev2.to_bytes());
    assert_eq!(ev2, ev2_recon);
}

#[test]
fn test_gammatone_filter_resonance_and_attenuation() {
    let sample_rate = 16000.0f32;
    let target_fc = 1000.0f32;
    let mut filter = GammatoneFilter::new(target_fc, sample_rate);

    assert_eq!(filter.center_freq(), target_fc);
    assert!(filter.bandwidth() > 0.0);

    // 1. Resonant excitation at center frequency (1000 Hz)
    let num_samples = 2000;
    let mut peak_res = 0.0f32;
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let s = (2.0 * PI * target_fc * t).sin();
        let y = filter.step(s);
        if i > 500 {
            peak_res = peak_res.max(y.abs());
        }
    }

    assert!(
        peak_res > 0.85,
        "Gammatone filter resonance at center frequency must be strong (~1.0), got {peak_res}"
    );

    // 2. Off-resonance excitation at 3000 Hz (attenuation band)
    filter.reset();
    let mut peak_off = 0.0f32;
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let s = (2.0 * PI * 3000.0 * t).sin();
        let y = filter.step(s);
        if i > 500 {
            peak_off = peak_off.max(y.abs());
        }
    }

    assert!(
        peak_off < 0.25,
        "Gammatone filter off-band response must be attenuated, got {peak_off}"
    );

    let attenuation_ratio = peak_res / peak_off;
    assert!(
        attenuation_ratio > 3.5,
        "Gammatone filter selectivity must exceed 3.5x, got {attenuation_ratio:.2}x"
    );
}

#[test]
fn test_silence_quiescence_zero_spikes() {
    let sample_rate = 16000.0f32;
    let num_channels = 16;
    let mut cochlea = NeuromorphicCochlea::new(sample_rate, num_channels, 100.0, 7500.0);

    // Process 1.0 second of pure silence
    let silence = vec![0.0f32; 16000];
    let mut spikes = Vec::new();
    cochlea.process_buffer(&silence, 0, &mut spikes);

    assert_eq!(
        spikes.len(),
        0,
        "Neuromorphic silicon cochlea must emit exactly 0 spikes in pure silence, got {}",
        spikes.len()
    );

    // Process sub-threshold micro-noise (ambient electrical noise floor)
    let low_noise: Vec<f32> = (0..16000)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.0001)
        .collect();
    let mut noise_spikes = Vec::new();
    cochlea.process_buffer(&low_noise, 1_000_000, &mut noise_spikes);

    assert_eq!(
        noise_spikes.len(),
        0,
        "Silicon cochlea must emit zero spikes under sub-threshold noise floor, got {}",
        noise_spikes.len()
    );
}

#[test]
fn test_transient_onset_spike_generation() {
    let sample_rate = 16000.0f32;
    let num_channels = 16;
    let mut cochlea = NeuromorphicCochlea::new(sample_rate, num_channels, 100.0, 7500.0);

    // Generate quiet period followed by an energetic acoustic transient (simulating plosive attack)
    let mut audio = vec![0.0f32; 320]; // 20 ms silence
    for i in 0..800 {
        // 50 ms tone burst at 1200 Hz
        let t = i as f32 / sample_rate;
        audio.push(0.9 * (2.0 * PI * 1200.0 * t).sin());
    }
    audio.extend(vec![0.0f32; 800]); // 50 ms trailing silence for decay

    let mut spikes = Vec::new();
    cochlea.process_buffer(&audio, 0, &mut spikes);

    assert!(
        !spikes.is_empty(),
        "Acoustic transient must trigger asynchronous spike emissions"
    );

    let on_spikes = spikes.iter().filter(|s| s.is_on()).count();
    let off_spikes = spikes.iter().filter(|s| s.is_off()).count();

    assert!(on_spikes > 0, "Must contain ON spikes on acoustic onset, got {on_spikes}");
    assert!(off_spikes > 0, "Must contain OFF spikes on acoustic offset, got {off_spikes}");

    // Verify channel bounds
    for s in &spikes {
        assert!((s.channel as usize) < num_channels);
    }
}

#[test]
fn test_spiking_kws_phoneme_sequence_matching() {
    let num_channels = 16;
    let mut kws = SpikingKwsCell::for_plank(num_channels);

    // Synthetic spike train matching Plank phoneme stages:
    // Stage 0: [P/L] - onset burst in low/high channels
    // Stage 1: [AE] - vowel formant in mid channels
    // Stage 2: [NG/K] - velar closure/release in mid-high channels
    let mut events = Vec::new();

    // Stage 0 spikes at t = 10 ms to 30 ms
    for t_step in 0..10 {
        let ts = 10_000 + t_step * 2_000;
        events.push(SpikeEvent::new(ts, 1, 1));
        events.push(SpikeEvent::new(ts, 2, 1));
        events.push(SpikeEvent::new(ts, 14, 1));
    }

    // Stage 1 spikes at t = 60 ms to 100 ms
    for t_step in 0..12 {
        let ts = 60_000 + t_step * 3_000;
        events.push(SpikeEvent::new(ts, 6, 1));
        events.push(SpikeEvent::new(ts, 7, 1));
        events.push(SpikeEvent::new(ts, 8, 1));
    }

    // Stage 2 spikes at t = 130 ms to 170 ms
    for t_step in 0..12 {
        let ts = 130_000 + t_step * 3_000;
        events.push(SpikeEvent::new(ts, 11, 1));
        events.push(SpikeEvent::new(ts, 12, 1));
        events.push(SpikeEvent::new(ts, 13, 1));
    }

    let detections = kws.step_events(&events);

    assert_eq!(
        detections.len(),
        1,
        "Spiking KWS cell must spot wake-word 'Plank' exactly once, got {}",
        detections.len()
    );

    let event: &KeywordEvent = &detections[0];
    assert_eq!(event.keyword, "Plank");
    assert!(event.confidence >= 0.90);
    assert!(
        event.timestamp_sec > 0.12 && event.timestamp_sec < 0.20,
        "Keyword trigger timestamp must align with final phoneme stage, got {}",
        event.timestamp_sec
    );
}

#[test]
fn test_spiking_kws_sequence_timeout_rejection() {
    let num_channels = 16;
    let mut kws = SpikingKwsCell::for_plank(num_channels);

    let mut events = Vec::new();

    // Stage 0 spikes at t = 10 ms
    for t_step in 0..10 {
        let ts = 10_000 + t_step * 2_000;
        events.push(SpikeEvent::new(ts, 1, 1));
        events.push(SpikeEvent::new(ts, 2, 1));
    }

    // Delay 600 ms (> 400 ms coincidence window)
    // Stage 1 spikes at t = 650 ms
    for t_step in 0..12 {
        let ts = 650_000 + t_step * 3_000;
        events.push(SpikeEvent::new(ts, 6, 1));
        events.push(SpikeEvent::new(ts, 7, 1));
    }

    // Stage 2 spikes at t = 750 ms
    for t_step in 0..12 {
        let ts = 750_000 + t_step * 3_000;
        events.push(SpikeEvent::new(ts, 11, 1));
        events.push(SpikeEvent::new(ts, 12, 1));
    }

    let detections = kws.step_events(&events);

    assert_eq!(
        detections.len(),
        0,
        "Coincidence timeout must prevent wake-word false trigger across broken pauses"
    );
}

#[test]
fn test_spiking_kws_throughput() {
    let num_channels = 16;
    let mut kws = SpikingKwsCell::for_plank(num_channels);

    let total_spikes = 1_000_000;
    let mut events = Vec::with_capacity(total_spikes);

    for i in 0..total_spikes {
        let ts = (i / 10) as u32;
        let ch = (i % num_channels) as u16;
        let pol = if (i % 3) == 0 { -1 } else { 1 };
        events.push(SpikeEvent::new(ts, ch, pol));
    }

    let start = Instant::now();
    let mut count = 0;
    for ev in &events {
        if kws.step_event(ev).is_some() {
            count += 1;
        }
    }
    let elapsed = start.elapsed();

    let spikes_per_sec = (total_spikes as f64) / elapsed.as_secs_f64();
    println!(
        "Spiking KWS Event Throughput: {spikes_per_sec:.0} spikes/sec ({count} detections)"
    );

    let min_target = if cfg!(debug_assertions) {
        1_500_000.0
    } else {
        2_500_000.0
    };

    assert!(
        spikes_per_sec > min_target,
        "Spike processing throughput must exceed {min_target:.0} spikes/sec, got {spikes_per_sec:.0}"
    );
}

#[test]
fn test_end_to_end_audio_to_spikes_to_kws() {
    let sample_rate = 16000.0f32;
    let num_channels = 16;
    let mut cochlea = NeuromorphicCochlea::new(sample_rate, num_channels, 100.0, 7500.0);
    let mut kws = SpikingKwsCell::for_plank(num_channels);

    // Synthesize an end-to-end "Plank" acoustic waveform:
    // Silence (20 ms) -> Low-freq attack (30 ms) -> Formant tone (60 ms) -> Velar burst (30 ms) -> Silence (20 ms)
    let mut waveform = vec![0.0f32; 320]; // 20 ms silence

    // Stage 0: Plosive attack
    for i in 0..480 {
        let t = i as f32 / sample_rate;
        waveform.push(0.7 * (2.0 * PI * 350.0 * t).sin() + 0.3 * (2.0 * PI * 4200.0 * t).sin());
    }

    // Stage 1: Formant resonance
    for i in 0..960 {
        let t = i as f32 / sample_rate;
        waveform.push(0.6 * (2.0 * PI * 750.0 * t).sin() + 0.4 * (2.0 * PI * 1800.0 * t).sin());
    }

    // Stage 2: Velar release burst
    for i in 0..480 {
        let t = i as f32 / sample_rate;
        waveform.push(0.7 * (2.0 * PI * 2800.0 * t).sin() + 0.3 * (2.0 * PI * 3500.0 * t).sin());
    }

    waveform.extend(vec![0.0f32; 320]); // Trailing silence

    let mut spikes = Vec::new();
    cochlea.process_buffer(&waveform, 0, &mut spikes);

    assert!(
        !spikes.is_empty(),
        "Audio waveform must generate spikes through neuromorphic cochlea"
    );

    let detections = kws.step_events(&spikes);
    println!(
        "End-to-End Cochlea Spike Count: {}, Detections: {}",
        spikes.len(),
        detections.len()
    );
    assert_eq!(kws.total_spikes_processed(), spikes.len() as u64);
}
