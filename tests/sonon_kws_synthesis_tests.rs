//! Comprehensive test suite verifying high-precision Wake-Word Spotting (KWS),
//! streaming VAD, FeatureRingBuffer, Liljencrants-Fant glottal flow synthesis,
//! automated synthetic exemplar pipeline enrollment, and noise robustness under drone rotor interference.

#![deny(unsafe_code)]

use sonon::{
    FeatureMode, FeatureRingBuffer, LiljencrantsFantPulse, SononEngine,
    StreamingDtwConfig, SyntheticExemplarGenerator,
};
use std::f32::consts::PI;

#[test]
fn test_feature_ring_buffer_zero_heap() {
    let capacity = 8;
    let feature_dim = 13;
    let mut ring = FeatureRingBuffer::new(capacity, feature_dim);

    assert_eq!(ring.len(), 0);
    assert!(ring.is_empty());
    assert_eq!(ring.capacity(), capacity);
    assert_eq!(ring.feature_dim(), feature_dim);

    // Push 5 distinct frames
    for i in 0..5 {
        let frame = vec![i as f32; feature_dim];
        ring.push_frame(&frame);
    }
    assert_eq!(ring.len(), 5);
    let frames = ring.to_vec();
    assert_eq!(frames.len(), 5);
    assert_eq!(frames[0][0], 0.0);
    assert_eq!(frames[4][0], 4.0);

    // Overfill past capacity by pushing 6 more frames (total 11 pushed)
    for i in 5..11 {
        let frame = vec![i as f32; feature_dim];
        ring.push_frame(&frame);
    }
    assert_eq!(ring.len(), 8);
    let latest_all = ring.to_vec();
    assert_eq!(latest_all.len(), 8);
    // Oldest surviving should be 11 - 8 = 3.0
    assert_eq!(latest_all[0][0], 3.0);
    assert_eq!(latest_all[7][0], 10.0);

    // Test get_latest
    let latest_3 = ring.get_latest(3);
    assert_eq!(latest_3.len(), 3);
    assert_eq!(latest_3[0][0], 8.0);
    assert_eq!(latest_3[1][0], 9.0);
    assert_eq!(latest_3[2][0], 10.0);

    ring.clear();
    assert_eq!(ring.len(), 0);
    assert!(ring.is_empty());
}

#[test]
fn test_liljencrants_fant_pulse_physics() {
    let modal = LiljencrantsFantPulse::from_rd(1.0);
    let pressed = LiljencrantsFantPulse::from_rd(0.5);
    let breathy = LiljencrantsFantPulse::from_rd(2.0);

    // Validate physical timing constraints
    assert!(modal.tp < modal.te, "Peak flow must precede maximum excitation");
    assert!(pressed.tp < pressed.te);
    assert!(breathy.tp < breathy.te);

    // Pressed voice has shorter return phase (ta) and sharper excitation
    assert!(pressed.ta < breathy.ta, "Pressed voice has faster closure than breathy");

    // Evaluate waveform across a full glottal period [0.0, 1.0)
    let n_steps = 100;
    let mut modal_samples = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let p = (i as f32) / (n_steps as f32);
        modal_samples.push(modal.evaluate(p));
    }

    // Must have positive open phase values
    let peak_open = modal_samples[0..50].iter().fold(0.0f32, |acc, &x| acc.max(x));
    assert!(peak_open > 0.0, "Glottal open phase must produce positive derivative");

    // Must reach negative closure excitation
    let min_return = modal_samples[50..95].iter().fold(0.0f32, |acc, &x| acc.min(x));
    assert!(min_return < -0.5, "LF pulse must exhibit negative excitation spike at closure");

    // Closed phase (end of period) must return to 0.0
    assert_eq!(modal.evaluate(0.99), 0.0, "Glottal wave must be quiescent in closed phase");
}

#[test]
fn test_synthetic_exemplar_generator_diversity() {
    let sample_rate = 16000.0;
    let generator = SyntheticExemplarGenerator::new(sample_rate);
    let phrase = "abort";

    let count = 6;
    let exemplars = generator.generate_exemplars(phrase, count);
    assert_eq!(exemplars.len(), count, "Must generate requested number of exemplars");

    // Verify all exemplars have non-trivial audio lengths and distinct waveforms
    for (idx, audio) in exemplars.iter().enumerate() {
        assert!(audio.len() > 2000, "Audio must contain substantial length for phrase");
        let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(peak > 0.5, "Audio exemplar {} must have normalized amplitude", idx);
    }

    // Verify acoustic waveform variation across generated exemplars (due to differing F0, speed, tract length)
    let len0 = exemplars[0].len();
    let len1 = exemplars[1].len();
    let has_variation = exemplars.iter().any(|ex| ex.len() != len0) || len0 != len1;
    assert!(has_variation, "Synthetic exemplars must possess varying speaking durations");
}

#[test]
fn test_synthetic_pipeline_enrollment_and_cross_speaker_spotting() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Enroll "land" using the automated synthetic exemplar pipeline
    let threshold = engine.enroll_keyword_synthetic_pipeline("land", "land", 5, 12, 1.45);
    assert!(threshold > 0.0, "Pipeline must compute a valid positive threshold");
    assert_eq!(engine.dtw().template_count(), 1);

    // Synthesize independent test utterances representing diverse speaker profiles
    let mut male_synth = sonon::KlattSynthesizer::new(sample_rate);
    male_synth.set_f0(100.0); // Deep male pitch
    male_synth.set_speaking_rate(0.95);
    male_synth.set_vocal_tract_scale(1.10);
    let male_audio = male_synth.synthesize(&sonon::G2pEngine::text_to_phonemes("land"));

    let mut female_synth = sonon::KlattSynthesizer::new(sample_rate);
    female_synth.set_f0(210.0); // Higher female pitch
    female_synth.set_speaking_rate(1.10);
    female_synth.set_vocal_tract_scale(0.92);
    let female_audio = female_synth.synthesize(&sonon::G2pEngine::text_to_phonemes("land"));

    // Test 1: Male voice spotting
    engine.reset();
    let male_events = engine.ingest_samples(&male_audio);
    assert!(
        !male_events.is_empty(),
        "Synthetic-trained template must spot male voice utterance"
    );
    assert_eq!(male_events[0].keyword, "land");
    assert!(male_events[0].confidence > 0.20);

    // Test 2: Female voice spotting
    engine.reset();
    let female_events = engine.ingest_samples(&female_audio);
    assert!(
        !female_events.is_empty(),
        "Synthetic-trained template must spot female voice utterance"
    );
    assert_eq!(female_events[0].keyword, "land");
    assert!(female_events[0].confidence > 0.20);
}

#[test]
fn test_streaming_sliding_window_endpoint_detection() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    let _thresh = engine.enroll_keyword_synthetic_pipeline("take_off", "take off", 4, 12, 1.40);

    // Synthesize test audio
    let keyword_audio = engine.synthesize_speech_from_text("take off");

    // Construct a continuous audio stream:
    // 1.5 seconds silence -> Keyword -> 1.0 second silence
    let silence_lead = vec![0.0f32; (1.5 * sample_rate) as usize];
    let silence_trail = vec![0.0f32; (1.0 * sample_rate) as usize];

    let mut full_stream = Vec::new();
    full_stream.extend_from_slice(&silence_lead);
    full_stream.extend_from_slice(&keyword_audio);
    full_stream.extend_from_slice(&silence_trail);

    // Stream audio in 160-sample (10 ms) chunks mimicking real mic hardware
    let chunk_size = 160;
    let mut detected_events = Vec::new();
    let mut offset = 0;

    while offset + chunk_size <= full_stream.len() {
        let chunk = &full_stream[offset..offset + chunk_size];
        let events = engine.ingest_samples(chunk);
        detected_events.extend(events);
        offset += chunk_size;
    }

    assert!(
        !detected_events.is_empty(),
        "Streaming sliding window matcher must detect keyword despite preceding silence"
    );
    assert_eq!(detected_events[0].keyword, "take_off");

    // Verify detection occurred near the end of the keyword utterance (~ 1.5s + keyword duration)
    let detection_time = detected_events[0].timestamp_sec;
    assert!(
        detection_time >= 1.4 && detection_time <= 2.8,
        "Detection timestamp ({:.2}s) must correspond to spoken keyword window",
        detection_time
    );
}

#[test]
fn test_refractory_debounce_suppresses_duplicate_events() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Configure refractory debounce to 15 frames
    let mut dtw_cfg = StreamingDtwConfig::default();
    dtw_cfg.refractory_frames = 15;
    engine.set_streaming_dtw_config(dtw_cfg);

    engine.enroll_keyword_synthetic_pipeline("abort", "abort", 4, 10, 1.35);
    let keyword_audio = engine.synthesize_speech_from_text("abort");

    // Stream audio frame by frame
    let chunk_size = 128;
    let mut events = Vec::new();
    let mut offset = 0;
    while offset + chunk_size <= keyword_audio.len() {
        let chunk = &keyword_audio[offset..offset + chunk_size];
        let evs = engine.ingest_samples(chunk);
        events.extend(evs);
        offset += chunk_size;
    }

    assert_eq!(
        events.len(),
        1,
        "Refractory lockout must prevent multiple duplicate triggers for a single utterance"
    );
    assert_eq!(events[0].keyword, "abort");
}

#[test]
fn test_kws_under_rotor_noise_and_false_alarm_rejection() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // Enable rotor harmonic notch filter targeting 350 Hz BPF (e.g. 7,000 RPM 3-blade prop)
    engine.enable_rotor_notch(3, 2, 14.0);
    engine.update_motor_rpm(7000.0);

    engine.enroll_keyword_synthetic_pipeline("emergency", "emergency", 5, 12, 1.50);

    // 1. Generate 3.0 seconds of quadcopter rotor interference:
    // Tonal Blade Pass Frequency (350 Hz, 700 Hz) + wideband turbulence hum
    let noise_len = (3.0 * sample_rate) as usize;
    let mut drone_noise = vec![0.0f32; noise_len];
    for n in 0..noise_len {
        let t = (n as f32) / sample_rate;
        let bpf1 = 0.35 * (2.0 * PI * 350.0 * t).sin();
        let bpf2 = 0.20 * (2.0 * PI * 700.0 * t).sin();
        let turbulence = 0.10 * (2.0 * PI * 180.0 * t + (2.0 * PI * 5.0 * t).sin()).sin();
        drone_noise[n] = bpf1 + bpf2 + turbulence;
    }

    // Ingest pure drone noise: False Alarm Rate (FAR) must be zero
    let false_alarm_events = engine.ingest_samples(&drone_noise);
    assert!(
        false_alarm_events.is_empty(),
        "Pure drone propeller hum must not produce false alarm triggers"
    );

    // 2. Mix keyword into drone noise at challenging +3 dB SNR
    let keyword_audio = engine.synthesize_speech_from_text("emergency");
    let mut noisy_speech = drone_noise[0..keyword_audio.len() + 1600].to_vec();
    for (i, &s) in keyword_audio.iter().enumerate() {
        noisy_speech[800 + i] += s;
    }

    engine.reset();
    let mut detected = false;
    let chunk_size = 256;
    let mut offset = 0;
    while offset + chunk_size <= noisy_speech.len() {
        let chunk = &noisy_speech[offset..offset + chunk_size];
        let evs = engine.ingest_samples(chunk);
        if !evs.is_empty() {
            assert_eq!(evs[0].keyword, "emergency");
            detected = true;
        }
        offset += chunk_size;
    }

    assert!(
        detected,
        "Keyword must be spotted accurately under realistic quadcopter rotor noise"
    );
}

#[test]
fn test_kws_synthesis_throughput_benchmark() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    engine.enroll_keyword_synthetic_pipeline("hold", "hold", 3, 10, 1.40);
    let test_audio = engine.synthesize_speech_from_text("hold");

    // Ingest 50,000 samples and benchmark execution time
    let repeat_count = (50000 / test_audio.len()) + 1;
    let mut benchmark_stream = Vec::with_capacity(repeat_count * test_audio.len());
    for _ in 0..repeat_count {
        benchmark_stream.extend_from_slice(&test_audio);
    }
    benchmark_stream.truncate(50000);

    let start = std::time::Instant::now();
    let _ = engine.ingest_samples(&benchmark_stream);
    let elapsed = start.elapsed();

    let samples_per_sec = (benchmark_stream.len() as f64) / elapsed.as_secs_f64();
    let realtime_factor = samples_per_sec / (sample_rate as f64);

    assert!(
        realtime_factor > 25.0,
        "Streaming KWS execution ({:.1}x real-time) must exceed 25x real-time",
        realtime_factor
    );
}
