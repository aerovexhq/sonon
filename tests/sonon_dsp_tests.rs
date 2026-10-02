use sonon::{
    AudioRingBuffer, EnergyVad, FftProcessor, MelFilterbank, SononEngine, Window, WindowType,
};
use std::f32::consts::PI;
use std::time::Instant;

#[test]
fn test_window_symmetry_and_application() {
    let size = 256;
    let window = Window::new(WindowType::Hann, size);
    let coeffs = window.coefficients();
    assert_eq!(coeffs.len(), size);

    // Verify Hann symmetry: coeffs[i] == coeffs[size - 1 - i]
    for i in 0..(size / 2) {
        assert!(
            (coeffs[i] - coeffs[size - 1 - i]).abs() < 1e-5,
            "Hann window must be symmetric"
        );
    }

    let mut frame = vec![1.0f32; size];
    window.apply(&mut frame);
    assert_eq!(frame[0], coeffs[0]);
    assert_eq!(frame[size / 2], coeffs[size / 2]);
}

#[test]
fn test_ring_buffer_fifo_ordering() {
    let mut ring = AudioRingBuffer::new(8);
    ring.push_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ring.len(), 4);

    let mut out = vec![0.0f32; 4];
    assert!(ring.read_latest(4, &mut out));
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

    // Push more to trigger wrap-around
    ring.push_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    assert_eq!(ring.len(), 8); // Capacity capped at 8

    let mut latest = vec![0.0f32; 4];
    assert!(ring.read_latest(4, &mut latest));
    assert_eq!(latest, vec![7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn test_fft_spectral_peak_sine_wave() {
    let size = 512;
    let sample_rate = 16000.0f32;
    let target_freq = 1000.0f32; // 1 kHz sine wave
    let fft = FftProcessor::new(size);

    let mut signal = Vec::with_capacity(size);
    for i in 0..size {
        let t = (i as f32) / sample_rate;
        signal.push((2.0 * PI * target_freq * t).sin());
    }

    let power = fft.power_spectrum(&signal);
    assert_eq!(power.len(), size / 2 + 1);

    // Find peak frequency bin
    let mut max_bin = 0;
    let mut max_power = 0.0f32;
    for (bin, &p) in power.iter().enumerate() {
        if p > max_power {
            max_power = p;
            max_bin = bin;
        }
    }

    let bin_width = sample_rate / (size as f32);
    let detected_freq = (max_bin as f32) * bin_width;
    assert!(
        (detected_freq - target_freq).abs() <= bin_width,
        "Detected freq {} must match target 1000 Hz within bin resolution {}",
        detected_freq,
        bin_width
    );
}

#[test]
fn test_mel_filterbank_and_mfcc() {
    let fft_size = 512;
    let sample_rate = 16000.0;
    let mel = MelFilterbank::new(26, fft_size, sample_rate, 80.0, 7600.0);
    assert_eq!(mel.num_filters(), 26);

    let power = vec![1.0f32; fft_size / 2 + 1];
    let log_energies = mel.compute_log_energies(&power);
    assert_eq!(log_energies.len(), 26);

    let mfcc = mel.compute_mfcc(&log_energies, 13);
    assert_eq!(mfcc.len(), 13);
}

#[test]
fn test_energy_vad_speech_detection() {
    let mut vad = EnergyVad::new(3.0, 0.95, 2);

    // Process silent frames
    let silence = vec![0.001f32; 160];
    for _ in 0..10 {
        vad.process_frame(&silence);
    }
    assert!(!vad.is_speech_active(), "Silence should not trigger VAD");

    // Ingest speech burst
    let speech = vec![0.5f32; 160];
    let active = vad.process_frame(&speech);
    assert!(active, "Loud energy burst must trigger VAD");
    assert!(vad.is_speech_active());
}

#[test]
fn test_dtw_keyword_spotting_engine() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Synthesize template audio (e.g. 500 Hz tone for 0.2s)
    let num_samples = (0.2 * sample_rate) as usize;
    let mut template_audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = (i as f32) / sample_rate;
        template_audio.push((2.0 * PI * 500.0 * t).sin());
    }

    let features = engine.extract_features(&template_audio);
    assert!(!features.is_empty(), "Must extract non-empty features");

    engine.enroll_keyword("takeoff", features, 5.0);

    // Ingest identical signal and verify match event
    let events = engine.ingest_samples(&template_audio);
    assert!(
        events.iter().any(|e| e.keyword == "takeoff"),
        "Engine must recognize enrolled keyword"
    );
}

#[test]
fn test_sonon_throughput_benchmark() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Generate 1 second (16000 samples) of audio
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = engine.ingest_samples(&samples);
    }
    let elapsed = start.elapsed();

    let total_samples = 16000 * 10;
    let samples_per_sec = (total_samples as f64) / elapsed.as_secs_f64();

    // Verify throughput exceeds 500,000 samples/sec (> 30x real-time speed)
    assert!(
        samples_per_sec > 500_000.0,
        "Throughput was {:.0} samples/sec, below 500,000 threshold",
        samples_per_sec
    );
}
