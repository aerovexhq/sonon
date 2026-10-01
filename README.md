# Sonon (`sonon`)

**High-Performance Robotics-Aimed Acoustic DSP, Voice Activity Detection & Few-Shot Phrase Spotting Engine in Pure Safe Rust**

[![Ecosystem](https://img.shields.io/badge/Ecosystem-Aerovex%20HQ-0ea5e9?style=flat-square)](https://github.com/aerovexhq)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue?style=flat-square)](LICENSE)

---

## Overview

**Sonon** is a minimalist, hard real-time acoustic signal processing and keyword spotting engine engineered for embedded robotics edge platforms (microcontrollers, Raspberry Pi, NVIDIA Jetson, FPGA companion compute).

Designed for edge audio interaction without reliance on multi-gigabyte neural network models or cloud connections, Sonon delivers:
1. **Low-Latency Streaming Audio Pipeline**: Contiguous lock-free circular ring buffer (`AudioRingBuffer`) for microsecond microphone sample ingestion.
2. **Spectral Analysis**: Real-to-complex Radix-2 Cooley-Tukey FFT (`FftProcessor`) paired with configurable windowing functions (Hann, Hamming, Blackman).
3. **Mel-Scale Filterbank & MFCC Extraction**: Triangular Mel filterbanks spanning arbitrary frequency envelopes with DCT-II cepstral representations.
4. **Adaptive Voice Activity Detection (VAD)**: Dynamic noise-floor tracking with hangover smoothing to reject drone propeller/rotor background noise.
5. **Few-Shot Phrase Spotting (DTW)**: Dynamic Time Warping template matcher recognizing custom user keywords (e.g., *"take off"*, *"land"*, *"halt"*) from 1-2 voice exemplars.

---

## Architecture

```
sonon/
├── Cargo.toml                  # Manifest (pure safe Rust, #![deny(unsafe_code)])
├── src/
│   ├── lib.rs                  # Public module exports
│   ├── ring_buffer.rs          # Circular sample buffer
│   ├── window.rs               # Precomputed windowing functions
│   ├── stft.rs                 # Radix-2 FFT and power spectrum estimation
│   ├── mel.rs                  # Triangular Mel filterbank & MFCC extraction
│   ├── vad.rs                  # Adaptive energy-based Voice Activity Detection
│   ├── dtw.rs                  # Dynamic Time Warping phrase template matcher
│   └── engine.rs               # Unified streaming SononEngine coordinator
└── tests/
    └── sonon_dsp_tests.rs      # Unit tests & 1M+ samples/sec throughput benchmark
```

---

## Quick Example

```rust
use sonon::{SononEngine, WindowType};

fn main() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);

    // Extract features from reference exemplar
    let template_samples = vec![0.0f32; 3200]; // 0.2s exemplar
    let template_features = engine.extract_features(&template_samples);

    // Enroll custom keyword
    engine.enroll_keyword("emergency_stop", template_features, 3.5);

    // Stream live microphone samples
    let incoming_mic_frame = vec![0.1f32; 160];
    let detections = engine.ingest_samples(&incoming_mic_frame);

    for event in detections {
        println!("Recognized keyword '{}' (conf: {:.2})", event.keyword, event.confidence);
    }
}
```

---

## Verification

```bash
cargo test
```
