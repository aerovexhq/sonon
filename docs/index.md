---
layout: home

hero:
  name: Sonon
  text: Embedded Acoustic Intelligence & Voice Control
  tagline: Hard real-time acoustic DSP, spatial beamforming, rotor noise suppression, and few-shot wake-word spotting in 100% pure safe Rust.
  image:
    src: /favicon.svg
    alt: Sonon Logo
  actions:
    - theme: brand
      text: Open Interactive Playground
      link: /playground
    - theme: alt
      text: Documentation & Physics
      link: /guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/aerovexhq/sonon

features:
  - icon: ⚡
    title: 100% Pure Safe Rust
    details: Strictly zero external C audio libraries and `#![deny(unsafe_code)]` crate-wide. Runs bare-metal on microcontrollers, companion computers, or in browser WebAssembly.
  - icon: 🎯
    title: Dynamic Rotor Notch Filtering
    details: Telemetry-coupled IIR biquad notch bank locked to motor RPM telemetry, attenuating propeller blade pass frequencies by up to 40 dB.
  - icon: 🔊
    title: Few-Shot Wake-Word Spotting
    details: Banded Sakoe-Chiba DTW and DBA barycenter averaging enabling immediate 1-2 exemplar voice enrollment without gigabyte neural models.
  - icon: 🗣️
    title: Physical Speech Synthesis
    details: Klatt + Liljencrants-Fant physical glottal flow model and cascade vocal tract resonators delivering smooth coarticulation in WebAssembly.
  - icon: 🩺
    title: Airframe Acoustic Health
    details: Complex Morlet continuous wavelet transform (CWT) evaluating scale-wise kurtosis and blade flutter before structural failure.
  - icon: 🦇
    title: Biosonar Echolocation & 3D Mapping
    details: Active linear chirp compression and CA-CFAR echo detection providing 5-axis clearance and collision hazard guards in GPS-denied environments.
---

## Multi-Platform Quickstart

```typescript
// WebAssembly in Browser (JavaScript / TypeScript)
const { instance } = await WebAssembly.instantiateStreaming(fetch('sonon.wasm'));
const api = resolveSononWasmExports(instance);
const handle = api.sonon_wasm_create(16000, 512, 160, 13);

// Register Custom Word from Live Microphone Voice Recording
api.sonon_wasm_set_string_bytes("falcon");
api.sonon_wasm_set_input_samples(recordedVoiceSamples);
api.sonon_wasm_enroll_audio_buffer(handle, 6, recordedVoiceSamples.length, 0.58);

// Streaming Ingestion via AudioWorkletNode
const detections = api.sonon_wasm_ingest(handle, samples.length);
if (detections > 0) {
    const keyword = api.sonon_wasm_get_last_keyword();
    console.log(`Detected: ${keyword}`);
}
```

```rust
// Pure Safe Rust Native Ingestion
use sonon::engine::SononEngine;

let mut engine = SononEngine::new(16_000, 512, 160, 13)?;
engine.enable_telemetry_coupled_notch(4, 2); // 4 blades, 2 harmonics
engine.update_motor_rpm(5_400.0);

let samples: [f32; 160] = read_mic_buffer();
if let Some(event) = engine.process_frame(&samples)? {
    println!("Spotting Event: {} (confidence: {:.2})", event.keyword, event.confidence);
}
```
