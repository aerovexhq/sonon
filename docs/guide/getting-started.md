# Getting Started with Sonon

**Sonon** is a minimalist, hard real-time acoustic signal processing, phrase spotting, and airframe acoustic diagnostics engine engineered for embedded robotics edge platforms, autopilot companion computers, and microcontrollers.

Engineered in **100% pure safe Rust** (`#![deny(unsafe_code)]` at line 1 of every module) with strictly zero external C audio dependencies, Sonon provides a unified edge auditory perception stack spanning classical physics, structured state-space models, and biological neuromorphic silicon cochlea dynamics.

## Key Capabilities

1. **Deterministic Edge Audio Processing**: Continuous 16 kHz stream ingestion with zero dynamic allocations during steady-state processing.
2. **Propeller Noise Attenuation**: Live ESC telemetry synchronization notches out blade pass frequencies (BPF) by up to 40 dB.
3. **Few-Shot Wake-Word Recognition**: Dynamic Time Warping (DTW) with Sakoe-Chiba band corridors allows instant 1-2 exemplar voice enrollment offline.
4. **Physical Formant Synthesis**: Klatt and Liljencrants-Fant glottal flow models generate speech without neural network weights.
5. **Airframe Health Diagnostics**: Continuous Wavelet Transform (CWT) detects motor bearing wear and rotor blade flutter.
6. **Cross-Platform Portability**: Compiles to native binaries, C-ABI shared libraries, Python wheels, and WebAssembly (`wasm32-unknown-unknown`).

## Quick Navigation

- [Try the Interactive WASM Playground](/playground)
- [Architecture & Core Principles](/guide/architecture)
- [Acoustic DSP Overview](/dsp/overview)
- [Rust Engine API](/api/rust)
