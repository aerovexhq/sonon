# Sonon (`sonon`)

**High-Performance Robotics-Aimed Acoustic DSP, Spatial Beamforming, Rotor Noise Suppression, AeroSSM State-Space Models, Neuromorphic Silicon Cochlea & Spiking Wake-Word Spotting in Pure Safe Rust**

[![Ecosystem](https://img.shields.io/badge/Ecosystem-Aerovex%20HQ-0ea5e9?style=flat-square)](https://github.com/aerovexhq)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue?style=flat-square)](LICENSE)
[![Safety](https://img.shields.io/badge/Rust-100%25%20Safe%20Code-brightgreen?style=flat-square)](#)

---

## Overview

**Sonon** is a minimalist, hard real-time acoustic signal processing, phrase spotting, and airframe acoustic diagnostics engine engineered for embedded robotics edge platforms, autopilot companion computers, and microcontrollers.

Engineered in 100% pure safe Rust (`#![deny(unsafe_code)]` at line 1 of every module) with strictly zero external C audio dependencies and zero unicode emojis, Sonon provides a unified edge auditory perception stack spanning classical physics, structured state-space neural models, and biological neuromorphic silicon cochlea dynamics.

---

## Capabilities Across 9 Operational Phases

### 1. Classical Acoustic DSP & Feature Extraction
- **Lock-Free Audio Ring Buffer** (`AudioRingBuffer`): Contiguous circular buffer with zero allocations during steady-state streaming.
- **Spectral Decomposition** (`FftProcessor`, `Window`): In-place Radix-2 Cooley-Tukey FFT with precomputed Hann, Hamming, and Blackman windows.
- **Mel Filterbank & MFCC** (`MelFilterbank`): Triangular Mel filterbanks with DCT-II cepstral extraction.
- **Adaptive Voice Activity Detection** (`EnergyVad`): Dynamic noise-floor tracking with hangover smoothing.

### 2. Few-Shot Dynamic Time Warping (DTW) & PCEN Normalization
- **Sakoe-Chiba Banded DTW** (`DtwMatcher`): Global path corridor constraint reducing computational complexity from $O(N \cdot M)$ to $O(N \cdot R)$.
- **DBA Barycenter Averaging** (`dtw_barycenter_averaging`): Iterative multi-exemplar template centroid synthesis.
- **Per-Channel Energy Normalization** (`PcenFilter`): Adaptive gain control with feed-forward IIR smoothing, dynamic range compression, and temporal differencing for far-field robustness.

### 3. Autopilot Telemetry-Coupled Rotor Harmonic Notch Bank
- **IIR Biquad Notch Filter** (`BiquadNotchFilter`): Direct Form II Transposed notch delivering $> 86\text{ dB}$ attenuation depth.
- **Dynamic Rotor Harmonic Bank** (`RotorHarmonicNotchBank`): Couples directly to flight controller ESC RPM telemetry ($f_k = k \cdot \frac{N_{\text{blades}} \cdot \text{RPM}}{60}$), continuously notching blade pass tones under $+15\text{ dB}$ motor whine.
- **Spectral Subtraction** (`SpectralSubtractionSuppressor`): Running spectral noise-floor estimator with over-subtraction and spectral floor protection.

### 4. Multi-Microphone Spatial Beamforming & Direction of Arrival (DoA)
- **Arbitrary Array Geometries** (`ArrayGeometry`): Pre-calibrated linear (wing-mount), circular (360-degree fuselage), and tetrahedral (3D) array configurations.
- **GCC-PHAT TDoA Estimation** (`GccPhatEstimator`): Generalized Cross-Correlation with Phase Transform and sub-sample parabolic interpolation.
- **Delay-and-Sum Beamformer** (`DelayAndSumBeamformer`): Real-time fractional-delay spatial steering providing $+5.59\text{ dB}$ directional signal-to-noise improvement.

### 5. AeroSSM Structured State-Space Models & Anticipatory Prefix Interlock
- **SincNet Parametric Frontend** (`SincConvFrontend`): Bandpass sinc filterbank with live ESC notch modulation.
- **Selective State-Space Recurrence** (`AeroSsmCell`): Input-dependent state-space kernel delivering $O(1)$ constant memory and high streaming throughput.
- **Wald's Sequential Probability Ratio Test** (`AnticipatoryPrefixDecoder`): Anticipatory prefix decoding triggering flight actions at $70\%$ phrase completion with a two-phase speculative flight actuator interlock (`Listening`, `PreArm`, `Commit`, `Rollback`).

### 6. Airframe Acoustic Health Monitoring & Anomaly Diagnostics
- **Blade Damage & Imbalance Detection** (`AcousticHealthMonitor`): Detects chipped or cracked blades via rotational subharmonic energy emergence ($0.5\times$, $1.5\times$ rotor fundamental).
- **Motor Bearing Wear Diagnostics**: Spectral kurtosis and ISO 10816 high-frequency friction ratios detecting impending mechanical failure.
- **MAVLink Telemetry Integration**: Formats health telemetry as standard `NAMED_VALUE_FLOAT` packets (`SONON_HLTH`, `SONON_STAT`, `SON_IMB*`, `SON_BRG*`).

### 7. Universal C-ABI FFI Layer & Zero-Copy POSIX Shared Memory
- **C-ABI FFI Layer** (`capi.rs`): Pure safe Rust handle registry providing C-compatible dynamic symbols (`sonon_engine_create`, `sonon_engine_destroy`, `sonon_shm_pump`).
- **POSIX Shared Memory Bridge** (`shm.rs`): Zero-copy inter-process audio channel (`/dev/shm/sonon_audio`) sustaining $> 15,000,000\text{ samples/sec}$.
- **Python SDK** (`bindings/python/sonon.py`): Zero-copy NumPy bindings.
- **C++20 Header-Only Client** (`bindings/cpp/include/sonon.hpp`): Modern RAII client library.

### 8. Microcontroller Portability & Fixed-Point DSP Math
- **Signed Q15 & Q31 Fixed-Point Math** (`src/fixed.rs`): Saturating arithmetic, widening dot products, and quarter-wave sine/cosine look-up tables with zero FPU instructions.
- **Heapless Circular Audio Buffer** (`StaticAudioBuffer<const N: usize>`): Deterministic memory allocation with zero dynamic heap usage for Cortex-M microcontrollers.
- **Pure Integer Manhattan DTW** (`FixedDtwMatcher`): Deterministic keyword matching running in integer arithmetic.

### 9. Neuromorphic Silicon Cochlea & Event-Driven Spiking Wake-Word Spotting
- **Greenwood Place-Frequency Mapping** (`NeuromorphicCochlea`): Physiologically mapped 16/32-channel basilar membrane filterbank with Greenwood distribution.
- **Gammatone Filterbank** (`GammatoneFilter`): 4th-order cascade complex IIR resonators with inner hair cell half-wave mechanical rectification and logarithmic compression.
- **Address-Event Representation (AER)** (`SpikeEvent`): Asynchronous delta contrast modulator emitting 8-byte binary spike events. In acoustic silence, active dynamic power collapses to zero spikes.
- **Leaky Integrate-and-Fire (LIF) SNN Decoder** (`SpikingKwsCell`): Multi-stage phoneme coincidence detection with exponential synaptic traces, lateral receptive field inhibition, and refractory lockout running in $O(1)$ constant time per event ($> 25,000,000\text{ spikes/sec}$).

---

## Directory Layout

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
│   ├── dtw.rs                  # Sakoe-Chiba banded DTW & Barycenter Averaging
│   ├── pcen.rs                 # Per-Channel Energy Normalization (PCEN)
│   ├── notch.rs                # Telemetry-guided rotor harmonic notch bank
│   ├── spectral_subtraction.rs # Spectral subtraction noise suppression
│   ├── beamforming.rs          # Spatial Delay-and-Sum beamforming & GCC-PHAT DoA
│   ├── aerossm.rs              # SincNet frontend & AeroSSM state-space kernel
│   ├── anticipatory.rs         # Wald's SPRT 70% early prefix flight interlock
│   ├── health.rs               # Acoustic airframe health & MAVLink telemetry
│   ├── shm.rs                  # Zero-copy /dev/shm/sonon_audio ring buffer
│   ├── capi.rs                 # Safe C-ABI dynamic library handle registry
│   ├── fixed.rs                # Q15/Q31 fixed-point DSP math & heapless buffers
│   ├── neuromorphic.rs         # Silicon cochlea, AER spikes & LIF spiking KWS
│   └── engine.rs               # Unified streaming SononEngine coordinator
├── bindings/
│   ├── python/                 # Python NumPy zero-copy client SDK
│   └── cpp/                    # Modern C++20 RAII client SDK
├── tests/
│   ├── sonon_dsp_tests.rs      # Phase 1: FFT, Mel, VAD, RingBuffer tests
│   ├── sonon_phase2_tests.rs   # Phase 2: Banded DTW, DBA, PCEN tests
│   ├── sonon_human_voice_tests.rs # Phase 2.5: Real human voice verification
│   ├── sonon_phase3_tests.rs   # Phase 3: Rotor harmonic notch & spectral subtraction
│   ├── sonon_phase4_tests.rs   # Phase 4: Beamforming & GCC-PHAT DoA tests
│   ├── sonon_phase5_tests.rs   # Phase 5: AeroSSM & Anticipatory prefix tests
│   ├── sonon_phase6_tests.rs   # Phase 6: Acoustic health & MAVLink tests
│   ├── sonon_phase7_tests.rs   # Phase 7: C-ABI FFI & Shared Memory IPC tests
│   ├── sonon_phase8_tests.rs   # Phase 8: Fixed-point Q15/Q31 & heapless buffer tests
│   └── sonon_phase9_tests.rs   # Phase 9: Neuromorphic cochlea & spiking KWS tests
└── analysis/                   # 11-part doctoral research monograph suite
```

---

## Verification & Testing

```bash
# Execute entire 56-test verification suite in release profile
cargo test --release

# Execute Python FFI tests
python3 bindings/python/test_sonon_py.py

# Execute C++20 RAII tests
g++ -std=c++20 -Ibindings/cpp/include bindings/cpp/test_sonon_cpp.cpp -o /tmp/test_cpp && /tmp/test_cpp
```
