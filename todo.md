# Sonon Project Roadmap & Task Registry

> **Sonon Autonomous Roadmap**: All Sonon-specific roadmap phases, technical specifications, and development milestones are tracked exclusively within this document in accordance with Aerovex's modular repository decoupling architecture.

---

## Grand End-Goal: Sonon Embedded Acoustic Intelligence & Voice Control Engine

The ultimate objective of **Sonon** (`sonon` / `aerovexhq/sonon`) is an ultra-low-latency, zero-cloud, deterministic acoustic signal processing and phrase spotting engine tailored for edge companion computers, robotics microcontrollers, and aerospace autopilots:

1. **Ultra-Low Resource Footprint**:
   - Zero dependence on multi-gigabyte deep learning models or heavy Python/C++ runtimes.
   - High-throughput execution: > 1,000,000 samples/sec on a single embedded core (> 60x real-time speed).
   - Deterministic memory usage with bounded ring buffers and precomputed filterbanks.

2. **Acoustic Drone Noise Rejection**:
   - Dynamic noise floor tracking and rotor Blade Pass Frequency (BPF) acoustic suppression.
   - Live telemetry integration with Kestrel/Chronos motor RPM data to dynamically notch out propeller tonal frequencies.
   - Multi-microphone delay-and-sum and MVDR beamforming for spatial speaker isolation.

3. **Few-Shot Adaptive Voice Control**:
   - 1-2 exemplar enrollment for immediate offline keyword recognition (e.g., "take off", "land", "hold position", "emergency stop").
   - Dynamic Time Warping (DTW) with Sakoe-Chiba band pruning and normalized confidence scoring.

4. **Universal Embedded & Cross-Language Deployment**:
   - Pure safe Rust (`#![deny(unsafe_code)]`) with optional `#![no_std]` support for Cortex-M microcontrollers.
   - C-ABI dynamic library (`capi.rs`) and zero-copy shared memory integration for companion computers (NVIDIA Jetson, Raspberry Pi).

---

## Current

- [ ] **Phase 2: Dynamic Time Warping (DTW) Sakoe-Chiba Band Pruning, Multi-Exemplar Template Averaging & Adaptive Threshold Tuning - [P1]**
  - [ ] Implement Sakoe-Chiba global path constraint band with configurable radius $R$, reducing DTW computational complexity from $O(N \cdot M)$ to $O(N \cdot R)$ and accelerating long-keyword matching.
  - [ ] Implement multi-exemplar template clustering and soft DTW barycenter averaging (DBA) to fuse 3-5 voice recordings into a robust reference template.
  - [ ] Implement adaptive distance threshold calibration based on exemplar self-similarity and ambient SNR.
  - [ ] Author automated verification test suite verifying Sakoe-Chiba constraint bounds and benchmarking throughput speedup.

---

## Future

- [ ] **Phase 3: Telemetry-Informed Rotor Blade Pass Frequency (BPF) Harmonic Notch Filter Bank - [P1]**
  - [ ] Acoustic drone noise modeling: calculate rotor fundamental and harmonic frequencies $f_k = k \cdot \frac{N_{\text{blades}} \cdot \text{RPM}}{60}$.
  - [ ] Real-time IIR biquad notch filter bank with dynamic center-frequency shifting synchronized to ESC/motor RPM telemetry.
  - [ ] Spectral subtraction noise reduction module with running noise spectrum estimation during non-speech intervals.
  - [ ] Author acoustic suppression verification test with synthetic multi-rotor noise injection.

- [ ] **Phase 4: Multi-Microphone Delay-and-Sum Spatial Beamforming & Direction of Arrival (DoA) Estimation - [P2]**
  - [ ] Geometry abstraction for linear, circular, and tetrahedral microphone arrays.
  - [ ] Generalized Cross-Correlation with Phase Transform (GCC-PHAT) for sub-millisecond acoustic Direction of Arrival (DoA) triangulation.
  - [ ] Real-time delay-and-sum spatial beamformer steering listening lobes toward detected speaker azimuth/elevation while attenuating ambient drone noise.
  - [ ] Benchmark multi-channel ingestion throughput.

- [ ] **Phase 5: Drone Acoustic Health Monitoring & Propeller Anomaly Diagnostics - [P2]**
  - [ ] Blade damage and imbalance acoustic signature detection (asymmetric spectral peak emergence).
  - [ ] Motor bearing wear and high-frequency friction acoustic monitoring.
  - [ ] Autonomous health state emitter generating MAVLink `NAMED_VALUE_FLOAT` / telemetry events for Kestrel autopilot.

- [ ] **Phase 6: C-ABI FFI Layer, Shared Memory Audio Ingestion & C/C++/Python Bindings - [P3]**
  - [ ] C-compatible FFI interface (`sonon_create`, `sonon_ingest`, `sonon_enroll`, `sonon_destroy`) in `capi.rs`.
  - [ ] Zero-copy ring buffer reader over POSIX shared memory (`/dev/shm/sonon_audio`).
  - [ ] Lightweight Python wrapper (`sonon-py`) with NumPy array zero-copy passing.
  - [ ] C++20 header-only wrapper with RAII handle.

- [ ] **Phase 7: `#![no_std]` Embedded HAL Support & Microcontroller Portability - [P3]**
  - [ ] Feature flag `no_std` using `alloc` or fixed-capacity `heapless` buffers.
  - [ ] Verification on ARM Cortex-M7 (STM32H7) and ESP32-S3 targets.
  - [ ] Fixed-point Q15/Q31 DSP math mode for microcontrollers without double-precision FPUs.

---

## Done

- [x] **Phase 1: Sonon Core Acoustic Signal Processing Pipeline & Few-Shot Keyword Spotting Engine Baseline**
  - [x] **Contiguous Audio Ring Buffer (`src/ring_buffer.rs`)**: High-throughput circular sample buffer with bounded capacity and overwrite-oldest semantics for uninterrupted streaming ingestion.
  - [x] **Precomputed Windowing Suite (`src/window.rs`)**: In-place windowing coefficient application supporting Rectangular, Hann, Hamming, and Blackman functions.
  - [x] **Radix-2 Cooley-Tukey FFT Processor (`src/stft.rs`)**: Real-to-complex in-place Fast Fourier Transform with bit-reversal permutations and one-sided power spectral density calculation.
  - [x] **Triangular Mel Filterbank & MFCC Extraction (`src/mel.rs`)**: 26-band Mel-scale filterbank with sparse bin representation, logarithmic energy compression, and DCT-II cepstral extraction (13 coefficients).
  - [x] **Adaptive Energy Voice Activity Detector (`src/vad.rs`)**: Dynamic noise-floor tracking during silence intervals, RMS energy thresholding, and configurable hangover smoothing.
  - [x] **Few-Shot Phrase Spotting via Dynamic Time Warping (`src/dtw.rs`)**: Symmetric dynamic programming sequence alignment with Euclidean distance metric and path-length normalized scoring.
  - [x] **Unified Streaming Coordinator (`src/engine.rs`)**: `SononEngine` orchestrating streaming ingestion, sliding feature history buffers, real-time phrase matching, and confidence score dispatch.
  - [x] **Automated Verification Suite (`tests/sonon_dsp_tests.rs`)**: 7 comprehensive unit tests (7/7 PASS) verifying window symmetry, ring buffer ordering, FFT 1 kHz peak detection, Mel filterbanks, VAD speech bursts, DTW phrase recognition, and 1M+ samples/sec throughput benchmark (> 60x real-time speed).
