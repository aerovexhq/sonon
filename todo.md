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

- [ ] **Phase 3: Telemetry-Informed Rotor Blade Pass Frequency (BPF) Harmonic Notch Filter Bank - [P1]**
  - [ ] Acoustic drone noise modeling: calculate rotor fundamental and harmonic frequencies $f_k = k \cdot \frac{N_{\text{blades}} \cdot \text{RPM}}{60}$.
  - [ ] Real-time IIR biquad notch filter bank with dynamic center-frequency shifting synchronized to ESC/motor RPM telemetry.
  - [ ] Spectral subtraction noise reduction module with running noise spectrum estimation during non-speech intervals.
  - [ ] Author acoustic suppression verification test with synthetic multi-rotor noise injection.

---

## Future

- [ ] **Phase 4: Multi-Microphone Delay-and-Sum Spatial Beamforming & Direction of Arrival (DoA) Estimation - [P2]**
  - [ ] Geometry abstraction for linear, circular, and tetrahedral microphone arrays.
  - [ ] Generalized Cross-Correlation with Phase Transform (GCC-PHAT) for sub-millisecond acoustic Direction of Arrival (DoA) triangulation.
  - [ ] Real-time delay-and-sum spatial beamformer steering listening lobes toward detected speaker azimuth/elevation while attenuating ambient drone noise.
  - [ ] Benchmark multi-channel ingestion throughput.

- [ ] **Phase 5: AeroSSM Next-Gen Structured State-Space Duality Engine & Anticipatory Prefix Decoding - [P2]**
  - [ ] Selective State-Space recurrence kernel (`AeroSSM`) with $O(1)$ streaming state memory in `#![no_std]` Rust.
  - [ ] SincNet physical convolutional layer dynamically modulated by Kestrel ESC motor RPM telemetry.
  - [ ] Anticipatory Prefix-CTC decoder with Wald's SPRT stopping boundary triggering flight actions at $70\%$ phrase completion.
  - [ ] Two-phase speculative flight actuator interlock (`PreArm` and `Commit` MAVLink triggers).

- [ ] **Phase 6: Drone Acoustic Health Monitoring & Propeller Anomaly Diagnostics - [P3]**
  - [ ] Blade damage and imbalance acoustic signature detection (asymmetric spectral peak emergence).
  - [ ] Motor bearing wear and high-frequency friction acoustic monitoring.
  - [ ] Autonomous health state emitter generating MAVLink `NAMED_VALUE_FLOAT` / telemetry events for Kestrel autopilot.

- [ ] **Phase 7: C-ABI FFI Layer, Shared Memory Audio Ingestion & C/C++/Python Bindings - [P3]**
  - [ ] C-compatible FFI interface (`sonon_create`, `sonon_ingest`, `sonon_enroll`, `sonon_destroy`) in `capi.rs`.
  - [ ] Zero-copy ring buffer reader over POSIX shared memory (`/dev/shm/sonon_audio`).
  - [ ] Lightweight Python wrapper (`sonon-py`) with NumPy array zero-copy passing.
  - [ ] C++20 header-only wrapper with RAII handle.

- [ ] **Phase 8: `#![no_std]` Embedded HAL Support & Microcontroller Portability - [P3]**
  - [ ] Feature flag `no_std` using `alloc` or fixed-capacity `heapless` buffers.
  - [ ] Verification on ARM Cortex-M7 (STM32H7) and ESP32-S3 targets.
  - [ ] Fixed-point Q15/Q31 DSP math mode for microcontrollers without double-precision FPUs.

---

## Done

- [x] **Phase 2.5: Empirical Human Voice Dataset Ingestion, Phonetic Dissection & Real-Speech Automated Testing - [P1]**
  - [x] Ingested operator speech recordings (`me_saying_plank` and `randomrecordingmesayingthings_and_plank`).
  - [x] Extracted, trimmed, and segmented 15 isolated citation exemplars into `tests/fixtures/plank_exemplars/*.wav`.
  - [x] Phonetically dissected $[p] + [l] + [æ] + [ŋ] + [k]$: formants ($F_1 \approx 606\text{ Hz}, F_2 \approx 1480\text{ Hz}$), pitch octave spread ($130.2\text{ to }295.2\text{ Hz}$), and duration elasticity ($210\text{ to }540\text{ ms}$).
  - [x] Discovered mathematical acoustic medoid (Exemplar #02, mean DTW distance 1.779).
  - [x] Pinpointed and verified all 9 operator-reported timestamps in continuous speech stream to within $50\text{--}200\text{ ms}$ physical precision.
  - [x] Authored automated integration test suite (`tests/sonon_human_voice_tests.rs`, 5/5 PASS) sustaining $> 10,000,000\text{ samples/sec}$ ($> 640\times$ real-time speed) on natural human speech.
  - [x] Authored Monograph 10: [`analysis/10_empirical_voice_analysis_and_dataset_validation.md`](file:///root/Projects/aerovex/modules/sonon/analysis/10_empirical_voice_analysis_and_dataset_validation.md).

- [x] **Phase 2: Dynamic Time Warping (DTW) Sakoe-Chiba Band Pruning, Multi-Exemplar DBA Template Averaging & Per-Channel Energy Normalization (PCEN) - [P1]**
  - [x] Implement Sakoe-Chiba global path constraint band with configurable radius $R$, reducing DTW computational complexity from $O(N \cdot M)$ to $O(N \cdot R)$ and accelerating long-keyword matching.
  - [x] Implement multi-exemplar template clustering and soft DTW barycenter averaging (DBA) to fuse 3-5 voice recordings into a robust reference template.
  - [x] Implement Per-Channel Energy Normalization (`PcenFilter`) with adaptive recursive noise smoothing ($s \approx 0.025$), exponent compression ($r \approx 0.25$), and feed-forward gain control for drone noise immunity.
  - [x] Implement adaptive distance threshold calibration (`calibrate_threshold`) based on exemplar self-similarity and ambient SNR.
  - [x] Author automated verification test suite (`tests/sonon_phase2_tests.rs`) verifying Sakoe-Chiba constraint bounds, PCEN dynamic compression, DBA convergence, 70% partial phrase anticipatory early detection, and > 1,000,000 samples/sec throughput speedup (6/6 PASS).

- [x] **Phase 1.5: World-Class Wake-Word & Speech Recognition Deep Research Monograph Suite (`analysis/`)**
  - [x] **Vocal Acoustics & Biomechanics Monograph (`analysis/01_vocal_acoustics_and_biomechanics.md`)**: Formulated subglottal pressure aerodynamics, Ishizaka-Flanagan two-mass vocal fold model, Fant source-filter formulation, $-6\text{ dB/octave}$ radiated speech tilt, quarter-wave acoustic tube resonators, formant perturbation theory, and complete phoneme taxonomy.
  - [x] **Speaker Variability, Accents & Prosody Monograph (`analysis/02_speaker_variability_accents_and_prosody.md`)**: Formulated vocal tract length scaling laws ($L_{\text{tract}} \approx 17.5\text{ cm} \to 11.0\text{ cm}$), VTLN frequency warping, $F_0$ distributions, jitter/shimmer/HNR, phonation regimes, English regional dialect shifts, 6-language L2 phonological transfer matrix, and drone rotor Lombard effect dynamics (+17 dB SPL gain, $+180\text{ Hz } F_1$ elevation, spectral tilt flattening, vowel stretching).
  - [x] **Acoustic Environmental & Hardware Transduction Monograph (`analysis/03_acoustic_environmental_and_hardware_factors.md`)**: Formulated MEMS/ECM/dynamic transducer physics, polar patterns, proximity effect bass boost ($+20\text{ dB}$ at $5\text{ cm}$), transducer AOP clipping, drone BPF harmonics ($f_{\text{BPF}} = B \cdot \text{RPM} / 60$), Gutin-Deming loading dipoles, inverse-square propagation ($-25\text{ dB}$ SNR at $10\text{ m}$), $25\text{ m/s}$ Doppler shifts ($+7.8\%$), and Sabine reverberation time ($T_{60}$) forward masking.
  - [x] **Compact Embedded KWS Architectures Monograph (`analysis/04_compact_embedded_kws_architectures.md`)**: Formulated microcontroller constraints ($< 50\text{ KB RAM}$, $< 50\text{ MIPS}$), Sakoe-Chiba DTW pruning corridor ($|i - j| \le R$, $79\%$ compute reduction), DTW Barycenter Averaging (DBA), Depthwise Separable CNNs (DS-CNN, $87\%$ FLOP reduction), TC-ResNet-8 dilated 1D temporal convolutions ($420\text{ kMACs/sec}$), streaming unidirectional GRUs, symmetric INT8 integer-only arithmetic, and deterministic Rust memory arena architectures.
  - [x] **Scalable Hierarchical Wake-Word Cascade Monograph (`analysis/05_scalable_hierarchical_kws_cascade.md`)**: Formulated multi-tier cascade (Stage 0 $< 10\text{ }\mu\text{W}$ analog gate $\to$ Stage 1 $10\text{ MIPS}$ streaming micro-KWS $\to$ Stage 2 $100\text{ MIPS}$ streaming Conformer verifier $\to$ Stage 3 companion NPU intent parser), Per-Channel Energy Normalization (PCEN) adaptive gain control, SincNet learnable frontends, multi-condition data augmentation ($-15\text{ dB}$ rotor noise, synthetic Lombard, 5,000 RIR virtual spaces, SpecAugment), DET curve optimization, and sub-$35\text{ ms}$ latency-to-fire.

- [x] **Phase 1: Sonon Core Acoustic Signal Processing Pipeline & Few-Shot Keyword Spotting Engine Baseline**
  - [x] **Contiguous Audio Ring Buffer (`src/ring_buffer.rs`)**: High-throughput circular sample buffer with bounded capacity and overwrite-oldest semantics for uninterrupted streaming ingestion.
  - [x] **Precomputed Windowing Suite (`src/window.rs`)**: In-place windowing coefficient application supporting Rectangular, Hann, Hamming, and Blackman functions.
  - [x] **Radix-2 Cooley-Tukey FFT Processor (`src/stft.rs`)**: Real-to-complex in-place Fast Fourier Transform with bit-reversal permutations and one-sided power spectral density calculation.
  - [x] **Triangular Mel Filterbank & MFCC Extraction (`src/mel.rs`)**: 26-band Mel-scale filterbank with sparse bin representation, logarithmic energy compression, and DCT-II cepstral extraction (13 coefficients).
  - [x] **Adaptive Energy Voice Activity Detector (`src/vad.rs`)**: Dynamic noise-floor tracking during silence intervals, RMS energy thresholding, and configurable hangover smoothing.
  - [x] **Few-Shot Phrase Spotting via Dynamic Time Warping (`src/dtw.rs`)**: Symmetric dynamic programming sequence alignment with Euclidean distance metric and path-length normalized scoring.
  - [x] **Unified Streaming Coordinator (`src/engine.rs`)**: `SononEngine` orchestrating streaming ingestion, sliding feature history buffers, real-time phrase matching, and confidence score dispatch.
  - [x] **Automated Verification Suite (`tests/sonon_dsp_tests.rs`)**: 7 comprehensive unit tests (7/7 PASS) verifying window symmetry, ring buffer ordering, FFT 1 kHz peak detection, Mel filterbanks, VAD speech bursts, DTW phrase recognition, and 1M+ samples/sec throughput benchmark (> 60x real-time speed).
