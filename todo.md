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

*(All 13 Roadmap Milestones Completed & Fully Verified across DSP, AI, Acoustics, FFI, Multi-Language, Embedded, Neuromorphic, PyPI, NPM, and Live Microphone Capture Targets - 56/56 Rust Tests + Python + JS PASS)*

---

## Future

- [ ] **Phase 14 (Future Hardware Integration): SynSense Xylo / BrainChip Akida Hardware-in-the-Loop AER Driver - [P3]**
  - [ ] SPI/I2C streaming driver binding `SpikeEvent` directly to physical neuromorphic accelerator silicon.
  - [ ] Sub-10 microwatt continuous edge listening verification on benchtop power analyzer.

---

## Done

- [x] **Phase 13: Live Microphone Enrollment, Auditory Playback Feedback & Gain-Invariant Acoustic Spotting - [P1]**
  - [x] Implemented energy-based VAD silence trimming (`SononEngine.trim_silence` / `SononEngine.trimSilence`) isolating active speech utterances from dead background silence.
  - [x] Implemented immediate acoustic playback confirmation via `aplay` (Linux) / `sounddevice` (cross-platform) / `afplay` (macOS), playing back enrolled wake-words to the user before listening starts.
  - [x] Engineered gain-invariant spectral feature normalization (Cepstral Mean Subtraction and L2 unit-norm) in Python and JS/Node.js engines, ensuring acoustic distance is invariant to mic gain and RMS levels.
  - [x] Fixed DTW Euclidean distance metric (`math.sqrt(...)` / `Math.sqrt(...)`) and multi-rate candidate window matching ($L \pm 2$).
  - [x] Added dynamic terminal visual telemetry displaying live microphone RMS levels and DTW distance meters in real-time.
  - [x] Rebuilt and verified redistributable Python wheel (`dist/sonon-0.1.0-py3-none-any.whl`), sdist, and NPM package (`dist/aerovexhq-sonon-0.1.0.tgz`).

- [x] **Phase 12: Empirical Train/Test Cross-Validation Accuracy Benchmark & Evaluation Report - [P1]**
  - [x] Evaluated 1-shot medoid enrollment and 3-shot DBA centroid on 14 unseen citation test exemplars and 40.8s continuous distractor stream.
  - [x] Documented exact empirical metrics: 100% recall on clean citation speech, 90.0% precision on continuous natural speech stream (9/9 targets spotted, 1 distractor false positive), 100% precision under Wald's SPRT anticipatory prefix interlock.
  - [x] Authored Monograph 12: [`analysis/12_empirical_accuracy_and_train_test_generalization.md`](file:///root/Projects/aerovex/modules/sonon/analysis/12_empirical_accuracy_and_train_test_generalization.md).

- [x] **Phase 11: Production NPM Package for JavaScript & Node.js (v0.1.0) - [P1]**
  - [x] Manifest (`package.json`) and TypeScript definitions (`index.d.ts`) for `@aerovexhq/sonon` v0.1.0.
  - [x] Pure zero-dependency Node.js engine and POSIX SHM (`/dev/shm/sonon_audio`) streaming client (`index.js`).
  - [x] Built redistributable npm tarball (`dist/aerovexhq-sonon-0.1.0.tgz`, 3.9 KB) and verified clean `npm install` and execution.

- [x] **Phase 10: Production Python Packaging & PyPI Wheel Distribution (v0.1.0) - [P1]**
  - [x] Modern Python package structure in `bindings/python/` with `pyproject.toml`, `setup.py`, and `sonon/__init__.py`.
  - [x] Packaged native safe Rust dynamic library (`libsonon.so`) into wheel distribution.
  - [x] Built binary wheel `dist/sonon-0.1.0-py3-none-any.whl` (181 KB) and source tarball `dist/sonon-0.1.0.tar.gz` (182 KB).
  - [x] Verified clean global `pip install` and runtime execution of `SononEngine` and `SononShm`.

- [x] **Phase 9: Neuromorphic Event-Based Silicon Cochlea & Spike-Driven Wake-Word Spotting - [P3]**
  - [x] Mammalian basilar membrane biomechanics model with Greenwood place-frequency mapping across 16/32 critical channels (`src/neuromorphic.rs`).
  - [x] Discrete 4th-order cascade complex IIR Gammatone filterbanks (`GammatoneFilter`) with inner hair cell half-wave mechanical rectification and logarithmic compression.
  - [x] Asynchronous Address-Event Representation (AER) delta contrast modulator (`NeuromorphicCochlea`) producing standard 8-byte `SpikeEvent` wire format and zero spikes during silence.
  - [x] Event-driven Leaky Integrate-and-Fire (LIF) Spiking Neural Network wake-word decoder (`SpikingKwsCell`) with exponential synaptic traces, receptive field lateral inhibition, and post-detection refractory lockout.
  - [x] Authored Phase 9 verification test suite (`tests/sonon_phase9_tests.rs`, 8/8 PASS) sustaining $> 25,000,000\text{ spikes/sec}$ throughput.

- [x] **Phase 8: `#![no_std]` Embedded HAL Support & Microcontroller Portability - [P3]**
  - [x] Fixed-point Q15 and Q31 DSP math mode (`src/fixed.rs`) with saturating arithmetic, widening dot products, and quarter-wave sine/cosine look-up tables with zero FPU instructions.
  - [x] Static heapless circular audio buffer (`StaticAudioBuffer<const N: usize>`) operating with deterministic memory and zero heap allocations.
  - [x] Pure integer Sakoe-Chiba Dynamic Time Warping matcher (`FixedDtwMatcher`) operating with Manhattan distance on Q15 feature frames.
  - [x] Authored Phase 8 verification test suite (`tests/sonon_phase8_tests.rs`, 7/7 PASS) sustaining $> 18,000,000\text{ samples/sec}$ throughput (> 1100x real-time speed).

- [x] **Phase 7: C-ABI FFI Layer, Shared Memory Audio Ingestion & C/C++/Python Bindings - [P3]**
  - [x] C-compatible FFI interface (`sonon_engine_create`, `sonon_engine_destroy`, `sonon_shm_pump`, etc.) in `src/capi.rs` with safe handle registries in pure safe Rust (`#![deny(unsafe_code)]`).
  - [x] Zero-copy POSIX shared memory ring buffer reader and writer (`/dev/shm/sonon_audio`) in `src/shm.rs` sustaining $> 4,000,000\text{ samples/sec}$ in debug and $> 15,700,000\text{ samples/sec}$ in release mode (> 980x real-time speed).
  - [x] Lightweight Python wrapper (`bindings/python/sonon.py`) with NumPy float32 zero-copy passing and automated unit testing (`bindings/python/test_sonon_py.py`).
  - [x] Modern C++20 header-only wrapper with RAII handle (`bindings/cpp/include/sonon.hpp`) and automated test (`bindings/cpp/test_sonon_cpp.cpp`).
  - [x] Authored Phase 7 verification test suite (`tests/sonon_phase7_tests.rs`, 6/6 PASS).

- [x] **Phase 6: Drone Acoustic Health Monitoring & Propeller Anomaly Diagnostics - [P3]**
  - [x] Blade damage and imbalance acoustic signature detection via rotational subharmonic energy emergence (`src/health.rs`).
  - [x] Motor bearing wear and high-frequency friction acoustic monitoring via spectral kurtosis and ISO 10816 high-frequency ratios.
  - [x] Autonomous airframe health evaluator and standard MAVLink `NAMED_VALUE_FLOAT` telemetry packet emitter (`SONON_HLTH`, `SONON_STAT`, `SON_IMB*`, `SON_BRG*`).
  - [x] Multi-motor quadcopter differential acoustic diagnostic identification isolating damaged blades on individual rotors.
  - [x] Integrated acoustic health inspection into streaming `SononEngine` pipeline.
  - [x] Authored Phase 6 verification test suite (`tests/sonon_phase6_tests.rs`, 6/6 PASS) sustaining $> 4,380,000\text{ samples/sec}$ throughput (> 270x real-time speed).

- [x] **Phase 5: AeroSSM Next-Gen Structured State-Space Duality Engine & Anticipatory Prefix Decoding - [P2]**
  - [x] Selective State-Space recurrence kernel (`AeroSSM`) with $O(1)$ streaming state memory in `#![no_std]` Rust (`src/aerossm.rs`).
  - [x] SincNet physical convolutional layer dynamically modulated by Kestrel ESC motor RPM telemetry (`SincConvFrontend`).
  - [x] Anticipatory Prefix decoder with Wald's SPRT stopping boundary triggering flight actions at $70\%$ phrase completion (`src/anticipatory.rs`).
  - [x] Two-phase speculative flight actuator interlock (`Listening`, `PreArm`, `Commit`, `Rollback`).
  - [x] Authored Phase 5 verification test suite (`tests/sonon_phase5_tests.rs`, 5/5 PASS) achieving $> 1,150,000\text{ samples/sec}$ in unoptimized debug and $> 8,280,000\text{ samples/sec}$ in release mode (> 500x real-time speed).

- [x] **Phase 4: Multi-Microphone Delay-and-Sum Spatial Beamforming & Direction of Arrival (DoA) Estimation - [P2]**
  - [x] Geometry abstraction for linear (lateral wing-mount), circular (360-degree), and tetrahedral (3D) microphone arrays in `src/beamforming.rs`.
  - [x] Generalized Cross-Correlation with Phase Transform (`GccPhatEstimator`) with sub-millisecond parabolic interpolation.
  - [x] Direction of Arrival (`DoAEstimator`) with Steered Response Power (SRP-PHAT) 360-degree azimuth triangulation.
  - [x] Real-time spatial Delay-and-Sum beamformer (`DelayAndSumBeamformer`) with power-of-2 bitmask circular buffers delivering $+5.59\text{ dB}$ spatial noise rejection.
  - [x] Authored Phase 4 verification test suite (`tests/sonon_phase4_tests.rs`, 6/6 PASS) sustaining $> 2,200,000\text{ samples/sec}$ throughput (> 140x real-time speed).

- [x] **Phase 3: Telemetry-Informed Rotor Blade Pass Frequency (BPF) Harmonic Notch Filter Bank & Spectral Subtraction - [P1]**
  - [x] Acoustic drone noise modeling: calculated rotor fundamental and harmonic frequencies $f_k = k \cdot \frac{N_{\text{blades}} \cdot \text{RPM}}{60}$.
  - [x] Implemented Direct Form II Transposed IIR biquad notch filter (`BiquadNotchFilter`) in `src/notch.rs` delivering $> 86\text{ dB}$ attenuation depth.
  - [x] Implemented real-time dynamic rotor notch filter bank (`RotorHarmonicNotchBank`) tracking autopilot/ESC telemetry with instantaneous retuning and multi-motor RPM averaging.
  - [x] Implemented recursive spectral subtraction noise suppression module (`SpectralSubtractionSuppressor`) in `src/spectral_subtraction.rs` with running noise floor adaptation during VAD silence.
  - [x] Integrated notch bank and spectral subtraction directly into `SononEngine` streaming ingestion and feature extraction pipelines.
  - [x] Authored Phase 3 verification suite (`tests/sonon_phase3_tests.rs`, 6/6 PASS) verifying $86\text{ dB}$ notch attenuation, dynamic RPM frequency shifts, multi-motor averaging, wake-word spotting under $+15\text{ dB}$ rotor whine, and $> 3,500,000\text{ samples/sec}$ throughput (> 220x real-time speed).

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
