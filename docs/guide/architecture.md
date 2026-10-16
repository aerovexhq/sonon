# System Architecture &amp; Execution Pipeline

<p class="doc-lead">
Sonon operates as an end-to-end, multi-stage acoustic processing kernel. Raw multi-channel PCM streams pass through telemetry-coupled IIR notch filters, spatial array beamforming, temporal voice activity detection, and banded dynamic time warping before routing to autopilot mission controllers.
</p>

<div class="key-features-box">
  <div class="key-features-header">Architecture Tenets</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero Dynamic Allocations (bounded scratchbuffers)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Deterministic Bounded Execution (&lt;5 us per filter block)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> POSIX Lock-Free Shared Memory Ring Buffers</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> SIMD AVX2 / ARM NEON Vectorized FFT Pipelines</div>
  </div>
</div>

## Detailed System Dataflow Pipeline

```
[ Raw Audio Source ] ── (8-Ch MEMS Array / POSIX SHM / I2S DMA)
       │
       ▼
[ Lock-Free Circular Ring Buffer ] ── Seqlock memory bus (>15M samples/sec)
       │
       ▼
[ ESC Telemetry Synchronizer ] ── MAVLink / DShot 600 motor RPM feed
       │
       ▼
[ Dynamic Harmonic Notch Filter Bank ] ── Biquad Direct Form II Transposed (>86 dB notch)
       │
       ▼
[ Spatial STFT Beamformer ] ── GCC-PHAT TDOA & Adaptive MVDR directional gain (1.5 deg DOA)
       │
       ▼
[ Streaming VAD Gate ] ── Sub-band energy ratio & spectral centroid gate (12 us decision)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
[ Few-Shot DTW Spotter ]   [ CWT Wavelet Profiler ]  [ Formant Vocal Synthesizer ]
  Sakoe-Chiba Corridor       Continuous Morlet         Klatt + LF Differential Flow
  (0.42 ms KWS latency)      Scale-Wise Kurtosis       (60x faster than real-time)
```

## Memory Layout &amp; Buffer Bounds

All data structures in Sonon are allocated during the initialization phase. The memory budget is strictly partitioned:

| Subsystem Component | Static Allocation | Cache Footprint | Memory Alignment | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Audio Ring Buffer** | 65,536 Samples | 256 KB | 64-byte aligned | Lock-free Seqlock circular stream storage |
| **FFT Scratch Space** | 512 Complex f32 | 4 KB | 32-byte aligned | SIMD Radix-2 butterfly calculation scratch |
| **Mel Filterbank Tables** | 40 x 257 f32 | 41.1 KB | Cache-line aligned | Pre-computed triangular frequency weights |
| **DTW Warping Corridor** | 200 x 21 f32 | 16.8 KB | 16-byte aligned | Sakoe-Chiba corridor matrix (R=10) |
| **Vocal Tract Waveguide** | 8 Resonator States | 1.8 KB | L1-D cache resident | Glottal flow delay line and digital waveguide |
| **Total Engine State** | 1,180 KB | ~1.2 MB | 64-byte aligned | Bound guaranteed across all targets |

::: info ARCHITECTURE DIRECTIVE: MEMORY BOUNDEDNESS
Sonon never calls `malloc`, `realloc`, or Rust's `Vec::push` during the `process_frame` call loop. If an audio buffer exceeds the pre-allocated ring capacity, samples overwrite the oldest unread frames with atomic sequence counter increments, preventing memory fragmentation or unbounded heap growth on mission flight hardware.
:::

## SIMD Acceleration &amp; Hardware Targets

Sonon implements compile-time target detection and leverages platform SIMD vector extensions:
- **x86_64**: AVX2 and FMA3 intrinsics for parallel 8-lane biquad filter step evaluation.
- **AArch64 / ARMv7E-M**: ARM NEON and Helium intrinsics for 4-lane single-precision float operations.
- **WebAssembly**: WASM 128-bit SIMD (`simd128`) instruction set for 60 FPS in-browser synthesis.

```rust
// Compile-time SIMD dispatch in sonon_dsp
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use self::avx2_biquad::process_biquad_cascade_simd;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use self::neon_biquad::process_biquad_cascade_simd;

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
pub use self::fallback_biquad::process_biquad_cascade_scalar;
```

## Next Steps

- [Quickstart &amp; Installation](/guide/quickstart)
- [30 Research Monograph Phases](/dsp/overview)
- [Rotor Notch Filter Physics](/dsp/rotor-notch-filter)
