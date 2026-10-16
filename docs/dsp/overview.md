# Acoustic DSP &amp; Physics Overview

<p class="doc-lead">
Sonon's acoustic digital signal processing architecture spans 30 exhaustive research monographs detailing classical acoustic physics, bio-inspired neuromorphic cochlea dynamics, flight vehicle rotor aeromechanics, and physical articulatory speech synthesis.
</p>

<div class="key-features-box">
  <div class="key-features-header">Theoretical Foundations &amp; Research Monographs</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Dynamic Formant &amp; Acoustic Filterbanks (Phases 01-04)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Banded Dynamic Time Warping &amp; Barycenters (Phases 05-07)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Propeller Aeroacoustic Notch Cascades (Phases 08-10)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> GCC-PHAT Spatial MVDR Beamforming (Phases 11-13)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Physical Liljencrants-Fant Vocal Tract (Phases 14-16)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Morlet Wavelet Structural Health (Phases 17-18)</div>
  </div>
</div>

## Complete 30-Phase Research Topology

The engine is engineered across 30 distinct mathematical phases:

| Phase Range | Subsystem Focus | Mathematical Foundation | Measured Engineering Performance |
| :--- | :--- | :--- | :--- |
| **Phases 01-04** | Feature Extraction &amp; PCEN | STFT, Hann/Blackman, Mel DCT-II, PCEN | 10 ms hop, 512-pt FFT in 18 us |
| **Phases 05-07** | Few-Shot DTW Spotter | Sakoe-Chiba corridor ($R=10$), DBA | 0.42 ms spotting, 60 KB RAM |
| **Phases 08-10** | Rotor Notch Filter Bank | Transposed Direct Form II Biquads | &gt;86.4 dB attenuation across 8 harmonics |
| **Phases 11-13** | Spatial Array Beamforming | GCC-PHAT TDOA, Adaptive MVDR | 1.5 degree DOA resolution, 8 channels |
| **Phases 14-16** | Articulatory Vocal Synthesis | Liljencrants-Fant glottis + Formants | 64x faster than real-time, 1.2 MB RAM |
| **Phases 17-18** | CWT Health Diagnostics | Complex Morlet wavelet scalograms | Scale-wise kurtosis &gt;4.5 bearing alert |
| **Phases 19-21** | Neuromorphic Silicon Cochlea | Gammatone filterbanks + LIF spikes | Spike-timing dependent phrase spotting |
| **Phases 22-25** | Aeroacoustics &amp; Biosonar | Corcos TBL noise, FW-H, CA-CFAR | 5-axis obstacle hazard detection |
| **Phases 26-30** | Autopilot Integration &amp; FFI | POSIX SHM Seqlock, C-ABI, WASM | &gt;15M samples/sec zero-copy throughput |

::: info ARCHITECTURE DIRECTIVE: NUMERICAL STABILITY
All biquad notch filters utilize Transposed Direct Form II topologies to prevent catastrophic pole-zero cancellation and limit cycle oscillations under 32-bit single-precision floating point arithmetic.
:::

## Key Subsystems In Depth

Explore detailed monographs and mathematical derivations for each primary subsystem:

- [Feature Extraction &amp; Mel Filterbanks](/dsp/feature-extraction)
- [Dynamic Rotor Harmonic Notch Bank](/dsp/rotor-notch-filter)
- [Sakoe-Chiba Banded DTW Spotting](/dsp/wake-word-spotting)
- [Physical Formant Speech Synthesis](/dsp/speech-synthesis)
- [Continuous Wavelet Diagnostics (CWT)](/dsp/wavelet-diagnostics)
- [Acoustic Echolocation &amp; 3D Mapping](/dsp/echolocation)
