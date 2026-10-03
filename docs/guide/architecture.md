# Sonon System Architecture

Sonon operates as an autonomous, multi-stage acoustic processing pipeline:

```
[ Audio Source ] (Microphone / POSIX SHM / WAV)
       │
       ▼
[ Ring Buffer ] ── Lock-free circular audio buffer (AudioRingBuffer)
       │
       ▼
[ Pre-Emphasis & Windowing ] ── Hann / Hamming / Blackman (Window)
       │
       ▼
[ Telemetry Notch Filter ] ── ESC RPM-locked Biquad BPF Notch Bank
       │
       ▼
[ Spectral Decomposition ] ── Radix-2 Cooley-Tukey FFT & Mel Filterbank
       │
       ▼
[ Per-Channel Normalization ] ── PCEN (Adaptive gain & compression)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
[ DTW Wake-Word Spotter ]  [ CWT Wavelet Profiler ]  [ Formant Synthesizer ]
  Sakoe-Chiba Corridor       Bearing / Rotor Wear      Klatt + LF Resonators
```

## Guarantees

- **Pure Safe Rust**: Zero `unsafe` blocks. Verified with `#![deny(unsafe_code)]`.
- **Zero Allocations**: Ring buffers and scratch matrices are bounded and reused.
- **Real-Time Execution**: Processes $> 1,000,000$ audio samples/sec on a single core ($> 60\times$ real-time).
