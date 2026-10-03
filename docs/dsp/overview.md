# Acoustic DSP & Physics Overview

Sonon's acoustic DSP kernel spans 30 research monographs detailing classical acoustic physics, bio-inspired cochlea mechanics, and flight vehicle noise rejection:

| Phase | Subsystem | Mathematical Foundation | Target Capability |
| :--- | :--- | :--- | :--- |
| **01-04** | Feature Extraction | STFT, Hann/Blackman, Mel DCT-II | Real-time MFCC feature extraction |
| **05-07** | Few-Shot DTW | Sakoe-Chiba corridor, DBA barycenter | 1-2 exemplar voice enrollment |
| **08-10** | Rotor Notch Bank | Biquad Direct Form II Transposed | Dynamic BPF motor whine suppression |
| **11-13** | Spatial Beamforming | GCC-PHAT TDoA, Delay-and-Sum | Direction of Arrival (DoA) steering |
| **14-16** | Adaptive AEC & Speech | NLMS FIR + Geigel DTD, Klatt + LF | Double-talk cancel & formant synthesis |
| **17-18** | CWT Health Diagnostics | Complex Morlet wavelet scalograms | Rotor blade fatigue & bearing wear |
| **19-21** | Neuromorphic Silicon Cochlea | Gammatone filterbanks + LIF spikes | Spike-timing dependent phrase spotting |
| **22-25** | Aeroacoustics & Biosonar | Corcos TBL noise, FW-H, CA-CFAR | Airspeed inversion & 3D echolocation |

## Explore Further

- [Rotor Notch Bank & BPF Telemetry](/dsp/rotor-notch-filter)
- [Banded Sakoe-Chiba DTW](/dsp/wake-word-spotting)
- [Klatt + LF Speech Synthesis](/dsp/speech-synthesis)
- [Continuous Wavelet Diagnostics (CWT)](/dsp/wavelet-diagnostics)
- [Interactive Audio Lab](/playground)
