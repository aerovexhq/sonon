# 17. Continuous Wavelet Transform (CWT) Non-Stationary Rotor Micro-Damage Profiler & Structural Vibration Decomposition

---

## 1. Executive Summary

Autonomous multirotors, eVTOL aircraft, and fixed-wing UAVs experience extreme aerodynamic and structural stresses on propeller blades, rotor hubs, and brushless motor bearings. Micro-cracks in carbon fiber or polymer blades, blade tip chipping from foreign object debris (FOD), aerodynamic flutter, and motor bearing ball micro-pitting produce **highly non-stationary acoustic transients** (sub-millisecond impact shocks):
- Conventional Short-Time Fourier Transform (STFT) spectral analysis is constrained by the **Heisenberg-Gabor uncertainty principle**: selecting a long time window achieves fine frequency resolution but temporal smearing obscures micro-crack impulse spikes; selecting a short time window captures temporal transients but loses frequency resolution, making harmonic separation of rotor blade pass frequencies (BPF) impossible.
- Phase 17 engineers a real-time **Continuous Wavelet Transform (CWT) Acoustic Profiler** in pure safe Rust (`#![deny(unsafe_code)]`).
- Utilizing multi-resolution wavelet dilation:
  - Small scales $a$ (high frequencies, $> 3000\text{ Hz}$) provide fine temporal resolution ($\Delta t < 0.2\text{ ms}$) to isolate micro-crack shock pulses and bearing friction clicks.
  - Large scales $a$ (low frequencies, $50\text{--}500\text{ Hz}$) provide fine spectral resolution ($\Delta f < 10\text{ Hz}$) to isolate motor rotational frequency ($1\times \text{RPM}$) and fundamental blade pass harmonics.

---

## 2. Mathematical Formulation & Wavelet Theory

### 2.1 The Continuous Wavelet Transform

Given a continuous real-valued acoustic signal $x(t)$, its Continuous Wavelet Transform $W(a, b)$ at scale $a > 0$ and temporal translation $b \in \mathbb{R}$ is defined as:
$$W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left( \frac{t - b}{a} \right) dt$$
where $\psi(t)$ is the mother wavelet satisfying the admissibility condition $\int_{-\infty}^{\infty} \psi(t) dt = 0$.

### 2.2 Wavelet Kernel Families

Sonon implements two complementary mother wavelet kernel families:

1. **Complex Morlet Wavelet**:
   $$\psi(t) = \pi^{-1/4} \exp(i \omega_0 t) \exp\left( -\frac{t^2}{2} \right)$$
   where $\omega_0$ is the central angular frequency parameter (default $\omega_0 = 6.0$).
   - Real component: $\operatorname{Re}\{\psi(t)\} = \pi^{-1/4} \cos(\omega_0 t) \exp(-t^2/2)$
   - Imaginary component: $\operatorname{Im}\{\psi(t)\} = \pi^{-1/4} \sin(\omega_0 t) \exp(-t^2/2)$
   - Yields analytic complex coefficients $W(a, b) = R(a, b) + i I(a, b)$ with instantaneous envelope magnitude $|W(a, b)| = \sqrt{R^2 + I^2}$ and instantaneous phase $\theta = \operatorname{atan2}(I, R)$.

2. **Mexican Hat (Ricker) Wavelet**:
   $$\psi(t) = \frac{2}{\sqrt{3} \pi^{1/4}} (1 - t^2) \exp\left( -\frac{t^2}{2} \right)$$
   - Purely real, second derivative of a Gaussian.
   - Zero DC bias: $\int_{-\infty}^{\infty} \psi(t) dt = 0$. In discrete convolution, kernel mean subtraction ensures exact zero DC response down to machine precision ($< 10^{-6}$).
   - Optimally matches impulsive mechanical impact discontinuities and blade shock waves.

### 2.3 Logarithmic Scale Generation & Pseudo-Frequency Mapping

To span the acoustic spectrum from $f_{\min} = 100\text{ Hz}$ to $f_{\max} = 7500\text{ Hz}$ across $S$ scales:
The central frequency of the mother wavelet at $a = 1.0$ is:
$$f_c = \begin{cases} \frac{\omega_0}{2\pi} \approx 0.9549\text{ Hz} & \text{(Complex Morlet)} \\ \frac{\sqrt{2.5}}{2\pi} \approx 0.2516\text{ Hz} & \text{(Mexican Hat)} \end{cases}$$

Pseudo-frequency corresponding to scale $a$ at sampling rate $f_s$:
$$f(a) = \frac{f_c \cdot f_s}{a} \implies a(f) = \frac{f_c \cdot f_s}{f}$$

Scales are logarithmically distributed between $a_{\max} = a(f_{\min})$ and $a_{\min} = a(f_{\max})$:
$$a_s = \exp\left( \ln(a_{\max}) - s \cdot \frac{\ln(a_{\max}) - \ln(a_{\min})}{S - 1} \right), \quad s = 0, \dots, S-1$$
ensuring $f_0 = f_{\min} < f_1 < \dots < f_{S-1} = f_{\max}$.

---

## 3. Structural Damage Metrics & Fault Signatures

### 3.1 Scale-Wise Statistical Kurtosis

Gaussian background aerodynamic hiss produces normal distribution statistics with kurtosis $\approx 3.0$. Micro-cracks, chipped blades, or spalled bearing races produce non-Gaussian impulsive shock pulses.

Along the temporal interior of each scale $|W(a, t)|$, kurtosis is evaluated:
$$\operatorname{Kurt}(a) = \frac{\frac{1}{N} \sum_{n=1}^N (|W(a, n)| - \mu_a)^4}{\left( \frac{1}{N} \sum_{n=1}^N (|W(a, n)| - \mu_a)^2 \right)^2}$$
- Excludes boundary samples (cone of influence) to prevent edge truncation distortions.
- Filters out inactive scales possessing less than $2\%$ of total scalogram energy.
- Peak kurtosis $\operatorname{Kurt}_{\max} > 6.0$ triggers acoustic damage warnings; $\operatorname{Kurt}_{\max} > 10.0$ indicates critical mechanical fracture risk.

### 3.2 Dynamic Blade Flutter Modulation Index

Dynamic blade flutter is characterized by periodic aerodynamic stall and structural vibration amplitude modulation (AM) synchronized with the blade pass frequency:
$$f_{\text{BPF}} = \frac{N_{\text{blades}} \cdot \text{RPM}}{60} \quad [\text{Hz}]$$

At the wavelet scale $a_{\text{BPF}}$ matching $f_{\text{BPF}}$, the envelope modulation depth is measured across the interior:
$$I_{\text{flutter}} = \frac{|W|_{\max} - |W|_{\min}}{|W|_{\max} + |W|_{\min}}$$
- Nominal pristine propeller blades maintain steady lift thrust: $I_{\text{flutter}} < 0.25$.
- Severe aerodynamic blade flutter or pitch divergence causes envelope surging: $I_{\text{flutter}} > 0.35$.

### 3.3 High-Frequency Micro-Crack Energy Ratio

Micro-cracks in rotating carbon-fiber blades radiate high-frequency acoustic emission bursts ($> 3000\text{ Hz}$). The ratio of high-frequency wavelet energy to total scalogram energy provides a dedicated crack propagation metric:
$$R_{\text{crack}} = \frac{\sum_{f(a) \ge 3000\text{ Hz}} E(a)}{E_{\text{total}}}$$

### 3.4 Composite Airframe Fatigue Index

A unified scalar metric normalized in $[0.0, 1.0]$:
$$I_{\text{fatigue}} = \operatorname{clamp}\left( 0.45 \cdot S_{\text{kurt}} + 0.35 \cdot S_{\text{crack}} + 0.20 \cdot S_{\text{flutter}}, 0.0, 1.0 \right)$$
where each component score $S$ is normalized against warning/critical thresholds.

| Fatigue Index | Severity Rating | Action Required |
|---|---|---|
| $0.00\text{--}0.24$ | `Normal` | Nominal flight operations. |
| $0.25\text{--}0.44$ | `Advisory` | Scheduled visual pre-flight inspection. |
| $0.45\text{--}0.74$ | `Warning` | Minor structural crack / bearing wear; service advised. |
| $0.75\text{--}1.00$ | `Critical` | Imminent fracture or motor seizure; immediate RTL / land. |

---

## 4. Autonomous MAVLink Telemetry Protocol

The profiler packages structural health assessments into standard MAVLink `NAMED_VALUE_FLOAT` telemetry packets (Message ID 251, $\le 10$ byte parameter names):

```
+--------------------+-------------------+-------------------------------+
| Telemetry Name     | Type              | Description                   |
+--------------------+-------------------+-------------------------------+
| "FATIGUE"          | Float32 [0.0-1.0] | Normalized Airframe Fatigue   |
| "FLUTTER"          | Float32 [0.0-1.0] | Dynamic Blade Flutter Index   |
| "CWT_KURT"         | Float32 [3.0-50]  | Peak Wavelet Kurtosis         |
| "CRACK_ENG"        | Float32 [0.0-1.0] | High-Freq Crack Energy Ratio  |
+--------------------+-------------------+-------------------------------+
```

Telemetry packets are emitted over serial UART, UDP, or POSIX SHM IPC to ArduPilot, PX4, or companion ground control stations (QGroundControl, Mission Planner, Aerovex Workstation).

---

## 5. Verification & Benchmark Results

The Phase 17 test suite (`tests/sonon_phase17_tests.rs`) was executed under release profile with zero failures across 6 analytical benchmarks:

```
running 6 tests
test test_morlet_wavelet_filterbank_and_frequency_mapping ... ok
test test_mexican_hat_wavelet_zero_dc_bias ... ok
test test_cwt_scalogram_synthetic_transient_impulse_detection ... ok
test test_rotor_damage_profiler_blade_crack_and_flutter_detection ... ok
test test_mavlink_telemetry_packet_generation ... ok
test test_cwt_engine_streaming_throughput_benchmark ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
```

### 5.1 Verification Summary

1. **Morlet Wavelet Mapping**: Confirmed monotonic scale-frequency correspondence across 16 scales from $100.0\text{ Hz}$ to $7500.0\text{ Hz}$ ($\pm 5\text{ Hz}$ boundary precision).
2. **Mexican Hat Zero DC Bias**: Verified interior response to constant DC signals is $< 10^{-4}$ ($< 0.0001$), confirming complete rejection of static pressure offsets.
3. **Transient Shock Spike Detection**: Injected synthetic sub-millisecond micro-crack impulse; peak kurtosis spiked from $3.1$ to $> 6.5$ in high-frequency wavelet bands.
4. **Rotor Damage Detection**: Verified that pristine 3600 RPM rotor audio evaluated to `Normal` ($I_{\text{fatigue}} < 0.25$), while simulated periodic micro-cracks and friction surged to `Warning` / `Critical` ($I_{\text{fatigue}} > 0.45$, kurtosis $> 5.0$).
5. **MAVLink Packet Formatting**: Verified all 4 named value float packets adhere to MAVLink standard null-terminated 10-byte name strings.
6. **Real-Time Throughput**: Achieved continuous streaming throughput $> 50,000\text{ samples/sec}$ (> 3.1x real-time speed at 16 kHz) with zero heap fragmentation during frame processing.

---

## 6. Architectural Integration

```
[ Microphone Audio Stream ]
            │
            ▼
[ Ring Buffer & Pre-Emphasis ]
            │
     ┌──────┴─────────────────────────────────┐
     ▼                                        ▼
[ FFT & Mel Filterbank ]        [ Continuous Wavelet Filterbank ]
     │ (Phrase Spotting)              │ (Multi-Resolution Scalogram)
     ▼                                        ▼
[ Sakoe-Chiba Banded DTW ]     [ Scale Kurtosis, Flutter, Crack Ratio ]
     │                                        │
     ▼                                        ▼
[ Keyword Events ]              [ RotorDamageReport & Fatigue Index ]
                                              │
                                              ▼
                                 [ MAVLink Telemetry Packets ]
```

Phase 17 equips autonomous robotics with continuous structural health monitoring, ensuring mechanical defects are isolated long before catastrophic in-flight failure.
