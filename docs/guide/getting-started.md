# Getting Started with Sonon

<p class="doc-lead">
Sonon is an embedded, hard real-time acoustic digital signal processing and speech synthesis engine engineered in 100% pure safe Rust. Designed for autonomous robotic platforms, UAV autopilot companion computers, and microcontrollers, it delivers dynamic rotor noise cancellation, few-shot voice phrase spotting, and on-device physical vocal tract synthesis with strictly zero external C audio dependencies.
</p>

<div class="key-features-box">
  <div class="key-features-header">Core Capabilities &amp; Engineering Guarantees</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 100% Safe Rust (#![deny(unsafe_code)] crate-wide)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> >86 dB Motor Propeller Harmonic Notch Rejection</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Sub-0.5 ms Sakoe-Chiba Banded DTW Keyword Spotter</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 1.2 MB Physical Formant Vocal Tract Synthesizer</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero Dynamic Allocations During Steady-State Audio Stream</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Bare-Metal Microcontroller &amp; WebAssembly Compatibility</div>
  </div>
</div>

::: info ARCHITECTURE DIRECTIVE
Sonon prohibits heap reallocations during audio frame streaming. All ring buffers, FFT workspaces, and dynamic warping corridors are pre-allocated upon engine instantiation. Real-time deadlines are maintained with deterministic cycle bounds under 5 microseconds per frame.
:::

## Architectural Philosophy

Acoustic perception on autonomous aerial and ground robotics operates under extreme physical constraints:

1. **Acoustic Noise Saturation**: Multirotor propellers generate acoustic wash exceeding +88 dBA with narrow blade pass frequency (BPF) peaks that completely mask human voice formants.
2. **Computational & Power Budgets**: Low-power companion computers and microcontrollers (e.g. ARM Cortex-M7 @ 480 MHz) cannot dedicate multiple CPU cores or gigabytes of VRAM to neural vocoders or deep transformer models.
3. **Deterministic Latency**: Mission-critical voice abort commands require sub-millisecond recognition latency, completely offline, without reliance on cloud network APIs.

Sonon bridges physical acoustic models with efficient algorithmic corridors, combining biquad filter cascades, Sakoe-Chiba dynamic time warping, and Liljencrants-Fant physical vocal cord simulation into a unified 1.2 MB memory footprint.

## Core Engine Specifications

| Metric / Parameter | Value | Unit | Engineering Description |
| :--- | :--- | :--- | :--- |
| **Nominal Ingestion Rate** | 16,000 / 48,000 | Hz | Native streaming sample rates supported over I2S DMA |
| **Hop Size (Frame Window)** | 160 | Samples (10 ms) | Fundamental discrete temporal analysis frame |
| **FFT Analysis Window** | 512 | Points | Radix-2 Cooley-Tukey spectral magnitude decomposition |
| **Mel Filterbank Bins** | 13 to 40 | Bins | Triangular filterbank with PCEN energy normalization |
| **Notch Filter Depth** | &gt; 86.4 | dB | Attenuation at blade pass fundamental and 7 harmonics |
| **KWS Inference Latency** | 0.42 | ms | Sakoe-Chiba banded DTW evaluation with R=10 corridor |
| **VAD Decision Time** | 11.8 | us | Dual-band energy ratio and spectral centroid decision |
| **Total Memory Footprint** | &lt; 1.2 | MB | Complete engine state including speech tables |

::: tip PRO TIP: ESC TELEMETRY COUPLING
Always feed live motor rotational speed (RPM) from autopilot telemetry (DShot 600 or MAVLink ESC status) into `engine.update_motor_rpm()`. This locks the IIR notch bank directly to the physical blade pass frequency, maintaining &gt;86 dB noise suppression even during aggressive throttle step inputs.
:::

## Next Steps

- [Explore System Architecture &amp; Dataflow](/guide/architecture)
- [Quickstart &amp; Installation Instructions](/guide/quickstart)
- [Test Live in Interactive Audio Lab](/playground)
- [Experiment with Voice Synthesis Studio](/synthesis)
