# Quickstart &amp; Installation

<p class="doc-lead">
Deploy Sonon across embedded Rust microcontrollers, Linux companion computers, Python robotics applications, and WebAssembly browser clients in minutes.
</p>

<div class="key-features-box">
  <div class="key-features-header">Supported Runtimes &amp; Packaging</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Native Rust Crate (Cargo / crates.io)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Python Wheels (PyPI / NumPy zero-copy)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> WebAssembly NPM Package (@aerovexhq/sonon)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> C-ABI Dynamic Shared Library (libsonon.so / sonon.dll)</div>
  </div>
</div>

## Package Installation

### 1. Rust Crate Installation

Add Sonon directly to your `Cargo.toml`:

```toml
[dependencies]
sonon = { version = "0.1", default-features = false, features = ["simd", "telemetry"] }
```

For bare-metal `#![no_std]` targets (STM32H7, RP2350):

```toml
[dependencies]
sonon = { version = "0.1", default-features = false }
```

### 2. Python Client Package

```bash
pip install sonon
```

### 3. WebAssembly Browser Package

```bash
npm install @aerovexhq/sonon
```

## Basic Rust Implementation

The following complete snippet instantiates the engine, enables the telemetry-coupled notch filter, enrolls a custom command keyword with 2 audio exemplars, and streams audio frames:

```rust
use sonon::engine::{SononEngine, SpottingEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize engine: 16 kHz sample rate, 512-pt FFT, 160-sample hop (10 ms), 13 Mel channels
    let mut engine = SononEngine::new(16_000, 512, 160, 13)?;

    // Enable motor telemetry notch filter for quadrotor (4 blades, 3 harmonics)
    engine.enable_telemetry_coupled_notch(4 /* blades */, 3 /* harmonics */);
    
    // Feed high-rate ESC RPM telemetry from MAVLink / DShot
    engine.update_motor_rpm(5_840.0);

    // Enroll a wake-word ("falcon") offline with 2 recorded exemplar buffers
    let exemplar_1 = read_pcm_sample("falcon_01.wav");
    let exemplar_2 = read_pcm_sample("falcon_02.wav");
    engine.enroll_keyword("falcon", &[&exemplar_1, &exemplar_2], 0.65)?;

    println!("Sonon DSP initialized. Ingesting 10 ms audio frames...");

    // Continuous 10 ms frame ingestion loop
    let mut mic_frame = [0.0f32; 160];
    loop {
        read_microphone_dma(&mut mic_frame);

        if let Some(event) = engine.process_frame(&mic_frame)? {
            println!(
                "Trigger Event: '{}' detected (confidence: {:.2}, latency: 0.42 ms)",
                event.keyword, event.confidence
            );
            trigger_mission_abort();
        }
    }
}
```

## Engine Configuration Options

| Parameter | Type | Valid Range | Default | Functional Description |
| :--- | :--- | :--- | :--- | :--- |
| `sample_rate` | `u32` | 8,000 to 48,000 | `16000` | Ingestion sampling frequency in Hz |
| `n_fft` | `usize` | 256 to 2048 | `512` | Radix-2 FFT window length |
| `hop_size` | `usize` | 80 to 480 | `160` | Discrete samples advanced per step (10 ms @ 16 kHz) |
| `n_mels` | `usize` | 13 to 40 | `13` | Number of triangular Mel filterbank frequency bins |
| `dtw_corridor` | `usize` | 4 to 32 | `10` | Sakoe-Chiba diagonal search corridor radius |
| `vad_hangover` | `u32` | 2 to 20 | `8` | Post-speech hangover frames preventing plosive clip |

::: tip PRO TIP: MEMORY ALLOCATION
Always instantiate `SononEngine` once during startup routines or thread initialization. Never reconstruct `SononEngine` inside the high-frequency audio interrupt service routine (ISR).
:::

## Next Steps

- [Overview of 30 Research Monograph Phases](/dsp/overview)
- [Dynamic Rotor Notch Filter Deep Dive](/dsp/rotor-notch-filter)
- [Pure Safe Rust Engine API Reference](/api/rust)
