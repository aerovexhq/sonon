# Quickstart Guide

## Installation

### Rust Crate

Add `sonon` to your `Cargo.toml`:

```toml
[dependencies]
sonon = { path = "modules/sonon" }
```

### Python Package

```bash
pip install sonon
```

### WebAssembly (NPM)

```bash
npm install @aerovexhq/sonon
```

## Basic Rust Usage

```rust
use sonon::engine::SononEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 16 kHz sample rate, 512-point FFT, 160-sample hop (10 ms), 13 Mel bins
    let mut engine = SononEngine::new(16_000, 512, 160, 13)?;

    // Enroll a wake-word with 2 exemplars
    engine.enroll_keyword("falcon", &[&exemplar_1, &exemplar_2], 0.65)?;

    // Stream audio buffer
    let chunk = [0.0f32; 160];
    if let Some(event) = engine.process_frame(&chunk)? {
        println!("Keyword detected: {} with confidence {:.2}", event.keyword, event.confidence);
    }

    Ok(())
}
```

## Try In Browser

You can immediately test Sonon using your microphone in our [WASM Playground](/playground).
