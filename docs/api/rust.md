# Pure Safe Rust API Reference

The core entry point is `SononEngine` in `sonon::engine`:

```rust
use sonon::engine::{SononEngine, SpottingEvent};

// 1. Initialize engine
let mut engine = SononEngine::new(
    16_000, // sample rate (Hz)
    512,    // FFT window size
    160,    // hop size (10 ms)
    13      // Mel filterbank channels
)?;

// 2. Enable rotor notch filtering
engine.enable_telemetry_coupled_notch(4, 3); // 4-blade rotor, 3 harmonics
engine.update_motor_rpm(5800.0);

// 3. Ingest audio samples
let frame = [0.0f32; 160];
if let Some(event) = engine.process_frame(&frame)? {
    println!("Wake word detected: {}", event.keyword);
}
```
