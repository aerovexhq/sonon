# Pure Safe Rust API Reference

<p class="doc-lead">
The core Sonon engine is written in 100% safe Rust with `#![deny(unsafe_code)]` enforced at the root of every module. The primary interface is provided by the `SononEngine` struct in `sonon::engine`.
</p>

<div class="key-features-box">
  <div class="key-features-header">Rust Engine Directives</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Deny Unsafe Code Everywhere (#![deny(unsafe_code)])</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero Dynamic Allocations in Ingestion Loop</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Compile-Time Target SIMD (AVX2, NEON, WASM)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Send + Sync Deterministic Multithreading</div>
  </div>
</div>

## Core Engine Interface

```rust
use sonon::engine::{SononEngine, SpottingEvent, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure engine parameters
    let config = EngineConfig {
        sample_rate: 16_000,
        n_fft: 512,
        hop_size: 160,
        n_mels: 13,
        enable_pcen: true,
    };

    // 2. Instantiate engine state
    let mut engine = SononEngine::with_config(config)?;

    // 3. Configure rotor notch tracking
    engine.enable_telemetry_coupled_notch(4 /* blades */, 3 /* harmonics */);
    engine.update_motor_rpm(5_420.0);

    // 4. Enroll custom voice trigger offline
    let exemplar_a = [0.0f32; 1600]; // 100 ms audio slice
    let exemplar_b = [0.0f32; 1600];
    engine.enroll_keyword("land_now", &[&exemplar_a, &exemplar_b], 0.62)?;

    // 5. Ingest 10 ms audio frames
    let pcm_frame = [0.0f32; 160];
    if let Some(event) = engine.process_frame(&pcm_frame)? {
        println!("Keyword spotted: {} (confidence: {:.2})", event.keyword, event.confidence);
    }

    Ok(())
}
```

## API Method Specifications

| Method | Signature | Return Type | Thread Safety | Description |
| :--- | :--- | :--- | :--- | :--- |
| `new` | `(sample_rate: u32, n_fft: usize, hop_size: usize, n_mels: usize)` | `Result<Self, EngineError>` | Safe | Instantiates engine with default parameters |
| `enable_telemetry_coupled_notch` | `(&mut self, blades: u32, harmonics: usize)` | `()` | Safe | Activates BPF harmonic notch cascade |
| `update_motor_rpm` | `(&mut self, rpm: f32)` | `()` | Lock-Free | Updates rotor frequency for notch recalculation |
| `enroll_keyword` | `(&mut self, word: &str, exemplars: &[&[f32]], threshold: f32)` | `Result<(), EngineError>` | Safe | Computes DBA barycenter prototype template |
| `process_frame` | `(&mut self, pcm: &[f32; 160])` | `Result<Option<SpottingEvent>, EngineError>` | Real-Time | Evaluates frame through filterbank, VAD, and DTW |
| `synthesize_speech` | `(&mut self, target: &PhonemeTarget, out: &mut [f32])` | `Result<usize, EngineError>` | Real-Time | Liljencrants-Fant physical vocal synthesis |

::: info ARCHITECTURE DIRECTIVE: ZERO HEAP CHURN
`process_frame` is marked `#[inline(always)]` and guarantees zero heap allocations. Internal ring buffers and scratch matrices are initialized exclusively inside `new` or `with_config`.
:::

## Next Steps

- [C-ABI &amp; POSIX Shared Memory API](/api/c-abi)
- [Python NumPy SDK](/api/python)
- [WebAssembly Browser Runtime](/api/wasm)
