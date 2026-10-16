# Python Client SDK

<p class="doc-lead">
The Sonon Python SDK provides high-performance zero-copy bindings for robotics researchers and AI validation suites, interoperating directly with NumPy arrays, PyTorch tensors, and ROS 2 Python nodes.
</p>

<div class="key-features-box">
  <div class="key-features-header">Python SDK Specifications</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero-Copy NumPy Array Buffer Exchange</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> GIL Release During Heavy DSP FFT Filtering</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Pre-Compiled Multi-Platform Wheels (Linux / macOS / Windows)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Type Annotations &amp; Python 3.9 - 3.13 Support</div>
  </div>
</div>

## Installation &amp; Setup

```bash
pip install sonon
```

## Complete Python Example

```python
import numpy as np
from sonon import SononEngine

def main():
    # Initialize engine: 16 kHz sample rate, 512 FFT, 160 hop, 13 Mel bins
    engine = SononEngine(sample_rate=16000, n_fft=512, hop_size=160, n_mels=13)

    # Enable rotor notch suppression for 4-blade quadrotor
    engine.enable_rotor_notch(blade_count=4, max_harmonics=3)
    engine.update_motor_rpm(5420.0)

    # Load and enroll custom voice command from WAV file
    template_audio = np.fromfile("voice_trigger.raw", dtype=np.float32)
    engine.enroll_keyword("abort_mission", [template_audio], threshold=0.62)

    print("Sonon Python Engine Armed. Streaming microphone frames...")

    # Simulated audio streaming loop (10 ms per step)
    frame = np.zeros(160, dtype=np.float32)
    while True:
        # Fill frame from soundcard / ROS / stream
        event = engine.process_frame(frame)
        if event is not None:
            print(f"Detected keyword: {event.keyword} (score: {event.confidence:.3f})")
            break

if __name__ == "__main__":
    main()
```

## Python Class Specifications

| Class / Method | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `SononEngine(...)` | `sample_rate=16000, n_fft=512, hop_size=160, n_mels=13` | `SononEngine` | Initializes underlying safe Rust engine state |
| `enable_rotor_notch` | `blade_count: int, max_harmonics: int = 8` | `None` | Instantiates IIR biquad notch cascade |
| `update_motor_rpm` | `rpm: float` | `None` | Dynamically updates center frequencies |
| `enroll_keyword` | `keyword: str, audio_exemplars: List[np.ndarray], threshold: float` | `None` | Computes DBA barycenter prototype |
| `process_frame` | `frame: np.ndarray (shape: (160,), dtype: float32)` | `Optional[SpottingEvent]` | Evaluates single 10 ms audio frame |
| `synthesize_speech` | `phoneme_string: str` | `np.ndarray` | Renders speech PCM waveform via physical synthesis |

::: tip PRO TIP: NUMPY CONTIGUITY
Ensure input NumPy arrays are C-contiguous (`np.ascontiguousarray(frame, dtype=np.float32)`). Non-contiguous slices force an internal memory copy.
:::

## Next Steps

- [WebAssembly Browser Runtime](/api/wasm)
- [Pure Safe Rust Engine API](/api/rust)
