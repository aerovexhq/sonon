# WebAssembly (WASM 32-bit) API

<p class="doc-lead">
Sonon compiles directly to WebAssembly (`wasm32-unknown-unknown`) with zero C dependencies. Running inside browser AudioWorklet threads, it delivers real-time rotor noise cancellation, keyword spotting, and speech synthesis without cloud server latency.
</p>

<div class="key-features-box">
  <div class="key-features-header">WebAssembly Architecture Highlights</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero External C Dependencies (Safe Rust WASM)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> AudioWorklet Dedicated Thread Ingestion</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero-Copy ArrayBuffer Memory Exchange</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Sub-300 KB Compressed WASM Bytecode Binary</div>
  </div>
</div>

## Browser AudioWorklet Integration

The following TypeScript example loads the WASM module into a dedicated `AudioWorkletProcessor`, streaming live microphone audio with zero UI thread contention:

```typescript
// sonon_worklet.ts
import { resolveSononWasmExports } from '@aerovexhq/sonon';

class SononWorkletProcessor extends AudioWorkletProcessor {
  private engineHandle: number = 0;
  private wasmApi: any;

  constructor(options: AudioWorkletNodeOptions) {
    super();
    this.initWasm(options.processorOptions.wasmBinary);
  }

  private async initWasm(binary: ArrayBuffer) {
    const { instance } = await WebAssembly.instantiate(binary);
    this.wasmApi = resolveSononWasmExports(instance);
    
    // 16 kHz sample rate, 512 FFT, 160 hop size, 13 Mel bins
    this.engineHandle = this.wasmApi.sonon_wasm_create(16000, 512, 160, 13);
    this.wasmApi.sonon_wasm_enable_rotor_notch(this.engineHandle, 4, 3);
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const inputChannel = inputs[0]?.[0];
    if (!inputChannel || !this.engineHandle) return true;

    // Zero-copy transfer into WASM memory
    this.wasmApi.sonon_wasm_set_input_samples(inputChannel);
    const detected = this.wasmApi.sonon_wasm_process(this.engineHandle, inputChannel.length);

    if (detected > 0) {
      const keyword = this.wasmApi.sonon_wasm_get_last_keyword(this.engineHandle);
      this.port.postMessage({ type: 'KWS_TRIGGER', keyword });
    }

    return true;
  }
}

registerProcessor('sonon-worklet-processor', SononWorkletProcessor);
```

## Exported WASM Functions

| Function Name | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `sonon_wasm_create` | `(sample_rate, n_fft, hop_size, n_mels)` | `number` (Pointer) | Allocates engine instance in WASM linear memory |
| `sonon_wasm_destroy` | `(handle)` | `void` | Frees engine memory allocations |
| `sonon_wasm_enable_rotor_notch` | `(handle, blades, harmonics)` | `void` | Activates biquad notch filterbank |
| `sonon_wasm_update_rpm` | `(handle, rpm)` | `void` | Dynamically updates motor tracking RPM |
| `sonon_wasm_set_input_samples` | `(samplesFloat32Array)` | `void` | Copies input chunk into pre-allocated WASM buffer |
| `sonon_wasm_process` | `(handle, length)` | `number` (Count) | Evaluates audio chunk; returns 1 on trigger |
| `sonon_wasm_get_last_keyword` | `(handle)` | `string` | Retrieves triggered keyword identifier string |

::: tip PRO TIP: MEMORY ISOLATION
Run Sonon strictly inside an `AudioWorkletNode`. Avoid running continuous FFT operations inside the main UI thread to prevent 60 FPS animation frame drops during heavy browser rendering.
:::

## Next Steps

- [Test Live in the WASM Playground](/playground)
- [Voice Synthesis Studio](/synthesis)
- [Pure Safe Rust Engine API](/api/rust)
