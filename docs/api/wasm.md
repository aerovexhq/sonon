# WebAssembly (WASM 32-bit) API

Sonon compiles to WebAssembly with zero C dependencies:

```typescript
// Load WASM module
const { instance } = await WebAssembly.instantiateStreaming(fetch('/sonon.wasm'));
const api = resolveSononWasmExports(instance);

// Create instance
const engine = api.sonon_wasm_create(16000, 512, 160, 13);

// Ingest samples from AudioWorklet
api.sonon_wasm_set_input_samples(pcmFloat32Array);
const detections = api.sonon_wasm_ingest(engine, pcmFloat32Array.length);
if (detections > 0) {
    const keyword = api.sonon_wasm_get_last_keyword();
    console.log(`Detected: ${keyword}`);
}
```

Try this directly in our [Playground](/playground).
