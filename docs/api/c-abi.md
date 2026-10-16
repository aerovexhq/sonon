# C-ABI &amp; POSIX Shared Memory API

<p class="doc-lead">
Sonon exports dynamic C symbols through `sonon_capi.rs` with C calling conventions (`extern "C"`), enabling direct inter-process communication with ROS 2, PX4 autopilot, ArduPilot, and native C/C++ avionics controllers over POSIX shared memory ring buffers.
</p>

<div class="key-features-box">
  <div class="key-features-header">C-ABI &amp; IPC Architecture</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Standard C-ABI Symbols (libsonon.so / sonon.dll)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> POSIX Lock-Free SHM Ring Buffer (/dev/shm/sonon_audio)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> >15,000,000 Samples/Sec Zero-Copy IPC Throughput</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Compatible with C, C++, C#, Zig, and Go</div>
  </div>
</div>

## C Header Definition (`sonon.h`)

```c
#ifndef SONON_H
#define SONON_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine handle
typedef struct SononEngineHandle SononEngineHandle;

// Engine lifecycle
SononEngineHandle* sonon_engine_create(uint32_t sample_rate, uint32_t n_fft, uint32_t hop_size, uint32_t n_mels);
void sonon_engine_destroy(SononEngineHandle* handle);

// Dynamic rotor notch configuration
void sonon_engine_enable_rotor_notch(SononEngineHandle* handle, uint32_t blade_count, uint32_t harmonics);
void sonon_engine_update_motor_rpm(SononEngineHandle* handle, float motor_rpm);

// Keyword spotting & processing
int32_t sonon_engine_enroll_keyword(
    SononEngineHandle* handle,
    const char* keyword,
    const float* pcm_samples,
    uint32_t sample_count,
    float threshold
);

int32_t sonon_engine_process_frame(
    SononEngineHandle* handle,
    const float* input_frame,
    uint32_t frame_len,
    char* detected_keyword_buffer,
    uint32_t buffer_len,
    float* confidence_out
);

#ifdef __cplusplus
}
#endif

#endif // SONON_H
```

## POSIX Shared Memory Ring Buffer

For high-throughput avionics architectures where latency jitter must be strictly eliminated, Sonon communicates over a lock-free circular ring buffer mounted at `/dev/shm/sonon_audio`:

```c
// Lock-free shared memory layout
struct SononShmBuffer {
    uint32_t sequence_head;     // Incremented by audio driver on write
    uint32_t sequence_tail;     // Incremented by Sonon on read
    uint32_t sample_rate;       // 16000 or 48000 Hz
    uint32_t channel_count;     // Up to 8 I2S channels
    float    samples[65536];    // Cyclic ring buffer
};
```

| C API Function | Parameter Types | Return Status | Notes |
| :--- | :--- | :--- | :--- |
| `sonon_engine_create` | `(uint32_t, uint32_t, uint32_t, uint32_t)` | Pointer / NULL | Returns opaque pointer to safe Rust state |
| `sonon_engine_destroy` | `(SononEngineHandle*)` | Void | Safely drops heap allocations |
| `sonon_engine_update_motor_rpm` | `(SononEngineHandle*, float)` | Void | Recomputes biquad filter poles in &lt;5 us |
| `sonon_engine_process_frame` | `(SononEngineHandle*, const float*, ...)` | `1` if detected, `0` if silent | Populates detected word and confidence score |

::: info ARCHITECTURE DIRECTIVE: MEMORY LIFETIMES
All pointers passed into C-ABI functions must remain valid for the duration of the synchronous call. Sonon never retains external raw pointers across frame boundaries.
:::

## Next Steps

- [Python NumPy SDK Bindings](/api/python)
- [WebAssembly Browser Runtime](/api/wasm)
