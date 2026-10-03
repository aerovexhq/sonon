# C-ABI & POSIX Shared Memory API

Sonon exposes dynamic C symbols from `capi.rs`:

```c
#include <stdint.h>

void* sonon_engine_create(uint32_t sample_rate, uint32_t n_fft, uint32_t hop_size, uint32_t n_mels);
void sonon_engine_destroy(void* handle);
int32_t sonon_engine_process(void* handle, const float* input, uint32_t length);
int32_t sonon_engine_get_last_keyword(void* handle, char* buffer, uint32_t buffer_len);
```

## Shared Memory Ring Buffer

POSIX shared memory channel `/dev/shm/sonon_audio` enables zero-copy streaming between flight software and the DSP engine at over 15,000,000 samples/sec.
