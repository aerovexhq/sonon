# Python Client SDK

The Python bindings provide zero-copy NumPy array integration:

```python
import numpy as np
from sonon import SononEngine

engine = SononEngine(sample_rate=16000, n_fft=512, hop_size=160, n_mels=13)
engine.enroll_keyword("falcon", [audio_template], threshold=0.60)

# Process 10 ms audio frame
frame = np.zeros(160, dtype=np.float32)
detection = engine.process_frame(frame)
if detection:
    print(f"Keyword: {detection.keyword}, confidence: {detection.confidence:.2f}")
```
