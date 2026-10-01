# Sonon Python SDK (`sonon`)

High-performance robotics-aimed acoustic DSP, Voice Activity Detection, and few-shot phrase spotting package.

## Quickstart

```python
import sonon

# 1. Initialize engine at 16 kHz
engine = sonon.SononEngine(sample_rate=16000.0)

# 2. Extract reference features and enroll keyword
ref_audio = [0.0] * 3200  # 200 ms exemplar
ref_features = engine.extract_features(ref_audio)
engine.enroll_keyword("emergency_stop", ref_features, threshold=3.5)

# 3. Stream live microphone samples
incoming_chunk = [0.05] * 160
events = engine.ingest_samples(incoming_chunk)
for ev in events:
    print(f"Recognized '{ev.keyword}' with confidence {ev.confidence:.2f}")

# 4. Zero-copy POSIX Shared Memory Interface (/dev/shm/sonon_audio)
shm = sonon.SononShm()
shm.write_samples([0.1, -0.2, 0.3])
health_score, severity = shm.read_health()
print(f"Airframe Acoustic Health: {health_score:.2f}")
```
