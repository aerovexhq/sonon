# @aerovexhq/sonon

High-performance robotics-aimed acoustic DSP, Voice Activity Detection, and few-shot wake-word spotting engine for Node.js and Browser.

## Installation

```bash
npm install @aerovexhq/sonon
```

## Quickstart

```javascript
const { SononEngine, SononShm } = require('@aerovexhq/sonon');

// 1. Initialize engine at 16 kHz
const engine = new SononEngine(16000.0);

// 2. Extract reference features and enroll keyword
const refAudio = new Float32Array(3200); // 200 ms exemplar
const refFeatures = engine.extractFeatures(refAudio);
engine.enrollKeyword('emergency_stop', refFeatures, 3.5);

// 3. Stream live microphone samples
const incomingChunk = new Float32Array(160).fill(0.05);
const events = engine.ingestSamples(incomingChunk);
for (const ev of events) {
  console.log(`Recognized '${ev.keyword}' with confidence ${ev.confidence.toFixed(2)}`);
}

// 4. POSIX Shared Memory Client (/dev/shm/sonon_audio)
const shm = new SononShm();
shm.writeSamples([0.1, -0.2, 0.3]);
const health = shm.readHealth();
console.log(`Airframe Acoustic Health: ${health.score.toFixed(2)}`);
```
