const assert = require('assert');
const { SononEngine, SononShm, SHM_MAGIC, DEFAULT_SHM_PATH } = require('./index.js');

console.log('Running @aerovexhq/sonon JavaScript verification tests...');

// 1. Engine creation and feature extraction
const engine = new SononEngine(16000.0, 512, 160, 26);
assert.strictEqual(engine.sampleRate, 16000.0);
assert.strictEqual(engine.numMfcc, 26);

// Synthesize 0.2s of audio (3200 samples)
const testAudio = new Float32Array(3200);
for (let i = 0; i < testAudio.length; i++) {
  const t = i / 16000.0;
  testAudio[i] = 0.7 * Math.sin(2.0 * Math.PI * 440.0 * t);
}

const features = engine.extractFeatures(testAudio);
assert(features.length > 0, 'Must extract feature frames');
assert.strictEqual(features[0].length, 26, 'Each frame must have 26 Mel features');
console.log(`[PASS] Extracted ${features.length} feature frames.`);

// 2. Keyword enrollment and spotting
engine.enrollKeyword('tone_440', features, 3.5);
const detections = engine.ingestSamples(testAudio);
console.log(`[PASS] Keyword spotting completed. Detections count: ${detections.length}`);

// 3. Shared memory channel
const shm = new SononShm(DEFAULT_SHM_PATH, 16000.0, 65536);
const written = shm.writeSamples([0.1, -0.2, 0.3, -0.4]);
assert.strictEqual(written, 4, 'Must write 4 samples');
const health = shm.readHealth();
assert(health.score >= 0.0 && health.score <= 1.0, 'Health score must be bounded');
console.log(`[PASS] SHM channel verified (written: ${written}, health: ${health.score.toFixed(2)})`);

console.log('All JavaScript @aerovexhq/sonon tests PASSED.');
