#!/usr/bin/env node
/**
 * Sonon Live Microphone Listening Example (Node.js).
 *
 * Captures streaming microphone PCM audio from system utility (arecord / sox / ffmpeg)
 * and spots wake-words in real-time with zero external npm dependencies.
 *
 * Usage:
 *   node live_mic_node.js
 */

'use strict';

const { spawn } = require('child_process');
const readline = require('readline');
const { SononEngine } = require('../bindings/npm/index.js');

const SAMPLE_RATE = 16000;
const engine = new SononEngine(SAMPLE_RATE, 512, 160, 26);

// Find available audio recorder
function getRecorderCommand() {
  const isLinux = process.platform === 'linux';
  const isMac = process.platform === 'darwin';

  if (isLinux) {
    return { cmd: 'arecord', args: ['-q', '-r', '16000', '-c', '1', '-f', 'S16_LE'] };
  } else if (isMac) {
    return { cmd: 'sox', args: ['-d', '-q', '-r', '16000', '-c', '1', '-b', '16', '-e', 'signed-integer', '-t', 'raw', '-'] };
  }
  return { cmd: 'ffmpeg', args: ['-nostdin', '-loglevel', 'quiet', '-f', 'pulse', '-i', 'default', '-ar', '16000', '-ac', '1', '-f', 's16le', '-'] };
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log('==================================================');
console.log('Sonon Node.js Live Microphone Wake-Word Spotter');
console.log('==================================================');

rl.question('Press Enter and say your keyword to enroll (e.g. "Take Off"): ', () => {
  console.log('Recording 1.5 seconds for keyword enrollment...');

  const { cmd, args } = getRecorderCommand();
  const rec = spawn(cmd, args);

  const enrollChunks = [];
  let totalBytes = 0;
  const targetBytes = SAMPLE_RATE * 2 * 1.5; // 1.5s of 16-bit PCM

  rec.stdout.on('data', (data) => {
    enrollChunks.push(data);
    totalBytes += data.length;

    if (totalBytes >= targetBytes) {
      rec.kill();
      const fullBuffer = Buffer.concat(enrollChunks);
      const numSamples = Math.floor(fullBuffer.length / 2);
      const floatSamples = new Float32Array(numSamples);

      for (let i = 0; i < numSamples; i++) {
        floatSamples[i] = fullBuffer.readInt16LE(i * 2) / 32768.0;
      }

      const features = engine.extractFeatures(floatSamples);
      if (features.length === 0) {
        console.log('No speech detected. Please run again.');
        process.exit(1);
      }

      engine.enrollKeyword('wake_word', features, 3.5);
      console.log(`Enrolled! Extracted ${features.length} feature frames.`);
      console.log();
      console.log('Now listening continuously. Speak naturally! (Press Ctrl+C to exit)');
      console.log('--------------------------------------------------');

      startLiveListening(cmd, args);
    }
  });

  rec.on('error', (err) => {
    console.error('Failed to spawn recorder:', err.message);
    console.log('Please ensure `arecord` (Linux) or `sox` (macOS) is installed.');
    process.exit(1);
  });
});

function startLiveListening(cmd, args) {
  const listener = spawn(cmd, args);

  listener.stdout.on('data', (chunk) => {
    const numSamples = Math.floor(chunk.length / 2);
    const floatSamples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      floatSamples[i] = chunk.readInt16LE(i * 2) / 32768.0;
    }

    const detections = engine.ingestSamples(floatSamples);
    for (const ev of detections) {
      console.log(
        `[TRIGGER] Wake-Word Detected: '${ev.keyword}' | Confidence: ${ev.confidence.toFixed(2)} | Time: ${ev.timestampSec.toFixed(2)}s`
      );
    }
  });

  listener.on('error', (err) => {
    console.error('Microphone stream error:', err.message);
  });
}
