#!/usr/bin/env node
/**
 * Sonon Live Microphone Listening Example (Node.js).
 *
 * Captures streaming microphone PCM audio from system utility (arecord / sox / ffmpeg),
 * trims silence via VAD, plays back the enrolled wake word once through speakers (aplay / afplay),
 * and spots wake-words in real-time with visual level and DTW distance meters.
 *
 * Usage:
 *   node live_mic_node.js
 */

'use strict';

const { spawn, spawnSync } = require('child_process');
const readline = require('readline');
const fs = require('fs');
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

function playAudioFile(filepath) {
  if (process.platform === 'linux') {
    spawnSync('aplay', ['-q', filepath]);
  } else if (process.platform === 'darwin') {
    spawnSync('afplay', [filepath]);
  } else {
    spawnSync('ffplay', ['-nodisp', '-autoexit', '-loglevel', 'quiet', filepath]);
  }
}

function writeWavFile(filepath, samples, sampleRate = 16000) {
  const numSamples = samples.length;
  const buffer = Buffer.alloc(44 + numSamples * 2);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + numSamples * 2, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buffer.writeUInt16LE(2, 32); // block align
  buffer.writeUInt16LE(16, 34); // bits per sample
  buffer.write('data', 36);
  buffer.writeUInt32LE(numSamples * 2, 40);

  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1.0, Math.min(1.0, samples[i]));
    const val = Math.floor(clamped * 32767.0);
    buffer.writeInt16LE(val, 44 + i * 2);
  }

  fs.writeFileSync(filepath, buffer);
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log('==================================================');
console.log('Sonon Node.js Live Microphone Wake-Word Spotter');
console.log('==================================================');

rl.question('Press Enter and say your keyword to enroll (e.g. "Take Off" or "Plank"): ', () => {
  console.log('Recording 2.0 seconds for keyword enrollment... Speak now!');

  const { cmd, args } = getRecorderCommand();
  const rec = spawn(cmd, args);

  const enrollChunks = [];
  let totalBytes = 0;
  const targetBytes = SAMPLE_RATE * 2 * 2.0; // 2.0s of 16-bit PCM

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

      // VAD silence trimming
      const trimmed = SononEngine.trimSilence(floatSamples, SAMPLE_RATE);
      const trimmedDuration = (trimmed.length / SAMPLE_RATE).toFixed(2);
      console.log(`Captured 2.00s -> Trimmed to ${trimmedDuration}s of active speech.`);

      const wavPath = '/tmp/sonon_wake_word.wav';
      writeWavFile(wavPath, trimmed, SAMPLE_RATE);

      // Play back to user immediately
      console.log(`\n[PLAYBACK] Playing back your enrolled wake-word via audio output...`);
      playAudioFile(wavPath);

      const features = engine.extractFeatures(trimmed);
      if (features.length === 0) {
        console.log('No speech detected. Please run again.');
        process.exit(1);
      }

      const threshold = 0.58;
      engine.enrollKeyword('wake_word', features, threshold);
      console.log(`Enrolled! Extracted ${features.length} feature frames (threshold: ${threshold}).`);
      console.log();
      console.log('Step 2: Continuous Real-Time Listening');
      console.log('Speak naturally! When you say your wake word, it will trigger below.');
      console.log('Press Ctrl+C to stop.');
      console.log('--------------------------------------------------');

      startLiveListening(cmd, args, threshold);
    }
  });

  rec.on('error', (err) => {
    console.error('Failed to spawn recorder:', err.message);
    console.log('Please ensure `arecord` (Linux) or `sox` (macOS) is installed.');
    process.exit(1);
  });
});

function startLiveListening(cmd, args, threshold) {
  const listener = spawn(cmd, args);
  let lastTriggerTime = 0;

  listener.stdout.on('data', (chunk) => {
    const numSamples = Math.floor(chunk.length / 2);
    const floatSamples = new Float32Array(numSamples);
    let sumSq = 0.0;

    for (let i = 0; i < numSamples; i++) {
      const val = chunk.readInt16LE(i * 2) / 32768.0;
      floatSamples[i] = val;
      sumSq += val * val;
    }

    const detections = engine.ingestSamples(floatSamples);
    const dist = engine.getLastDistance('wake_word');
    const rms = Math.sqrt(sumSq / Math.max(1, numSamples));
    const bars = Math.min(15, Math.floor(rms * 40));
    const meter = '#'.repeat(bars) + ' '.repeat(15 - bars);
    const distStr = dist !== undefined && isFinite(dist) ? dist.toFixed(3) : '---';

    process.stdout.write(`\r[LISTENING] Level: [${meter}] (RMS: ${rms.toFixed(3)}) | Best Dist: ${distStr} (Thresh: ${threshold.toFixed(2)})  `);

    const now = Date.now();
    for (const ev of detections) {
      if (now - lastTriggerTime > 1000) {
        lastTriggerTime = now;
        process.stdout.write(
          `\n\n==================================================\n` +
          `*** [TRIGGER] WAKE-WORD DETECTED: '${ev.keyword}'! ***\n` +
          `Confidence: ${(ev.confidence * 100).toFixed(1)}% | Time: ${ev.timestampSec.toFixed(2)}s\n` +
          `==================================================\n\n`
        );
      }
    }
  });

  listener.on('error', (err) => {
    console.error('Microphone stream error:', err.message);
  });
}

