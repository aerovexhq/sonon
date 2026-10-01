/**
 * Sonon JavaScript / Node.js Client: High-Performance Acoustic DSP & Wake-Word Engine.
 *
 * Provides zero-cloud few-shot keyword spotting, feature extraction, and
 * zero-copy POSIX shared memory (/dev/shm/sonon_audio) streaming.
 */

'use strict';

const fs = require('fs');

const SHM_MAGIC = 0x534F4E4F; // "SONO"
const SHM_VERSION = 1;
const DEFAULT_SHM_PATH = '/dev/shm/sonon_audio';
const DEFAULT_CAPACITY = 65536;
const HEADER_SIZE = 64;

class SononShm {
  constructor(path = DEFAULT_SHM_PATH, sampleRate = 16000.0, capacity = DEFAULT_CAPACITY) {
    this.path = path;
    this.sampleRate = Number(sampleRate);
    this.capacity = Number(capacity);
    this.fd = null;
    this._ensureChannel();
  }

  _ensureChannel() {
    const totalSize = HEADER_SIZE + this.capacity * 4;
    const exists = fs.existsSync(this.path);

    this.fd = fs.openSync(this.path, 'r+');

    const stats = fs.fstatSync(this.fd);
    if (!exists || stats.size < totalSize) {
      fs.ftruncateSync(this.fd, totalSize);
      const headerBuf = Buffer.alloc(HEADER_SIZE);
      headerBuf.writeUInt32LE(SHM_MAGIC, 0);
      headerBuf.writeUInt32LE(SHM_VERSION, 4);
      headerBuf.writeFloatLE(this.sampleRate, 8);
      headerBuf.writeUInt32LE(1, 12); // channels
      headerBuf.writeUInt32LE(this.capacity, 16);
      headerBuf.writeBigUInt64LE(0n, 20); // write_head
      headerBuf.writeBigUInt64LE(0n, 28); // read_head
      headerBuf.writeFloatLE(1.0, 36); // health_score
      headerBuf.writeUInt32LE(0, 40); // worst_severity
      headerBuf.writeFloatLE(0.0, 44); // last_kw_conf
      headerBuf.writeUInt32LE(0, 48); // last_kw_len
      fs.writeSync(this.fd, headerBuf, 0, HEADER_SIZE, 0);
    }
  }

  writeSamples(samples) {
    const count = samples.length;
    if (count === 0) return 0;

    const rawBytes = Buffer.alloc(count * 4);
    for (let i = 0; i < count; i++) {
      rawBytes.writeFloatLE(samples[i], i * 4);
    }

    const headBuf = Buffer.alloc(8);
    fs.readSync(this.fd, headBuf, 0, 8, 20);
    const writeHead = headBuf.readBigUInt64LE(0);
    const writeHeadNum = Number(writeHead);

    const startSlot = writeHeadNum % this.capacity;
    const endSlot = startSlot + count;

    if (endSlot <= this.capacity) {
      fs.writeSync(this.fd, rawBytes, 0, count * 4, HEADER_SIZE + startSlot * 4);
    } else {
      const firstCount = this.capacity - startSlot;
      fs.writeSync(this.fd, rawBytes, 0, firstCount * 4, HEADER_SIZE + startSlot * 4);
      fs.writeSync(this.fd, rawBytes, firstCount * 4, (count - firstCount) * 4, HEADER_SIZE);
    }

    const newHeadBuf = Buffer.alloc(8);
    newHeadBuf.writeBigUInt64LE(BigInt(writeHeadNum + count), 0);
    fs.writeSync(this.fd, newHeadBuf, 0, 8, 20);

    return count;
  }

  readHealth() {
    const buf = Buffer.alloc(8);
    fs.readSync(this.fd, buf, 0, 8, 36);
    const score = buf.readFloatLE(0);
    const severity = buf.readUInt32LE(4);
    return { score, severity };
  }

  readDetection() {
    const buf = Buffer.alloc(20);
    fs.readSync(this.fd, buf, 0, 20, 44);
    const conf = buf.readFloatLE(0);
    const kwLen = buf.readUInt32LE(4);
    if (kwLen === 0 || conf <= 0.0) return null;
    const kwStr = buf.toString('utf8', 8, 8 + Math.min(12, kwLen)).replace(/\0/g, '');
    return { keyword: kwStr, confidence: conf };
  }

  close() {
    if (this.fd !== null) {
      fs.closeSync(this.fd);
      this.fd = null;
    }
  }
}

class SononEngine {
  constructor(sampleRate = 16000.0, frameSize = 512, hopSize = 160, numMfcc = 26) {
    this.sampleRate = Number(sampleRate);
    this.frameSize = Number(frameSize);
    this.hopSize = Number(hopSize);
    this.numMfcc = Number(numMfcc);
    this.templates = [];
    this.buffer = [];
    this.history = [];
    this.totalSamples = 0;
    this.dcXPrev = 0.0;
    this.dcYPrev = 0.0;
    this.dcR = 0.995;

    this.window = new Float32Array(frameSize);
    for (let i = 0; i < frameSize; i++) {
      this.window[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (frameSize - 1)));
    }
  }

  static removeDc(samples) {
    const n = samples.length;
    if (n === 0) return Array.from(samples);
    let sum = 0.0;
    for (let i = 0; i < n; i++) sum += samples[i];
    const mean = sum / n;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = samples[i] - mean;
    return Array.from(out);
  }

  extractFeatures(samples) {
    const features = [];
    let pos = 0;
    const n = samples.length;

    // Remove local DC bias
    let sampleSum = 0.0;
    for (let i = 0; i < n; i++) sampleSum += samples[i];
    const sampleMean = n > 0 ? sampleSum / n : 0.0;

    while (pos + this.frameSize <= n) {
      const frame = new Float32Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) {
        frame[i] = (samples[pos + i] - sampleMean) * this.window[i];
      }

      const mels = new Float32Array(this.numMfcc);
      for (let m = 0; m < this.numMfcc; m++) {
        const kStart = Math.floor((m * (this.frameSize / 2)) / this.numMfcc);
        const kEnd = Math.max(kStart + 1, Math.floor(((m + 1) * (this.frameSize / 2)) / this.numMfcc));
        let bandPower = 0.0;

        for (let k = kStart; k < kEnd; k++) {
          let re = 0.0;
          let im = 0.0;
          for (let s = 0; s < this.frameSize; s += 4) {
            const angle = (2.0 * Math.PI * k * s) / this.frameSize;
            re += frame[s] * Math.cos(angle);
            im -= frame[s] * Math.sin(angle);
          }
          bandPower += re * re + im * im;
        }
        mels[m] = Math.log(Math.max(1e-6, bandPower));
      }

      // Gain-invariant normalization: Cepstral Mean Subtraction and L2 unit-norm
      let sum = 0.0;
      for (let m = 0; m < this.numMfcc; m++) sum += mels[m];
      const mean = sum / this.numMfcc;
      let normSq = 0.0;
      for (let m = 0; m < this.numMfcc; m++) {
        mels[m] -= mean;
        normSq += mels[m] * mels[m];
      }
      const norm = Math.sqrt(normSq);
      if (norm > 1e-4) {
        for (let m = 0; m < this.numMfcc; m++) mels[m] /= norm;
      }

      features.push(Array.from(mels));
      pos += this.hopSize;
    }

    return features;
  }

  static trimSilence(samples, sampleRate = 16000, frameSize = 160, thresholdRatio = 0.15, marginFrames = 5) {
    const nSamples = samples.length;
    if (nSamples === 0) return Array.from(samples);

    // 1. Remove DC bias
    let totalSum = 0.0;
    for (let i = 0; i < nSamples; i++) totalSum += samples[i];
    const meanVal = totalSum / nSamples;
    const acSamples = new Float32Array(nSamples);
    for (let i = 0; i < nSamples; i++) acSamples[i] = samples[i] - meanVal;

    const numFrames = Math.floor(nSamples / frameSize);
    if (numFrames === 0) return Array.from(acSamples);

    const energies = new Float32Array(numFrames);
    let maxEnergy = 0.0;

    for (let i = 0; i < numFrames; i++) {
      let sumSq = 0.0;
      const offset = i * frameSize;
      for (let s = 0; s < frameSize; s++) {
        const val = acSamples[offset + s];
        sumSq += val * val;
      }
      const rms = Math.sqrt(sumSq / frameSize);
      energies[i] = rms;
      if (rms > maxEnergy) maxEnergy = rms;
    }

    if (maxEnergy < 0.01) return Array.from(acSamples);

    const thresh = Math.max(0.015, maxEnergy * thresholdRatio);
    let firstSpeech = -1;
    let lastSpeech = -1;

    for (let i = 0; i < numFrames; i++) {
      // Discard boundary clicks at extreme frames
      if (energies[i] >= thresh && i >= 2 && i <= numFrames - 3) {
        if (firstSpeech === -1) firstSpeech = i;
        lastSpeech = i;
      }
    }

    if (firstSpeech === -1) return Array.from(acSamples);

    const startFrame = Math.max(0, firstSpeech - marginFrames);
    const endFrame = Math.min(numFrames, lastSpeech + marginFrames + 1);

    const startSample = startFrame * frameSize;
    const endSample = Math.min(nSamples, endFrame * frameSize);
    const trimmed = Array.from(acSamples.slice(startSample, endSample));

    // Soft Hann fade to eliminate clicks
    const fadeLen = Math.min(Math.floor(0.02 * sampleRate), Math.floor(trimmed.length / 4));
    for (let i = 0; i < fadeLen; i++) {
      const ramp = 0.5 * (1.0 - Math.cos((Math.PI * i) / Math.max(1, fadeLen)));
      trimmed[i] *= ramp;
      trimmed[trimmed.length - 1 - i] *= ramp;
    }

    return trimmed;
  }

  getLastDistance(keyword) {
    return this.lastDistances ? this.lastDistances[keyword] : undefined;
  }

  enrollKeyword(name, features, threshold = 0.60) {
    if (!this.lastDistances) this.lastDistances = {};
    this.templates.push({
      name: String(name),
      features: features,
      threshold: Number(threshold),
    });
  }

  ingestSamples(samples) {
    if (!this.lastDistances) this.lastDistances = {};

    // Real-time DC blocker filter
    const dcR = this.dcR;
    let xPrev = this.dcXPrev;
    let yPrev = this.dcYPrev;
    for (let i = 0; i < samples.length; i++) {
      const x = samples[i];
      const y = x - xPrev + dcR * yPrev;
      this.buffer.push(y);
      xPrev = x;
      yPrev = y;
    }
    this.dcXPrev = xPrev;
    this.dcYPrev = yPrev;

    this.totalSamples += samples.length;
    const detections = [];

    const maxTpl = this.templates.reduce((acc, t) => Math.max(acc, t.features.length), 50);
    const maxHist = Math.max(250, maxTpl + 60);

    while (this.buffer.length >= this.frameSize) {
      const chunk = this.buffer.slice(0, this.frameSize);
      this.buffer = this.buffer.slice(this.hopSize);

      const feats = this.extractFeatures(chunk);
      if (feats.length > 0) {
        for (const f of feats) {
          this.history.push(f);
        }
        if (this.history.length > maxHist) {
          this.history = this.history.slice(-maxHist);
        }

        for (const tpl of this.templates) {
          const tplLen = tpl.features.length;
          const band = Math.max(4, Math.floor(tplLen / 4));
          let bestDist = Infinity;

          for (const candLen of [tplLen, tplLen - 2, tplLen + 2]) {
            if (candLen > 0 && this.history.length >= candLen) {
              const obs = this.history.slice(-candLen);
              const dist = this._sakoeChibaDistance(obs, tpl.features, band);
              if (dist < bestDist) {
                bestDist = dist;
              }
            }
          }

          this.lastDistances[tpl.name] = bestDist;

          if (bestDist < tpl.threshold) {
            const conf = Math.max(0.0, Math.min(1.0, 1.0 - bestDist / tpl.threshold));
            const ts = this.totalSamples / this.sampleRate;
            detections.push({
              keyword: tpl.name,
              confidence: conf,
              timestampSec: ts,
            });
          }
        }
      }
    }

    return detections;
  }

  _sakoeChibaDistance(s, t, r) {
    const n = s.length;
    const m = t.length;
    if (n === 0 || m === 0 || Math.abs(n - m) > r) return Infinity;

    const cost = Array.from({ length: n + 1 }, () => new Float32Array(m + 1).fill(Infinity));
    cost[0][0] = 0.0;

    for (let i = 1; i <= n; i++) {
      const jMin = Math.max(1, i - r);
      const jMax = Math.min(m, i + r);

      for (let j = jMin; j <= jMax; j++) {
        let diffSq = 0.0;
        for (let f = 0; f < this.numMfcc; f++) {
          const diff = s[i - 1][f] - t[j - 1][f];
          diffSq += diff * diff;
        }
        const d = Math.sqrt(diffSq);
        const minPrev = Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1]);
        cost[i][j] = d + minPrev;
      }
    }

    if (!isFinite(cost[n][m])) return Infinity;
    return cost[n][m] / Math.max(1, n + m);
  }
}

module.exports = {
  SononEngine,
  SononShm,
  SHM_MAGIC,
  SHM_VERSION,
  DEFAULT_SHM_PATH,
};
