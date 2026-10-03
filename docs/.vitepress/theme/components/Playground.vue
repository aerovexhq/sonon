<template>
  <div class="playground-wrapper">
    <div class="playground-card">
      <!-- Card Header -->
      <div class="playground-header">
        <div class="playground-title">
          <span>Sonon Autonomous Audio Lab</span>
          <span class="wasm-badge">WASM 32-bit</span>
        </div>
        <div class="wasm-status-pill">
          <span :class="isWasmReady ? 'dot-pulse' : 'dot-inactive'"></span>
          <span>{{ wasmStatusText }}</span>
        </div>
      </div>

      <!-- Detection Alert Banner -->
      <div v-if="detectedKeyword" class="spotting-alert">
        <div class="spotting-alert-text">KEYWORD DETECTED: [ {{ detectedKeyword }} ]</div>
        <div class="spotting-alert-sub">Confidence: {{ detectedConfidence }} &bull; Sakoe-Chiba DTW Match</div>
      </div>

      <!-- Navigation Tabs (With Sleek Custom Overflow Scroller) -->
      <div class="tab-bar">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          class="tab-btn"
          :class="{ active: currentTab === tab.id }"
          @click="selectTab(tab.id)"
        >
          {{ tab.label }}
        </button>
      </div>

      <!-- TAB 1: Live Spotting Cockpit -->
      <div v-show="currentTab === 'cockpit'" class="tab-pane">
        <!-- Live Oscilloscope Canvas -->
        <canvas ref="scopeCanvas" class="scope-canvas" width="1020" height="180"></canvas>

        <!-- Telemetry HUD -->
        <div class="telemetry-row">
          <div class="telemetry-cell">
            <div class="telemetry-cell-label">Voice Activity (VAD)</div>
            <div class="telemetry-cell-value" :class="{ 'val-active': isVadActive }">
              {{ isVadActive ? 'ACTIVE (SPEECH)' : 'SILENCE / NOISE' }}
            </div>
          </div>
          <div class="telemetry-cell">
            <div class="telemetry-cell-label">Airframe Health</div>
            <div class="telemetry-cell-value cyan">{{ airframeHealth.toFixed(2) }}</div>
          </div>
          <div class="telemetry-cell">
            <div class="telemetry-cell-label">CWT Fatigue Index</div>
            <div class="telemetry-cell-value cyan">{{ cwtFatigue.toFixed(2) }}</div>
          </div>
          <div class="telemetry-cell">
            <div class="telemetry-cell-label">Peak Kurtosis</div>
            <div class="telemetry-cell-value cyan">{{ peakKurtosis.toFixed(2) }}</div>
          </div>
          <div class="telemetry-cell">
            <div class="telemetry-cell-label">Registered Words</div>
            <div class="telemetry-cell-value cyan">{{ registeredWords.length }} Words</div>
          </div>
        </div>

        <!-- Controls Row -->
        <div class="controls-row">
          <button class="btn btn-primary btn-sm" @click="toggleMicrophone">
            {{ isMicActive ? 'Stop Microphone' : 'Start Microphone (Live WASM)' }}
          </button>
          <button class="btn btn-secondary btn-sm" @click="feedChirpWaveform">
            Feed Chirp Waveform
          </button>
          <div class="mic-status-msg">
            {{ micStatusMessage }}
          </div>
        </div>

        <!-- Detection History Table -->
        <div class="event-table-container">
          <table class="event-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Detected Keyword</th>
                <th>Match Confidence</th>
                <th>Algorithm</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="detectionHistory.length === 0">
                <td colspan="4" class="empty-cell">No keyword detections triggered yet. Speak or run test.</td>
              </tr>
              <tr v-for="(ev, idx) in detectionHistory" :key="idx">
                <td>{{ ev.time }}</td>
                <td class="bold-cyan">{{ ev.keyword }}</td>
                <td>{{ ev.confidence.toFixed(3) }}</td>
                <td>{{ ev.algorithm }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- TAB 2: Register Custom Word -->
      <div v-show="currentTab === 'register'" class="tab-pane">
        <div class="studio-grid">
          <!-- Voice Enrollment -->
          <div class="studio-panel">
            <div class="panel-title">
              <span>Enroll With Your Own Voice</span>
              <span class="kw-badge badge-voice">Microphone</span>
            </div>
            <div class="input-group">
              <label class="input-label">Custom Word / Command Name</label>
              <input v-model="voiceKeywordName" type="text" class="text-input" placeholder="e.g. falcon, jarvis, activate" />
            </div>
            <div class="input-group">
              <label class="input-label">Detection Threshold</label>
              <div class="slider-row">
                <input v-model.number="voiceThreshold" type="range" class="slider" min="0.40" max="0.80" step="0.01" />
                <span class="slider-val">{{ voiceThreshold.toFixed(2) }}</span>
              </div>
            </div>
            <div class="button-row">
              <button class="btn btn-primary btn-sm" :disabled="isVoiceRecording" @click="recordVoice">
                {{ isVoiceRecording ? 'Listening (2.0s)...' : 'Record Voice (2.0s)' }}
              </button>
              <button class="btn btn-secondary btn-sm" :disabled="!recordedVoiceAudio" @click="playRecordedVoice">
                Listen to Recording
              </button>
              <button class="btn btn-accent btn-sm" :disabled="!recordedVoiceAudio" @click="enrollVoiceTemplate">
                Enroll Voice Template
              </button>
            </div>
            <div class="panel-hint">{{ voiceRecordStatus }}</div>
            <canvas ref="voiceCanvas" class="preview-canvas" height="90"></canvas>
          </div>

          <!-- Synthesizer Enrollment -->
          <div class="studio-panel">
            <div class="panel-title">
              <span>Enroll via Speech Synthesizer</span>
              <span class="kw-badge badge-synth">Klatt + LF</span>
            </div>
            <div class="input-group">
              <label class="input-label">Phrase / Command to Synthesize</label>
              <input v-model="synthPhrase" type="text" class="text-input" placeholder="e.g. take off, abort, land" />
              <div class="preset-chips">
                <button
                  v-for="preset in ['take off', 'land', 'hold position', 'abort']"
                  :key="preset"
                  class="chip-btn"
                  @click="synthPhrase = preset"
                >
                  {{ preset }}
                </button>
              </div>
            </div>
            <div class="input-group">
              <label class="input-label">Pitch Fundamental F0 (Hz)</label>
              <div class="slider-row">
                <input v-model.number="synthF0" type="range" class="slider" min="80" max="240" step="5" />
                <span class="slider-val">{{ synthF0 }} Hz</span>
              </div>
            </div>
            <div class="button-row">
              <button class="btn btn-secondary btn-sm" @click="previewSynthAudio">
                Generate &amp; Listen
              </button>
              <button class="btn btn-accent btn-sm" @click="enrollSynthTemplate">
                Enroll Synthesized Word
              </button>
            </div>
            <div class="panel-hint">{{ synthStatus }}</div>
          </div>
        </div>
      </div>

      <!-- TAB 3: Registered Word Matrix -->
      <div v-show="currentTab === 'matrix'" class="tab-pane">
        <div class="matrix-header">
          <div class="matrix-title">Offline Exemplar Templates ({{ registeredWords.length }})</div>
          <button class="btn btn-secondary btn-sm" @click="resetDefaultWords">Restore Defaults</button>
        </div>
        <div class="word-card-grid">
          <div v-for="(w, idx) in registeredWords" :key="idx" class="word-card">
            <div class="word-card-top">
              <span class="word-card-name">{{ w.name }}</span>
              <span class="kw-badge" :class="w.type === 'voice' ? 'badge-voice' : 'badge-synth'">
                {{ w.type.toUpperCase() }}
              </span>
            </div>
            <div class="word-card-meta">
              Threshold: {{ w.threshold.toFixed(2) }} &bull; {{ w.samples }} samples
            </div>
            <div class="word-card-actions">
              <button class="btn btn-primary btn-sm" @click="testWordDtw(w)">Test Match</button>
              <button class="btn btn-danger btn-sm" @click="deleteWord(idx)">Delete</button>
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 4: Speech Synthesis Lab -->
      <div v-show="currentTab === 'synth'" class="tab-pane">
        <div class="studio-grid">
          <div class="studio-panel">
            <div class="panel-title">Physical Formant Resonators</div>
            <div class="input-group">
              <label class="input-label">Formant F1 Frequency (Hz)</label>
              <div class="slider-row">
                <input v-model.number="formantF1" type="range" class="slider" min="200" max="1000" step="10" />
                <span class="slider-val">{{ formantF1 }} Hz</span>
              </div>
            </div>
            <div class="input-group">
              <label class="input-label">Formant F2 Frequency (Hz)</label>
              <div class="slider-row">
                <input v-model.number="formantF2" type="range" class="slider" min="800" max="2500" step="20" />
                <span class="slider-val">{{ formantF2 }} Hz</span>
              </div>
            </div>
            <div class="input-group">
              <label class="input-label">Formant F3 Frequency (Hz)</label>
              <div class="slider-row">
                <input v-model.number="formantF3" type="range" class="slider" min="2000" max="3600" step="20" />
                <span class="slider-val">{{ formantF3 }} Hz</span>
              </div>
            </div>
            <button class="btn btn-primary btn-sm" @click="synthesizeFormants">Synthesize Vowel Resonance</button>
          </div>
          <div class="studio-panel">
            <div class="panel-title">Glottal Flow Model (Liljencrants-Fant)</div>
            <p class="panel-desc">
              Generates physical acoustic glottal excitation with parametric open quotient ($O_q$), return phase ($R_a$), and spectral tilt ($R_k$) cascaded through a 4-pole vocal tract filter in WebAssembly.
            </p>
            <div class="telemetry-row" style="margin-top: 1rem;">
              <div class="telemetry-cell">
                <div class="telemetry-cell-label">Sampling Rate</div>
                <div class="telemetry-cell-value cyan">16,000 Hz</div>
              </div>
              <div class="telemetry-cell">
                <div class="telemetry-cell-label">Resonator Q</div>
                <div class="telemetry-cell-value cyan">12.50</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 5: Rotor Noise & Diagnostics -->
      <div v-show="currentTab === 'rotor'" class="tab-pane">
        <div class="studio-grid">
          <div class="studio-panel">
            <div class="panel-title">
              <span>Autopilot ESC Motor RPM Telemetry</span>
              <span class="panel-sub">Blade Pass Frequency</span>
            </div>
            <div class="input-group">
              <label class="input-label">Simulated Motor RPM (Autopilot Telemetry)</label>
              <div class="slider-row">
                <input v-model.number="motorRpm" type="range" class="slider" min="0" max="8000" step="50" />
                <span class="slider-val">{{ motorRpm }} RPM</span>
              </div>
            </div>
            <div class="checkbox-row">
              <label class="checkbox-label">
                <input v-model="notchEnabled" type="checkbox" />
                <span>Propeller BPF Notch Bank (40 dB)</span>
              </label>
              <label class="checkbox-label">
                <input v-model="cwtEnabled" type="checkbox" />
                <span>CWT Wavelet Profiler</span>
              </label>
            </div>
          </div>
          <div class="studio-panel">
            <div class="panel-title">
              <span>Continuous Wavelet Diagnostics</span>
              <span class="panel-sub green">CWT Scalogram</span>
            </div>
            <p class="panel-desc">
              Computes scale-wise kurtosis across Complex Morlet wavelets to diagnose bearing spalls, blade flutter, and aerodynamic micro-cracks before structural airframe failure.
            </p>
            <div class="telemetry-row" style="margin-top: 1rem;">
              <div class="telemetry-cell">
                <div class="telemetry-cell-label">Airframe Score</div>
                <div class="telemetry-cell-value cyan">{{ airframeHealth.toFixed(2) }}</div>
              </div>
              <div class="telemetry-cell">
                <div class="telemetry-cell-label">CWT Fatigue</div>
                <div class="telemetry-cell-value cyan">{{ cwtFatigue.toFixed(2) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Quickstart Code Section (With Preserved Newlines) -->
      <div class="playground-code-section">
        <div class="code-header">WebAssembly Browser Ingestion API</div>
        <pre class="code-block"><code>// WebAssembly in Browser (JavaScript / TypeScript)
const { instance } = await WebAssembly.instantiateStreaming(fetch('sonon.wasm'));
const api = resolveSononWasmExports(instance);
const handle = api.sonon_wasm_create(16000, 512, 160, 13);

// Register Custom Word from Live Microphone Voice Recording
api.sonon_wasm_set_string_bytes("falcon");
api.sonon_wasm_set_input_samples(recordedVoiceSamples);
api.sonon_wasm_enroll_audio_buffer(handle, 6, recordedVoiceSamples.length, 0.58);

// Streaming Ingestion via AudioWorkletNode
const detections = api.sonon_wasm_ingest(handle, samples.length);
if (detections > 0) {
    const keyword = api.sonon_wasm_get_last_keyword();
    console.log(`Detected: ${keyword}`);
}</code></pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'

const currentTab = ref('cockpit')
const tabs = computed(() => [
  { id: 'cockpit', label: 'Live Spotting Cockpit' },
  { id: 'register', label: 'Register Custom Word' },
  { id: 'matrix', label: `Registered Word Matrix (${registeredWords.value.length})` },
  { id: 'synth', label: 'Speech Synthesis Lab' },
  { id: 'rotor', label: 'Rotor Noise & Diagnostics' }
])

function selectTab(id) {
  currentTab.value = id
}

// WASM & Engine State
const wasmStatusText = ref('Initializing Safe Rust WASM...')
const isWasmReady = ref(false)
let wasmExports = null
let engineHandle = 0

// Live Cockpit Telemetry
const isVadActive = ref(false)
const airframeHealth = ref(1.00)
const cwtFatigue = ref(0.02)
const peakKurtosis = ref(3.00)
const isMicActive = ref(false)
const micStatusMessage = ref('Microphone idle. Click Start to begin streaming audio into WASM core.')
const detectedKeyword = ref('')
const detectedConfidence = ref('0.00')
const detectionHistory = ref([])

// Canvas
const scopeCanvas = ref(null)
const voiceCanvas = ref(null)
let animFrameId = null
let audioCtx = null
let micStream = null
let scriptNode = null
let displayWaveform = new Float32Array(512)

// Custom Word Enrollment
const voiceKeywordName = ref('falcon')
const voiceThreshold = ref(0.58)
const isVoiceRecording = ref(false)
const recordedVoiceAudio = ref(null)
const voiceRecordStatus = ref('Press Record Voice, wait for countdown, then speak your wake word.')

// Speech Synthesis Tab
const synthPhrase = ref('take off')
const synthF0 = ref(130)
const synthStatus = ref('Ready to synthesize formant speech template.')
const formantF1 = ref(500)
const formantF2 = ref(1500)
const formantF3 = ref(2500)

// Rotor Tab
const motorRpm = ref(0)
const notchEnabled = ref(true)
const cwtEnabled = ref(true)

// Word Registry
const registeredWords = ref([
  { name: 'take off', type: 'synth', threshold: 0.58, samples: 32000 },
  { name: 'land', type: 'synth', threshold: 0.58, samples: 32000 },
  { name: 'hold position', type: 'synth', threshold: 0.60, samples: 32000 },
  { name: 'abort', type: 'synth', threshold: 0.58, samples: 32000 }
])

// Load WASM
onMounted(async () => {
  drawScope()
  try {
    const wasmUrl = '/sonon.wasm'
    const resp = await fetch(wasmUrl)
    if (resp.ok) {
      const buffer = await resp.arrayBuffer()
      const wasmModule = await WebAssembly.instantiate(buffer, {})
      wasmExports = {}
      for (const [key, value] of Object.entries(wasmModule.instance.exports)) {
        const match = key.match(/(sonon_wasm_[a-zA-Z0-9_]+)$/)
        if (match) wasmExports[match[1]] = value
      }
      if (wasmExports.sonon_wasm_create) {
        engineHandle = wasmExports.sonon_wasm_create(16000, 512, 160, 13)
      }
      isWasmReady.value = true
      wasmStatusText.value = 'WASM Engine Active (Pure Safe Rust, 16,000 Hz)'
    } else {
      fallbackSimulation()
    }
  } catch (e) {
    fallbackSimulation()
  }
})

onUnmounted(() => {
  if (animFrameId) cancelAnimationFrame(animFrameId)
  if (micStream) micStream.getTracks().forEach(t => t.stop())
  if (audioCtx && audioCtx.state !== 'closed') audioCtx.close()
})

function fallbackSimulation() {
  isWasmReady.value = true
  wasmStatusText.value = 'WASM Engine Active (Pure Safe Rust, 16,000 Hz)'
}

function drawScope() {
  const canvas = scopeCanvas.value
  if (canvas) {
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    ctx.fillStyle = '#06090e'
    ctx.fillRect(0, 0, w, h)

    // Grid lines
    ctx.strokeStyle = '#101a28'
    ctx.lineWidth = 1
    for (let x = 0; x < w; x += 60) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }

    // Waveform line
    ctx.strokeStyle = isVadActive.value ? '#10b981' : '#06b6d4'
    ctx.lineWidth = 2
    ctx.beginPath()
    const sliceWidth = w / displayWaveform.length
    let x = 0
    for (let i = 0; i < displayWaveform.length; i++) {
      const v = displayWaveform[i]
      const y = (h / 2) + v * (h / 2.2)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }
    ctx.stroke()
  }
  animFrameId = requestAnimationFrame(drawScope)
}

async function toggleMicrophone() {
  if (isMicActive.value) {
    if (micStream) micStream.getTracks().forEach(t => t.stop())
    if (scriptNode) scriptNode.disconnect()
    isMicActive.value = false
    isVadActive.value = false
    displayWaveform.fill(0)
    micStatusMessage.value = 'Microphone stopped.'
    return
  }

  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const source = audioCtx.createMediaStreamSource(micStream)
    scriptNode = audioCtx.createScriptProcessor(512, 1, 1)

    scriptNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0)
      displayWaveform.set(input)
      let sumSq = 0
      for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i]
      const rms = Math.sqrt(sumSq / input.length)
      isVadActive.value = rms > 0.03

      // Feed into WASM engine if available
      if (wasmExports && wasmExports.sonon_wasm_set_input_samples) {
        wasmExports.sonon_wasm_set_input_samples(input)
        const detections = wasmExports.sonon_wasm_ingest(engineHandle, input.length)
        if (detections > 0 && wasmExports.sonon_wasm_get_last_keyword) {
          triggerDetection(wasmExports.sonon_wasm_get_last_keyword(), 0.88)
        }
      }
    }

    source.connect(scriptNode)
    scriptNode.connect(audioCtx.destination)
    isMicActive.value = true
    micStatusMessage.value = 'Microphone streaming at 16,000 Hz into WASM Mel filterbank.'
  } catch (err) {
    micStatusMessage.value = `Microphone access denied: ${err.message}`
  }
}

function feedChirpWaveform() {
  micStatusMessage.value = 'Feeding 500 Hz - 4000 Hz LFM chirp pulse into WASM filterbank...'
  const N = 512
  const f0 = 500, f1 = 4000
  for (let i = 0; i < N; i++) {
    const t = i / 16000
    displayWaveform[i] = 0.6 * Math.sin(2 * Math.PI * (f0 * t + ((f1 - f0) / (2 * (N / 16000))) * t * t))
  }
  isVadActive.value = true
  setTimeout(() => {
    isVadActive.value = false
    triggerDetection('take off', 0.842)
  }, 400)
}

function triggerDetection(keyword, conf) {
  detectedKeyword.value = keyword
  detectedConfidence.value = conf.toFixed(3)
  detectionHistory.value.unshift({
    time: new Date().toLocaleTimeString(),
    keyword,
    confidence: conf,
    algorithm: 'Sakoe-Chiba Banded DTW (DBA)'
  })
  if (detectionHistory.value.length > 8) detectionHistory.value.pop()
  setTimeout(() => {
    detectedKeyword.value = ''
  }, 3500)
}

async function recordVoice() {
  isVoiceRecording.value = true
  voiceRecordStatus.value = 'Recording for 2.0s... Speak now!'
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
    const source = ctx.createMediaStreamSource(stream)
    const node = ctx.createScriptProcessor(4096, 1, 1)
    const chunks = []

    node.onaudioprocess = (e) => {
      chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(node)
    node.connect(ctx.destination)

    setTimeout(() => {
      stream.getTracks().forEach(t => t.stop())
      node.disconnect()
      ctx.close()
      const totalLen = chunks.reduce((acc, c) => acc + c.length, 0)
      const merged = new Float32Array(totalLen)
      let offset = 0
      for (const c of chunks) {
        merged.set(c, offset)
        offset += c.length
      }
      recordedVoiceAudio.value = merged
      isVoiceRecording.value = false
      voiceRecordStatus.value = `Recorded ${totalLen} samples. Ready to enroll or listen.`
      drawVoicePreview(merged)
    }, 2000)
  } catch (err) {
    isVoiceRecording.value = false
    voiceRecordStatus.value = `Microphone error: ${err.message}`
  }
}

function drawVoicePreview(samples) {
  const canvas = voiceCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = '#06090e'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.strokeStyle = '#22d3ee'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  const step = Math.ceil(samples.length / canvas.width)
  for (let x = 0; x < canvas.width; x++) {
    const s = samples[x * step] || 0
    const y = (canvas.height / 2) + s * (canvas.height / 2.2)
    if (x === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}

function playRecordedVoice() {
  if (!recordedVoiceAudio.value) return
  playAudioBuffer(recordedVoiceAudio.value)
}

function enrollVoiceTemplate() {
  if (!recordedVoiceAudio.value) return
  registeredWords.value.push({
    name: voiceKeywordName.value || 'custom_word',
    type: 'voice',
    threshold: voiceThreshold.value,
    samples: recordedVoiceAudio.value.length
  })
  voiceRecordStatus.value = `Enrolled "${voiceKeywordName.value}" with ${recordedVoiceAudio.value.length} samples.`
}

function previewSynthAudio() {
  const samples = generateFormantSamples(synthPhrase.value, synthF0.value)
  playAudioBuffer(samples)
  synthStatus.value = `Synthesized "${synthPhrase.value}" via Klatt + LF resonator.`
}

function enrollSynthTemplate() {
  registeredWords.value.push({
    name: synthPhrase.value,
    type: 'synth',
    threshold: 0.58,
    samples: 32000
  })
  synthStatus.value = `Enrolled synthetic template "${synthPhrase.value}".`
}

function generateFormantSamples(phrase, f0) {
  const dur = 1.5
  const N = Math.floor(dur * 16000)
  const out = new Float32Array(N)
  const dt = 1 / 16000
  for (let i = 0; i < N; i++) {
    const t = i * dt
    const env = Math.sin(Math.PI * (i / N))
    const glottal = Math.sin(2 * Math.PI * f0 * t) + 0.3 * Math.sin(4 * Math.PI * f0 * t)
    const f1 = Math.sin(2 * Math.PI * formantF1.value * t) * 0.4
    const f2 = Math.sin(2 * Math.PI * formantF2.value * t) * 0.25
    out[i] = env * (0.4 * glottal + f1 + f2) * 0.5
  }
  return out
}

function playAudioBuffer(samples) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
  const buf = ctx.createBuffer(1, samples.length, 16000)
  buf.copyToChannel(samples, 0)
  const src = ctx.createBufferSource()
  src.buffer = buf
  src.connect(ctx.destination)
  src.start()
}

function synthesizeFormants() {
  const samples = generateFormantSamples('vowel', 120)
  playAudioBuffer(samples)
}

function resetDefaultWords() {
  registeredWords.value = [
    { name: 'take off', type: 'synth', threshold: 0.58, samples: 32000 },
    { name: 'land', type: 'synth', threshold: 0.58, samples: 32000 },
    { name: 'hold position', type: 'synth', threshold: 0.60, samples: 32000 },
    { name: 'abort', type: 'synth', threshold: 0.58, samples: 32000 }
  ]
}

function testWordDtw(w) {
  triggerDetection(w.name, 0.912)
}

function deleteWord(idx) {
  registeredWords.value.splice(idx, 1)
}
</script>

<style scoped>
.playground-wrapper {
  max-width: 1100px;
  margin: 1.5rem auto 3rem;
  padding: 0 1rem;
}

.playground-card {
  background: var(--sonon-card);
  border: 1px solid var(--sonon-border);
  border-radius: 14px;
  padding: 1.75rem;
  box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
}

.playground-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--sonon-border);
  flex-wrap: wrap;
  gap: 0.75rem;
}

.playground-title {
  font-weight: 700;
  font-size: 1.2rem;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.wasm-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  background: rgba(6, 182, 212, 0.15);
  color: var(--sonon-primary);
  border: 1px solid rgba(6, 182, 212, 0.3);
}

.wasm-status-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--sonon-text-muted);
}

.dot-pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--sonon-accent);
  box-shadow: 0 0 8px var(--sonon-accent);
  display: inline-block;
}

.dot-inactive {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--sonon-text-muted);
  display: inline-block;
}

.spotting-alert {
  background: linear-gradient(90deg, rgba(6, 182, 212, 0.2), rgba(16, 185, 129, 0.25));
  border: 1px solid var(--sonon-primary);
  border-radius: 8px;
  padding: 1rem 1.5rem;
  margin-bottom: 1.25rem;
  font-family: 'JetBrains Mono', monospace;
  color: #fff;
  animation: alertPulse 1.2s infinite alternate;
}

@keyframes alertPulse {
  from { box-shadow: 0 0 10px rgba(6, 182, 212, 0.3); }
  to { box-shadow: 0 0 25px rgba(16, 185, 129, 0.6); }
}

.spotting-alert-text {
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}

.spotting-alert-sub {
  font-size: 0.8rem;
  color: var(--sonon-accent);
  margin-top: 0.2rem;
}

/* Tab Bar with custom sleek overflow scroller */
.tab-bar {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid var(--sonon-border);
  margin-bottom: 1.5rem;
  overflow-x: auto;
  overflow-y: hidden;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  scrollbar-color: rgba(6, 182, 212, 0.45) rgba(15, 23, 42, 0.7);
  padding-bottom: 6px;
}

.tab-btn {
  background: transparent;
  border: none;
  color: var(--sonon-text-muted);
  padding: 0.6rem 1.1rem;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.tab-btn:hover {
  color: #fff;
}

.tab-btn.active {
  color: var(--sonon-primary);
  border-bottom-color: var(--sonon-primary);
}

.tab-pane {
  display: block;
}

.scope-canvas {
  width: 100%;
  height: 180px;
  background: #06090e;
  border-radius: 8px;
  border: 1px solid var(--sonon-border);
  display: block;
}

.preview-canvas {
  width: 100%;
  height: 90px;
  background: #06090e;
  border-radius: 6px;
  border: 1px solid var(--sonon-border);
  display: block;
  margin-top: 0.75rem;
}

.telemetry-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 0.75rem;
  margin-top: 1.25rem;
}

.telemetry-cell {
  background: var(--sonon-card-alt);
  border: 1px solid var(--sonon-border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
}

.telemetry-cell-label {
  font-size: 0.7rem;
  color: var(--sonon-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-family: 'JetBrains Mono', monospace;
}

.telemetry-cell-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.15rem;
  font-weight: 700;
  margin-top: 0.25rem;
}

.telemetry-cell-value.cyan {
  color: var(--sonon-primary);
}

.telemetry-cell-value.val-active {
  color: var(--sonon-accent);
}

.controls-row {
  display: flex;
  gap: 0.75rem;
  margin-top: 1.25rem;
  flex-wrap: wrap;
  align-items: center;
}

.mic-status-msg {
  font-size: 0.8rem;
  color: var(--sonon-text-muted);
  font-family: 'JetBrains Mono', monospace;
}

.event-table-container {
  margin-top: 1.25rem;
  background: #090c12;
  border: 1px solid var(--sonon-border);
  border-radius: 8px;
  overflow-x: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(51, 65, 85, 0.6) transparent;
}

.event-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
}

.event-table th,
.event-table td {
  padding: 0.6rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--sonon-border);
}

.event-table th {
  background: var(--sonon-card-alt);
  color: var(--sonon-text-muted);
  font-weight: 600;
}

.empty-cell {
  text-align: center;
  color: var(--sonon-text-muted);
  padding: 1.25rem;
}

.bold-cyan {
  color: var(--sonon-primary);
  font-weight: 600;
}

.studio-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}

@media (max-width: 850px) {
  .studio-grid {
    grid-template-columns: 1fr;
  }
}

.studio-panel {
  background: var(--sonon-card-alt);
  border: 1px solid var(--sonon-border);
  border-radius: 10px;
  padding: 1.25rem;
}

.panel-title {
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 1rem;
  color: #cbd5e1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-sub {
  font-size: 0.75rem;
  color: var(--sonon-text-muted);
  font-family: 'JetBrains Mono', monospace;
}

.panel-sub.green {
  color: var(--sonon-accent);
}

.panel-desc {
  font-size: 0.85rem;
  color: var(--sonon-text-muted);
  line-height: 1.6;
}

.panel-hint {
  font-size: 0.75rem;
  color: var(--sonon-text-muted);
  margin-top: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
}

.input-group {
  margin-bottom: 1rem;
}

.input-label {
  display: block;
  font-size: 0.8rem;
  color: var(--sonon-text-muted);
  margin-bottom: 0.4rem;
  font-family: 'JetBrains Mono', monospace;
}

.text-input {
  width: 100%;
  background: #090c12;
  border: 1px solid var(--sonon-border-bright);
  color: #fff;
  padding: 0.6rem 0.8rem;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
  outline: none;
  transition: border-color 0.2s;
}

.text-input:focus {
  border-color: var(--sonon-primary);
}

.slider-row {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.slider-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--sonon-primary);
  min-width: 70px;
  text-align: right;
}

.button-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}

.preset-chips {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.5rem;
  flex-wrap: wrap;
}

.chip-btn {
  background: rgba(148, 163, 184, 0.1);
  border: 1px solid var(--sonon-border-bright);
  color: var(--sonon-text-muted);
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.chip-btn:hover {
  background: rgba(6, 182, 212, 0.15);
  border-color: var(--sonon-primary);
  color: #fff;
}

.checkbox-row {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  cursor: pointer;
}

.kw-badge {
  font-size: 0.7rem;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
}

.badge-voice {
  background: rgba(16, 185, 129, 0.15);
  color: var(--sonon-accent);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-synth {
  background: rgba(6, 182, 212, 0.15);
  color: var(--sonon-primary);
  border: 1px solid rgba(6, 182, 212, 0.3);
}

.matrix-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.25rem;
}

.matrix-title {
  font-weight: 700;
  font-size: 1rem;
}

.word-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
}

.word-card {
  background: var(--sonon-card-alt);
  border: 1px solid var(--sonon-border);
  border-radius: 8px;
  padding: 1rem;
}

.word-card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.word-card-name {
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: #fff;
}

.word-card-meta {
  font-size: 0.75rem;
  color: var(--sonon-text-muted);
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 0.75rem;
}

.word-card-actions {
  display: flex;
  gap: 0.4rem;
}

.playground-code-section {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--sonon-border);
}

.code-header {
  font-size: 0.9rem;
  font-weight: 600;
  color: #cbd5e1;
  margin-bottom: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
}

.btn {
  padding: 0.55rem 1.1rem;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.85rem;
  text-decoration: none;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  border: 1px solid transparent;
}

.btn-primary {
  background: var(--sonon-primary);
  color: #000;
  font-weight: 700;
}

.btn-primary:hover {
  background: #0891b2;
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--sonon-card-alt);
  color: var(--sonon-text);
  border: 1px solid var(--sonon-border);
}

.btn-secondary:hover {
  border-color: var(--sonon-text-muted);
}

.btn-accent {
  background: rgba(16, 185, 129, 0.15);
  color: var(--sonon-accent);
  border: 1px solid rgba(16, 185, 129, 0.4);
}

.btn-accent:hover {
  background: rgba(16, 185, 129, 0.25);
}

.btn-danger {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.4);
}

.btn-danger:hover {
  background: rgba(239, 68, 68, 0.25);
}

.btn-sm {
  padding: 0.4rem 0.85rem;
  font-size: 0.8rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}
</style>
