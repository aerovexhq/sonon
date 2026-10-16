<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { SUBSYSTEMS, BENCHMARK_COMPARISONS, type Subsystem } from '../subsystems'

// Subsystem Modal state
const selectedSubsystem = ref<Subsystem | null>(null)
const copiedSnippet = ref(false)

// Domain filter tabs for Bento Grid
type FilterKey = 'all' | 'cancellation' | 'kws' | 'synth'
const activeFilter = ref<FilterKey>('all')

const filteredSubsystems = computed(() => {
  if (activeFilter.value === 'cancellation') {
    return SUBSYSTEMS.filter(s => s.id === 'rotor-notch' || s.id === 'spatial-beamforming')
  }
  if (activeFilter.value === 'kws') {
    return SUBSYSTEMS.filter(s => s.id === 'wake-word' || s.id === 'streaming-vad')
  }
  if (activeFilter.value === 'synth') {
    return SUBSYSTEMS.filter(s => s.id === 'vocal-tract')
  }
  return SUBSYSTEMS
})

// Ecosystem dropdown state
const ecosystemOpen = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)

function toggleEcosystem() {
  ecosystemOpen.value = !ecosystemOpen.value
}

function closeEcosystem() {
  ecosystemOpen.value = false
}

// Mouse-tracking spotlight for Bento Grid cards
const cardRefs = ref<HTMLElement[]>([])

function setCardRef(el: any, index: number) {
  if (el) cardRefs.value[index] = el
}

function handleMouseMove(e: MouseEvent, index: number) {
  const card = cardRefs.value[index]
  if (!card) return
  const rect = card.getBoundingClientRect()
  const x = e.clientX - rect.left
  const y = e.clientY - rect.top
  card.style.setProperty('--mouse-x', `${x}px`)
  card.style.setProperty('--mouse-y', `${y}px`)
  card.style.setProperty('--spotlight-opacity', '1')
}

function handleMouseLeave(index: number) {
  const card = cardRefs.value[index]
  if (!card) return
  card.style.setProperty('--spotlight-opacity', '0')
}

// Architecture Modal Open / Close
function openArchitecture(subsystem: Subsystem) {
  selectedSubsystem.value = subsystem
  copiedSnippet.value = false
  if (typeof document !== 'undefined') {
    document.body.style.overflow = 'hidden'
  }
}

function closeArchitecture() {
  selectedSubsystem.value = null
  copiedSnippet.value = false
  if (typeof document !== 'undefined') {
    document.body.style.overflow = ''
  }
}

function copySnippetText(code: string) {
  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    navigator.clipboard.writeText(code)
    copiedSnippet.value = true
    setTimeout(() => {
      copiedSnippet.value = false
    }, 2000)
  }
}

// Keydown listener for Modal ESC & Dropdown ESC
function handleGlobalKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    if (selectedSubsystem.value) {
      closeArchitecture()
    }
    if (ecosystemOpen.value) {
      closeEcosystem()
    }
  }
}

function handleClickOutside(e: MouseEvent) {
  if (dropdownRef.value && !dropdownRef.value.contains(e.target as Node)) {
    ecosystemOpen.value = false
  }
}

// Code Tabs in Architecture Pipeline Showcase
type PipelineTabKey = 'notch' | 'kws' | 'synth' | 'wasm'
const activePipelineTab = ref<PipelineTabKey>('notch')
const pipelineCopied = ref(false)

const pipelineSnippets: Record<PipelineTabKey, { filename: string; language: string; code: string }> = {
  notch: {
    filename: 'rotor_notch.rs',
    language: 'rust',
    code: `use sonon_dsp::{DynamicRotorNotch, EscTelemetry, PcmFrame};

pub fn process_rotor_audio(raw_frame: &[f32; 160], telemetry: &EscTelemetry) -> [f32; 160] {
    let mut notch_bank = DynamicRotorNotch::new(48_000 /* Hz */, 4 /* blades */);
    // Lock notch frequencies directly to high-speed ESC motor RPM
    notch_bank.sync_motor_rpm(telemetry.motor_rpm);
    // Attenuate fundamental blade pass frequency & 7 harmonics by >86 dB
    notch_bank.filter_frame(raw_frame)
}`
  },
  kws: {
    filename: 'kws_stream.rs',
    language: 'rust',
    code: `use sonon_kws::{BandedDtw, Mfcc13Extractor, VoiceTemplate};

pub fn evaluate_voice_command(stream_chunk: &[f32; 160], registered_template: &VoiceTemplate) -> bool {
    let mut extractor = Mfcc13Extractor::new(16_000, 512, 160);
    let mfcc_features = extractor.extract(stream_chunk);
    
    // Evaluate Sakoe-Chiba diagonal corridor (R=10) in sub-0.5 ms
    let mut dtw = BandedDtw::with_corridor(10);
    let alignment_cost = dtw.align(&mfcc_features, registered_template);
    alignment_cost < registered_template.threshold()
}`
  },
  synth: {
    filename: 'vocal_tract.rs',
    language: 'rust',
    code: `use sonon_synth::{FormantCascade, LiljencrantsFantGlottis, VocalTractTarget};

pub fn render_tactical_alert(target: &VocalTractTarget, pcm_out: &mut [f32; 1024]) {
    let mut glottis = LiljencrantsFantGlottis::new(target.pitch_f0, target.open_quotient);
    let mut formants = FormantCascade::new(target.formants_hz, target.bandwidths_hz);

    for sample in pcm_out.iter_mut() {
        let glottal_pulse = glottis.tick();
        *sample = formants.tick(glottal_pulse);
    }
}`
  },
  wasm: {
    filename: 'wasm_worklet.ts',
    language: 'typescript',
    code: `// AudioWorkletNode running zero-copy Sonon WebAssembly
class SononWorkletProcessor extends AudioWorkletProcessor {
  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const mic = inputs[0][0];
    const out = outputs[0][0];
    
    // Zero-copy pointer exchange into WASM linear memory
    this.wasm.set_input_samples(mic);
    this.wasm.filter_rotor_telemetry(this.currentRpm);
    this.wasm.get_filtered_stream(out);
    return true;
  }
}`
  }
}

function copyPipelineCode() {
  const item = pipelineSnippets[activePipelineTab.value]
  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    navigator.clipboard.writeText(item.code)
    pipelineCopied.value = true
    setTimeout(() => {
      pipelineCopied.value = false
    }, 2000)
  }
}

// Hero Canvas Interactive Oscilloscope & Live DSP Spectrum
const canvasRef = ref<HTMLCanvasElement | null>(null)
const motorWhineEnabled = ref(true)
const notchFilterActive = ref(true)
let animId: number | null = null

function initHeroCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  let time = 0
  const dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1

  function resize() {
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
  }

  resize()
  if (typeof window !== 'undefined') {
    window.addEventListener('resize', resize)
  }

  function draw() {
    if (!canvas || !ctx) return
    const rect = canvas.getBoundingClientRect()
    const width = rect.width
    const height = rect.height

    ctx.clearRect(0, 0, width, height)

    // Background subtle grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
    ctx.lineWidth = 1
    const gridSize = 32
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Split viewport: Top half = Raw / Filtered Waveforms, Bottom half = FFT Spectrum
    const waveH = height * 0.52
    const specY = waveH + 10
    const specH = height - specY - 14

    // Horizontal split line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)'
    ctx.beginPath()
    ctx.moveTo(16, waveH)
    ctx.lineTo(width - 16, waveH)
    ctx.stroke()

    time += 0.024

    // Simulation parameters: 4-blade rotor @ 5420 RPM -> f0 = 361.3 Hz (scaled for visual display)
    const motorFreq = 0.08
    const voiceFreq1 = 0.018
    const voiceFreq2 = 0.034

    // Waveform 1: Raw Microphone Input (Red / Amber trace)
    ctx.beginPath()
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)'
    ctx.lineWidth = 1.6
    for (let i = 0; i < width - 32; i++) {
      const x = 16 + i
      // Voice component
      const voice = Math.sin((i + time * 60) * voiceFreq1) * 16 + Math.cos((i + time * 40) * voiceFreq2) * 10
      // Motor hum component (strong if motorWhineEnabled)
      const motor = motorWhineEnabled.value
        ? Math.sin((i + time * 120) * motorFreq) * 32 +
          Math.sin((i + time * 120) * motorFreq * 2) * 18 +
          Math.sin((i + time * 120) * motorFreq * 3) * 10
        : 0
      const noise = (Math.random() - 0.5) * 3
      const y = (waveH * 0.3) + voice + motor + noise
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Waveform 2: Filtered Audio Stream (Emerald / Cyan trace)
    ctx.beginPath()
    ctx.strokeStyle = notchFilterActive.value && motorWhineEnabled.value
      ? '#10b981'
      : (notchFilterActive.value ? '#38bdf8' : 'rgba(239, 68, 68, 0.85)')
    ctx.lineWidth = 2.0
    for (let i = 0; i < width - 32; i++) {
      const x = 16 + i
      const voice = Math.sin((i + time * 60) * voiceFreq1) * 18 + Math.cos((i + time * 40) * voiceFreq2) * 12
      // If notch is active, motor hum is attenuated by >86 dB (effectively 0.005)
      const residualMotor = (motorWhineEnabled.value && !notchFilterActive.value)
        ? Math.sin((i + time * 120) * motorFreq) * 32 + Math.sin((i + time * 120) * motorFreq * 2) * 18
        : (motorWhineEnabled.value ? Math.sin((i + time * 120) * motorFreq) * 0.4 : 0)
      const y = (waveH * 0.72) + voice + residualMotor
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Labels for waveforms
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.fillStyle = '#f59e0b'
    ctx.fillText('CH 01 // RAW MIC INPUT (ROTOR NOISE: 88.2 dBA)', 20, 22)

    ctx.fillStyle = notchFilterActive.value ? '#10b981' : '#ef4444'
    const statusText = notchFilterActive.value
      ? 'CH 02 // FILTERED OUTPUT (>86.4 dB NOTCH LOCKED TO 5,420 RPM)'
      : 'CH 02 // NOTCH FILTER BYPASSED (UNFILTERED MOTOR SATURATION)'
    ctx.fillText(statusText, 20, waveH * 0.52 + 16)

    // Bottom Section: Real-Time FFT Spectrum Bar Graph
    ctx.fillStyle = '#a1a1aa'
    ctx.fillText('SPECTRUM ANALYSIS // 0 Hz - 8,000 Hz (NOTCH BANDS: BPF, 2x, 3x, 4x)', 20, specY + 14)

    const numBars = Math.floor((width - 48) / 7)
    const barW = 4
    const barGap = 3

    for (let b = 0; b < numBars; b++) {
      const bx = 20 + b * (barW + barGap)
      const normFreq = b / numBars
      // Base speech spectral envelope (declines with frequency)
      let mag = Math.max(0.08, Math.exp(-normFreq * 2.8) * 0.82)
      mag += Math.sin(b * 0.4 + time * 4) * 0.08

      // Check if frequency aligns with motor harmonic peaks
      const isHarmonic1 = Math.abs(b - Math.floor(numBars * 0.16)) <= 1
      const isHarmonic2 = Math.abs(b - Math.floor(numBars * 0.32)) <= 1
      const isHarmonic3 = Math.abs(b - Math.floor(numBars * 0.48)) <= 1
      const isHarmonic4 = Math.abs(b - Math.floor(numBars * 0.64)) <= 1

      const isMotorSpike = isHarmonic1 || isHarmonic2 || isHarmonic3 || isHarmonic4

      if (isMotorSpike && motorWhineEnabled.value) {
        if (!notchFilterActive.value) {
          // Spike blows out spectrum
          mag = 0.96
        } else {
          // Notch cuts deep hole into the spectrum
          mag = 0.03
        }
      }

      const barH = Math.max(3, mag * (specH - 24))
      const by = specY + specH - barH

      if (isMotorSpike && motorWhineEnabled.value && notchFilterActive.value) {
        // Deep notch indicator
        ctx.fillStyle = '#38bdf8'
      } else if (isMotorSpike && motorWhineEnabled.value && !notchFilterActive.value) {
        ctx.fillStyle = '#ef4444'
      } else {
        ctx.fillStyle = 'rgba(94, 106, 210, 0.55)'
      }

      ctx.fillRect(bx, by, barW, barH)
    }

    animId = requestAnimationFrame(draw)
  }

  draw()
}

onMounted(() => {
  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', handleGlobalKeydown)
    document.addEventListener('click', handleClickOutside)
  }
  initHeroCanvas()
})

onUnmounted(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', handleGlobalKeydown)
    document.removeEventListener('click', handleClickOutside)
  }
  if (animId) {
    cancelAnimationFrame(animId)
  }
})

function getStatusToneClass(tone: string) {
  switch (tone) {
    case 'emerald': return 'sonon-status-emerald'
    case 'cyan': return 'sonon-status-cyan'
    case 'amber': return 'sonon-status-amber'
    default: return 'sonon-status-primary'
  }
}
</script>

<template>
  <div class="sonon-landing-root">
    <!-- Deep Atmospheric Canvas Ambient Glow -->
    <div class="nx-ambient-glow" aria-hidden="true"></div>

    <!-- Fixed Blur Navbar (Standard Aerovex Design) -->
    <header class="nx-navbar">
      <div class="nx-navbar-inner">
        <!-- Brand Mark -->
        <div class="nx-navbar-brand-group">
          <a href="/" class="nx-navbar-brand">
            <svg class="nx-brand-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 10v4" />
              <path d="M6 6v12" />
              <path d="M10 3v18" />
              <path d="M14 7v10" />
              <path d="M18 9v6" />
              <path d="M22 11v2" />
            </svg>
            <span class="nx-brand-title">SONON.</span>
            <span class="nx-brand-badge">DSP</span>
            <span class="nx-telemetry-dot" title="Operational Telemetry Active"></span>
          </a>
        </div>

        <!-- Desktop Navigation Links -->
        <nav class="nx-navbar-links" aria-label="Main Navigation">
          <a href="#subsystems" class="nx-nav-link">Subsystems</a>
          <a href="#pipeline" class="nx-nav-link">Pipeline</a>
          <a href="#benchmarks" class="nx-nav-link">Benchmarks</a>
          <a href="/synthesis" class="nx-nav-link">Voice Studio</a>
          <a href="/playground" class="nx-nav-link">Audio Lab</a>

          <!-- Ecosystem Subdomains Dropdown -->
          <div class="nx-ecosystem-dropdown-wrap" ref="dropdownRef">
            <button
              type="button"
              class="nx-ecosystem-trigger"
              :class="{ 'nx-ecosystem-trigger-open': ecosystemOpen }"
              @click.stop="toggleEcosystem"
              aria-haspopup="true"
              :aria-expanded="ecosystemOpen"
            >
              <span>Ecosystem</span>
              <svg
                class="nx-ecosystem-chevron"
                :class="{ 'nx-ecosystem-chevron-open': ecosystemOpen }"
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            <!-- Dropdown Panel -->
            <transition name="dropdown-fade">
              <div v-if="ecosystemOpen" class="nx-ecosystem-panel" @click.stop>
                <div class="nx-ecosystem-header">
                  <span class="nx-ecosystem-title">Aerovex Architecture</span>
                  <span class="nx-ecosystem-subdomain-note">*.aerovex.net</span>
                </div>

                <div class="nx-ecosystem-items">
                  <a href="https://aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Aerovex Portal
                        <span class="nx-ecosystem-item-tag">Operations</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">Unified operations portal, multi-world sim &amp; fleet management</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>portal</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>

                  <a href="https://axiom.aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Axiom
                        <span class="nx-ecosystem-item-tag">Silicon HDL</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">In-RAM Verilog/SystemVerilog simulator &amp; Cranelift JIT compiler</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>axiom</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>

                  <a href="https://phonon.aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Phonon
                        <span class="nx-ecosystem-item-tag">Semiconductor TCAD</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">Universal electro-thermal SPICE &amp; 4.2K Cryo-CMOS simulator</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>phonon</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>

                  <a href="https://atoma.aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Atoma
                        <span class="nx-ecosystem-item-tag">Quantum &amp; CAD</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">3D atomistic studio, 14 Bravais crystal lattices &amp; Dirac cones</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>atoma</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>

                  <div class="nx-ecosystem-item nx-ecosystem-item-active">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Sonon
                        <span class="nx-ecosystem-item-badge">Active</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">Rotor noise notch filtering (&gt;86 dB) &amp; articulatory speech</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>sonon</span>
                    </div>
                  </div>

                  <a href="https://vexel.aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Vexel Language
                        <span class="nx-ecosystem-item-tag">Compiler</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">Statically-typed systems language with LLVM 18 &amp; borrow checker</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>vexel</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>

                  <a href="https://appify.aerovex.net" target="_blank" rel="noopener noreferrer" class="nx-ecosystem-item">
                    <div class="nx-ecosystem-item-left">
                      <div class="nx-ecosystem-item-name">
                        Appify
                        <span class="nx-ecosystem-item-tag">Desktop Bundler</span>
                      </div>
                      <div class="nx-ecosystem-item-desc">Modern web-to-native desktop application packager (Tauri 2.0)</div>
                    </div>
                    <div class="nx-ecosystem-subdomain-pill">
                      <span>appify</span>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </div>
                  </a>
                </div>
              </div>
            </transition>
          </div>
        </nav>

        <!-- Navbar Right Actions -->
        <div class="nx-navbar-actions">
          <a href="/guide/getting-started" class="nx-nav-btn-primary">
            <span>Documentation</span>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </a>
          <a
            href="https://github.com/aerovexhq/sonon"
            target="_blank"
            rel="noopener noreferrer"
            class="nx-nav-github"
            aria-label="GitHub Repository"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
          </a>
        </div>
      </div>
    </header>

    <!-- Main Content Container -->
    <main class="sonon-main">
      <!-- 00 // HERO SECTION -->
      <section id="home" class="nx-hero container">
        <div class="nx-hero-content">
          <!-- Status Pill -->
          <div class="nx-hero-eyebrow">
            <span class="nx-telemetry-dot"></span>
            <span>EDGE ACOUSTIC DSP &amp; SPEECH SYNTHESIS // HARD REAL-TIME</span>
          </div>

          <!-- Main Monolithic Headline -->
          <h1 class="nx-hero-title">
            <span>Mission-Critical Acoustic Signal Processing for </span>
            <span class="nx-hero-title-accent">Autonomous Robotics.</span>
          </h1>

          <!-- Elevator Pitch -->
          <p class="nx-hero-sub">
            Pure safe Rust digital signal processing suite engineered for autonomous drone operations. Delivers real-time blade-pass frequency tracking with &gt;86 dB acoustic cancellation, microphone array spatial beamforming, sub-millisecond keyword spotting, and zero-cloud articulatory speech synthesis.
          </p>

          <!-- Primary CTAs -->
          <div class="nx-hero-actions">
            <a href="/guide/getting-started" class="nx-hero-cta-primary">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
              <span>Explore Documentation</span>
            </a>

            <a href="/synthesis" class="nx-hero-cta-secondary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
              </svg>
              <span>Voice Synthesis Studio</span>
            </a>

            <a href="/playground" class="nx-hero-cta-secondary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              <span>Audio Lab Playground</span>
            </a>
          </div>

          <!-- Live Metrics Ticker -->
          <div class="nx-hero-metrics">
            <div class="nx-hero-metric-item">
              <span class="nx-metric-val">&gt;86 dB</span>
              <span class="nx-metric-lbl">Rotor Notch Filter</span>
            </div>
            <div class="nx-hero-metric-item">
              <span class="nx-metric-val">0.42 ms</span>
              <span class="nx-metric-lbl">Sub-ms KWS Latency</span>
            </div>
            <div class="nx-hero-metric-item">
              <span class="nx-metric-val">16 kHz</span>
              <span class="nx-metric-lbl">Streaming VAD</span>
            </div>
            <div class="nx-hero-metric-item">
              <span class="nx-metric-val">1.2 MB</span>
              <span class="nx-metric-lbl">Liljencrants-Fant Vocal Tract</span>
            </div>
          </div>
        </div>

        <!-- Hero Viewport: Live Simulated / Real-Time DSP Telemetry Scope -->
        <div class="nx-hero-viewport">
          <div class="nx-hero-viewport-topbar">
            <div class="nx-frame-dots">
              <span class="nx-dot nx-dot-red"></span>
              <span class="nx-dot nx-dot-yellow"></span>
              <span class="nx-dot nx-dot-green"></span>
            </div>
            <div class="nx-frame-title">
              SONON REAL-TIME DSP TELEMETRY // 48 KHZ STREAM
            </div>
            <div class="nx-frame-telemetry">
              <span class="nx-telemetry-dot"></span>
              <span>MAVLINK ESC: 5,420 RPM</span>
            </div>
          </div>

          <div class="nx-hero-viewport-body">
            <canvas ref="canvasRef" class="nx-hero-canvas"></canvas>

            <!-- Interactive HUD overlay -->
            <div class="nx-viewport-hud-overlay">
              <span class="nx-hud-pill nx-hud-pill-active">BPF: 361.3 HZ // 4 BLADES</span>
              <span class="nx-hud-pill nx-hud-pill-active">NOTCH DEPTH: -86.4 DB</span>
              <span class="nx-hud-pill">VAD: SPEECH ACTIVE (0.94)</span>
              <span class="nx-hud-pill nx-hud-pill-active">KWS: "FALCON" ARMED</span>
            </div>

            <!-- Viewport Controls -->
            <div class="nx-hero-canvas-controls">
              <button
                type="button"
                class="nx-canvas-ctrl-btn"
                :class="{ 'nx-ctrl-active': motorWhineEnabled }"
                @click="motorWhineEnabled = !motorWhineEnabled"
              >
                <span>Motor Prop Wash: {{ motorWhineEnabled ? 'ACTIVE (88 dBA)' : 'MUTED' }}</span>
              </button>
              <button
                type="button"
                class="nx-canvas-ctrl-btn"
                :class="{ 'nx-ctrl-active': notchFilterActive }"
                @click="notchFilterActive = !notchFilterActive"
              >
                <span>Harmonic Notch: {{ notchFilterActive ? 'ENABLED (>86 dB)' : 'BYPASS' }}</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      <!-- 01 // BALANCED BENTO GRID WITH SPOTLIGHT -->
      <section id="subsystems" class="nx-ecosystem-section container">
        <div class="nx-ecosystem-header-wrap">
          <div class="nx-ecosystem-eyebrow">
            01 // CORE ACOUSTIC SUBSYSTEMS
          </div>
          <h2 class="nx-ecosystem-title">
            Embedded Acoustic Architecture
          </h2>
          <p class="nx-ecosystem-lead">
            Modular, hard real-time acoustic signal processing modules engineered in 100% pure safe Rust. Zero dynamic allocations during audio processing, deterministic bounded cycle latencies, and direct integration with companion computers and flight microcontrollers.
          </p>

          <!-- Domain Filter Tabs -->
          <div class="nx-ecosystem-filter-bar" role="tablist">
            <button
              type="button"
              role="tab"
              :aria-selected="activeFilter === 'all'"
              class="nx-filter-tab"
              :class="{ 'nx-filter-tab-active': activeFilter === 'all' }"
              @click="activeFilter = 'all'"
            >
              <span>All Subsystems</span>
              <span class="nx-filter-count">5</span>
            </button>
            <button
              type="button"
              role="tab"
              :aria-selected="activeFilter === 'cancellation'"
              class="nx-filter-tab"
              :class="{ 'nx-filter-tab-active': activeFilter === 'cancellation' }"
              @click="activeFilter = 'cancellation'"
            >
              <span>Noise Cancellation &amp; Beamforming</span>
              <span class="nx-filter-count">2</span>
            </button>
            <button
              type="button"
              role="tab"
              :aria-selected="activeFilter === 'kws'"
              class="nx-filter-tab"
              :class="{ 'nx-filter-tab-active': activeFilter === 'kws' }"
              @click="activeFilter = 'kws'"
            >
              <span>Voice Spotting &amp; VAD</span>
              <span class="nx-filter-count">2</span>
            </button>
            <button
              type="button"
              role="tab"
              :aria-selected="activeFilter === 'synth'"
              class="nx-filter-tab"
              :class="{ 'nx-filter-tab-active': activeFilter === 'synth' }"
              @click="activeFilter = 'synth'"
            >
              <span>Physical Speech Synthesis</span>
              <span class="nx-filter-count">1</span>
            </button>
          </div>
        </div>

        <!-- 2-Column Balanced Bento Grid with Radial Mouse Spotlight -->
        <div class="nx-bento-2col">
          <article
            v-for="(subsystem, idx) in filteredSubsystems"
            :key="subsystem.id"
            :ref="(el) => setCardRef(el, idx)"
            class="sonon-spotlight-card nx-project-card"
            @mousemove="handleMouseMove($event, idx)"
            @mouseleave="handleMouseLeave(idx)"
          >
            <!-- Card Header -->
            <div class="nx-project-card-header">
              <div class="nx-project-meta">
                <span class="nx-project-tier-tag">{{ subsystem.tierLabel }}</span>
                <span class="nx-project-category">{{ subsystem.category }}</span>
              </div>
              <div class="nx-project-status-badge" :class="getStatusToneClass(subsystem.statusTone)">
                <span class="nx-status-dot"></span>
                <span>{{ subsystem.status }}</span>
              </div>
            </div>

            <!-- Card Body -->
            <div class="nx-project-body">
              <div class="nx-project-title-row">
                <h3 class="nx-project-name">{{ subsystem.name }}</h3>
              </div>
              <p class="nx-project-tagline">{{ subsystem.tagline }}</p>

              <!-- Empirical Benchmark Metric Well -->
              <div class="nx-project-metric-box">
                <div class="nx-project-metric-val">{{ subsystem.metric }}</div>
                <div class="nx-project-metric-desc">{{ subsystem.metricDetail }}</div>
              </div>

              <p class="nx-project-desc">{{ subsystem.description }}</p>

              <!-- Architecture Highlights -->
              <ul class="nx-project-highlights">
                <li v-for="(hl, hIdx) in subsystem.highlights" :key="hIdx" class="nx-project-highlight-item">
                  <span class="nx-highlight-bullet"></span>
                  <span>{{ hl }}</span>
                </li>
              </ul>
            </div>

            <!-- Card Footer -->
            <div class="nx-project-card-footer">
              <button
                type="button"
                class="nx-project-btn-deepdive"
                @click="openArchitecture(subsystem)"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="12 2 2 7 12 12 22 7 12 2" />
                  <polyline points="2 17 12 22 22 17" />
                  <polyline points="2 12 12 17 22 12" />
                </svg>
                <span>Inspect Architecture</span>
              </button>

              <a :href="subsystem.docsLink" class="nx-project-btn-subdomain">
                <span>View Monograph</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </a>
            </div>
          </article>
        </div>
      </section>

      <!-- 02 // ARCHITECTURE SHOWCASE & PIPELINE DIAGRAM SECTION -->
      <section id="pipeline" class="nx-section container">
        <div class="nx-section-header">
          <div class="nx-section-eyebrow">02 // DATAFLOW &amp; EXECUTION PIPELINE</div>
          <h2 class="nx-section-title">End-to-End Real-Time Acoustic Engine Dataflow</h2>
          <p class="nx-section-lead">
            From raw multi-channel I2S DMA buffers to telemetry-guided notch cancellation, spatial MVDR beamforming, and on-device articulatory speech synthesis.
          </p>
        </div>

        <!-- Full Engine Pipeline ASCII Flowchart Container -->
        <div class="nx-pipeline-showcase-box">
          <div class="nx-pipeline-topbar">
            <span class="nx-pipeline-title">SONON ACOUSTIC PROCESSING KERNEL // DATAFLOW BUS</span>
            <span class="nx-pipeline-rate">48,000 HZ DMA INGESTION // ZERO-ALLOC</span>
          </div>
          <pre class="nx-pipeline-ascii-block"><code>[8-Ch MEMS Array / Raw PCM] ──► [Telemetry-Locked Notch Filter Bank (>86 dB)]
                                                    │
                                                    ▼
                                     [SIMD STFT &amp; Spatial Beamformer]
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                                                     ▼
             [Streaming VAD Gate]                                    [Acoustic Diagnostics]
                         │                                                     │
                         ▼                                                     ▼
             [Few-Shot Sakoe-Chiba KWS]                              [CWT Bearing / Blade Flutter]
                         │
                         ▼
             [Autopilot Mission Event] ◄──► [Liljencrants-Fant Articulatory Synth]</code></pre>
        </div>

        <!-- Interactive Code Tabs Interface -->
        <div class="nx-code-tabs-container">
          <div class="nx-code-tabs-header">
            <div class="nx-code-tab-buttons">
              <button
                type="button"
                class="nx-code-tab-btn"
                :class="{ 'nx-code-tab-active': activePipelineTab === 'notch' }"
                @click="activePipelineTab = 'notch'"
              >
                rotor_notch.rs
              </button>
              <button
                type="button"
                class="nx-code-tab-btn"
                :class="{ 'nx-code-tab-active': activePipelineTab === 'kws' }"
                @click="activePipelineTab = 'kws'"
              >
                kws_stream.rs
              </button>
              <button
                type="button"
                class="nx-code-tab-btn"
                :class="{ 'nx-code-tab-active': activePipelineTab === 'synth' }"
                @click="activePipelineTab = 'synth'"
              >
                vocal_tract.rs
              </button>
              <button
                type="button"
                class="nx-code-tab-btn"
                :class="{ 'nx-code-tab-active': activePipelineTab === 'wasm' }"
                @click="activePipelineTab = 'wasm'"
              >
                wasm_worklet.ts
              </button>
            </div>

            <button
              type="button"
              class="nx-code-copy-btn"
              @click="copyPipelineCode"
            >
              <svg v-if="!pipelineCopied" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <svg v-else width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>{{ pipelineCopied ? 'Copied' : 'Copy Code' }}</span>
            </button>
          </div>

          <pre class="nx-code-viewport"><code>{{ pipelineSnippets[activePipelineTab].code }}</code></pre>
        </div>
      </section>

      <!-- 03 // BENCHMARK COMPARISON MATRIX TABLE -->
      <section id="benchmarks" class="nx-section container">
        <div class="nx-section-header">
          <div class="nx-section-eyebrow">03 // EMPIRICAL BENCHMARK SPECIFICATIONS</div>
          <h2 class="nx-section-title">Acoustic Engine Benchmark Comparison</h2>
          <p class="nx-section-lead">
            Measured empirical metrics comparing Sonon against industry standard acoustic DSP and speech recognition alternatives under severe drone aerodynamic interference.
          </p>
        </div>

        <div class="nx-bench-table-wrap">
          <table class="nx-bench-table">
            <thead>
              <tr>
                <th class="nx-col-metric">Metric / Capability</th>
                <th class="nx-col-sonon">Sonon (Aerovex)</th>
                <th>Picovoice Porcupine</th>
                <th>AlphaCephei Vosk</th>
                <th>WebRTC VAD</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in BENCHMARK_COMPARISONS" :key="idx">
                <td class="nx-cell-metric-label">{{ row.metric }}</td>
                <td class="nx-cell-sonon-val">{{ row.sonon }}</td>
                <td class="nx-cell-baseline">{{ row.porcupine }}</td>
                <td class="nx-cell-baseline">{{ row.vosk }}</td>
                <td class="nx-cell-baseline">{{ row.webrtcVad }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>

    <!-- UNIFIED FOOTER DIRECTORY (Aerovex Standard) -->
    <footer class="nx-footer">
      <div class="nx-footer-container container">
        <div class="nx-footer-grid">
          <!-- Brand & Mission Column -->
          <div class="nx-footer-col nx-footer-col-brand">
            <div class="nx-footer-brand">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 10v4" />
                <path d="M6 6v12" />
                <path d="M10 3v18" />
                <path d="M14 7v10" />
                <path d="M18 9v6" />
                <path d="M22 11v2" />
              </svg>
              <span class="nx-footer-title">SONON.</span>
            </div>
            <p class="nx-footer-tagline">
              High-performance embedded acoustic signal processing, dynamic rotor noise cancellation, spatial beamforming, and on-device speech synthesis engineered for autonomous robotics.
            </p>
            <div class="nx-footer-status-pill">
              <span class="nx-telemetry-dot"></span>
              <span>ALL DSP CHANNELS NOMINAL // 48 KHZ LOCK</span>
            </div>
          </div>

          <!-- Column 1: Core Subsystems -->
          <div class="nx-footer-col">
            <div class="nx-footer-heading">Acoustic Subsystems</div>
            <ul class="nx-footer-links">
              <li><a href="#subsystems">Dynamic Rotor Notch</a></li>
              <li><a href="#subsystems">Few-Shot KWS Spotting</a></li>
              <li><a href="#subsystems">Streaming Dual-Band VAD</a></li>
              <li><a href="#subsystems">Physical Vocal Tract Synth</a></li>
              <li><a href="#subsystems">Spatial MVDR Beamformer</a></li>
              <li><a href="/dsp/wavelet-diagnostics">Wavelet CWT Diagnostics</a></li>
            </ul>
          </div>

          <!-- Column 2: Aerovex Ecosystem -->
          <div class="nx-footer-col">
            <div class="nx-footer-heading">Aerovex Ecosystem</div>
            <ul class="nx-footer-links">
              <li><a href="https://aerovex.net" target="_blank" rel="noopener noreferrer">Aerovex Platform</a></li>
              <li><a href="https://axiom.aerovex.net" target="_blank" rel="noopener noreferrer">Axiom Silicon EDA</a></li>
              <li><a href="https://phonon.aerovex.net" target="_blank" rel="noopener noreferrer">Phonon SPICE &amp; TCAD</a></li>
              <li><a href="https://atoma.aerovex.net" target="_blank" rel="noopener noreferrer">Atoma Atomistic CAD</a></li>
              <li><a href="https://vexel.aerovex.net" target="_blank" rel="noopener noreferrer">Vexel Systems Language</a></li>
              <li><a href="https://appify.aerovex.net" target="_blank" rel="noopener noreferrer">Appify Bundler</a></li>
            </ul>
          </div>

          <!-- Column 3: Documentation & Guides -->
          <div class="nx-footer-col">
            <div class="nx-footer-heading">Documentation</div>
            <ul class="nx-footer-links">
              <li><a href="/guide/getting-started">Getting Started Guide</a></li>
              <li><a href="/guide/architecture">Architecture Principles</a></li>
              <li><a href="/dsp/overview">30 Research Monographs</a></li>
              <li><a href="/dsp/rotor-notch-filter">Notch Filter Physics</a></li>
              <li><a href="/dsp/wake-word-spotting">Sakoe-Chiba DTW</a></li>
              <li><a href="/dsp/speech-synthesis">Formant Vocal Physics</a></li>
            </ul>
          </div>

          <!-- Column 4: Engine APIs & Interactive Labs -->
          <div class="nx-footer-col">
            <div class="nx-footer-heading">Engine Labs &amp; APIs</div>
            <ul class="nx-footer-links">
              <li><a href="/synthesis">Voice Synthesis Studio</a></li>
              <li><a href="/playground">Audio Lab Playground</a></li>
              <li><a href="/api/rust">Pure Safe Rust API</a></li>
              <li><a href="/api/c-abi">C-ABI &amp; POSIX SHM</a></li>
              <li><a href="/api/wasm">WebAssembly API</a></li>
              <li><a href="/api/python">Python NumPy SDK</a></li>
            </ul>
          </div>
        </div>

        <div class="nx-footer-bottom">
          <div class="nx-footer-copy">
            &copy; 2026 Aerovex HQ &amp; Sonon Contributors. All rights reserved.
          </div>
          <div class="nx-footer-security">
            CANONICAL NETWORK: *.AEROVEX.NET // AES-GCM-128
          </div>
        </div>
      </div>
    </footer>

    <!-- INTERACTIVE ARCHITECTURE MODAL -->
    <teleport to="body">
      <transition name="modal-fade">
        <div
          v-if="selectedSubsystem"
          class="nx-arch-modal-overlay"
          @click.self="closeArchitecture"
          role="dialog"
          aria-modal="true"
        >
          <div class="nx-arch-modal-card">
            <!-- Close Button -->
            <button
              type="button"
              class="nx-arch-modal-close"
              @click="closeArchitecture"
              aria-label="Close Architecture Modal"
            >
              &times;
            </button>

            <!-- Modal Header -->
            <div class="nx-arch-modal-header">
              <div class="nx-arch-modal-tags">
                <span class="nx-arch-modal-tier">{{ selectedSubsystem.tierLabel }}</span>
                <span class="nx-arch-stack-chip">{{ selectedSubsystem.category }}</span>
                <span class="nx-arch-stack-chip" :class="getStatusToneClass(selectedSubsystem.statusTone)">
                  {{ selectedSubsystem.status }}
                </span>
              </div>
              <h2 class="nx-arch-modal-title">{{ selectedSubsystem.name }}</h2>
              <p class="nx-arch-modal-tagline">{{ selectedSubsystem.tagline }}</p>

              <div class="nx-arch-modal-stack">
                <span
                  v-for="(tech, sIdx) in selectedSubsystem.architectureDetails.stack"
                  :key="sIdx"
                  class="nx-arch-stack-chip"
                >
                  {{ tech }}
                </span>
              </div>
            </div>

            <!-- Section 1: Dataflow & Execution Pipeline -->
            <div class="nx-arch-section">
              <div class="nx-arch-section-title">
                01 // Dataflow &amp; Execution Pipeline
              </div>
              <div class="nx-arch-pipeline-box">
                <pre class="nx-arch-pipeline-code">{{ selectedSubsystem.architectureDetails.pipelineAscii }}</pre>
              </div>
            </div>

            <!-- Section 2: Empirical Benchmark Matrix -->
            <div class="nx-arch-section">
              <div class="nx-arch-section-title">
                02 // Measured Empirical Benchmark Matrix
              </div>
              <div class="nx-arch-table-wrap">
                <table class="nx-arch-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Measured Value</th>
                      <th>Unit</th>
                      <th>Industry Baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(b, bIdx) in selectedSubsystem.architectureDetails.benchmarks" :key="bIdx">
                      <td>{{ b.label }}</td>
                      <td class="nx-arch-table-val">{{ b.value }}</td>
                      <td>{{ b.unit }}</td>
                      <td class="nx-arch-table-baseline">{{ b.baseline }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Section 3: Systems Trade-offs Analysis -->
            <div class="nx-arch-section">
              <div class="nx-arch-section-title">
                03 // Systems Engineering Trade-Offs
              </div>
              <div class="nx-arch-tradeoff-box">
                {{ selectedSubsystem.architectureDetails.tradeoffs }}
              </div>
            </div>

            <!-- Section 4: Mathematical Formulation -->
            <div class="nx-arch-section">
              <div class="nx-arch-section-title">
                04 // Mathematical Formulation
              </div>
              <div class="nx-arch-math-box">
                <code>{{ selectedSubsystem.architectureDetails.mathematicalFormulation }}</code>
              </div>
            </div>

            <!-- Section 5: Core Implementation Interface -->
            <div class="nx-arch-section">
              <div class="nx-arch-section-title">
                05 // Core Implementation Interface
              </div>
              <div class="nx-arch-code-wrap">
                <div class="nx-arch-code-topbar">
                  <span class="nx-arch-code-filename">
                    {{ selectedSubsystem.architectureDetails.codeSnippet.filename }}
                  </span>
                  <button
                    type="button"
                    class="nx-arch-code-copy"
                    @click="copySnippetText(selectedSubsystem.architectureDetails.codeSnippet.code)"
                  >
                    {{ copiedSnippet ? 'Copied' : 'Copy Code' }}
                  </button>
                </div>
                <pre class="nx-arch-code-pre"><code>{{ selectedSubsystem.architectureDetails.codeSnippet.code }}</code></pre>
              </div>
            </div>

            <!-- Modal Footer -->
            <div class="nx-arch-modal-footer">
              <button
                type="button"
                class="nx-arch-footer-close"
                @click="closeArchitecture"
              >
                Close Specification
              </button>

              <a :href="selectedSubsystem.docsLink" class="nx-arch-footer-link">
                <span>Open {{ selectedSubsystem.name }} Documentation</span>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </transition>
    </teleport>
  </div>
</template>

<style scoped>
/* Scoped styles are kept minimal as core rules are registered in custom.css */
.sonon-landing-root {
  min-height: 100vh;
  background-color: var(--bg-canvas, #08090a);
  color: var(--text-primary, #f7f8f8);
  position: relative;
  overflow-x: hidden;
}
</style>
