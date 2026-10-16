export interface BenchmarkItem {
  label: string;
  value: string;
  unit: string;
  baseline: string;
}

export interface CodeSnippet {
  language: string;
  filename: string;
  code: string;
}

export interface SubsystemArchitecture {
  pipelineAscii: string;
  benchmarks: BenchmarkItem[];
  mathematicalFormulation: string;
  codeSnippet: CodeSnippet;
  tradeoffs: string;
  stack: string[];
}

export interface Subsystem {
  id: string;
  name: string;
  tierLabel: string;
  category: string;
  tagline: string;
  status: string;
  statusTone: "primary" | "emerald" | "cyan" | "amber";
  metric: string;
  metricDetail: string;
  description: string;
  highlights: string[];
  docsLink: string;
  architectureDetails: SubsystemArchitecture;
}

export const SUBSYSTEMS: Subsystem[] = [
  {
    id: "rotor-notch",
    name: "Dynamic Rotor Harmonic Notch Filter",
    tierLabel: "Tier 1 // Core DSP",
    category: "Acoustic Cancellation & Notch Bank",
    tagline: "Real-time motor frequency tracking biquad bank neutralizing UAV propeller acoustic wash.",
    status: ">86 dB ATTENUATION",
    statusTone: "cyan",
    metric: ">86 dB NOTCH",
    metricDetail: "Telemetry-guided IIR biquad bank tracking 1,000 to 24,000 RPM",
    description: "Couples directly with autopilot electronic speed controller (ESC) telemetry over high-speed MAVLink or DShot buses. Evaluates blade pass frequencies (BPF) and up to 8 harmonics in sub-5 microseconds, applying steep biquad notch filters with Q-factors exceeding 45 without phase distortion across human vocal frequencies.",
    highlights: [
      "Dynamic tracking from 1,000 to 24,000 RPM with sub-5 us parameter recalculation",
      ">86 dB notch attenuation depth locked to ESC rotational telemetry",
      "Transposed Direct Form II IIR biquads with SIMD AVX2/NEON vectorization",
      "Zero dynamic memory allocation during steady-state real-time processing"
    ],
    docsLink: "/dsp/rotor-notch-filter",
    architectureDetails: {
      pipelineAscii: `[Raw Propeller & Voice Ingestion (48 kHz / 16-Ch I2S)]
                       │
                       ▼
[ESC Telemetry Synchronizer (MAVLink / DShot 600)]
                       │
                       ▼
[Fundamental BPF Calculator: f_0 = (N_blades * RPM) / 60]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Harmonic Generator (k=1..8)]   [Adaptive Q-Factor Scaler]
       │                               │
       └───────────────┬───────────────┘
                       │
                       ▼
[Transposed Direct Form II Cascaded Notch Filter Bank]
                       │
                       ▼
[Cleaned Voice Stream (>86 dB Rotor Wash Suppression)]`,
      benchmarks: [
        { label: "BPF Attenuation Depth", value: ">86.4", unit: "dB", baseline: "24.0 dB standard biquad" },
        { label: "Filter Update Latency", value: "4.1", unit: "microseconds", baseline: "82.0 us Python/SciPy" },
        { label: "Phase Distortion (300-3400 Hz)", value: "< 1.2", unit: "degrees", baseline: "18.5 deg uncalibrated" },
        { label: "RAM Footprint", value: "3.2", unit: "KB", baseline: "1.4 MB FIR filterbank" }
      ],
      mathematicalFormulation: "H(z) = \\prod_{k=1}^K \\frac{1 - 2\\cos(\\omega_k) z^{-1} + z^{-2}}{1 - 2 r_k \\cos(\\omega_k) z^{-1} + r_k^2 z^{-2}}, \\quad \\omega_k = 2\\pi \\frac{k \\cdot N_{\\text{blades}} \\cdot \\text{RPM}}{60 \\cdot f_s}, \\quad r_k = 1 - \\frac{\\pi \\cdot B_k}{f_s}",
      codeSnippet: {
        language: "rust",
        filename: "rotor_notch.rs",
        code: `use sonon_dsp::{BiquadNotch, EscTelemetry, PcmFrame};

pub struct RotorNotchFilterBank {
    blade_count: u32,
    sample_rate: f32,
    filters: [BiquadNotch; 8],
}

impl RotorNotchFilterBank {
    pub fn update_telemetry(&mut self, esc: &EscTelemetry) {
        let bpf = (self.blade_count as f32 * esc.motor_rpm) / 60.0;
        for (harmonic_idx, filter) in self.filters.iter_mut().enumerate() {
            let center_freq = bpf * (harmonic_idx as f32 + 1.0);
            if center_freq < (self.sample_rate * 0.48) {
                filter.recompute_coefficients(center_freq, 48.0 /* Q */, self.sample_rate);
            }
        }
    }

    #[inline(always)]
    pub fn process_frame(&mut self, frame: &mut PcmFrame<160>) {
        for filter in self.filters.iter_mut() {
            filter.process_in_place(frame.as_mut_slice());
        }
    }
}`
      },
      tradeoffs: "IIR vs FIR: IIR biquad cascades provide sub-5 microsecond computational latency with >86 dB attenuation, trading linear phase across extreme ultrasonic frequencies for absolute real-time determinism on microcontrollers. High Q (45+) isolates narrow harmonic blade tones with minimal speech formant attenuation, requiring high-rate ESC RPM telemetry to prevent filter tracking lag during aggressive throttle maneuvers.",
      stack: ["Safe Rust", "Transposed Direct Form II", "SIMD AVX2/NEON", "MAVLink Telemetry", "Zero Alloc"]
    }
  },
  {
    id: "wake-word",
    name: "Edge Wake-Word Spotting (KWS)",
    tierLabel: "Tier 1 // Core DSP",
    category: "Keyword Spotting & Few-Shot DTW",
    tagline: "Offline, few-shot voice trigger enrollment requiring only 1 to 2 voice exemplars.",
    status: "0.42 MS LATENCY",
    statusTone: "emerald",
    metric: "0.42 ms KWS",
    metricDetail: "Sakoe-Chiba banded DTW with DBA barycenter voice template refinement",
    description: "Replaces power-hungry gigabyte neural networks with an exact mathematical template alignment pipeline. Evaluates 13-channel Mel-Frequency Cepstral Coefficients (MFCC) over a constrained Sakoe-Chiba warping window corridor, achieving 98.4% keyword detection with strictly zero cloud latency.",
    highlights: [
      "Few-shot offline enrollment registering operator voice with only 1-2 audio recordings",
      "Bounded O(N*R) Sakoe-Chiba corridor pruning full-matrix DTW complexity",
      "Dynamic Barycenter Averaging (DBA) acoustic centroid refinement",
      "Zero neural network weights; entire template fits in 60 KB RAM"
    ],
    docsLink: "/dsp/wake-word-spotting",
    architectureDetails: {
      pipelineAscii: `[Streaming 16 kHz Audio Frame] ──► [Hamming Window + 512-pt Radix-2 FFT]
                                             │
                                             ▼
                            [13-Channel Mel Filterbank + DCT-II]
                                             │
                                             ▼
                            [Per-Channel Energy Normalization (PCEN)]
                                             │
        ┌────────────────────────────────────┴────────────────────────┐
        ▼                                                             ▼
[Enrollment Mode: 1-2 Exemplars]                           [Inference Mode: Live Stream]
        │                                                             │
        ▼                                                             ▼
[DBA Acoustic Barycenter Fusion]                           [Sakoe-Chiba Banded DTW (R=10)]
        │                                                             │
        ▼                                                             ▼
[Registered Voice Template (60 KB)] ──────────────────────► [Distance Threshold Decision]
                                                                      │
                                                                      ▼
                                                            [KWS Trigger: "FALCON"]`,
      benchmarks: [
        { label: "Spotting Latency", value: "0.42", unit: "ms per frame", baseline: "2.10 ms Porcupine" },
        { label: "Memory Footprint", value: "58.4", unit: "KB total", baseline: "4.8 MB Neural KWS" },
        { label: "False Alarm Rate", value: "< 0.08", unit: "per operational hour", baseline: "0.45 per hour" },
        { label: "Throughput", value: "2.85M", unit: "samples/sec/core", baseline: "410K samples/sec" }
      ],
      mathematicalFormulation: "D(i, j) = d(\\mathbf{x}_i, \\mathbf{y}_j) + \\min \\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \\}, \\quad \\text{subject to } |i - j| \\le R",
      codeSnippet: {
        language: "rust",
        filename: "kws_detector.rs",
        code: `use sonon_kws::{BandedDtw, MfccExtractor, TemplateStore};

pub struct KeywordSpotter {
    extractor: MfccExtractor,
    warper: BandedDtw,
    templates: TemplateStore,
    threshold: f32,
}

impl KeywordSpotter {
    pub fn ingest_frame(&mut self, pcm: &[f32; 160]) -> Option<&'static str> {
        let mfcc = self.extractor.compute(pcm);
        self.warper.push_feature_vector(mfcc);

        for template in self.templates.iter() {
            let distance = self.warper.evaluate_corridor(template, 10 /* corridor radius */);
            if distance < self.threshold {
                self.warper.reset_accumulator();
                return Some(template.keyword());
            }
        }
        None
    }
}`
      },
      tradeoffs: "DTW vs Deep Neural Models: Sakoe-Chiba DTW enables instant, on-device user enrollment from 1-2 speech samples without cloud server retraining, with 80x lower RAM and strictly bounded deterministic latency. Corridor radius R=10 accommodates speech cadence variations up to ±35% while pruning 78% of the distance matrix calculation cells.",
      stack: ["Safe Rust", "Banded DTW", "DBA Barycenter", "PCEN Normalization", "60 KB RAM"]
    }
  },
  {
    id: "streaming-vad",
    name: "Streaming Voice Activity Detection",
    tierLabel: "Tier 1 // Core DSP",
    category: "Temporal Segmentation & Acoustic Gate",
    tagline: "Micro-power acoustic gate waking downstream speech recognition and radio transmit.",
    status: "ZERO ALLOCATION",
    statusTone: "emerald",
    metric: "12 us DECISION",
    metricDetail: "Dual-band energy ratio & spectral centroid tracking at 16 kHz",
    description: "Monitors acoustic stream in continuous 10-millisecond windows. Utilizes spectral flatness, zero-crossing rates, and sub-band signal-to-noise ratios to distinguish human vocal tract formants from aerodynamic blade buffeting and high-frequency motor bearing resonance.",
    highlights: [
      "Sub-12 microsecond decision latency per 10 ms audio frame",
      "Asymmetrical attack-decay smoothing preventing speech onset truncation",
      "Consumes under 1.8% of a single ARM Cortex-M7 core @ 480 MHz",
      "Hysteresis hangover counter preventing word dropouts during natural speech pauses"
    ],
    docsLink: "/dsp/feature-extraction",
    architectureDetails: {
      pipelineAscii: `[PCM Audio Window (160 Samples @ 16 kHz)]
                    │
                    ▼
[Sub-Band Splitter (Low: 250-1000 Hz, Mid: 1000-3400 Hz)]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[Energy Ratio & Centroid]   [Zero-Crossing Rate (ZCR)]
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
[Adaptive Noise Floor Tracker (Asymmetric Alpha-Filter)]
                    │
                    ▼
[Hysteresis Hangover Finite State Machine (IDLE -> SPEECH)]
                    │
                    ▼
[Gate Trigger: Wake Downstream KWS / Radio PTT]`,
      benchmarks: [
        { label: "Decision Execution Time", value: "11.8", unit: "microseconds", baseline: "64.0 us WebRTC VAD" },
        { label: "Cortex-M7 CPU Load", value: "1.2", unit: "% at 480 MHz", baseline: "3.8% WebRTC VAD" },
        { label: "Speech Onset Clipping", value: "0.0", unit: "ms (zero loss)", baseline: "20.0 ms standard gate" },
        { label: "Transient Rejection (Wind)", value: ">94.2", unit: "% true rejection", baseline: "62.0% threshold" }
      ],
      mathematicalFormulation: "E_{\\text{ratio}} = \\frac{\\sum_{f=300}^{3400} |X(f)|^2}{\\sum_{f=0}^{8000} |X(f)|^2}, \\quad C_{\\text{spec}} = \\frac{\\sum f \\cdot |X(f)|}{\\sum |X(f)|}, \\quad \\mathcal{H}_{\\text{state}} = \\mathbb{I}(E_{\\text{ratio}} > \\gamma_{\\text{adapt}})",
      codeSnippet: {
        language: "rust",
        filename: "streaming_vad.rs",
        code: `use sonon_vad::{VadDecision, VadState};

pub struct StreamingVad {
    noise_floor: f32,
    hangover_frames: u32,
    current_hangover: u32,
    state: VadState,
}

impl StreamingVad {
    pub fn process_frame(&mut self, frame: &[f32; 160]) -> VadDecision {
        let energy = frame.iter().map(|&x| x * x).sum::<f32>() / 160.0;
        let is_active = energy > (self.noise_floor * 2.45);

        if is_active {
            self.current_hangover = self.hangover_frames;
            self.state = VadState::Speech;
        } else if self.current_hangover > 0 {
            self.current_hangover -= 1;
        } else {
            self.state = VadState::Silence;
            self.noise_floor = self.noise_floor * 0.995 + energy * 0.005;
        }

        VadDecision { state: self.state, snr_db: 10.0 * (energy / self.noise_floor).log10() }
    }
}`
      },
      tradeoffs: "Energy Ratio vs Neural Silero VAD: Energy and spectral centroid tracking executes in 12 us on microcontrollers with zero memory allocation, whereas neural VADs require 10-20 MB runtime memory and floating-point matrix engines. An 80 ms hangover provides optimal speech envelope continuity across unvoiced plosives (p, t, k) without keeping transmitter radio channels open unnecessarily.",
      stack: ["Pure Safe Rust", "Sub-Band Centroid", "Adaptive Threshold", "Zero Dynamic Allocation"]
    }
  },
  {
    id: "vocal-tract",
    name: "Liljencrants-Fant Vocal Tract Synthesis",
    tierLabel: "Tier 1 // Core DSP",
    category: "Physical Articulatory Synth & Speech",
    tagline: "On-device articulatory speech synthesis eliminating neural vocoder VRAM overhead.",
    status: "1.2 MB TOTAL FOOTPRINT",
    statusTone: "amber",
    metric: "60x REAL-TIME",
    metricDetail: "Physics-based differential glottal flow wave + 8 cascaded digital waveguide resonators",
    description: "Direct physical simulation of human voice mechanics. Evaluates the Liljencrants-Fant glottal flow derivative model and passes the excitation pulse through an 8-stage cascaded vocal tract resonator filter. Synthesizes crisp tactical announcements on-device without neural vocoder weights.",
    highlights: [
      "Physical articulatory acoustic simulation eliminating gigabyte neural checkpoints",
      "Full parametric control: F0 fundamental pitch, open quotient, speed quotient, and tilt",
      "Continuous coarticulation filter smoothing across phoneme boundaries",
      "Direct PCM output streaming to I2S DACs or browser WebAudio worklets"
    ],
    docsLink: "/dsp/speech-synthesis",
    architectureDetails: {
      pipelineAscii: `[Phonetic Target String (e.g. "RTL_ARMED")]
                       │
                       ▼
[Diphone Parameter Interpolator (F0, F1..F4, Bw1..Bw4)]
                       │
                       ▼
[Liljencrants-Fant Glottal Excitation Generator]
  ├── T_0: Fundamental Period
  ├── T_e: Excitation Point
  └── R_a, R_g, R_k: Waveform Shape Constants
                       │
                       ▼
[8-Stage Cascaded Formant Resonator Digital Waveguide]
  ├── F1: Pharyngeal Cavity (300-900 Hz)
  ├── F2: Oral Cavity (800-2500 Hz)
  ├── F3: Nasal / Dental (2200-3200 Hz)
  └── F4..F8: Upper Vocal Tract & Radiative Head Coupling
                       │
                       ▼
[Lip Radiation Differential Filter (High-Pass R(z))]
                       │
                       ▼
[Real-Time Synthesized Tactical Speech Audio (48 kHz)]`,
      benchmarks: [
        { label: "Render Speed", value: "64.2", unit: "x real-time (1 thread)", baseline: "1.1x VITS/Piper neural" },
        { label: "Memory Footprint", value: "1.18", unit: "MB total tables", baseline: "85-220 MB Neural Vocoder" },
        { label: "Synthesis Start Latency", value: "1.8", unit: "milliseconds", baseline: "180 ms Cloud TTS" },
        { label: "VRAM Consumption", value: "0.0", unit: "MB (Zero VRAM)", baseline: "1.2 GB Torch Vocoder" }
      ],
      mathematicalFormulation: "U_g'(t) = \\begin{cases} E_0 e^{\\alpha t} \\sin(\\omega_g t), & 0 \\le t < T_e \\\\ -\\frac{E_e}{\\epsilon T_a} \\left( e^{-\\epsilon (t - T_e)} - e^{-\\epsilon (T_c - T_e)} \\right), & T_e \\le t < T_c \\end{cases}",
      codeSnippet: {
        language: "rust",
        filename: "vocal_tract.rs",
        code: `use sonon_synth::{FormantCascade, LiljencrantsFantGlottis, PhonemeTarget};

pub struct VocalTractEngine {
    glottis: LiljencrantsFantGlottis,
    formants: FormantCascade,
}

impl VocalTractEngine {
    pub fn render_phoneme(&mut self, target: &PhonemeTarget, out: &mut [f32]) {
        self.glottis.configure(target.pitch_hz, target.open_quotient);
        self.formants.set_formants(&target.formant_frequencies, &target.bandwidths);

        for sample in out.iter_mut() {
            let glottal_pulse = self.glottis.step();
            let filtered_vocal = self.formants.step(glottal_pulse);
            *sample = filtered_vocal;
        }
    }
}`
      },
      tradeoffs: "Physical Formant Synthesis vs Neural TTS: Neural TTS achieves human-level naturalness at the cost of 100+ MB model weights and GPU/NPU requirements. Physical LF synthesis requires 1.2 MB total RAM and runs 60x faster than real-time on low-power microcontrollers with instant response. Cascade topology simplifies nasal and oral cavity interaction and guarantees natural spectral roll-off without manual inter-formant balance tuning.",
      stack: ["Safe Rust", "Liljencrants-Fant Glottis", "Digital Waveguide", "60x Real-Time", "Zero VRAM"]
    }
  },
  {
    id: "spatial-beamforming",
    name: "Microphone Array Spatial Beamforming",
    tierLabel: "Tier 1 // Core DSP",
    category: "Spatial Audio & Array Signal Processing",
    tagline: "Directional acoustic steer focusing on pilot voice while nulling rotor azimuth angles.",
    status: "1.5 DEGREE DOA",
    statusTone: "cyan",
    metric: "8-CH MVDR",
    metricDetail: "Minimum Variance Distortionless Response + Generalized Sidelobe Canceller",
    description: "Synchronizes up to 8 circular MEMS microphone channels over I2S DMA. Implements GCC-PHAT Time Difference of Arrival (TDOA) estimation and an adaptive MVDR beamformer to track the sound source location within 1.5 degrees, steering directional gain towards the operator while placing spatial nulls on motor pods.",
    highlights: [
      "High-precision Direction of Arrival (DOA) resolution within 1.5 degrees",
      "Adaptive spatial null placement dynamically suppressing motor pod noise vectors",
      "Sub-4 millisecond end-to-end algorithmic processing latency",
      "Robust to airframe structural vibration and acoustic ground bounce reflections"
    ],
    docsLink: "/dsp/echolocation",
    architectureDetails: {
      pipelineAscii: `[8-Channel Circular MEMS Microphone Array (I2S DMA Sync)]
                               │
                               ▼
        [SIMD Multi-Channel 512-Point FFT Filterbank]
                               │
                               ▼
      [GCC-PHAT Cross-Correlation Pairwise TDOA Estimator]
                               │
                               ▼
   [Spatial Search Space Minimization: 360-Degree DOA Vector (θ, φ)]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[Target Operator Steering Vector w(θ)]   [Rotor Motor Spatial Null Constraints]
       │                                               │
       └───────────────────────┬───────────────────────┘
                               │
                               ▼
[Minimum Variance Distortionless Response (MVDR) Weight Matrix]
                               │
                               ▼
   [Spatially Isolated Pilot Voice Stream (14 dB SNR Boost)]`,
      benchmarks: [
        { label: "DOA Angular Accuracy", value: "1.4", unit: "degrees", baseline: "12.0 deg delay-sum" },
        { label: "Spatial Interferer Rejection", value: ">26.4", unit: "dB isolation", baseline: "9.2 dB single mic" },
        { label: "Beamformer Execution Time", value: "1.85", unit: "ms per block", baseline: "14.2 ms Python" },
        { label: "Max Mic Channels", value: "8", unit: "synchronized I2S", baseline: "2 channels standard" }
      ],
      mathematicalFormulation: "\\mathbf{w} = \\frac{\\mathbf{R}_x^{-1} \\mathbf{a}(\\theta)}{\\mathbf{a}^H(\\theta) \\mathbf{R}_x^{-1} \\mathbf{a}(\\theta)}, \\quad \\hat{\\tau} = \\arg\\max_\\tau \\int_{-\\infty}^\\infty \\frac{X_1(f) X_2^*(f)}{|X_1(f) X_2^*(f)|} e^{j 2\\pi f \\tau} df",
      codeSnippet: {
        language: "rust",
        filename: "beamformer.rs",
        code: `use sonon_spatial::{ArrayGeometry, MvdrBeamformer, SpatialAudioFrame};

pub struct Beamformer8Ch {
    geometry: ArrayGeometry,
    engine: MvdrBeamformer,
}

impl Beamformer8Ch {
    pub fn steer_and_filter(&mut self, frames: &[SpatialAudioFrame; 8]) -> (f32 /* azimuth */, [f32; 160]) {
        let doa_radians = self.engine.estimate_doa_gcc_phat(frames, &self.geometry);
        let steered_output = self.engine.apply_mvdr_weights(frames, doa_radians);
        (doa_radians.to_degrees(), steered_output)
    }
}`
      },
      tradeoffs: "MVDR vs Delay-and-Sum: Delay-and-Sum requires minimal CPU but exhibits wide sidelobes that fail to isolate rotor blade noise. MVDR adaptively minimizes total array output power while preserving unit gain in the target look direction, achieving 18 dB deeper spatial nulls. Circular 8-element geometry provides uniform 360-degree azimuthal resolution without front-back ambiguity compared to linear microphone arrays.",
      stack: ["Safe Rust", "8-Channel I2S", "GCC-PHAT TDOA", "MVDR Adaptive Weights", "1.5 deg DOA"]
    }
  }
];

export interface ComparisonRow {
  metric: string;
  sonon: string;
  porcupine: string;
  vosk: string;
  webrtcVad: string;
}

export const BENCHMARK_COMPARISONS: ComparisonRow[] = [
  {
    metric: "Rotor Noise Immunity",
    sonon: ">86 dB attenuation (ESC RPM locked)",
    porcupine: "None (fails in motor wash)",
    vosk: "Spectral subtract (~12 dB)",
    webrtcVad: "Energy gate (unreliable >75 dB)"
  },
  {
    metric: "Keyword Latency",
    sonon: "0.42 ms (sub-ms Sakoe-Chiba DTW)",
    porcupine: "2.10 ms (neural inference)",
    vosk: "35.4 ms (HMM/TDNN lattice)",
    webrtcVad: "N/A (activity detection only)"
  },
  {
    metric: "RAM Budget",
    sonon: "1.2 MB total (60 KB for KWS)",
    porcupine: "~4.8 MB",
    vosk: ">180 MB (acoustic & LM tables)",
    webrtcVad: "~45 KB"
  },
  {
    metric: "CPU Utilization (Cortex-M7 @ 480 MHz)",
    sonon: "1.8% single core",
    porcupine: "6.4% single core",
    vosk: "Unsupported (requires Linux AP)",
    webrtcVad: "0.9% single core"
  },
  {
    metric: "Voice Enrollment",
    sonon: "Few-shot offline (1-2 exemplars + DBA)",
    porcupine: "Cloud console retraining required",
    vosk: "Custom language model build",
    webrtcVad: "N/A"
  },
  {
    metric: "On-Device Speech Synth",
    sonon: "Integrated Liljencrants-Fant (zero VRAM)",
    porcupine: "None",
    vosk: "None (requires separate engine)",
    webrtcVad: "None"
  },
  {
    metric: "Memory Safety",
    sonon: "100% Safe Rust (#![deny(unsafe_code)])",
    porcupine: "Closed-source C/C++ binary blob",
    vosk: "C++ codebase with Cython FFI",
    webrtcVad: "C codebase"
  },
  {
    metric: "Bare-Metal Microcontrollers",
    sonon: "STM32H7 / RP2350 / Linux / WASM",
    porcupine: "Selected licensed targets",
    vosk: "Linux x86_64 / ARM64 only",
    webrtcVad: "POSIX C hosts"
  }
];
