export interface DocSectionContent {
  title: string;
  badge: string;
  description: string;
  rustCode?: string;
  wavScript?: string;
  pythonCode?: string;
  callout?: {
    type: 'note' | 'tip' | 'warning' | 'spec';
    title: string;
    text: string;
  };
  specs?: { param: string; val: string; note: string }[];
  steps?: { title: string; desc: string; cmd?: string }[];
}

export interface DocSection {
  id: string;
  title: string;
  badge: string;
  content: DocSectionContent;
}

export interface DocCategory {
  id: string;
  title: string;
  badge: string;
  sections: DocSection[];
}

export const SONON_DOC_TREE: DocCategory[] = [
  {
    id: 'getting_started',
    title: '1. Getting Started',
    badge: 'CORE',
    sections: [
      {
        id: 'why_sonon',
        title: 'Why Sonon',
        badge: 'OVERVIEW',
        content: {
          title: 'Edge Acoustic DSP, Voice Engine & Speech Synthesis in Pure Safe Rust',
          badge: 'ARCHITECTURE',
          description: 'Sonon is a hard real-time acoustic signal processing and keyword spotting engine designed for autonomous robotics operating in extreme acoustic noise environments (100+ dB SPL rotor noise). It provides dynamic rotor notch filtering, streaming VAD, INT8 wake-word spotting, and Liljencrants-Fant articulatory speech synthesis.',
          callout: {
            type: 'spec',
            title: 'Acoustic Rejection Invariant',
            text: 'Sonon is explicitly designed for drone flight controllers. Its dynamic IIR biquad cascade tracks instantaneous ESC motor RPM telemetries with zero latency, providing >86 dB rejection at primary blade-pass harmonics.'
          },
          specs: [
            { param: 'Rotor Rejection Depth', val: '>86 dB', note: 'Blade pass fundamental and harmonics' },
            { param: 'KWS Latency', val: '<0.85 ms', note: 'INT8 DS-CNN inference on Cortex-M7' },
            { param: 'Streaming VAD Latency', val: '<2.0 ms', note: 'Spectral entropy sub-band segmenter' },
            { param: 'Static RAM Footprint', val: '<48 KB', note: 'Zero heap allocations in audio loop' }
          ],
          rustCode: `// Sonon: Synchronous Rotor Notch Filter & VAD
use sonon_dsp::filter::{RotorNotchCascade, BiquadConfig};
use sonon_dsp::vad::{SpectralEntropyVad, VadDecision};

fn main() {
    let mut notch = RotorNotchCascade::new(16000.0, 4); // 16 kHz, 4 cascaded rotors
    let mut vad = SpectralEntropyVad::new(16000.0, 160); // 10ms frame size
    
    // Dynamically update rotor RPM from Kestrel telemetry
    notch.update_rotor_rpm(0, 14250.0, 2); // Rotor 0: 14,250 RPM, 2 blades (475 Hz BPF)
    
    let mut audio_frame = [0.0f32; 160];
    notch.process_in_place(&mut audio_frame);
    
    let decision = vad.process_frame(&audio_frame);
    if decision == VadDecision::SpeechActive {
        println!("Voice activity detected. Forwarding to KWS classifier.");
    }
}`,
          wavScript: `# Sonon CLI Benchmark under 100 dB SPL Drone Noise
$ sonon-bench --input drone_hover_100db.wav --rpm 14250 --blades 2
Processing 16000 Hz 16-bit PCM stream...
[Harmonic Notch] Tracking BPF: 475.0 Hz, 2nd: 950.0 Hz, 3rd: 1425.0 Hz
[Harmonic Notch] Rejection depth: 89.4 dB
[Streaming VAD] Attack latency: 1.84 ms, Release: 12.0 ms
[Keyword Spot] Wake word 'AEROVEX' detected (Confidence: 0.9842, Latency: 0.72 ms)`,
          pythonCode: `# Python Hardware-in-the-Loop Audio Testing via Sonon C-ABI
import sonon
import numpy as np

engine = sonon.AcousticEngine(sample_rate=16000)
engine.set_rotor_telemetry(rotor_idx=0, rpm=14200, blade_count=2)

raw_audio = np.random.randn(160).astype(np.float32)
filtered_audio = engine.process_frame(raw_audio)
print(f"Attenuation: {engine.get_rejection_depth_db():.1f} dB")`
        }
      },
      {
        id: 'installation',
        title: 'Installation & Embedded Setup',
        badge: 'SETUP',
        content: {
          title: 'Compiling Sonon for Native Linux & ARM Cortex-M7',
          badge: 'ENVIRONMENT',
          description: 'Deploy Sonon directly on host mission computers, edge companion daemons, or bare-metal flight controllers via embedded-hal.',
          steps: [
            {
              title: 'Add Cargo Dependency',
              desc: 'Include Sonon DSP in your Rust embedded workspace with #![no_std] support.',
              cmd: 'cargo add --path modules/sonon sonon-dsp --no-default-features'
            },
            {
              title: 'Build CLI Tools',
              desc: 'Compile native audio inspection, filtering, and synthesis binaries.',
              cmd: 'cargo build -p sonon-cli --release'
            },
            {
              title: 'Run Embedded Verification Suite',
              desc: 'Execute deterministic DSP test suite verifying >86 dB attenuation and pure safe Rust memory invariants.',
              cmd: 'cargo test -p sonon-dsp --release'
            }
          ]
        }
      }
    ]
  },
  {
    id: 'filtering_dsp',
    title: '2. Rotor Noise & Filtering',
    badge: 'FILTERS',
    sections: [
      {
        id: 'rotor_notch_architecture',
        title: 'Dynamic Rotor Notch Filter',
        badge: 'NOTCH',
        content: {
          title: 'Cascaded Direct Form II Biquads with RPM Telemetry Lockstep',
          badge: 'HARMONIC',
          description: 'Calculates instantaneous blade-pass frequencies (BPF) and harmonics from ESC RPM telemetry, dynamically repositioning high-Q notch zeros to cancel propeller whine while keeping human vocal formats intact.',
          specs: [
            { param: 'Rejection Depth', val: '>86 dB (Empirical)', note: 'Narrow Q > 35 biquad zeros' },
            { param: 'Tracking Latency', val: '0 ms (Lockstep)', note: 'Synchronized with ESC CAN/DShot telemetry' },
            { param: 'Filter Topology', val: 'Transposed Direct Form II', note: 'Minimal numerical roundoff noise' }
          ],
          callout: {
            type: 'tip',
            title: 'Q-Factor Optimization',
            text: 'Sonon uses an adaptive Q-factor: Q scales linearly with motor RPM to keep the notch 3 dB bandwidth strictly bounded to under 8 Hz, preventing vocal consonant attenuation.'
          }
        }
      },
      {
        id: 'spatial_beamforming_architecture',
        title: 'Spatial Microphone Array',
        badge: 'BEAMFORM',
        content: {
          title: 'Delay-and-Sum 4-Microphone Beamformer & MVDR Nulling',
          badge: 'SPATIAL',
          description: 'Performs fractional delay FIR interpolation across 4 planar MEMS microphones, steering acoustic directivity toward the operator while nulling laterally incident rotor noise.',
          specs: [
            { param: 'Microphone Geometry', val: '4-Mic Circular / Linear Planar', note: 'Configurable array radius' },
            { param: 'Directivity Gain', val: '>24 dB', note: 'Measured at 1 kHz target angle' },
            { param: 'Interpolation Method', val: 'Fractional Lagrange FIR', note: 'Sub-sample delay without FFT latency' }
          ]
        }
      }
    ]
  },
  {
    id: 'voice_detection',
    title: '3. Voice Activity & KWS',
    badge: 'VOICE',
    sections: [
      {
        id: 'streaming_vad_architecture',
        title: 'Streaming VAD',
        badge: 'ENERGY',
        content: {
          title: 'Sub-Band Energy Ratio & Spectral Entropy Speech Segmenter',
          badge: 'DETECTOR',
          description: 'Distinguishes human speech from turbulent aerodynamics and drone propeller noise using spectral entropy and human vocal band (250 Hz - 3.4 kHz) power concentration.',
          specs: [
            { param: 'Frame Duration', val: '10.0 ms (160 samples)', note: 'At 16 kHz sample rate' },
            { param: 'Decision Latency', val: '<2.0 ms', note: 'Attack time constant' },
            { param: 'RAM Usage', val: '<8 KB', note: 'No dynamic allocations' }
          ]
        }
      },
      {
        id: 'edge_kws_architecture',
        title: 'Neural Wake-Word Spotting (KWS)',
        badge: 'NEURAL',
        content: {
          title: 'INT8 Quantized Depthwise-Separable CNN Keyword Spotter',
          badge: 'CLASSIFIER',
          description: 'Runs lightweight depthwise-separable 2D convolutions on log-mel spectrogram frames, detecting mission wake words in real time on microcontrollers.',
          specs: [
            { param: 'Inference Time', val: '<0.85 ms', note: 'On STM32H7 480 MHz ARM Cortex-M7' },
            { param: 'Weights Footprint', val: '48 KB (INT8 Quantized)', note: 'Fits comfortably in embedded flash' },
            { param: 'Precision in Noise', val: '>98.5% at 95 dB SPL', note: 'Trained with physical rotor noise augmentation' }
          ]
        }
      }
    ]
  },
  {
    id: 'speech_synthesis',
    title: '4. Physical Speech Synthesis',
    badge: 'SYNTHESIS',
    sections: [
      {
        id: 'vocal_tract_architecture',
        title: 'Liljencrants-Fant Vocal Tract Model',
        badge: 'PHYSICAL',
        content: {
          title: 'Parametric Klatt Formants & Glottal Flow Synthesis',
          badge: 'ACOUSTICS',
          description: 'Generates synthetic speech exemplars by simulating physical glottal airflow pulses and cascaded vocal tract acoustic resonators, enabling on-device synthetic training exemplar generation.',
          specs: [
            { param: 'Glottal Model', val: 'Liljencrants-Fant (LF) Model', note: 'Analytical velocity derivative waveform' },
            { param: 'Formant Resonators', val: 'F1 through F5 Digital Biquads', note: 'Adjustable bandwidths and frequencies' },
            { param: 'Sample Rate', val: '48 kHz / 16 kHz', note: 'Studio high-fidelity synthesis' }
          ]
        }
      }
    ]
  }
];
