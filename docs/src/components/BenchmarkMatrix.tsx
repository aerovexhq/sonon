import React from 'react';
import { BarChart3, Zap } from 'lucide-react';

interface MatrixRow {
  metric: string;
  category: string;
  sonon: { text: string; highlight?: boolean };
  porcupine: { text: string };
  vosk: { text: string };
  webrtc: { text: string };
}

const COMPARISON_ROWS: MatrixRow[] = [
  {
    metric: 'Dynamic Rotor Harmonic Attenuation',
    category: 'Noise Immunity',
    sonon: { text: '>86 dB (Telemetry-Guided Biquad)', highlight: true },
    porcupine: { text: '0 dB (Requires external pre-filter)' },
    vosk: { text: '0 dB (Fails above 85 dB rotor noise)' },
    webrtc: { text: '~12 dB (Stationary noise suppression)' }
  },
  {
    metric: 'Keyword Spotting (KWS) Inference Latency',
    category: 'Wake-Word Speed',
    sonon: { text: '<0.85 ms (INT8 DS-CNN Cortex-M7)', highlight: true },
    porcupine: { text: '4.2 ms (Proprietary model)' },
    vosk: { text: '42.0 ms (Kaldi neural net)' },
    webrtc: { text: 'Unsupported (VAD only)' }
  },
  {
    metric: 'Embedded RAM Footprint',
    category: 'Memory Budget',
    sonon: { text: '<48 KB (Deterministic Static Bounded)', highlight: true },
    porcupine: { text: '180 KB' },
    vosk: { text: '48 MB (Requires Linux OS)' },
    webrtc: { text: '64 KB' }
  },
  {
    metric: 'Streaming VAD Frame Latency',
    category: 'Speech Segmenter',
    sonon: { text: '<2.0 ms (Spectral Entropy & Sub-Band)', highlight: true },
    porcupine: { text: '20.0 ms' },
    vosk: { text: '32.0 ms' },
    webrtc: { text: '10.0 ms' }
  },
  {
    metric: 'On-Device Physical Speech Synthesis',
    category: 'Exemplar Generation',
    sonon: { text: 'Liljencrants-Fant + Klatt (48 kHz)', highlight: true },
    porcupine: { text: 'Unsupported' },
    vosk: { text: 'Unsupported' },
    webrtc: { text: 'Unsupported' }
  },
  {
    metric: 'Language & Flight Computer Safety',
    category: 'Safety Standard',
    sonon: { text: 'Pure Safe Rust (#![deny(unsafe_code)])', highlight: true },
    porcupine: { text: 'Closed-source binary blob' },
    vosk: { text: 'C++ with dynamic memory allocs' },
    webrtc: { text: 'C/C++ legacy codebase' }
  }
];

export function BenchmarkMatrix() {
  return (
    <section className="py-20 px-6 max-w-7xl mx-auto w-full z-10 relative">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs font-mono text-rose-400 mb-3">
            <BarChart3 className="w-3.5 h-3.5" />
            <span>EMPIRICAL BENCHMARKS // ACOUSTIC DSP &amp; KWS PERFORMANCE</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight font-sans">
            Sonon vs Embedded Speech Engines
          </h2>
          <p className="text-sm text-slate-400 mt-2 max-w-2xl leading-relaxed">
            Direct comparison across multi-rotor noise rejection depth, wake-word spotting inference latency, RAM budget, and physical articulatory speech synthesis.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs text-slate-400 bg-[#0f1012] p-2 rounded-xl border border-white/[0.08]">
          <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></span>
          <span>BENCHMARK SUITE: 100 dB SPL QUADROTOR HOVER NOISE</span>
        </div>
      </div>

      <div className="bg-[#0f1012] border border-white/[0.08] rounded-2xl overflow-hidden shadow-2xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-white/[0.08] bg-[#15171a] font-mono text-[11px] text-slate-300">
                <th className="py-4 px-6 font-bold uppercase tracking-wider">Evaluation Dimension</th>
                <th className="py-4 px-6 font-bold text-rose-400 uppercase tracking-wider bg-rose-500/[0.06] border-x border-rose-500/20">
                  Sonon Engine
                </th>
                <th className="py-4 px-6 font-semibold uppercase tracking-wider text-slate-400">Porcupine v3</th>
                <th className="py-4 px-6 font-semibold uppercase tracking-wider text-slate-400">Vosk Kaldi</th>
                <th className="py-4 px-6 font-semibold uppercase tracking-wider text-slate-400">WebRTC VAD</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono">
              {COMPARISON_ROWS.map((row, idx) => (
                <tr key={idx} className="hover:bg-white/[0.02] transition">
                  <td className="py-4 px-6 text-slate-300 font-sans font-medium">
                    <div className="text-sm font-semibold text-white">{row.metric}</div>
                    <div className="text-xs text-slate-500 font-mono mt-0.5">{row.category}</div>
                  </td>

                  <td className="py-4 px-6 text-white font-bold bg-rose-500/[0.04] border-x border-rose-500/10">
                    <div className="flex items-center gap-2 text-rose-400">
                      <Zap className="w-3.5 h-3.5 fill-current shrink-0" />
                      <span>{row.sonon.text}</span>
                    </div>
                  </td>

                  <td className="py-4 px-6 text-slate-400">{row.porcupine.text}</td>
                  <td className="py-4 px-6 text-slate-400">{row.vosk.text}</td>
                  <td className="py-4 px-6 text-slate-400">{row.webrtc.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
