import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Sonon',
  description: 'High-Performance Embedded Acoustic DSP, Wake-Word Spotting & Speech Synthesis in Pure Safe Rust',
  cleanUrls: true,
  ignoreDeadLinks: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['link', { rel: 'alternate icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#06b6d4' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Sonon — Autonomous Acoustic Audio Lab & DSP Engine' }],
    ['meta', { property: 'og:description', content: 'Zero-cloud acoustic DSP, few-shot wake-word spotting, rotor noise suppression, and speech synthesis in WebAssembly.' }]
  ],
  themeConfig: {
    logo: '/favicon.svg',
    siteTitle: 'Sonon',
    nav: [
      { text: 'Voice Synthesis', link: '/synthesis' },
      { text: 'Playground', link: '/playground' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'DSP Physics', link: '/dsp/overview' },
      { text: 'API Reference', link: '/api/rust' },
      { text: 'GitHub', link: 'https://github.com/aerovexhq/sonon' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction & Overview', link: '/guide/getting-started' },
            { text: 'Architecture & Core Principles', link: '/guide/architecture' },
            { text: 'Quickstart & Installation', link: '/guide/quickstart' }
          ]
        }
      ],
      '/dsp/': [
        {
          text: 'Acoustic DSP & Physics',
          items: [
            { text: 'Overview of 30 Monograph Phases', link: '/dsp/overview' },
            { text: 'Feature Extraction & Mel Spectrogram', link: '/dsp/feature-extraction' },
            { text: 'Rotor Notch Bank & BPF Telemetry', link: '/dsp/rotor-notch-filter' },
            { text: 'Sakoe-Chiba Banded DTW Spotting', link: '/dsp/wake-word-spotting' },
            { text: 'Klatt + LF Formant Speech Synthesis', link: '/dsp/speech-synthesis' },
            { text: 'Continuous Wavelet Diagnostics (CWT)', link: '/dsp/wavelet-diagnostics' },
            { text: 'Acoustic Echolocation & 3D Mapping', link: '/dsp/echolocation' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'Engine APIs & Bindings',
          items: [
            { text: 'Pure Safe Rust API', link: '/api/rust' },
            { text: 'C-ABI & Shared Memory FFI', link: '/api/c-abi' },
            { text: 'Python Client (NumPy Zero-Copy)', link: '/api/python' },
            { text: 'WebAssembly (WASM 32-bit)', link: '/api/wasm' }
          ]
        }
      ],
      '/playground': [
        {
          text: 'Interactive Lab',
          items: [
            { text: 'Autonomous Audio Lab', link: '/playground' }
          ]
        }
      ],
      '/synthesis': [
        {
          text: 'Voice Synthesis',
          items: [
            { text: 'Voice Synthesis Studio', link: '/synthesis' },
            { text: 'Formant & Glottal Physics', link: '/dsp/speech-synthesis' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/aerovexhq/sonon' }
    ],
    footer: {
      message: 'Released under the Apache-2.0 / MIT License.',
      copyright: 'Copyright © 2026 Aerovex HQ & Sonon Contributors'
    },
    search: {
      provider: 'local'
    }
  }
})
