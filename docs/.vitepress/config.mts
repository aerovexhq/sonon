import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Sonon',
  description: 'High-Performance Embedded Acoustic DSP, Wake-Word Spotting & Speech Synthesis in Pure Safe Rust',
  cleanUrls: true,
  ignoreDeadLinks: true,
  appearance: 'dark',
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['link', { rel: 'alternate icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#08090a' }],
    ['meta', { name: 'color-scheme', content: 'dark' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Sonon — Mission-Critical Acoustic Signal Processing for Autonomous Robotics' }],
    ['meta', { property: 'og:description', content: 'Pure safe Rust digital signal processing suite engineered for autonomous drone operations: dynamic rotor notch filtering, spatial beamforming, sub-ms KWS, and physical articulatory speech synthesis.' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap', rel: 'stylesheet' }]
  ],
  themeConfig: {
    logo: '/favicon.svg',
    siteTitle: 'Sonon.',
    nav: [
      { text: 'Voice Synthesis', link: '/synthesis' },
      { text: 'Playground', link: '/playground' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'DSP Physics', link: '/dsp/overview' },
      { text: 'API Reference', link: '/api/rust' },
      {
        text: 'Aerovex Ecosystem',
        items: [
          { text: 'Aerovex Platform', link: 'https://aerovex.net', target: '_blank' },
          { text: 'Axiom EDA', link: 'https://axiom.aerovex.net', target: '_blank' },
          { text: 'Phonon CAD', link: 'https://phonon.aerovex.net', target: '_blank' },
          { text: 'Atoma CAD', link: 'https://atoma.aerovex.net', target: '_blank' },
          { text: 'Sonon DSP', link: 'https://sonon.aerovex.net', target: '_self' },
          { text: 'Vexel Lang', link: 'https://vexel.aerovex.net', target: '_blank' },
          { text: 'Appify', link: 'https://appify.aerovex.net', target: '_blank' }
        ]
      },
      { text: 'GitHub', link: 'https://github.com/aerovexhq/sonon' }
    ],
    sidebar: [
      {
        text: '01 // GETTING STARTED',
        collapsed: false,
        items: [
          { text: 'Introduction & Overview', link: '/guide/getting-started' },
          { text: 'Architecture & Core Principles', link: '/guide/architecture' },
          { text: 'Quickstart & Installation', link: '/guide/quickstart' }
        ]
      },
      {
        text: '02 // ACOUSTIC DSP & PHYSICS',
        collapsed: false,
        items: [
          { text: '30 Research Monograph Phases', link: '/dsp/overview' },
          { text: 'Feature Extraction & Mel Filterbanks', link: '/dsp/feature-extraction' },
          { text: 'Dynamic Rotor Harmonic Notch Bank', link: '/dsp/rotor-notch-filter' },
          { text: 'Sakoe-Chiba Banded DTW Spotting', link: '/dsp/wake-word-spotting' },
          { text: 'Physical Formant Speech Synthesis', link: '/dsp/speech-synthesis' },
          { text: 'Continuous Wavelet Diagnostics (CWT)', link: '/dsp/wavelet-diagnostics' },
          { text: 'Biosonar Echolocation & 3D Mapping', link: '/dsp/echolocation' }
        ]
      },
      {
        text: '03 // ENGINE APIS & INTEGRATION',
        collapsed: false,
        items: [
          { text: 'Pure Safe Rust Engine API', link: '/api/rust' },
          { text: 'C-ABI & POSIX Shared Memory', link: '/api/c-abi' },
          { text: 'Python Client (NumPy Zero-Copy)', link: '/api/python' },
          { text: 'WebAssembly AudioWorklet API', link: '/api/wasm' }
        ]
      },
      {
        text: '04 // INTERACTIVE LABS',
        collapsed: false,
        items: [
          { text: 'Voice Synthesis Studio', link: '/synthesis' },
          { text: 'Autonomous Audio Lab Playground', link: '/playground' }
        ]
      }
    ],
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
