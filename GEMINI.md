# Sonon Engineering Guidelines & Rules

## Professional Robotics Engineering Mandate
- **Commercial Robotics Standard**: Sonon is an enterprise acoustic DSP, wake-word detection, and realistic speech synthesis engine engineered by Aerovex HQ for production autonomous robotics and edge embedded systems. Non-professionalism is strictly intolerable. Every code modification, architectural decision, commit message, and documentation update must adhere to production-grade engineering standards.
- **Core Mission & Strategic Priorities**:
  1. **Wake-Word Spotting (KWS)**: Primary objective is ultra-low latency, hard real-time, low-power keyword spotting and streaming Voice Activity Detection (VAD) resilient to extreme ambient noise (rotor harmonics, propwash, motor interference).
  2. **Realistic Speech Synthesis for Instant Synthetic Data Generation**: High-fidelity, physically grounded articulatory and formant speech synthesis (Klatt filter banks, Liljencrants-Fant glottal flow dynamics) is a vital core subsystem. Its principal role is enabling instantaneous on-device/in-pipeline generation of diverse acoustic training exemplars, pitch/rate perturbations, and accent calibrations without dependence on external cloud models or bulky audio datasets.
  3. **Zero Scope Drift & Aerodynamics Freeze**: Maintain strict discipline. Do NOT improve, modify, or expand aerodynamics, aeroacoustic inverse modeling, or fluid-mechanical features under any circumstances. Aerodynamics features are strictly frozen. Peripheral research phases (such as neuromorphic hardware drivers, speculative radar, crypto-steganography, or terrain altimetry) are strictly eliminated. All work must exclusively advance wake-word spotting and realistic speech synthesis for instant synthetic data generation. Existing systems must be preserved with 100% test pass rates and zero regressions.
- **Tone & Code Quality**: Zero emojis across the entire codebase, documentation, tests, and commit messages. Maintain 100% pure safe Rust (`#![deny(unsafe_code)]`).

## Git & CI/CD Deployment Policy
- **Deploy on Every Commit**: Documentation (`docs/`) is hosted on GitHub Pages (`https://sonon.aerovex.net`). The deployment workflow runs automatically on every single push to `main`. Never restrict trigger paths in `.github/workflows/deploy-pages.yml`.
- **Zero Build Artifacts in Git**: Never commit `dist/`, `.vitepress/dist/`, `.vitepress/cache/`, `node_modules/`, or compiled WASM binaries to git history. All build artifacts must be generated within ephemeral CI runners.
- **Commit Message Hygiene**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`). Never reference development phase numbers or milestone identifiers in commit titles.
