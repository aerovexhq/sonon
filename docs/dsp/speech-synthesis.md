# Physical Formant Speech Synthesis

<p class="doc-lead">
Sonon embeds a pure physical articulatory speech synthesizer built on the Liljencrants-Fant glottal flow differential model and an 8-stage cascaded digital waveguide filter. It synthesizes intelligible tactical speech directly on-device without neural network weights or GPU VRAM.
</p>

<div class="key-features-box">
  <div class="key-features-header">Acoustic Synthesizer Highlights</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 64x Faster Than Real-Time Render on Microcontrollers</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 1.2 MB Total Static Memory Footprint</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero GPU / VRAM Requirements (Runs Bare-Metal)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Sub-2 ms Synthesis Start Latency</div>
  </div>
</div>

## Glottal Source &amp; Vocal Tract Physics

### 1. Liljencrants-Fant (LF) Glottal Flow Model

The glottal volume velocity derivative $U_g'(t)$ describes the physical airflow pulse expelled through vocal fold vibrations:

$$U_g'(t) = \begin{cases} E_0 e^{\alpha t} \sin(\omega_g t), & 0 \le t < T_e \\ -\frac{E_e}{\epsilon T_a} \left( e^{-\epsilon (t - T_e)} - e^{-\epsilon (T_c - T_e)} \right), & T_e \le t < T_c \end{cases}$$

Key physical parameters control voice personality:
- **Fundamental Pitch ($F_0$)**: $T_0 = 1 / F_0$
- **Open Quotient ($O_q = T_e / T_0$)**: Breathiness and vocal effort
- **Return Phase ($R_a = T_a / T_0$)**: Spectral tilt and high-frequency energy
- **Speed Quotient ($R_k = (T_e - T_p) / T_p$)**: Asymmetry of glottal pulse rise and fall

### 2. Cascaded Digital Waveguide Resonators

The acoustic resonance of the pharyngeal and oral cavities is modeled as an 8-stage cascade of second-order digital resonators:

$$H_{\text{tract}}(z) = \prod_{m=1}^8 \frac{1 - 2 r_m \cos(\theta_m) + r_m^2}{1 - 2 r_m \cos(\theta_m) z^{-1} + r_m^2 z^{-2}}$$

Where pole frequencies and bandwidths correspond directly to anatomical formants:

$$\theta_m = \frac{2\pi F_m}{f_s}, \quad r_m = e^{-\pi B_m / f_s}$$

- **Formant $F_1$ (300 to 900 Hz)**: Pharyngeal cavity volume (vowel height: /i/ vs /a/)
- **Formant $F_2$ (800 to 2500 Hz)**: Oral cavity length (tongue fronting: /u/ vs /i/)
- **Formant $F_3$ (2200 to 3200 Hz)**: Retroflexion and dental constrictions (/r/, /l/)
- **Formants $F_4 \dots F_8$**: Upper vocal tract head acoustics and radiation resistance

## Physical Synthesizer Parameters

| Parameter | Type | Default | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `pitch_f0` | `f32` | `120.0` | Hz | Fundamental pitch frequency (80 to 350 Hz) |
| `open_quotient` | `f32` | `0.62` | Dimensionless | LF glottal open quotient ratio ($T_e / T_0$) |
| `spectral_tilt` | `f32` | `0.05` | Dimensionless | High-frequency damping factor ($R_a$) |
| `f1_freq` | `f32` | `500.0` | Hz | First resonant formant frequency |
| `f2_freq` | `f32` | `1500.0` | Hz | Second resonant formant frequency |
| `f3_freq` | `f32` | `2500.0` | Hz | Third resonant formant frequency |
| `f4_freq` | `f32` | `3500.0` | Hz | Fourth resonant formant frequency |
| `flutter` | `f32` | `0.0` | Dimensionless | Glottal pulse jitter / flutter amplitude |

::: info ARCHITECTURE DIRECTIVE: ZERO NEURAL VRAM
Unlike neural TTS models (e.g. VITS, Tacotron 2, Piper) that require hundreds of megabytes of model checkpoints and GPU tensor accelerators, Sonon's physical formant synthesizer requires strictly 1.2 MB RAM and runs comfortably on embedded microcontrollers with zero heap churn.
:::

## Next Steps

- [Test Real-Time Formants in Voice Synthesis Studio](/synthesis)
- [Review Pure Safe Rust Engine API](/api/rust)
