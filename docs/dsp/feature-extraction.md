# Feature Extraction &amp; Mel Filterbanks

<p class="doc-lead">
Sonon converts continuous 16 kHz acoustic streams into normalized spectral representations in streaming 10-millisecond frames (160 samples), executing windowing, 512-point Radix-2 FFT, triangular Mel filterbank integration, and Per-Channel Energy Normalization (PCEN).
</p>

<div class="key-features-box">
  <div class="key-features-header">Feature Extraction Pipeline Highlights</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 512-point Cooley-Tukey Radix-2 FFT in 18 microseconds</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 13 to 40 Band Pre-Calculated Mel Filterbanks</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Per-Channel Energy Normalization (PCEN) Dynamic Compression</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Pre-Emphasis High-Pass Filter ($y[n] = x[n] - 0.97 x[n-1]$)</div>
  </div>
</div>

## Mathematical Formulation

### 1. Pre-Emphasis &amp; Windowing

High-frequency speech formants typically exhibit a $-6\text{ dB/octave}$ spectral roll-off due to glottal volume velocity. A pre-emphasis FIR filter balances the frequency spectrum:

$$y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha \approx 0.97$$

The pre-emphasized frame is windowed using a Hamming window to attenuate sidelobe leakage:

$$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N - 1}\right), \quad 0 \le n < N$$

### 2. Mel Filterbank Mapping

The Mel scale models the nonlinear human auditory frequency resolution:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

Triangular filter weighting functions $H_m[k]$ integrate energy across adjacent FFT bins $k$:

$$H_m[k] = \begin{cases} 0, & k < f[m-1] \\ \frac{k - f[m-1]}{f[m] - f[m-1]}, & f[m-1] \le k \le f[m] \\ \frac{f[m+1] - k}{f[m+1] - f[m]}, & f[m] \le k \le f[m+1] \\ 0, & k > f[m+1] \end{cases}$$

### 3. Per-Channel Energy Normalization (PCEN)

Standard logarithmic compression amplifies stationary background motor noise during quiet speech pauses. Sonon implements PCEN:

$$\text{PCEN}(t, f) = \left( \frac{E(t, f)}{(\epsilon + M(t, f))^\alpha} + \delta \right)^r - \delta^r$$

Where $M(t, f) = (1 - s) M(t-1, f) + s E(t, f)$ represents an autoregressive running energy smoother.

## Processing Parameter Specifications

| Parameter | Type | Default | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `frame_length` | `usize` | `160` | Samples | Discrete temporal analysis window (10 ms @ 16 kHz) |
| `n_fft` | `usize` | `512` | Points | Zero-padded FFT length |
| `pre_emphasis` | `f32` | `0.97` | Coefficient | High-pass spectral tilt compensator |
| `n_mels` | `usize` | `13` | Bins | Number of triangular filter channels |
| `pcen_s` | `f32` | `0.025` | Smoothing | Autoregressive smoother factor |
| `pcen_alpha` | `f32` | `0.98` | Exponent | Dynamic range gain exponent |
| `pcen_delta` | `f32` | `2.0` | Bias | Numerical floor stabilizer |
| `pcen_r` | `f32` | `0.5` | Compression | Power-law output compression |

::: tip PRO TIP: PCEN VS LOG-MEL
Under high-amplitude drone propeller noise, PCEN outperforms standard Log-Mel spectrograms by over 28% in wake-word detection accuracy by continuously adapting the gain per frequency channel.
:::

## Next Steps

- [Dynamic Rotor Harmonic Notch Bank](/dsp/rotor-notch-filter)
- [Sakoe-Chiba Banded DTW Spotting](/dsp/wake-word-spotting)
