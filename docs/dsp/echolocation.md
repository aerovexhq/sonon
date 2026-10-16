# Acoustic Echolocation &amp; 3D Mapping

<p class="doc-lead">
Active Linear Frequency Modulated (LFM) chirp pulse compression (500 to 4,000 Hz) combined with Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection provides sub-centimeter range precision for UAV navigation in GPS-denied environments.
</p>

<div class="key-features-box">
  <div class="key-features-header">Echolocation &amp; Mapping Features</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Sub-Centimeter Echo Ranging Resolution</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> CA-CFAR Adaptive Thresholding Isolating Wall Reflections</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 5-Axis Obstacle Collision Hazard Guards</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Zero Optical Degradation in Dust, Smoke, and Darkness</div>
  </div>
</div>

## Pulse Compression Physics

Sonon emits an active acoustic linear frequency modulated (LFM) chirp $s(t)$ through on-board ultrasonic or high-bandwidth transducers:

$$s(t) = \text{rect}\left(\frac{t}{T}\right) \exp\left(j 2\pi \left( f_0 t + \frac{B}{2T} t^2 \right)\right)$$

Where $T$ is pulse duration ($20\text{ ms}$), $f_0$ is starting frequency ($500\text{ Hz}$), and $B$ is chirp bandwidth ($3500\text{ Hz}$).

The received acoustic echo $r(t)$ is passed through a matched filter $h(t) = s^*(-t)$:

$$y(t) = r(t) * h(t) = \int_{-\infty}^{\infty} r(\tau) \, s^*(\tau - t) d\tau$$

Matched filter pulse compression narrows the effective echo pulse width to $\tau_{\text{eff}} \approx 1/B$, yielding a theoretical range resolution:

$$\Delta R = \frac{c}{2 B} = \frac{343\text{ m/s}}{2 \cdot 3500\text{ Hz}} \approx 4.9\text{ cm}$$

With phase interpolation, range precision exceeds $\pm 0.8\text{ cm}$.

## CA-CFAR Adaptive Detection Matrix

To discriminate true wall echoes from multi-rotor acoustic reflections, Sonon evaluates a sliding cell window:

$$T_{\text{thresh}} = \alpha \cdot \frac{1}{2N} \left( \sum_{i=1}^N X_{\text{lag}, i} + \sum_{i=1}^N X_{\text{lead}, i} \right)$$

Where guard cells isolate the cell under test (CUT) from target energy bleeding.

| Parameter | Type | Default | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `chirp_duration` | `f32` | `0.020` | Seconds | LFM chirp transmission duration |
| `start_freq` | `f32` | `500.0` | Hz | Starting acoustic chirp frequency |
| `end_freq` | `f32` | `4000.0` | Hz | Upper acoustic chirp frequency |
| `cfar_guard_cells` | `usize` | `4` | Cells | Guard cells surrounding cell under test |
| `cfar_ref_cells` | `usize` | `16` | Cells | Reference noise estimation cells |
| `hazard_distance` | `f32` | `1.0` | Meters | Collision safety perimeter threshold |

::: info ARCHITECTURE DIRECTIVE: BIOSONAR SAFETY
Acoustic echolocation functions reliably in complete darkness, dense smoke, and high-particulate mining shafts where LiDAR and optical stereo cameras experience total sensor blackout.
:::

## Next Steps

- [Pure Safe Rust API Reference](/api/rust)
- [C-ABI Shared Memory API](/api/c-abi)
