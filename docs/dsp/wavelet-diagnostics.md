# Continuous Wavelet Diagnostics (CWT)

<p class="doc-lead">
Using continuous Complex Morlet wavelets and scale-wise spectral kurtosis, Sonon continuously inspects UAV structural integrity, detecting propeller micro-fractures, motor bearing race spalling, and airframe harmonic flutter before catastrophic in-flight failure.
</p>

<div class="key-features-box">
  <div class="key-features-header">Acoustic Diagnostics Capabilities</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Complex Morlet Wavelet Scalogram Profiling</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Scale-Wise Kurtosis Bearing Race Damage Metric</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Propeller Blade Flutter Subharmonic Detection</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Autonomous Early-Warning Flight Warning Flags</div>
  </div>
</div>

## Wavelet Formulation

The continuous wavelet transform (CWT) projects the discrete acoustic signal $x(t)$ onto scaled and translated mother wavelets $\psi(t)$:

$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t - b}{a}\right) dt$$

Sonon selects the Complex Morlet mother wavelet with bandwidth parameter $f_b$ and center frequency $f_c$:

$$\psi(t) = \frac{1}{\sqrt{\pi f_b}} \exp\left(j 2\pi f_c t - \frac{t^2}{f_b}\right)$$

### Scale-Wise Spectral Kurtosis

Bearing impacts generate repetitive high-frequency impulses that manifest as extreme deviations from Gaussian noise statistics. The scale-wise kurtosis $K(a)$ is computed across time shifts $b$:

$$K(a) = \frac{\frac{1}{N} \sum_{b=1}^N (|W(a, b)| - \mu_{|W|})^4}{\left(\frac{1}{N} \sum_{b=1}^N (|W(a, b)| - \mu_{|W|})^2\right)^2}$$

An empirical kurtosis threshold $K(a) > 4.5$ indicates localized raceway micro-spalling or ball bearing degradation.

## Failure Mode Diagnostic Matrix

| Failure Mode | Frequency Signature | Detection Threshold | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Propeller Micro-Cracks** | Subharmonics ($0.5\times, 1.5\times$ rotor BPF) | Modulation Index &gt; 0.25 | Pilot advisory: land immediately |
| **Bearing Raceway Spall** | High-frequency kurtosis burst ($4\text{--}8\text{ kHz}$) | Kurtosis $K(a) &gt; 4.5$ | Maintenance flag logged |
| **Airframe Flutter** | Sub-30 Hz modal resonance shift | Energy ratio shift &gt; 35% | Restrict flight speed envelope |
| **Rotor Delamination** | 1st harmonic asymmetry between arms | Phase variance &gt; 22 deg | Autonomous return-to-launch (RTL) |

::: tip PRO TIP: PROPELLANT RESONANCE
Enable wavelet profiling during high-speed forward flight segments where aerodynamic stall flutter is most likely to initiate propeller blade failure.
:::

## Next Steps

- [Biosonar Echolocation &amp; 3D Mapping](/dsp/echolocation)
- [Pure Safe Rust Engine API](/api/rust)
