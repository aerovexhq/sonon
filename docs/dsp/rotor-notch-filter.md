# Dynamic Rotor Harmonic Notch Bank

When operating on multi-rotor UAVs, motor propeller noise can exceed human speech energy by up to $+25\text{ dB}$.

Sonon couples directly to autopilot electronic speed controller (ESC) telemetry to track motor rotational speed ($\text{RPM}$) and compute blade pass frequencies (BPF):

$$f_k = k \cdot \frac{N_{\text{blades}} \cdot \text{RPM}}{60}$$

## Attenuation Performance

- **Filter Topology**: Direct Form II Transposed Biquad Notch Filter.
- **Harmonics Covered**: Fundamental ($1\times$), 2nd ($2\times$), and 3rd ($3\times$) blade pass harmonics.
- **Attenuation Depth**: $> 40\text{ dB}$ at notch center frequency.
- **Q Factor**: Adaptive bandwidth narrowing ($Q \approx 15\text{--}25$) to minimize voice formant distortion.

Test this live in the [Playground Rotor Noise Tab](/playground).
