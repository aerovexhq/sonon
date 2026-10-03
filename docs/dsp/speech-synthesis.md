# Physical Formant Speech Synthesis

Sonon embeds a pure mathematical articulatory speech synthesizer built on:
- **Liljencrants-Fant Glottal Flow**: Parametric differential model generating natural vocal cord excitation pulses with adjustable open quotient ($O_q$), return phase ($R_a$), and spectral tilt ($R_k$).
- **4-Pole Cascade Vocal Tract Filter**: Second-order resonator poles implementing resonant formants $F_1, F_2, F_3, F_4$.

This allows generating synthesized command words directly on-device without neural network memory overhead.

Try synthesizing phrases in the [Playground Speech Synthesis Tab](/playground).
