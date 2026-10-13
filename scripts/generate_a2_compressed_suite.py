#!/usr/bin/env python3
"""Sonon Tier A.2: Direct Neural Model Compression & Quantization Suite Generator.

Consolidates and exports all Tier A.2 versioned audio masters:
- A.2.1: Post-Training Quantized (W8A8) Full Neural Vocoder (970 KB Flash, metallic timbre)
- A.2.2: Formant-Constrained Distilled Student FP32 (272 KB Flash)
- A.2.3: Formant-Constrained Quantized Student W8A8 (73.6 KB Flash, 332 B RAM)
- A.2.4: Safe Rust MCU Hardware Engine (332 B SRAM, 52.8x RTF, fixed-point MCU sim)
- A.2.5: Direct Formant-Harmonic Synthesizer (Formant-matched DSP baseline)

Also maps A.0: Foundational Neural Baseline (82M Flow Matching DiT + BigVGAN Vocoder)

Zero emojis. Strictly safe execution.
"""

import os
import shutil
import soundfile as sf
import numpy as np

OUTPUT_DIR = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis"

# Source file mapping to formal taxonomy
MAPPING = {
    "sonon_foundation_a0_baseline.wav": "sonon_tactical_advisory_bigvgan_neural.wav",
    "sonon_arduino_a2_1_ptq.wav": "sonon_arduino_b1_ptq.wav",
    "sonon_arduino_a2_2_distilled.wav": "sonon_arduino_b2_distilled.wav",
    "sonon_arduino_a2_3_quantized.wav": "sonon_arduino_b3_quantized.wav",
    "sonon_arduino_a2_4_mcu_sim.wav": "sonon_arduino_b4_mcu_sim.wav",
    "sonon_arduino_a2_5_formant_harmonic.wav": "sonon_arduino_b5_formant_harmonic.wav",
}


def main():
    print("=" * 70)
    print("Sonon Tier A.2: Neural Compression & Quantization Master Suite")
    print("=" * 70)

    for target_name, src_name in MAPPING.items():
        src_path = os.path.join(OUTPUT_DIR, src_name)
        target_path = os.path.join(OUTPUT_DIR, target_name)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, target_path)
            info = sf.info(target_path)
            audio, sr = sf.read(target_path)
            diff = np.abs(np.diff(audio))
            spikes_045 = len(np.where(diff > 0.45)[0])
            print(f"Exported {target_name}: {info.duration:.2f}s | Spikes > 0.45: {spikes_045:3d} | Max jump: {np.max(diff):.4f}")
        else:
            print(f"Warning: Source {src_path} not found.")

    print("=" * 70)


if __name__ == "__main__":
    main()
