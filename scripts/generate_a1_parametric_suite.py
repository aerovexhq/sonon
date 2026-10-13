#!/usr/bin/env python3
"""Sonon Tier A.1: Classical Parametric All-Pole Inversion Suite Generator.

Consolidates and exports all Tier A.1 versioned audio masters:
- A.1.1: Order-18 All-Pole Baseline (-23.4 dB Treble Muffle)
- A.1.2: Lip Radiation Differentiated (Treble restored, 371 clicks)
- A.1.3: Silk-Crisp Continuous S-Curve (Click-free, baseline favorite)
- A.1.4: Formant-Balanced Parametric (Noise fizz, deprecated)
- A.1.5: Direct Room Presence (A.1.3 core + 5 dB room presence filter)

Zero emojis. Strictly safe execution.
"""

import os
import shutil
import soundfile as sf
import numpy as np

OUTPUT_DIR = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis"

# Source file mapping to formal taxonomy
MAPPING = {
    "sonon_arduino_a1_1_muffle.wav": "sonon_arduino_v1_muffle.wav",
    "sonon_arduino_a1_2_crisp.wav": "sonon_arduino_v2_crisp.wav",
    "sonon_arduino_a1_3_silkcrisp.wav": "sonon_arduino_v3_silkcrisp.wav",
    "sonon_arduino_a1_4_perfected.wav": "sonon_arduino_v4_perfected.wav",
    "sonon_arduino_a1_5_presence.wav": "sonon_arduino_v5_presence_direct.wav",
}


def main():
    print("=" * 70)
    print("Sonon Tier A.1: Classical Parametric Inversion Master Suite")
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
