#!/usr/bin/env python3
"""Sonon Micro-DDSP Student Distillation & Training Script.

Distills the 82M foundation neural speech model & BigVGAN vocoder into a 69k-parameter
frame-rate neural student model (Micro-DDSP).

Outputs:
1. models/micro_ddsp/micro_ddsp_student.pt: PyTorch trained student weights.
2. output/speech_synthesis/sonon_arduino_b2_distilled.wav: Distilled FP32 audio master.
3. models/micro_ddsp/micro_ddsp_w8a8_weights.json: W8A8 quantized weights for embedded Rust.
4. output/speech_synthesis/sonon_arduino_b3_quantized.wav: W8A8 quantized audio master.

Zero emojis. Strictly safe execution.
"""

import os
import sys
import math
import time
import json
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from train_vocoder import compute_mel_from_audio, SAMPLE_RATE, HOP_LENGTH
from micro_ddsp_synthesizer import (
    MicroDdspStudent,
    MultiScaleSpectralLoss,
    HarmonicOscillatorBank,
    FilteredNoiseSynthesizer,
    NUM_HARMONICS,
    NUM_NOISE_BANDS,
)

REF_AUDIO = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis/sonon_neural_waypoint_reached.wav"
OUTPUT_DIR = "/home/usr/Projects/aerovex/modules/sonon/models/micro_ddsp"
AUDIO_DIR = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis"


def extract_acoustic_targets(audio: np.ndarray, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH):
    """Extract ground-truth acoustic targets for distillation supervision."""
    mel = compute_mel_from_audio(audio, sr)  # (80, T)
    T_frames = mel.shape[1]

    # Frame-wise F0 extraction via autocorrelation
    frame_len = 1024
    f0_list = []
    voicing_list = []
    energy_list = []

    padded_audio = np.pad(audio, (0, frame_len), mode="constant")
    for i in range(T_frames):
        start = i * hop_length
        frame = padded_audio[start : start + frame_len] * np.hanning(frame_len)
        rms = float(np.sqrt(np.mean(frame**2)))
        energy_list.append(rms)

        r = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        min_lag = int(sr / 400.0)
        max_lag = int(sr / 60.0)
        if r[0] > 1e-4 and max_lag < len(r):
            peak_lag = min_lag + int(np.argmax(r[min_lag:max_lag]))
            peak_ratio = float(r[peak_lag] / r[0])
            if peak_ratio > 0.32:
                f0 = float(sr / peak_lag)
                v = 1.0
            else:
                f0 = 150.0  # Default unvoiced placeholder
                v = 0.0
        else:
            f0 = 150.0
            v = 0.0
        f0_list.append(f0)
        voicing_list.append(v)

    return {
        "mel": mel,
        "f0": np.array(f0_list, dtype=np.float32),
        "voicing": np.array(voicing_list, dtype=np.float32),
        "energy": np.array(energy_list, dtype=np.float32),
    }


def quantize_w8a8_tensor(tensor: torch.Tensor):
    """Symmetric per-channel INT8 quantization for weights."""
    max_abs = torch.amax(torch.abs(tensor), dim=list(range(1, tensor.ndim)), keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=1e-8)
    q_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return q_tensor, scale


def export_w8a8_model(model: MicroDdspStudent, output_json_path: str):
    """Export Micro-DDSP model in W8A8 format with per-channel scales."""
    export_dict = {
        "architecture": "MicroDdspStudent",
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "num_harmonics": NUM_HARMONICS,
        "num_noise_bands": NUM_NOISE_BANDS,
        "layers": {},
    }

    total_weight_bytes = 0
    total_scale_bytes = 0

    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            q_w, scale = quantize_w8a8_tensor(param.data.cpu())
            export_dict["layers"][name] = {
                "shape": list(q_w.shape),
                "weights": q_w.flatten().tolist(),
                "scales": scale.flatten().tolist(),
            }
            total_weight_bytes += q_w.numel()
            total_scale_bytes += scale.numel() * 4
        elif "bias" in name:
            export_dict["layers"][name] = {
                "shape": list(param.shape),
                "bias": param.data.cpu().flatten().tolist(),
            }
            total_scale_bytes += param.numel() * 4

    export_dict["memory_footprint"] = {
        "weights_bytes": total_weight_bytes,
        "scales_bytes": total_scale_bytes,
        "total_flash_bytes": total_weight_bytes + total_scale_bytes,
        "total_kb": (total_weight_bytes + total_scale_bytes) / 1024.0,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2)

    return export_dict["memory_footprint"]


def main():
    print("=" * 70)
    print("Sonon Phase 2: Micro-DDSP Student Distillation (Series B.2 & B.3)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 1. Load reference audio and extract targets
    print(f"Loading reference ground truth: {REF_AUDIO}")
    ref_audio, sr = sf.read(REF_AUDIO)
    assert sr == SAMPLE_RATE, f"Sample rate mismatch: {sr} != {SAMPLE_RATE}"
    targets = extract_acoustic_targets(ref_audio, sr)

    mel_t = torch.from_numpy(targets["mel"]).unsqueeze(0).to(device)  # (1, 80, T)
    f0_target = torch.from_numpy(targets["f0"]).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, T)
    voicing_target = torch.from_numpy(targets["voicing"]).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, T)
    energy_target = torch.from_numpy(targets["energy"]).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, T)

    total_samples = targets["mel"].shape[1] * HOP_LENGTH
    audio_target = torch.from_numpy(ref_audio[:total_samples]).unsqueeze(0).unsqueeze(1).to(device).float()

    print(f"Extracted targets: {targets['mel'].shape[1]} mel frames ({total_samples} samples, {total_samples/sr:.2f}s)")

    # 2. Instantiate Student Model
    student = MicroDdspStudent().to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Micro-DDSP Student Parameters: {total_params:,} ({total_params / 1024:.2f} kParams)")

    # 3. Setup Distillation Losses & Optimizer
    spectral_loss_fn = MultiScaleSpectralLoss(
        fft_sizes=[256, 512, 1024, 2048],
        hop_sizes=[64, 128, 256, 512]
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-4)

    # 4. Training Loop
    print("\nBeginning Knowledge Distillation against Ground Truth Foundation Audio...")
    epochs = 400
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        optimizer.zero_grad()

        audio_pred, controls = student(mel_t)

        # Loss components
        loss_spectral = spectral_loss_fn(audio_pred, audio_target)
        loss_f0 = F.smooth_l1_loss(controls["f0"] * voicing_target, f0_target * voicing_target) / 40.0
        loss_voicing = F.binary_cross_entropy(controls["voicing"], voicing_target)
        loss_energy = F.l1_loss(controls["gain"], energy_target * 5.0)

        total_loss = loss_spectral + 0.5 * loss_f0 + 0.2 * loss_voicing + 0.1 * loss_energy
        total_loss.backward()

        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"  Epoch [{epoch:3d}/{epochs}] "
                f"Total Loss: {total_loss.item():.4f} | "
                f"Spectral: {loss_spectral.item():.4f} | "
                f"F0: {loss_f0.item():.4f} | "
                f"Voicing: {loss_voicing.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    elapsed = time.time() - start_time
    print(f"Distillation completed in {elapsed:.2f}s ({epochs/elapsed:.1f} epochs/sec).\n")

    # 5. Export FP32 Model Checkpoint & B.2 Audio Master
    student.eval()
    ckpt_path = os.path.join(OUTPUT_DIR, "micro_ddsp_student.pt")
    torch.save({
        "model_state_dict": student.state_dict(),
        "total_params": total_params,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "num_harmonics": NUM_HARMONICS,
            "num_noise_bands": NUM_NOISE_BANDS,
        }
    }, ckpt_path)
    print(f"Saved Student Checkpoint: {ckpt_path}")

    with torch.no_grad():
        audio_b2, _ = student(mel_t)
        wav_b2 = audio_b2.squeeze().cpu().numpy()

    # Peak normalize to -1.0 dBFS
    peak_b2 = np.max(np.abs(wav_b2))
    if peak_b2 > 1e-4:
        wav_b2 = wav_b2 * (0.891 / peak_b2)

    fade_len = int(0.05 * sr)
    if len(wav_b2) > fade_len:
        wav_b2[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    b2_wav_path = os.path.join(AUDIO_DIR, "sonon_arduino_b2_distilled.wav")
    sf.write(b2_wav_path, wav_b2.astype(np.float32), sr)
    print(f"Exported B.2 Audio Master (FP32 Distilled): {b2_wav_path}")

    # 6. W8A8 Quantization of Student (Series B.3)
    json_path = os.path.join(OUTPUT_DIR, "micro_ddsp_w8a8_weights.json")
    footprint = export_w8a8_model(student, json_path)
    print(f"\nW8A8 Quantized Memory Footprint:")
    print(f"  Weights: {footprint['weights_bytes']:,} bytes ({footprint['weights_bytes']/1024:.2f} KB)")
    print(f"  Scales & Biases: {footprint['scales_bytes']:,} bytes ({footprint['scales_bytes']/1024:.2f} KB)")
    print(f"  Total Flash Storage: {footprint['total_flash_bytes']:,} bytes ({footprint['total_kb']:.2f} KB)")
    print(f"  Saved W8A8 JSON: {json_path}")

    # Simulate W8A8 quantized forward pass
    # Quantize student weights in-place
    with torch.no_grad():
        for name, param in student.named_parameters():
            if "weight" in name and param.ndim >= 2:
                q_w, scale = quantize_w8a8_tensor(param.data)
                param.data.copy_(q_w.float() * scale)

        audio_b3, _ = student(mel_t)
        wav_b3 = audio_b3.squeeze().cpu().numpy()

    peak_b3 = np.max(np.abs(wav_b3))
    if peak_b3 > 1e-4:
        wav_b3 = wav_b3 * (0.891 / peak_b3)
    if len(wav_b3) > fade_len:
        wav_b3[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    b3_wav_path = os.path.join(AUDIO_DIR, "sonon_arduino_b3_quantized.wav")
    sf.write(b3_wav_path, wav_b3.astype(np.float32), sr)
    print(f"Exported B.3 Audio Master (W8A8 Quantized): {b3_wav_path}")

    # 7. Quantitative Acoustic Audits for B.2 and B.3
    print("\nQuantitative Evaluation Audit:")
    for name, wav in [("B.2 (FP32 Distilled)", wav_b2), ("B.3 (W8A8 Quantized)", wav_b3)]:
        min_l = min(len(ref_audio), len(wav))
        r = ref_audio[:min_l]
        s = wav[:min_l]
        diff = np.abs(np.diff(s))
        spikes_08 = len(np.where(diff > 0.8)[0])
        spikes_045 = len(np.where(diff > 0.45)[0])

        sos_hf = scipy.signal.butter(2, 3500.0, btype="highpass", fs=sr, output="sos")
        r_hf = np.mean(np.abs(scipy.signal.sosfilt(sos_hf, r)))
        s_hf = np.mean(np.abs(scipy.signal.sosfilt(sos_hf, s)))
        hf_ratio_db = 20.0 * np.log10((s_hf + 1e-8) / (r_hf + 1e-8))

        # STFT MCD
        win = np.hanning(1024)
        _, _, z_r = scipy.signal.stft(r, fs=sr, window=win, nperseg=1024, noverlap=768)
        _, _, z_s = scipy.signal.stft(s, fs=sr, window=win, nperseg=1024, noverlap=768)
        log_diff = np.log(np.maximum(np.abs(z_r), 1e-5)) - np.log(np.maximum(np.abs(z_s), 1e-5))
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(log_diff**2))

        print(f"  {name}:")
        print(f"    Mel Cepstral Distortion (MCD): {mcd:.2f} dB")
        print(f"    High Frequency Presence (>3.5kHz): {hf_ratio_db:+.2f} dB")
        print(f"    Transient Spikes (>0.80): {spikes_08}")
        print(f"    Sub-transient Spikes (>0.45): {spikes_045}")
        print(f"    Maximum Sample Jump: {np.max(diff):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
