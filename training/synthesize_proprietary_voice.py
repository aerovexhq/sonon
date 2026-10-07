#!/usr/bin/env python3
"""Sonon End-to-End Proprietary Voice Synthesis Inference Engine.

Generates continuous speech audio using 100% proprietary models trained from scratch:
1. FlowMatchingDiT: Continuous Normalizing Flow / Optimal Transport ODE solver.
2. BigVganVocoder: Multi-scale transposed conv upsampler with SnakeBeta activations.

Zero downloaded models. Strictly safe local execution.
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import soundfile as sf

# Import local architecture classes
from train_flow_matching_dit import FlowMatchingDiT, N_MELS, SAMPLE_RATE, HOP_LENGTH
from train_vocoder import BigVganVocoder

DEFAULT_DIT_CHECKPOINT = "/home/usr/Projects/aerovex/modules/sonon/models/flow_matching_dit/flow_matching_dit_epoch_005.pt"
DEFAULT_VOCODER_CHECKPOINT = "/home/usr/Projects/aerovex/modules/sonon/models/bigvgan_vocoder/bigvgan_vocoder_epoch_003.pt"
DEFAULT_OUTPUT_WAV = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis/sonon_proprietary_flow_matching_voice.wav"


def solve_flow_ode(
    model: FlowMatchingDiT,
    context: torch.Tensor,
    seq_len: int = 128,
    steps: int = 16,
    solver: str = "rk2",
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Solve Optimal Transport ODE dx/dt = v_theta(x_t, t, c) from t=0 (noise) to t=1 (mel latent)."""
    # Sample base Gaussian noise x_0 ~ N(0, I)
    x = torch.randn(1, seq_len, N_MELS, device=device)
    dt = 1.0 / steps

    t = 0.0
    for step in range(steps):
        t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

        if solver == "rk2":
            # Midpoint Runge-Kutta 2
            k1 = model(x, t_tensor, context)
            x_mid = x + 0.5 * dt * k1
            t_mid_tensor = torch.tensor([t + 0.5 * dt], device=device, dtype=torch.float32)
            k2 = model(x_mid, t_mid_tensor, context)
            x = x + dt * k2
        else:
            # Euler
            v = model(x, t_tensor, context)
            x = x + dt * v

        t += dt

    return x  # (1, seq_len, 80)


def invert_mel_analytical(mel_latents: torch.Tensor, n_iters: int = 32) -> np.ndarray:
    """Invert 80-channel mel latent spectrogram to time-domain audio via pseudo-inverse & iterative STFT."""
    from train_vocoder import MEL_BASIS, WIN_LENGTH, N_FFT
    import scipy.signal

    mel = mel_latents.squeeze(0).transpose(0, 1).cpu().numpy()  # (80, T)
    T_frames = mel.shape[1]

    log_mel_unnorm = mel * 4.0 - 4.0
    mel_linear = np.exp(np.clip(log_mel_unnorm, -10.0, 10.0))
    mel_pinv = np.linalg.pinv(MEL_BASIS)
    linear_mag = np.maximum(0.0, np.dot(mel_pinv, mel_linear))

    angles = np.exp(2j * np.pi * np.random.rand(*linear_mag.shape))
    spec = linear_mag * angles
    window = np.hanning(WIN_LENGTH)

    for _ in range(n_iters):
        _, x_recon = scipy.signal.istft(
            spec, fs=SAMPLE_RATE, window=window, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT
        )
        _, _, spec_new = scipy.signal.stft(
            x_recon, fs=SAMPLE_RATE, window=window, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT, boundary=None, padded=True
        )
        cur_angles = np.angle(spec_new)
        if cur_angles.shape[1] < T_frames:
            pad = np.zeros((cur_angles.shape[0], T_frames - cur_angles.shape[1]))
            cur_angles = np.hstack([cur_angles, pad])
        else:
            cur_angles = cur_angles[:, :T_frames]
        angles = np.exp(1j * cur_angles)
        spec = linear_mag * angles

    _, x_final = scipy.signal.istft(
        spec, fs=SAMPLE_RATE, window=window, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT
    )
    return x_final


def synthesize(
    text: str = "Waypoint Alpha reached. Maintaining altitude three thousand feet.",
    dit_checkpoint: str = DEFAULT_DIT_CHECKPOINT,
    vocoder_checkpoint: str = DEFAULT_VOCODER_CHECKPOINT,
    output_wav: str = DEFAULT_OUTPUT_WAV,
    vocoder_mode: str = "analytical",
    valence: float = 0.6,
    arousal: float = 0.7,
    dominance: float = 0.8,
    steps: int = 16,
    duration_seconds: float = 3.0,
    device_name: str = "auto"
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Proprietary Synthesis target device: {device}")
    print(f"Synthesizing text: '{text}' (Vocoder mode: {vocoder_mode})")
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    # 1. Load FlowMatchingDiT model
    print(f"Loading proprietary FlowMatchingDiT weights: {dit_checkpoint}...")
    dit_ckpt = torch.save if not os.path.exists(dit_checkpoint) else torch.load(dit_checkpoint, map_location=device)
    dit_model = FlowMatchingDiT(
        latent_dim=N_MELS,
        hidden_dim=384,
        context_dim=256,
        num_heads=8,
        num_layers=6
    ).to(device)
    dit_model.load_state_dict(dit_ckpt["model_state_dict"], strict=False)
    dit_model.eval()

    # 2. Build 3D VAD affective conditioning context and byte text tokens
    vad = np.array([valence, arousal, dominance], dtype=np.float32)
    lookahead = np.zeros(13, dtype=np.float32)
    lookahead[0] = 3.0   # 3 lookahead tokens
    lookahead[1] = 0.5   # 150 ms offset
    lookahead[2] = 0.25  # Low entropy
    cond_vector = torch.from_numpy(np.concatenate([vad, lookahead])).unsqueeze(0).to(device)

    byte_tokens = [min(255, b) for b in text.encode("utf-8")[:128]]
    if len(byte_tokens) < 128:
        byte_tokens = byte_tokens + [0] * (128 - len(byte_tokens))
    text_tokens = torch.tensor([byte_tokens], dtype=torch.int64, device=device)

    # Project to context space (1, 1 + L, context_dim)
    with torch.no_grad():
        context = dit_model.build_context(cond_vector, text_tokens)

        # 3. Solve Flow Matching ODE to synthesize 80-channel mel frames
        seq_len = int((duration_seconds * SAMPLE_RATE) / HOP_LENGTH)
        print(f"Solving Optimal Transport ODE ({steps} RK2 steps, {seq_len} mel frames)...")
        start_time = time.time()
        mel_latents = solve_flow_ode(dit_model, context, seq_len=seq_len, steps=steps, solver="rk2", device=device)
        ode_elapsed = time.time() - start_time
        print(f"Flow Matching latent generation completed in {ode_elapsed:.2f}s.")

        # 4. Invert mel-spectrogram to continuous audio waveform
        if vocoder_mode == "neural":
            print(f"Loading proprietary BigVganVocoder weights: {vocoder_checkpoint}...")
            voc_ckpt = torch.load(vocoder_checkpoint, map_location=device)
            voc_model = BigVganVocoder(
                in_channels=N_MELS,
                initial_channels=128,
                upsample_rates=[8, 4, 4, 2],
                upsample_kernel_sizes=[16, 8, 8, 4],
                resblock_kernel_sizes=[3, 7, 11]
            ).to(device)
            voc_model.load_state_dict(voc_ckpt["model_state_dict"])
            voc_model.eval()

            mel_in = mel_latents.transpose(1, 2)
            print("Synthesizing continuous 24,000 Hz audio waveform via BigVGAN-v2...")
            wav_tensor = voc_model(mel_in)
            audio = wav_tensor.squeeze().cpu().numpy()
        else:
            print("Synthesizing continuous 24,000 Hz audio waveform via analytical vocoder inversion...")
            audio = invert_mel_analytical(mel_latents)

        # Peak normalization to -1.0 dBFS
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_peak = 10.0 ** (-1.0 / 20.0)  # ~0.891
            audio = (audio / max_val) * target_peak

        # Write output WAV file
        sf.write(output_wav, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"Audio synthesized successfully: {output_wav}")
        print(f"Duration: {len(audio)/SAMPLE_RATE:.2f}s, Sample Rate: {SAMPLE_RATE} Hz, Channels: 1 (Mono)")


def main():
    parser = argparse.ArgumentParser(description="Sonon Proprietary Voice Synthesizer")
    parser.add_argument("--text", type=str, default="Waypoint Alpha reached. Maintaining altitude three thousand feet.")
    parser.add_argument("--dit_checkpoint", type=str, default=DEFAULT_DIT_CHECKPOINT)
    parser.add_argument("--vocoder_checkpoint", type=str, default=DEFAULT_VOCODER_CHECKPOINT)
    parser.add_argument("--output_wav", type=str, default=DEFAULT_OUTPUT_WAV)
    parser.add_argument("--vocoder_mode", type=str, default="analytical", choices=["analytical", "neural"])
    parser.add_argument("--valence", type=float, default=0.6)
    parser.add_argument("--arousal", type=float, default=0.7)
    parser.add_argument("--dominance", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    synthesize(
        text=args.text,
        dit_checkpoint=args.dit_checkpoint,
        vocoder_checkpoint=args.vocoder_checkpoint,
        output_wav=args.output_wav,
        vocoder_mode=args.vocoder_mode,
        valence=args.valence,
        arousal=args.arousal,
        dominance=args.dominance,
        steps=args.steps,
        duration_seconds=args.duration,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
