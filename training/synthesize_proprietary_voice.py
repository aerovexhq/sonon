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
    context_uncond: Optional[torch.Tensor] = None,
    cfg_scale: float = 2.0,
    seq_len: int = 128,
    steps: int = 24,
    solver: str = "rk2",
    key_padding_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Solve Optimal Transport ODE dx/dt = v_theta(x_t, t, c) with Classifier-Free Guidance."""
    x = torch.randn(1, seq_len, N_MELS, device=device)
    dt = 1.0 / steps

    t = 0.0
    for step in range(steps):
        t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

        if solver == "rk2":
            # Midpoint Runge-Kutta 2 with CFG
            k1_cond = model(x, t_tensor, context, key_padding_mask=key_padding_mask)
            if context_uncond is not None and cfg_scale > 1.0:
                k1_uncond = model(x, t_tensor, context_uncond)
                k1 = k1_uncond + cfg_scale * (k1_cond - k1_uncond)
            else:
                k1 = k1_cond

            x_mid = x + 0.5 * dt * k1
            t_mid_tensor = torch.tensor([t + 0.5 * dt], device=device, dtype=torch.float32)
            k2_cond = model(x_mid, t_mid_tensor, context, key_padding_mask=key_padding_mask)
            if context_uncond is not None and cfg_scale > 1.0:
                k2_uncond = model(x_mid, t_mid_tensor, context_uncond)
                k2 = k2_uncond + cfg_scale * (k2_cond - k2_uncond)
            else:
                k2 = k2_cond

            x = x + dt * k2
        else:
            # Euler
            v_cond = model(x, t_tensor, context, key_padding_mask=key_padding_mask)
            if context_uncond is not None and cfg_scale > 1.0:
                v_uncond = model(x, t_tensor, context_uncond)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
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


def invert_mel_source_filter(
    mel_latents: torch.Tensor,
    f0_base: float = 135.0,
    arousal: float = 0.7,
    valence: float = 0.6
) -> np.ndarray:
    """Invert mel latent spectrogram using warm human glottal phonation with psychoacoustic phase dispersion."""
    from train_vocoder import MEL_BASIS, WIN_LENGTH, N_FFT
    import scipy.signal
    import scipy.ndimage

    # Extract mel latents and apply articulatory temporal smoothing
    m_frames = mel_latents.squeeze(0).cpu().numpy()  # (T_frames, 80)
    T_frames = m_frames.shape[0]

    # Savitzky-Golay temporal smoothing (window=9, poly=2) matching vocal tract articulatory inertia
    if T_frames >= 9:
        m_frames = scipy.signal.savgol_filter(m_frames, window_length=9, polyorder=2, axis=0)

    # Monotonic end-of-phrase silence decay on the last 25 frames (eliminates end chirp / rising pitch)
    fade_frames = min(25, T_frames // 4)
    if fade_frames > 0:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_frames)))
        for i, f_idx in enumerate(range(T_frames - fade_frames, T_frames)):
            w = fade_curve[i]
            m_frames[f_idx] = w * m_frames[f_idx] + (1.0 - w) * (-2.0)

    mel = m_frames.T  # (80, T_frames)

    # Un-normalize log-mel to linear spectral magnitude
    log_mel_unnorm = mel * 4.0 - 4.0
    mel_linear = np.exp(np.clip(log_mel_unnorm, -8.0, 6.0))
    mel_pinv = np.linalg.pinv(MEL_BASIS)
    linear_mag = np.maximum(0.0, np.dot(mel_pinv, mel_linear))  # (513, T_frames)

    # Energy envelope from linear spectral magnitude
    frame_energy = np.sqrt(np.sum(linear_mag**2, axis=0))
    p95_energy = np.percentile(frame_energy, 95) + 1e-6
    norm_energy = np.clip(frame_energy / p95_energy, 0.0, 1.0)

    # 1. Natural pitch contour: base f0, gentle syllabic inflection, and natural declination
    f0_syllabic = f0_base + (20.0 * arousal) * (norm_energy**0.6) - 10.0 * np.linspace(0, 1, T_frames)
    f0 = np.clip(f0_syllabic, 85.0, 240.0)

    N_samples = (T_frames - 1) * HOP_LENGTH
    t_frames = np.arange(T_frames)
    t_samples = np.linspace(0, T_frames - 1, N_samples)
    f0_interp = np.interp(t_samples, t_frames, f0)

    # Add natural vocal fold micro-jitter (0.8% variation), breaking artificial robotic buzz
    np.random.seed(42)
    jitter = 1.0 + 0.008 * scipy.ndimage.gaussian_filter1d(np.random.randn(N_samples), sigma=40)
    f0_jittered = f0_interp * jitter

    # 2. Warm human glottal phonation with psychoacoustic phase dispersion
    # Eliminates mosquito buzz by dispersing phase above 1,500 Hz and enforcing natural -12 dB/oct tilt
    phase = 2 * np.pi * np.cumsum(f0_jittered) / SAMPLE_RATE
    glottal_harmonic = np.zeros(N_samples)
    max_harmonic = int(min(32, 4500.0 / f0_base))  # Bandlimited to warm vocal band <= 4.5 kHz
    for k in range(1, max_harmonic + 1):
        freq = k * f0_base
        # Natural warm human vocal fold spectral roll-off (-12 dB/octave)
        amp = 1.0 / (1.0 + (freq / 650.0)**1.85)
        if freq < 1500.0:
            # Low frequencies: phase-coherent for solid fundamental and F1 resonance
            disp_phase = 0.2 * np.sin(k * 0.5)
        else:
            # High frequencies: psychoacoustic phase dispersion eliminating insect buzz
            disp_phase = 0.5 * np.sin(k * 0.8) + 0.3 * np.cos(k * 1.4)
        glottal_harmonic += amp * np.cos(k * phase + disp_phase)

    # 3. Continuous voicing degree + warm turbulent aspiration
    low_freq_energy = np.sum(linear_mag[:35, :], axis=0)  # Energy < 820 Hz
    total_energy = np.sum(linear_mag, axis=0) + 1e-6
    low_ratio = low_freq_energy / total_energy
    voicing_deg = np.clip(0.40 + 1.1 * low_ratio, 0.40, 0.90)
    voicing_interp = np.interp(t_samples, t_frames, voicing_deg)

    # Natural vocal aspiration noise floor
    aspiration = scipy.ndimage.gaussian_filter1d(np.random.randn(N_samples), sigma=1) * 0.15
    mixed_excitation = voicing_interp * glottal_harmonic + (1.0 - voicing_interp * 0.6) * aspiration

    # Tail envelope decay to guarantee zero rising pitch / boundary chirp at end
    fade_tail_samples = min(int(0.25 * SAMPLE_RATE), N_samples // 4)
    if fade_tail_samples > 0:
        tail_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_tail_samples)))
        mixed_excitation[-fade_tail_samples:] *= tail_curve

    mixed_excitation = mixed_excitation / (np.max(np.abs(mixed_excitation)) + 1e-6)

    # 4. STFT of glottal source
    window = np.hanning(WIN_LENGTH)
    _, _, zxx_exc = scipy.signal.stft(
        mixed_excitation, fs=SAMPLE_RATE, window=window, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT, boundary=None, padded=True
    )

    min_T = min(linear_mag.shape[1], zxx_exc.shape[1])
    linear_mag_aligned = linear_mag[:, :min_T]
    zxx_exc_aligned = zxx_exc[:, :min_T]

    # Natural formant contrast (gamma = 1.03 for clean vowel articulation without needle spikes)
    linear_mag_natural = np.power(linear_mag_aligned + 1e-6, 1.03)
    linear_mag_natural = linear_mag_natural * (np.sum(linear_mag_aligned, axis=0, keepdims=True) / (np.sum(linear_mag_natural, axis=0, keepdims=True) + 1e-6))

    # Source-filter spectral shaping
    exc_mag = np.abs(zxx_exc_aligned) + 1e-6
    smooth_exc_mag = scipy.ndimage.gaussian_filter1d(exc_mag, sigma=10, axis=0)
    norm_exc = zxx_exc_aligned / smooth_exc_mag
    shaped_stft = linear_mag_natural * norm_exc

    # 5. Inverse STFT to continuous audio waveform
    _, audio = scipy.signal.istft(
        shaped_stft, fs=SAMPLE_RATE, window=window, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT
    )

    # Final audio tail fadeout
    audio_fade_samples = min(int(0.18 * SAMPLE_RATE), len(audio) // 4)
    if audio_fade_samples > 0:
        audio[-audio_fade_samples:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, audio_fade_samples)))

    if np.max(np.abs(audio)) > 1e-6:
        audio = audio / np.max(np.abs(audio)) * 0.89125
    return audio


def synthesize(
    text: str = "Waypoint Alpha reached. Maintaining altitude three thousand feet.",
    dit_checkpoint: str = DEFAULT_DIT_CHECKPOINT,
    vocoder_checkpoint: str = DEFAULT_VOCODER_CHECKPOINT,
    output_wav: str = DEFAULT_OUTPUT_WAV,
    vocoder_mode: str = "source_filter",
    valence: float = 0.6,
    arousal: float = 0.7,
    dominance: float = 0.8,
    steps: int = 24,
    cfg_scale: float = 2.0,
    duration_seconds: float = 3.0,
    f0_base: float = 145.0,
    device_name: str = "auto"
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Proprietary Synthesis target device: {device}")
    print(f"Synthesizing text: '{text}' (Vocoder mode: {vocoder_mode}, CFG: {cfg_scale})")
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

    # Ensure cross-attention gate alpha_2 in all blocks is active
    h_dim = dit_model.hidden_dim
    with torch.no_grad():
        for block in dit_model.blocks:
            block.ada_ln_2.linear.bias.data[2 * h_dim : 3 * h_dim] = 1.0

    # 2. Build 3D VAD affective conditioning context and byte text tokens (NO ZERO PADDING)
    vad = np.array([valence, arousal, dominance], dtype=np.float32)
    lookahead = np.zeros(13, dtype=np.float32)
    lookahead[0] = 3.0   # 3 lookahead tokens
    lookahead[1] = 0.5   # 150 ms offset
    lookahead[2] = 0.25  # Low entropy
    cond_vector = torch.from_numpy(np.concatenate([vad, lookahead])).unsqueeze(0).to(device)

    byte_tokens = [min(255, b) for b in text.encode("utf-8")]
    if not byte_tokens:
        byte_tokens = [32]
    text_tokens = torch.tensor([byte_tokens], dtype=torch.int64, device=device)

    # Project to context space (1, 1 + L, context_dim)
    with torch.no_grad():
        context = dit_model.build_context(cond_vector, text_tokens)
        context_uncond = torch.zeros_like(context)

        # 3. Solve Flow Matching ODE with CFG to synthesize 80-channel mel frames
        # Strictly bound within the 256-frame training distribution (max 250 frames)
        if duration_seconds is None or duration_seconds == 3.0:
            est_dur = min(2.65, max(1.8, len(text) / 25.0 + 0.35))
            seq_len = int((est_dur * SAMPLE_RATE) / HOP_LENGTH)
        else:
            seq_len = min(252, int((duration_seconds * SAMPLE_RATE) / HOP_LENGTH))
        print(f"Solving Optimal Transport ODE ({steps} RK2 steps, {seq_len} mel frames, CFG={cfg_scale})...")
        start_time = time.time()
        mel_latents = solve_flow_ode(
            dit_model, context,
            context_uncond=context_uncond,
            cfg_scale=cfg_scale,
            seq_len=seq_len,
            steps=steps,
            solver="rk2",
            device=device
        )
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
        elif vocoder_mode == "source_filter":
            print("Synthesizing continuous 24,000 Hz audio waveform via physical glottal source-filter synthesis...")
            audio = invert_mel_source_filter(mel_latents, f0_base=f0_base, arousal=arousal, valence=valence)
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
    parser.add_argument("--vocoder_mode", type=str, default="source_filter", choices=["source_filter", "analytical", "neural"])
    parser.add_argument("--valence", type=float, default=0.6)
    parser.add_argument("--arousal", type=float, default=0.7)
    parser.add_argument("--dominance", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--f0_base", type=float, default=145.0)
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
        cfg_scale=args.cfg_scale,
        duration_seconds=args.duration,
        f0_base=args.f0_base,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
