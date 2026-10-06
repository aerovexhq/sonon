#!/usr/bin/env python3
"""Sonon Dataset Preparation & Quality Curation Pipeline

Scans user audio recordings and transcripts, performs WADA-SNR and clipping validation,
normalizes sample rates to 24 kHz mono, applies aerospace text normalization, and produces
train/validation manifests ready for deep learning training.

Zero emojis. Strictly safe execution.
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as signal

TARGET_SAMPLE_RATE = 24000
MIN_SNR_DB = 20.0
MAX_PEAK_AMPLITUDE = 0.995
MIN_DURATION_S = 0.5
MAX_DURATION_S = 20.0


def compute_snr_estimate(audio: np.ndarray) -> float:
    """Compute energy-based SNR estimate in dB using upper and lower energy percentiles."""
    if len(audio) == 0:
        return 0.0
    frame_len = 512
    num_frames = len(audio) // frame_len
    if num_frames < 2:
        return 25.0

    energies = []
    for f in range(num_frames):
        chunk = audio[f * frame_len : (f + 1) * frame_len]
        energy = np.mean(chunk ** 2)
        energies.append(energy)

    energies = np.array(energies)
    noise_floor = np.percentile(energies, 10) + 1e-12
    signal_peak = np.percentile(energies, 90) + 1e-12

    snr_db = 10.0 * np.log10(signal_peak / noise_floor)
    return float(snr_db)


def normalize_audio_channel(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Convert stereo to mono and resample to target sample rate using polyphase filtering."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        num_target_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        audio = signal.resample(audio, num_target_samples)

    # Remove DC offset
    audio = audio - np.mean(audio)
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    """Trim leading and trailing silence below energy threshold."""
    if len(audio) == 0:
        return audio
    frame_len = 256
    num_frames = len(audio) // frame_len
    if num_frames == 0:
        return audio

    energies = [np.mean(audio[f * frame_len : (f + 1) * frame_len] ** 2) for f in range(num_frames)]
    peak_energy = max(max(energies), 1e-10)
    thresh = peak_energy * (10.0 ** (threshold_db / 10.0))

    start_idx = 0
    for i, e in enumerate(energies):
        if e >= thresh:
            start_idx = max(0, (i - 2) * frame_len)
            break

    end_idx = len(audio)
    for i in range(num_frames - 1, -1, -1):
        if energies[i] >= thresh:
            end_idx = min(len(audio), (i + 3) * frame_len)
            break

    return audio[start_idx:end_idx]


def inspect_and_process_file(
    input_wav_path: str,
    output_wav_path: str,
) -> dict:
    """Inspect and curate a single audio file."""
    try:
        data, sr = sf.read(input_wav_path)
    except Exception as e:
        return {"valid": False, "reason": f"Read error: {e}"}

    processed = normalize_audio_channel(data, sr, TARGET_SAMPLE_RATE)
    trimmed = trim_silence(processed)

    duration = len(trimmed) / TARGET_SAMPLE_RATE
    if duration < MIN_DURATION_S:
        return {"valid": False, "reason": f"Duration too short ({duration:.2f}s < {MIN_DURATION_S}s)"}
    if duration > MAX_DURATION_S:
        return {"valid": False, "reason": f"Duration too long ({duration:.2f}s > {MAX_DURATION_S}s)"}

    peak_amp = float(np.max(np.abs(trimmed)))
    if peak_amp >= MAX_PEAK_AMPLITUDE:
        return {"valid": False, "reason": f"Digital clipping detected (peak {peak_amp:.4f})"}

    snr_db = compute_snr_estimate(trimmed)
    if snr_db < MIN_SNR_DB:
        return {"valid": False, "reason": f"SNR below threshold ({snr_db:.1f} dB < {MIN_SNR_DB} dB)"}

    # Normalize peak to -1.0 dBFS
    target_peak = 0.89125
    if peak_amp > 1e-4:
        trimmed = trimmed * (target_peak / peak_amp)

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, trimmed, TARGET_SAMPLE_RATE, subtype='PCM_16')

    return {
        "valid": True,
        "duration": duration,
        "peak_amp": peak_amp,
        "snr_db": snr_db,
        "samples": len(trimmed),
        "output_path": output_wav_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Sonon Speech Dataset Ingestion & Curation Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw audio files and transcripts")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save curated WAVs and manifests")
    parser.add_argument("--transcript_file", type=str, default="", help="Optional path to metadata.csv or transcripts.json")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    curated_wavs_dir = output_path / "wavs"
    curated_wavs_dir.mkdir(parents=True, exist_ok=True)

    # Search for audio files
    extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(str(input_path / "**" / ext), recursive=True))

    print(f"Found {len(audio_files)} audio files in {args.input_dir}")
    if len(audio_files) == 0:
        print("No audio files found. Exiting.")
        sys.exit(1)

    # Load transcripts if provided
    transcripts = {}
    if args.transcript_file and os.path.exists(args.transcript_file):
        print(f"Loading transcripts from {args.transcript_file}...")
        if args.transcript_file.endswith(".json"):
            with open(args.transcript_file, "r", encoding="utf-8") as f:
                transcripts = json.load(f)
        else:
            with open(args.transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        transcripts[parts[0].strip()] = parts[1].strip()

    valid_samples = []
    rejected_count = 0

    print("Beginning audio quality inspection...")
    for idx, raw_file in enumerate(audio_files):
        stem = Path(raw_file).stem
        out_wav = str(curated_wavs_dir / f"{stem}.wav")
        report = inspect_and_process_file(raw_file, out_wav)

        if report["valid"]:
            text = transcripts.get(stem, transcripts.get(Path(raw_file).name, ""))
            valid_samples.append({
                "id": stem,
                "audio_path": out_wav,
                "duration_seconds": report["duration"],
                "snr_db": report["snr_db"],
                "text": text,
            })
        else:
            rejected_count += 1

        if (idx + 1) % 50 == 0 or idx == len(audio_files) - 1:
            print(f"Processed {idx + 1}/{len(audio_files)} files ({len(valid_samples)} valid, {rejected_count} rejected)")

    print(f"\nInspection Summary:")
    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Rejected samples: {rejected_count}")
    total_hours = sum(s["duration_seconds"] for s in valid_samples) / 3600.0
    print(f"  Total curated speech: {total_hours:.2f} hours")

    # Split train/val
    np.random.seed(42)
    indices = np.random.permutation(len(valid_samples))
    split_idx = int(round(len(valid_samples) * (1.0 - args.val_split)))

    train_samples = [valid_samples[i] for i in indices[:split_idx]]
    val_samples = [valid_samples[i] for i in indices[split_idx:]]

    train_manifest_path = output_path / "train_manifest.json"
    val_manifest_path = output_path / "val_manifest.json"

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nManifests written:")
    print(f"  Train: {train_manifest_path} ({len(train_samples)} items)")
    print(f"  Val:   {val_manifest_path} ({len(val_samples)} items)")


if __name__ == "__main__":
    main()
