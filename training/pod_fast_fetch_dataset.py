#!/usr/bin/env python3
"""
Aerovex Sonon - Datacenter Fast Dataset Staging & Shard Builder
===============================================================
Downloads clean multi-speaker studio speech (LibriTTS-R) and natural conversational
headset dialogue (Edinburgh AMI) directly on Runpod datacenter fiber (1-10 Gbps),
and packages them into standardized 24,000 Hz WebDataset shards and manifests.

Mastering specifications:
- 24,000 Hz broadcast mastering
- 45 Hz 4-pole Butterworth highpass filter
- -1.0 dBFS true peak normalization
- Zero emojis. Safe execution.
"""

import os
import sys
import math
import time
import json
import glob
import io
import re
import tarfile
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pyarrow.parquet as pq
import scipy.signal as signal
import soundfile as sf

TARGET_SAMPLE_RATE = 24000
DATA_DIR = Path("/workspace/data")
SHARDS_DIR = DATA_DIR / "shards"
DOWNLOADS_DIR = Path("/workspace/downloads")

# Hugging Face Datacenter Mirrors
LIBRITTS_CLEAN_URLS = [
    f"https://huggingface.co/datasets/mythicinfinity/libritts_r/resolve/main/data/train.clean.100/train.clean.100-{i:05d}-of-00018.parquet"
    for i in range(4) # 4 partitions = ~20 hours of pristine studio speech
]

AMI_URLS = [
    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/ihm/train-{i:05d}-of-00042.parquet"
    for i in range(4) # 4 partitions = ~8 hours of multi-speaker headset dialogue
]

def fetch_file(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
        print(f"Using cached file: {out_path.name}")
        return
    print(f"Downloading at datacenter speed: {out_path.name}...")
    tmp = out_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Aerovex-Sonon/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {out_path.name} ({size_mb:.2f} MB)")

def process_and_shard_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    sos_highpass = signal.butter(4, 45.0, btype="highpass", fs=TARGET_SAMPLE_RATE, output="sos")
    records = []

    shard_idx = 0
    curr_shard_samples = 0
    max_samples_per_shard = 500
    curr_tar_path = SHARDS_DIR / f"shard_{shard_idx:06d}.tar"
    curr_tar = tarfile.open(curr_tar_path, "w")

    print("\n--- Processing and Sharding Dataset ---")
    parquet_files = sorted(DOWNLOADS_DIR.glob("*.parquet"))
    total_valid = 0
    total_duration = 0.0

    for pq_path in parquet_files:
        print(f"Reading {pq_path.name}...")
        table = pq.read_table(str(pq_path))
        num_rows = table.num_rows

        is_libritts = "libritts" in pq_path.name.lower() or "train.clean" in pq_path.name.lower()
        col_audio = table["audio"]

        if is_libritts:
            col_text = table["text_normalized"] if "text_normalized" in table.column_names else table["text"]
            col_spk = table["speaker_id"]
        else:
            col_text = table["text"]
            col_spk = table["speaker_id"]

        for i in range(num_rows):
            text = (col_text[i].as_py() or "").strip()
            if not text or len(text) < 2:
                continue

            spk = str(col_spk[i].as_py() or "unknown")
            audio_struct = col_audio[i].as_py()
            if not audio_struct or not isinstance(audio_struct, dict):
                continue
            audio_bytes = audio_struct.get("bytes")
            if not audio_bytes or len(audio_bytes) < 1000:
                continue

            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            except Exception:
                continue

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            dur_s = len(audio_data) / sr
            if dur_s < 0.6 or dur_s > 15.0:
                continue

            # Resample to 24 kHz if needed
            if sr != TARGET_SAMPLE_RATE:
                target_len = int(len(audio_data) * TARGET_SAMPLE_RATE / sr)
                audio_data = signal.resample(audio_data, target_len).astype(np.float32)

            # Highpass filter
            audio_data = signal.sosfilt(sos_highpass, audio_data).astype(np.float32)

            # Normalize peak to -1.0 dBFS
            peak = float(np.max(np.abs(audio_data)))
            if peak < 1e-4:
                continue
            audio_data = (audio_data / peak) * 0.89125

            # Rotation of shards
            if curr_shard_samples >= max_samples_per_shard:
                curr_tar.close()
                shard_idx += 1
                curr_tar_path = SHARDS_DIR / f"shard_{shard_idx:06d}.tar"
                curr_tar = tarfile.open(curr_tar_path, "w")
                curr_shard_samples = 0

            # Write WAV to tar
            prefix = "libri" if is_libritts else "ami"
            sample_uuid = f"{prefix}_{spk}_{shard_idx:04d}_{curr_shard_samples:04d}"
            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio_data, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
            wav_bytes = wav_buf.getvalue()

            tar_info = tarfile.TarInfo(name=f"{sample_uuid}.wav")
            tar_info.size = len(wav_bytes)
            tar_info.mtime = int(time.time())
            curr_tar.addfile(tar_info, io.BytesIO(wav_bytes))

            record = {
                "uuid": sample_uuid,
                "shard": curr_tar_path.name,
                "speaker_id": f"{prefix}_{spk}",
                "duration_seconds": round(dur_s, 3),
                "text": text,
                "metacognitive": {
                    "affective_state": {
                        "valence": 0.5,
                        "arousal": 0.5,
                        "dominance": 0.5
                    },
                    "infilling_mask": {
                        "mask_start_ms": 500.0,
                        "mask_duration_ms": 250.0
                    }
                }
            }
            records.append(record)
            curr_shard_samples += 1
            total_valid += 1
            total_duration += dur_s

    curr_tar.close()

    # Split train and validation (95% / 5%)
    split_idx = int(len(records) * 0.95)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    train_manifest = DATA_DIR / "train_manifest.json"
    val_manifest = DATA_DIR / "val_manifest.json"

    with open(train_manifest, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(val_manifest, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)

    print(f"\nStaging complete:")
    print(f"Total valid utterances: {total_valid}")
    print(f"Total speech duration: {total_duration / 3600:.2f} hours")
    print(f"Created {shard_idx + 1} WebDataset shards in {SHARDS_DIR}")
    print(f"Train manifest: {train_manifest} ({len(train_records)} records)")
    print(f"Val manifest: {val_manifest} ({len(val_records)} records)")

def main():
    print("=== Aerovex Sonon Datacenter High-Speed Dataset Staging ===")
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Parquet partitions from HuggingFace
    for url in LIBRITTS_CLEAN_URLS:
        fname = Path(url).name
        fetch_file(url, DOWNLOADS_DIR / fname)

    for url in AMI_URLS:
        fname = Path(url).name
        fetch_file(url, DOWNLOADS_DIR / fname)

    # 2. Extract into standardized WebDataset shards
    process_and_shard_data()
    print("\nDataset ready for immediate GPU training.")

if __name__ == "__main__":
    main()
