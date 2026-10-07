#!/usr/bin/env python3
"""
Aerovex Sonon - Ingest Edinburgh AMI Meeting Conversational Dialogue Corpus
===========================================================================
Downloads, curates, and packages authentic multi-speaker headset meeting
dialogue (edinburghcstr/ami ihm split) into WebDataset shards and updates
canonical manifests.

Conforms strictly to 24,000 Hz broadcast mastering and safe Rust specifications.
Zero emojis.
"""

import argparse
import glob
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import scipy.signal as signal
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sonon.ami_ingest")

TARGET_SAMPLE_RATE = 24000
MIN_DURATION_S = 0.8
MAX_DURATION_S = 18.0
MIN_SNR_DB = 20.0

HF_BASE_URL = "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/ihm"


def download_partition(partition_idx: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"train-{partition_idx:05d}-of-00042.parquet"
    out_path = cache_dir / fname
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
        logger.info(f"Using cached partition: {out_path}")
        return out_path

    url = f"{HF_BASE_URL}/{fname}"
    logger.info(f"Downloading {url} to {out_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Aerovex-Sonon/1.0"})
    tmp_path = out_path.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(resp, f_out)
    tmp_path.rename(out_path)
    logger.info(f"Downloaded {fname} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_path


def stage_ami_partition(
    parquet_path: Path,
    staged_dir: Path,
    max_samples: Optional[int] = None,
    existing_uuids: Optional[set] = None
) -> Tuple[int, int, float]:
    table = pq.read_table(str(parquet_path))
    scanned = 0
    valid = 0
    total_dur = 0.0

    sos_highpass = signal.butter(4, 45.0, btype="highpass", fs=TARGET_SAMPLE_RATE, output="sos")

    num_rows = table.num_rows
    col_audio = table["audio"]
    col_text = table["text"]
    col_spk = table["speaker_id"]
    col_meeting = table["meeting_id"]

    for i in range(num_rows):
        if max_samples and valid >= max_samples:
            break

        meeting_id = col_meeting[i].as_py() or "unknown_meeting"
        spk_id = col_spk[i].as_py() or "spk_unknown"
        clip_id = f"ami_{meeting_id}_{spk_id}_{parquet_path.stem}_{i:05d}"
        safe_id = re.sub(r"[^\w\-.]", "_", clip_id)

        if existing_uuids and safe_id in existing_uuids:
            continue

        scanned += 1
        text = (col_text[i].as_py() or "").strip()
        if not text or len(text) < 2:
            continue

        audio_struct = col_audio[i].as_py()
        if not audio_struct or not isinstance(audio_struct, dict):
            continue

        audio_bytes = audio_struct.get("bytes")
        if not audio_bytes or len(audio_bytes) < 500:
            continue

        try:
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            continue

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        dur_sec = len(audio_data) / sr
        if dur_sec < MIN_DURATION_S or dur_sec > MAX_DURATION_S:
            continue

        # Resample to 24,000 Hz if needed
        if sr != TARGET_SAMPLE_RATE:
            target_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sr)
            audio_data = signal.resample(audio_data, target_samples)

        # 45 Hz 4-pole highpass filter
        audio_data = signal.sosfilt(sos_highpass, audio_data)

        # Normalize peak to -1.0 dBFS
        peak = float(np.max(np.abs(audio_data)))
        if peak < 1e-4:
            continue
        audio_data = (audio_data / peak) * 0.89125

        # SNR estimate
        frame_len = 512
        energies = [float(np.mean(audio_data[f * frame_len : (f + 1) * frame_len] ** 2))
                    for f in range(len(audio_data) // frame_len)]
        if len(energies) >= 2:
            noise_floor = float(np.percentile(energies, 10)) + 1e-12
            signal_peak = float(np.percentile(energies, 90)) + 1e-12
            snr_db = 10.0 * np.log10(signal_peak / noise_floor)
        else:
            snr_db = 28.0

        if snr_db < MIN_SNR_DB:
            continue

        # Save WAV
        wav_path = staged_dir / f"{safe_id}.wav"
        sf.write(str(wav_path), audio_data.astype(np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")

        # Save companion JSON
        meta = {
            "uuid": safe_id,
            "speaker_id": f"spk_ami_{spk_id}",
            "locale": "en-GB",
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_seconds": round(dur_sec, 3),
            "snr_db": round(snr_db, 2),
            "transcript_raw": text,
            "transcript_normalized": text.upper(),
            "dialogue_context": {
                "corpus": "edinburgh_ami",
                "meeting_id": meeting_id,
                "channel": "ihm_headset"
            },
            "paralinguistics": [],
            "affective_state": {
                "valence": 0.55,
                "arousal": 0.52,
                "dominance": 0.50
            },
            "turn_metadata": {
                "is_duplex": True,
                "has_interruption": False,
                "has_backchannel": False,
                "floor_yielded": True
            }
        }
        json_path = staged_dir / f"{safe_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        valid += 1
        total_dur += dur_sec

    logger.info(f"Parquet {parquet_path.name}: {valid}/{scanned} valid ({total_dur / 3600:.2f}h)")
    return scanned, valid, total_dur


def main():
    parser = argparse.ArgumentParser(description="Ingest Edinburgh AMI Meeting Corpus into WebDataset shards.")
    parser.add_argument(
        "--start_partition",
        type=int,
        default=0,
        help="Start partition index (0-41)",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=2,
        help="Number of partitions to process in this run",
    )
    parser.add_argument(
        "--download_cache",
        type=str,
        default="/D/aerovex_datasets/downloads/ami",
        help="Directory to cache downloaded parquet files",
    )
    parser.add_argument(
        "--staged_dir",
        type=str,
        default="/D/aerovex_datasets/pilot/staged_ami",
        help="Directory where staged WAV and JSON files are collected",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/D/aerovex_datasets/pilot",
        help="Dataset output directory containing shards/ and manifests",
    )
    parser.add_argument(
        "--max_samples_per_partition",
        type=int,
        default=None,
        help="Optional limit on samples per partition",
    )
    args = parser.parse_args()

    download_cache = Path(args.download_cache)
    staged_dir = Path(args.staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    train_manifest = output_dir / "train_manifest.json"
    val_manifest = output_dir / "val_manifest.json"

    existing_uuids = set()
    if train_manifest.exists():
        with open(train_manifest, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                existing_uuids.add(rec.get("uuid"))
    if val_manifest.exists():
        with open(val_manifest, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                existing_uuids.add(rec.get("uuid"))
    logger.info(f"Loaded {len(existing_uuids)} pre-existing record UUIDs from manifests.")

    total_valid = 0
    total_duration = 0.0

    # Step 1: Download and stage partitions
    for p_idx in range(args.start_partition, args.start_partition + args.num_partitions):
        parquet_path = download_partition(p_idx, download_cache)
        _, valid, dur = stage_ami_partition(
            parquet_path, staged_dir,
            max_samples=args.max_samples_per_partition,
            existing_uuids=existing_uuids
        )
        total_valid += valid
        total_duration += dur

    logger.info(f"Total Staged: {total_valid} utterances ({total_duration / 3600:.2f} hours)")

    if total_valid == 0:
        logger.warning("No utterances were staged. Exiting.")
        sys.exit(0)

    # Step 2: Determine next start shard index
    existing_shards = sorted(list(shards_dir.glob("shard_*.tar")))
    if existing_shards:
        last_num = int(existing_shards[-1].stem.split("_")[-1])
        start_shard_idx = last_num + 1
    else:
        start_shard_idx = 0

    logger.info(f"Sharding {total_valid} utterances starting at shard index {start_shard_idx}...")

    script_dir = Path(__file__).resolve().parent

    prepare_cmd = [
        sys.executable,
        str(script_dir / "prepare_dataset.py"),
        "--input_dir", str(staged_dir),
        "--output_dir", str(output_dir),
        "--manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--min_snr", str(MIN_SNR_DB),
        "--shard_size", "500",
        "--start_shard_idx", str(start_shard_idx),
        "--append_manifest",
    ]

    logger.info("Running prepare_dataset: " + " ".join(prepare_cmd))
    res = subprocess.run(prepare_cmd, check=True)
    if res.returncode != 0:
        logger.error(f"prepare_dataset failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    # Step 3: Run Metacognitive Conditioning
    logger.info("Running Metacognitive Conditioning on Updated Manifests...")
    cond_cmd = [
        sys.executable,
        str(script_dir / "stage_phase5_metacognitive_conditioning.py"),
        "--train_manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--in_place",
    ]
    res_cond = subprocess.run(cond_cmd, check=True)
    if res_cond.returncode != 0:
        logger.error(f"metacognitive conditioning failed with exit code {res_cond.returncode}")
        sys.exit(res_cond.returncode)

    # Step 4: Cleanup temporary staged WAVs
    shutil.rmtree(str(staged_dir), ignore_errors=True)
    logger.info(f"Cleaned up temporary directory: {staged_dir}")
    logger.info("AMI Ingestion Batch Complete!")


if __name__ == "__main__":
    main()
