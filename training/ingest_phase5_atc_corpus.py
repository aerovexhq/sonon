#!/usr/bin/env python3
"""
Aerovex Sonon - Ingest Tactical Aerospace Air Traffic Control Corpus
====================================================================
Downloads, curates, and packages authentic air traffic control (ATC)
transmissions (UWB ATCC and ATCO2) into WebDataset shards and updates
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
logger = logging.getLogger("sonon.atc_ingest")

TARGET_SAMPLE_RATE = 24000
MIN_DURATION_S = 0.8
MAX_DURATION_S = 18.0
MIN_SNR_DB = 16.0  # VHF radio channels have natural background RF hiss

INTENT_PATTERNS = [
    ("emergency_alert", re.compile(r"\b(mayday|pan pan|emergency|abort|collision|pull up|terrain)\b", re.IGNORECASE)),
    ("taxi_clearance", re.compile(r"\b(taxi|holding point|runway|rwy|stand|apron|via|cross|hold short)\b", re.IGNORECASE)),
    ("altitude_clearance", re.compile(r"\b(climb|descend|maintain|flight level|fl|feet|altitude|level)\b", re.IGNORECASE)),
    ("heading_vector", re.compile(r"\b(turn left|turn right|heading|hdg|direct|proceed|vector)\b", re.IGNORECASE)),
    ("transponder_squawk", re.compile(r"\b(squawk|ident|transponder)\b", re.IGNORECASE)),
    ("altimeter_qnh", re.compile(r"\b(qnh|altimeter|millibars|hectopascals)\b", re.IGNORECASE)),
    ("handover_contact", re.compile(r"\b(contact|frequency|tower|radar|approach|departure|ground|control|monitor)\b", re.IGNORECASE)),
    ("speed_control", re.compile(r"\b(reduce speed|knots|mach|speed|maintain)\b", re.IGNORECASE)),
]


def classify_tactical_intent(text: str) -> str:
    for intent_name, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return intent_name
    return "general_tactical_atc"


def stage_parquet_file(
    parquet_path: Path,
    staged_dir: Path,
    corpus_name: str,
    max_samples: Optional[int] = None,
    existing_uuids: Optional[set] = None
) -> Tuple[int, int, float]:
    table = pq.read_table(str(parquet_path))
    rows = table.to_pylist()

    scanned = 0
    valid = 0
    total_dur = 0.0

    sos_highpass = signal.butter(4, 45.0, btype="highpass", fs=TARGET_SAMPLE_RATE, output="sos")

    for i, row in enumerate(rows):
        if max_samples and valid >= max_samples:
            break

        clip_id = row.get("id") or f"{corpus_name}_{parquet_path.stem}_{i:05d}"
        safe_id = re.sub(r"[^\w\-.]", "_", clip_id)
        if existing_uuids and safe_id in existing_uuids:
            continue

        scanned += 1
        text = (row.get("text") or "").strip()
        if not text:
            continue

        audio_obj = row.get("audio")
        if not audio_obj or not isinstance(audio_obj, dict):
            continue

        audio_bytes = audio_obj.get("bytes")
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
            snr_db = 25.0

        if snr_db < MIN_SNR_DB:
            continue

        intent = classify_tactical_intent(text)

        # Save WAV
        wav_path = staged_dir / f"{safe_id}.wav"
        sf.write(str(wav_path), audio_data.astype(np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")

        # Save companion JSON
        meta = {
            "uuid": safe_id,
            "speaker_id": f"spk_atc_{corpus_name}",
            "locale": "en-US",
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_seconds": round(dur_sec, 3),
            "snr_db": round(snr_db, 2),
            "transcript_raw": text,
            "transcript_normalized": text.upper(),
            "tactical_intent": intent,
            "paralinguistics": [],
            "affective_state": {
                "valence": 0.5,
                "arousal": 0.7 if intent == "emergency_alert" else 0.5,
                "dominance": 0.75 if "clearance" in intent else 0.5
            },
            "turn_metadata": {
                "is_duplex": False,
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
    parser = argparse.ArgumentParser(description="Ingest Tactical Air Traffic Control audio into shards.")
    parser.add_argument(
        "--staged_dir",
        type=str,
        default="/D/aerovex_datasets/pilot/staged_atc",
        help="Directory where staged WAV and JSON files are collected",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/D/aerovex_datasets/pilot",
        help="Dataset output directory containing shards/ and manifests",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional maximum samples to ingest per parquet file",
    )
    args = parser.parse_args()

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

    # Step 1: Scan all ATC parquets
    uwb_files = sorted(glob.glob("/D/aerovex_datasets/downloads/aerospace_atc/uwb_atcc/data/*.parquet"))
    atco2_files = sorted(glob.glob("/D/aerovex_datasets/downloads/aerospace_atc/atco2/data/*.parquet"))
    all_files = [(Path(f), "uwb_atcc") for f in uwb_files] + [(Path(f), "atco2") for f in atco2_files]

    logger.info("=" * 70)
    logger.info(f"Step 1: Staging {len(all_files)} Tactical ATC Parquet Files to {staged_dir}")
    logger.info("=" * 70)

    total_valid = 0
    total_duration = 0.0

    for fpath, corpus in all_files:
        logger.info(f"Processing {fpath.name} ({corpus})...")
        _, valid, dur = stage_parquet_file(
            fpath, staged_dir, corpus,
            max_samples=args.max_samples_per_file,
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

    logger.info("=" * 70)
    logger.info(f"Step 2: Sharding Staged Utterances into WebDataset (Starting at shard index {start_shard_idx})")
    logger.info("=" * 70)

    train_manifest = output_dir / "train_manifest.json"
    val_manifest = output_dir / "val_manifest.json"
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
    logger.info("=" * 70)
    logger.info("Step 3: Running Metacognitive Conditioning on Updated Manifests")
    logger.info("=" * 70)

    cond_cmd = [
        sys.executable,
        str(script_dir / "stage_phase5_metacognitive_conditioning.py"),
        "--train_manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--in_place",
    ]

    logger.info("Running metacognitive conditioning: " + " ".join(cond_cmd))
    res_cond = subprocess.run(cond_cmd, check=True)
    if res_cond.returncode != 0:
        logger.error(f"metacognitive conditioning failed with exit code {res_cond.returncode}")
        sys.exit(res_cond.returncode)

    # Step 4: Cleanup temporary staged WAVs
    logger.info("=" * 70)
    logger.info("Step 4: Cleaning up staged directory to reclaim scratch disk space")
    logger.info("=" * 70)
    shutil.rmtree(str(staged_dir), ignore_errors=True)
    logger.info(f"Deleted temporary directory: {staged_dir}")

    logger.info("=" * 70)
    logger.info("Tactical Aerospace ATC Corpus Ingestion Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
