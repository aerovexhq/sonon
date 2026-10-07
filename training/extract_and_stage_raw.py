#!/usr/bin/env python3
"""Automated extraction and staging script for foundation voice datasets.

Unpacks incoming archives (LibriTTS-R, LaughterScape parquet, emotional corpora)
into organized tier directories under modules/sonon/data/raw/ or /D/aerovex_datasets/raw/.

Zero emojis. Strictly safe enterprise standard.
"""

import os
import sys
import io
import json
import tarfile
import zipfile
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def extract_libritts_r(archive_path: Path, target_dir: Path):
    """Extract LibriTTS-R dev_clean tar.gz archive."""
    print(f"Extracting LibriTTS-R from {archive_path} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"Extraction complete for {archive_path}")


def extract_parquet_audio(parquet_path: Path, target_dir: Path, max_samples: int = 500):
    """Extract audio files and transcripts from parquet dataset into target directory."""
    if pq is None:
        print("pyarrow not installed, skipping parquet extraction.")
        return

    print(f"Extracting audio samples from parquet {parquet_path} into {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path)
    audio_col = table["audio"] if "audio" in table.column_names else None
    text_col = table["text"] if "text" in table.column_names else None
    trans_col = table["transcription"] if "transcription" in table.column_names else text_col

    num_rows = min(len(table), max_samples)
    extracted = 0

    for i in range(num_rows):
        sample_id = f"laughter_{i:05d}"
        if "id" in table.column_names:
            sample_id = f"laughter_{table['id'][i].as_py()}"

        audio_dict = audio_col[i].as_py() if audio_col else None
        if not audio_dict:
            continue

        raw_bytes = audio_dict.get("bytes")
        sr = audio_dict.get("sampling_rate", 24000)

        out_wav = target_dir / f"{sample_id}.wav"
        out_txt = target_dir / f"{sample_id}.normalized.txt"

        if raw_bytes:
            with open(out_wav, "wb") as f:
                f.write(raw_bytes)
        elif "array" in audio_dict:
            arr = np.array(audio_dict["array"], dtype=np.float32)
            sf.write(str(out_wav), arr, sr)

        text_val = "[laughter] Spontaneous conversational laughter."
        if trans_col:
            val = trans_col[i].as_py()
            if val:
                text_val = f"[laughter] {val}"

        out_txt.write_text(text_val, encoding="utf-8")
        extracted += 1

    print(f"Extracted {extracted} samples from parquet {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and stage raw audio archives")
    parser.add_argument("--downloads_dir", type=str, default="/D/aerovex_datasets/downloads")
    parser.add_argument("--raw_dir", type=str, default="/D/aerovex_datasets/raw")
    args = parser.parse_args()

    dl_path = Path(args.downloads_dir)
    raw_path = Path(args.raw_dir)

    # 1. LibriTTS-R
    libritts_archive = dl_path / "dev_clean.tar.gz"
    if libritts_archive.exists() and libritts_archive.stat().st_size > 100_000_000:
        target_libritts = raw_path / "tier1_studio" / "libritts_r_dev"
        if not target_libritts.exists() or len(list(target_libritts.glob("**/*.wav"))) == 0:
            extract_libritts_r(libritts_archive, target_libritts)

    # 2. Laughter parquet
    parquet_files = list(dl_path.glob("*.parquet")) + list(Path.home().glob(".cache/huggingface/hub/**/data/*.parquet"))
    target_laughter = raw_path / "tier2_paralinguistic" / "laughterscape"
    for pq_file in parquet_files:
        if "laughter" in str(pq_file).lower():
            extract_parquet_audio(pq_file, target_laughter, max_samples=500)
            break

    print("Staging check complete.")


if __name__ == "__main__":
    main()
