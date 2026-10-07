#!/usr/bin/env python3
"""
Aerovex Sonon - Ingest Eurocontrol ATCOSIM Corpus (Phase 4 Extension)
====================================================================
Downloads, curates, and packages the full ATCOSIM Eurocontrol air traffic
control speech corpus into WebDataset shards and updates the canonical
dataset manifests.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sonon.atcosim")

SCRIPT_DIR = Path(__file__).resolve().parent
SONON_ROOT = SCRIPT_DIR.parent
if str(SONON_ROOT) not in sys.path:
    sys.path.insert(0, str(SONON_ROOT))

from training.stage_phase4_aerospace_atc import stage_atc_parquet

ATCOSIM_PARQUET_FILES = [
    # "data/train-00000-of-00004-c1d7fb31dcbf644a.parquet" already staged!
    "data/train-00001-of-00004-f165730df6bf7253.parquet",
    "data/train-00002-of-00004-67e682f17e32b703.parquet",
    "data/train-00003-of-00004-b0b05d4b243c95c6.parquet",
    "data/test-00000-of-00001-01544bdf54b4ccf3.parquet",
]


def main():
    parser = argparse.ArgumentParser(description="Ingest Eurocontrol ATCOSIM speech dataset.")
    parser.add_argument(
        "--staged_dir",
        type=str,
        default="/D/aerovex_datasets/pilot/staged_atcosim",
        help="Directory where staged WAV and JSON files are collected",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/D/aerovex_datasets/cache_hf",
        help="HuggingFace download cache directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/D/aerovex_datasets/pilot",
        help="Dataset output directory containing shards/ and manifests",
    )
    parser.add_argument(
        "--min_snr",
        type=float,
        default=18.0,
        help="Minimum WADA-SNR in dB for ATC radio gating",
    )
    args = parser.parse_args()

    staged_dir = Path(args.staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"

    # Step 1: Download and stage remaining partitions
    logger.info("=" * 70)
    logger.info("Step 1: Downloading & Staging ATCOSIM Partitions")
    logger.info("=" * 70)

    for rfilename in ATCOSIM_PARQUET_FILES:
        logger.info(f"Downloading {rfilename} from Jzuluaga/atcosim_corpus...")
        parquet_path_str = hf_hub_download(
            repo_id="Jzuluaga/atcosim_corpus",
            filename=rfilename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        parquet_path = Path(parquet_path_str)
        logger.info(f"Downloaded to {parquet_path}, staging audio...")

        scanned, valid, dur = stage_atc_parquet(
            parquet_path,
            staged_dir,
            corpus_name="atcosim",
            max_samples=None,
            min_snr_db=args.min_snr,
        )
        logger.info(f"Staged {valid}/{scanned} utterances ({dur / 3600.0:.2f} h) from {parquet_path.name}")

        # Delete raw parquet and its cached snapshot to conserve disk
        try:
            if parquet_path.exists():
                parquet_path.unlink()
                logger.info(f"Removed temporary raw file: {parquet_path}")
        except Exception as e:
            logger.warning(f"Failed to remove raw parquet {parquet_path}: {e}")

    # Count staged WAVs
    all_staged_wavs = list(staged_dir.glob("*.wav"))
    logger.info(f"Total staged ATCOSIM utterances in {staged_dir}: {len(all_staged_wavs)}")

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

    prepare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_dataset.py"),
        "--input_dir", str(staged_dir),
        "--output_dir", str(output_dir),
        "--manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--min_snr", str(args.min_snr),
        "--shard_size", "500",
        "--start_shard_idx", str(start_shard_idx),
        "--append_manifest",
    ]

    logger.info("Running prepare_dataset: " + " ".join(prepare_cmd))
    res = subprocess.run(prepare_cmd, check=True)
    if res.returncode != 0:
        logger.error(f"prepare_dataset failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    # Step 3: Run Phase 5 Metacognitive Conditioning on updated manifests
    logger.info("=" * 70)
    logger.info("Step 3: Running Metacognitive Conditioning on Updated Manifests")
    logger.info("=" * 70)

    cond_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "stage_phase5_metacognitive_conditioning.py"),
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

    # Summary
    logger.info("=" * 70)
    logger.info("ATCOSIM Ingestion & Metacognitive Conditioning Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
