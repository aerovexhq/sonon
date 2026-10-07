#!/usr/bin/env bash
# Aerovex Sonon - Remote Phase 3 Full-Duplex Pipeline Runner
set -euo pipefail

export PYTHONPATH="/root/sonon"
MOUNT_DIR="/mnt/volume"
VENV_PYTHON="$MOUNT_DIR/venv/bin/python3"
HF_BIN="$MOUNT_DIR/venv/bin/hf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "Aerovex Sonon: Executing Phase 3 Full-Duplex Pipeline"
echo "================================================================================"

DOWNLOAD_DIR="$MOUNT_DIR/downloads"
STAGING_DIR="$MOUNT_DIR/raw/duplex_staged"
SHARDS_DIR="$MOUNT_DIR/shards"
MANIFESTS_DIR="$MOUNT_DIR/manifests"

mkdir -p "$DOWNLOAD_DIR" "$STAGING_DIR" "$SHARDS_DIR" "$MANIFESTS_DIR"

# 1. High-throughput download of MagicHub spontaneous conversational dataset
echo "Step 1: Downloading MagicHub Spontaneous English Conversational Dataset..."
"$HF_BIN" download \
    MagicHub/multi-stream-spontaneous-conversation-training-datasets_english \
    --repo-type dataset \
    --local-dir "$DOWNLOAD_DIR/magichub_en"

# 2. High-throughput download of Expresso conversational dialogues
echo "Step 2: Downloading Expresso Conversational Dialogue Parquet Partitions..."
"$HF_BIN" download \
    nytopop/expresso-conversational \
    --repo-type dataset \
    --local-dir "$DOWNLOAD_DIR/expresso"

# 3. Stage & Inspect Dual-Channel Synchronized Turns
echo "Step 3: Staging and Curating Synchronized Conversational Turns..."
"$VENV_PYTHON" "$SCRIPT_DIR/stage_phase3_duplex_dialogue.py" \
    --magichub_dir "$DOWNLOAD_DIR/magichub_en" \
    --expresso_parquet "$DOWNLOAD_DIR/expresso/conversational/"*.parquet \
    --output_dir "$STAGING_DIR"

# 4. WebDataset Shard Packaging
echo "Step 4: Packaging WebDataset Shards & Updating Manifests..."
"$VENV_PYTHON" "$SCRIPT_DIR/prepare_dataset.py" \
    --input_dir "$STAGING_DIR" \
    --output_dir "$SHARDS_DIR" \
    --manifest "$MANIFESTS_DIR/train_manifest.json" \
    --val_manifest "$MANIFESTS_DIR/val_manifest.json" \
    --shard_size 500 \
    --start_shard_idx 202 \
    --min_snr 28.0 \
    --sample_rate 24000

echo "================================================================================"
echo "Phase 3 Ingestion & Sharding Complete."
echo "Current Shard Count:"
ls -lh "$SHARDS_DIR"/*.tar | wc -l || true
echo "Total Storage Used:"
du -sh "$MOUNT_DIR"
echo "================================================================================"
