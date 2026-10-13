#!/usr/bin/env bash
# Aerovex Sonon - Runpod End-to-End GPU Training Launcher
# ========================================================
# Executes dataset staging and high-throughput GPU training on Runpod instance.
# Zero emojis. Safe execution.

set -euo pipefail

echo "=========================================================="
echo "   Aerovex Sonon - High-Throughput Runpod Training Engine  "
echo "=========================================================="

# 1. Environment & GPU Verification
echo "[1/4] Verifying GPU acceleration..."
nvidia-smi || true
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 2. Dependency Installation
echo "[2/4] Ensuring audio processing dependencies..."
pip install -q pyarrow soundfile scipy

# 3. Datacenter High-Speed Dataset Staging
echo "[3/4] Staging 24kHz multi-speaker speech dataset..."
python3 pod_fast_fetch_dataset.py

# 4. Launch GPU Training
echo "[4/4] Launching FlowMatching Diffusion Transformer training on GPU..."
python3 train_flow_matching_dit.py \
    --device cuda \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --shards_dir /workspace/data/shards \
    --manifest /workspace/data/train_manifest.json \
    --output_dir /workspace/checkpoints/flow_matching_dit

echo "Launching BigVGAN Vocoder training on GPU..."
python3 train_vocoder.py \
    --device cuda \
    --batch_size 32 \
    --epochs 10 \
    --lr 2e-4 \
    --shards_dir /workspace/data/shards \
    --manifest /workspace/data/train_manifest.json \
    --output_dir /workspace/checkpoints/bigvgan_vocoder

echo "Training completed successfully."
