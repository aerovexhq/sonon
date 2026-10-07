#!/usr/bin/env bash
# Aerovex Sonon - Remote Worker Setup & Pipeline Runner
# Sets up 1TB Volume, Python environment, system libraries, and starts Phase 3 pipeline.
set -euo pipefail

echo "================================================================================"
echo "Aerovex Sonon: Provisioning Remote Data Worker"
echo "================================================================================"

# 1. Detect and mount 1TB Hetzner Volume if attached
MOUNT_DIR="/mnt/volume"
mkdir -p "$MOUNT_DIR"

echo "Checking block devices..."
lsblk

# Check if Hetzner volume is attached (usually /dev/disk/by-id/scsi-0HC_Volume_* or /dev/sdb)
VOLUME_DEV=$(ls /dev/disk/by-id/scsi-0HC_Volume_* 2>/dev/null | head -n 1 || true)
if [ -z "$VOLUME_DEV" ]; then
    # Fallback to /dev/sdb if exists and not mounted
    if [ -b "/dev/sdb" ]; then
        VOLUME_DEV="/dev/sdb"
    fi
fi

if [ -n "$VOLUME_DEV" ]; then
    echo "Detected volume device: $VOLUME_DEV"
    # Format if not already formatted
    if ! blkid "$VOLUME_DEV" > /dev/null; then
        echo "Formatting volume with ext4..."
        mkfs.ext4 -F "$VOLUME_DEV"
    fi
    # Mount if not already mounted
    if ! mountpoint -q "$MOUNT_DIR"; then
        echo "Mounting $VOLUME_DEV to $MOUNT_DIR..."
        mount -o discard,defaults "$VOLUME_DEV" "$MOUNT_DIR"
    fi
else
    echo "Warning: Dedicated block volume not detected yet. Using root partition scratch space at /mnt/volume."
fi

# Create directory hierarchy on storage
mkdir -p "$MOUNT_DIR/raw"
mkdir -p "$MOUNT_DIR/shards"
mkdir -p "$MOUNT_DIR/manifests"
mkdir -p "$MOUNT_DIR/logs"
mkdir -p "$MOUNT_DIR/downloads"

echo "Storage directories prepared on $MOUNT_DIR."
df -h "$MOUNT_DIR"

# 2. Install essential system dependencies
echo "Updating apt repositories and installing audio DSP toolchains..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv git curl wget \
    ffmpeg sox libsndfile1-dev rsync htop tmux build-essential

# 3. Setup Python virtual environment
VENV_DIR="$MOUNT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtualenv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing high-performance Python audio and ML libraries..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -q \
    soundfile scipy numpy pyarrow huggingface_hub tqdm requests webdataset

echo "Environment verification complete."
"$VENV_DIR/bin/python3" -c "import soundfile, scipy, pyarrow, huggingface_hub; print('Python libraries verified successfully.')"
echo "================================================================================"
echo "Worker Ready for Execution."
echo "================================================================================"
