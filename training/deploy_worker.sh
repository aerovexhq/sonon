#!/usr/bin/env bash
# Deploy worker to Hetzner server and initialize environment
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <SERVER_IP>"
    exit 1
fi

SERVER_IP="$1"
SSH_KEY="$HOME/.ssh/id_ed25519_hetzner"
SONON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Connecting to root@$SERVER_IP using $SSH_KEY..."
ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "root@$SERVER_IP" "mkdir -p /root/sonon/training /mnt/volume"

echo "Syncing Sonon training scripts to remote worker..."
scp -i "$SSH_KEY" -r \
    "$SONON_DIR/training/remote_setup.sh" \
    "$SONON_DIR/training/run_remote_pipeline.sh" \
    "$SONON_DIR/training/stage_phase3_duplex_dialogue.py" \
    "$SONON_DIR/training/prepare_dataset.py" \
    "root@$SERVER_IP:/root/sonon/training/"

echo "Running remote setup script on $SERVER_IP..."
ssh -i "$SSH_KEY" "root@$SERVER_IP" "bash /root/sonon/training/remote_setup.sh"

echo "Remote worker base environment is successfully initialized."
