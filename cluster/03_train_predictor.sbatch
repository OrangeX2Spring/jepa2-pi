#!/bin/bash
#SBATCH --job-name=b1k-train
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/chunquancheng/logs/train_%j.out
#SBATCH --error=/mnt/home/chunquancheng/logs/train_%j.err

# Resubmit this script as many times as needed — training resumes automatically
# from the latest checkpoint each time. Stop when the loss curve flattens.

set -e
mkdir -p /mnt/home/chunquancheng/logs
mkdir -p /mnt/home/chunquancheng/ckpt_predictor

REPO=/mnt/home/chunquancheng/jepa2-pi
IMAGE=/mnt/home/chunquancheng/jepa2-pi.tar

# ── 1. Load container image ──────────────────────────────────────────────────
echo "Loading container..."
podman load -i $IMAGE

# ── 2. Copy latent shards to fast local /tmp ─────────────────────────────────
# Training is I/O-bound on shard reads; local SSD is much faster than NFS.
echo "Copying latent shards to /tmp..."
mkdir -p /tmp/latents
cp /mnt/home/chunquancheng/latents/*.tar /tmp/latents/
echo "Shards ready: $(ls /tmp/latents/*.tar | wc -l) files"

# ── 3. Run Stage 1b: train predictor ─────────────────────────────────────────
# Training runs until convergence (ReduceLROnPlateau, stops at min_lr=1e-6).
# Checkpoints are written to network storage every 500 steps so they survive
# when this job is killed. latest.pt auto-resumes on relaunch.
echo "Starting training..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/vjepa2 \
    --device=nvidia.com/gpu=$CUDA_VISIBLE_DEVICES \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --network=host \
    jepa2-pi \
    python -m app.vjepa_behavior.train_predictor \
        --latent_dir /tmp/latents \
        --ckpt_dir   /mnt/home/chunquancheng/ckpt_predictor \
        --config     app/vjepa_behavior/configs/vitl-256-b1k.yaml

echo "Job finished. Check /mnt/home/chunquancheng/ckpt_predictor/ for checkpoints."
