#!/bin/bash
#SBATCH --job-name=b1k-train
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/at3dcv/world_model/logs/train_%j.out
#SBATCH --error=/mnt/projects/at3dcv/world_model/logs/train_%j.err

# Resubmit this script as many times as needed — training resumes automatically
# from the latest checkpoint each time. Stop when the loss curve flattens.
#
# NOTE: a latent-precaching variant was tried (encode once to /tmp, then train
# from shards) but the real dataset at --sample_stride 8 --max_episodes_per_task 200
# is ~1.14M samples (~1.1TB) — too big for /tmp (900GB), the 250GB project quota,
# and takes 37h+ to encode. Reverted to live on-the-fly encoding, which is
# already compliant with both kill rules (~12% GPU util, ~52% GPU mem).

set -e
mkdir -p /mnt/projects/at3dcv/world_model/logs
mkdir -p /mnt/projects/at3dcv/world_model/ckpt_predictor

REPO=/mnt/projects/at3dcv/world_model/jepa2-pi
IMAGE=/mnt/projects/at3dcv/world_model/jepa2-pi.tar

# ── 1. Load container image ──────────────────────────────────────────────────
echo "Loading container..."
podman load -i $IMAGE

# ── 2. Run Stage 1b: train predictor ─────────────────────────────────────────
# Training runs until convergence (ReduceLROnPlateau, stops at min_lr=1e-6).
# Checkpoints are written to network storage every 500 steps so they survive
# when this job is killed. latest.pt auto-resumes on relaunch.
# Copy data to local SSD for fast video I/O
echo "Copying b1k data to /tmp..."
mkdir -p /tmp/b1k_data
cp -r /mnt/projects/at3dcv/world_model/b1k_data/data /tmp/b1k_data/
cp -r /mnt/projects/at3dcv/world_model/b1k_data/videos /tmp/b1k_data/
echo "Data ready: $(du -sh /tmp/b1k_data)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Starting training..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/vjepa2 \
    --device=nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e DECORD_EOF_RETRY_MAX=512 \
    -e PYTHONUNBUFFERED=1 \
    --network=host \
    --ipc=host \
    jepa2-pi \
    python -m app.vjepa_behavior.train_predictor \
        --data_root             /tmp/b1k_data \
        --encoder_ckpt          /mnt/projects/at3dcv/world_model/vjepa2_vitl.pt \
        --ckpt_dir              /mnt/projects/at3dcv/world_model/ckpt_predictor \
        --config                app/vjepa_behavior/configs/vitl-256-b1k.yaml \
        --batch_size            192 \
        --max_episodes_per_task 200 \
        --num_workers           4 \
        --prefetch_factor       1 \
        --video_cache_size      1 \
        --episode_cache_size    1 \
        --sample_stride         8 \
        --no-shuffle_live

echo "Job finished. Check /mnt/projects/at3dcv/world_model/ckpt_predictor/ for checkpoints."
