#!/bin/bash
#SBATCH --job-name=b1k-train
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/at3dcv/world_model/logs/train_%j.out
#SBATCH --error=/mnt/projects/at3dcv/world_model/logs/train_%j.err

# Two-phase job, designed around the cluster constraints:
#   - only 4 CPU cores (no --cpus-per-task allowed) → video decoding is the
#     bottleneck, so pay it ONCE per job instead of once per epoch
#   - 250 GB network quota → latents live on node-local /tmp (900 GB SSD,
#     free, wiped at job end), never on network storage
#
# Phase A: copy videos to /tmp, encode latents to /tmp/latents (~25 GB at
#          stride 8 / 200 eps-per-task; ≈ one epoch's worth of decode time).
# Phase B: train the predictor from /tmp/latents — pure GPU work, fed from
#          local SSD, so GPU utilisation stays well above the 10% kill rule.
#
# Resubmit as many times as needed — training auto-resumes from latest.pt.

set -e
mkdir -p /mnt/projects/at3dcv/world_model/logs
mkdir -p /mnt/projects/at3dcv/world_model/ckpt_predictor

REPO=/mnt/projects/at3dcv/world_model/jepa2-pi
IMAGE=/mnt/projects/at3dcv/world_model/jepa2-pi.tar

# ── 1. Load container image ──────────────────────────────────────────────────
echo "Loading container..."
podman load -i $IMAGE

# ── 2. Copy b1k data to local SSD for fast video I/O ────────────────────────
echo "Copying b1k data to /tmp..."
mkdir -p /tmp/b1k_data
cp -r /mnt/projects/at3dcv/world_model/b1k_data/data /tmp/b1k_data/
cp -r /mnt/projects/at3dcv/world_model/b1k_data/videos /tmp/b1k_data/
echo "Data ready: $(du -sh /tmp/b1k_data)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── 3. Phase A: encode latents once to local SSD ─────────────────────────────
# max_episodes_per_task / sample_stride MUST match what training expects.
echo "Encoding latents to /tmp/latents..."
mkdir -p /tmp/latents
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
    python -m app.vjepa_behavior.encode_latents \
        --data_root             /tmp/b1k_data \
        --camera_key            observation.images.rgb.head \
        --encoder_ckpt          /mnt/projects/at3dcv/world_model/vjepa2_vitl.pt \
        --out_dir               /tmp/latents \
        --shard_size            2000 \
        --batch_size            512 \
        --num_workers           4 \
        --max_episodes_per_task 200 \
        --sample_stride         8
echo "Latents ready: $(du -sh /tmp/latents)"

# Videos are no longer needed — free the local SSD for other users.
rm -rf /tmp/b1k_data

# ── 4. Phase B: train predictor from cached latents ─────────────────────────
# No encoder on the GPU any more, so memory drops a lot — batch_size must be
# large enough to keep GPU memory >= 12 GB (24g kill rule). Check the
# "warmup step ... GPU xxx MB" lines in the log and adjust: the script also
# prints a WARNING if memory is below 12288 MB.
echo "Starting training..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/vjepa2 \
    --device=nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e PYTHONUNBUFFERED=1 \
    --network=host \
    --ipc=host \
    jepa2-pi \
    python -m app.vjepa_behavior.train_predictor \
        --latent_dir  /tmp/latents \
        --ckpt_dir    /mnt/projects/at3dcv/world_model/ckpt_predictor \
        --config      app/vjepa_behavior/configs/vitl-256-b1k.yaml \
        --batch_size  768 \
        --num_workers 4

echo "Job finished. Check /mnt/projects/at3dcv/world_model/ckpt_predictor/ for checkpoints."
