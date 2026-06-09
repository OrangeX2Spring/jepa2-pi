#!/bin/bash
#SBATCH --job-name=b1k-encode
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/at3dcv/world_model/logs/encode_%j.out
#SBATCH --error=/mnt/projects/at3dcv/world_model/logs/encode_%j.err

set -e
mkdir -p /mnt/projects/at3dcv/world_model/logs
mkdir -p /mnt/projects/at3dcv/world_model/latents

REPO=/mnt/projects/at3dcv/world_model/jepa2-pi
IMAGE=/mnt/projects/at3dcv/world_model/jepa2-pi.tar

# ── 1. Load container image ──────────────────────────────────────────────────
echo "Loading container..."
podman load -i $IMAGE

# ── 2. Copy B1K data to fast local storage and mount as virtual filesystem ───
echo "Extracting B1K data to /tmp..."
# Now data is at /tmp/b1k_raw

# ── 3. Run Stage 1a: encode latents ─────────────────────────────────────────
# Latents are written directly to network storage (sequential large files = OK over NFS).
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Encoding latents..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/vjepa2 \
    --device=nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --network=host \
    jepa2-pi \
    python -m app.vjepa_behavior.encode_latents \
        --data_root      /mnt/projects/at3dcv/world_model/b1k_data \
        --camera_key    observation.images.rgb.head \
        --camera_key    observation.images.rgb.head \
        --encoder_ckpt   /mnt/projects/at3dcv/world_model/vjepa2_vitl.pt \
        --out_dir        /mnt/projects/at3dcv/world_model/latents \
        --shard_size     2000 \
        --batch_size     32 \
        --num_workers    4

echo "Encoding done. Shards at /mnt/projects/at3dcv/world_model/latents/"
ls -lh /mnt/projects/at3dcv/world_model/latents/
