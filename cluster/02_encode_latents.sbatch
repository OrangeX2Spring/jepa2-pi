#!/bin/bash
#SBATCH --job-name=b1k-encode
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/chunquancheng/logs/encode_%j.out
#SBATCH --error=/mnt/home/chunquancheng/logs/encode_%j.err

set -e
mkdir -p /mnt/home/chunquancheng/logs
mkdir -p /mnt/home/chunquancheng/latents

REPO=/mnt/home/chunquancheng/jepa2-pi
IMAGE=/mnt/home/chunquancheng/jepa2-pi.tar

# ── 1. Load container image ──────────────────────────────────────────────────
echo "Loading container..."
podman load -i $IMAGE

# ── 2. Copy B1K data to fast local storage and mount as virtual filesystem ───
echo "Extracting B1K data to /tmp..."
tar -xf /mnt/home/chunquancheng/b1k_data.tar -C /tmp
# Now data is at /tmp/b1k_raw

# ── 3. Run Stage 1a: encode latents ─────────────────────────────────────────
# Latents are written directly to network storage (sequential large files = OK over NFS).
echo "Encoding latents..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/vjepa2 \
    --device=nvidia.com/gpu=$CUDA_VISIBLE_DEVICES \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --network=host \
    jepa2-pi \
    python -m app.vjepa_behavior.encode_latents \
        --data_root      /tmp/b1k_raw \
        --encoder_ckpt   /mnt/home/chunquancheng/vjepa2_vitl.pt \
        --out_dir        /mnt/home/chunquancheng/latents \
        --shard_size     2000 \
        --batch_size     32 \
        --num_workers    4

echo "Encoding done. Shards at /mnt/home/chunquancheng/latents/"
ls -lh /mnt/home/chunquancheng/latents/
