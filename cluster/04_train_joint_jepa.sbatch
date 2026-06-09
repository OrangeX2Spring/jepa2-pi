#!/bin/bash
#SBATCH --job-name=b1k-joint
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/at3dcv/world_model/logs/joint_%j.out
#SBATCH --error=/mnt/projects/at3dcv/world_model/logs/joint_%j.err

set -e
mkdir -p /mnt/projects/at3dcv/world_model/logs
mkdir -p /mnt/projects/at3dcv/world_model/ckpt_joint

REPO=/mnt/projects/at3dcv/world_model/jepa2-pi
IMAGE=/mnt/projects/at3dcv/world_model/jepa2-pi.tar
PI05_PYTORCH_CKPT=${PI05_PYTORCH_CKPT:-/mnt/projects/at3dcv/world_model/pi05_pytorch_ckpt}

if [ ! -f "$PI05_PYTORCH_CKPT/model.safetensors" ]; then
    echo "Missing PI0.5 PyTorch checkpoint: $PI05_PYTORCH_CKPT/model.safetensors"
    echo "Set PI05_PYTORCH_CKPT to a directory containing model.safetensors before submitting."
    exit 1
fi

echo "Loading container..."
podman load -i $IMAGE

echo "Copying b1k data to /tmp..."
mkdir -p /tmp/b1k_data
cp -r /mnt/projects/at3dcv/world_model/b1k_data/data /tmp/b1k_data/
cp -r /mnt/projects/at3dcv/world_model/b1k_data/videos /tmp/b1k_data/
cp -r /mnt/projects/at3dcv/world_model/b1k_data/meta /tmp/b1k_data/
echo "Data ready: $(du -sh /tmp/b1k_data)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Starting joint PI0.5 + V-JEPA training..."
podman run --rm \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w $REPO/openpi-comet \
    --device=nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e DECORD_EOF_RETRY_MAX=512 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=$REPO/openpi-comet/src:$REPO/vjepa2 \
    --network=host \
    --ipc=host \
    jepa2-pi \
    python scripts/train_pytorch.py pi05_b1k-jepa_joint \
        --exp_name pi05-b1k-jepa-joint \
        --pytorch_weight_path $PI05_PYTORCH_CKPT \
        --batch_size 8 \
        --num_workers 2 \
        --jepa_loss_weight 0.05 \
        --jepa_action_anneal_steps 5000

echo "Joint training finished. Check /mnt/projects/at3dcv/world_model/ckpt_joint/."
