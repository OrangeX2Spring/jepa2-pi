#!/bin/bash
#SBATCH --job-name=b1k-download
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=/mnt/projects/at3dcv/world_model/logs/download_%j.out
#SBATCH --error=/mnt/projects/at3dcv/world_model/logs/download_%j.err
# Install huggingface_hub to /tmp (avoids home file quota)
# Download encoder weights
# Download B1K dataset
# Tar and copy to network storage

set -e
BASE=/mnt/projects/at3dcv/world_model
mkdir -p $BASE/logs $BASE/b1k_data

pip install -q --target /tmp/hflib huggingface_hub
export PYTHONPATH=/tmp/hflib:$PYTHONPATH
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}"

[ ! -f $BASE/vjepa2_vitl.pt ] && wget -q https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -O $BASE/vjepa2_vitl.pt

echo "Downloading B1K dataset..."
python3 /mnt/projects/at3dcv/world_model/jepa2-pi/cluster/download_b1k.py
echo "Done: $(ls $BASE/b1k_data | wc -l) items"
