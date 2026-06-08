#!/bin/bash
#SBATCH --job-name=b1k-download
#SBATCH --partition=data
#SBATCH --qos=students_normal
#SBATCH --nodes=1
#SBATCH --output=/mnt/home/chunquancheng/logs/download_%j.out
#SBATCH --error=/mnt/home/chunquancheng/logs/download_%j.err

set -e
mkdir -p /mnt/home/chunquancheng/logs

# ── 1. Install huggingface_hub in /tmp (no network-storage file quota hit) ──
python3 -m venv /tmp/dl_env
source /tmp/dl_env/bin/activate
pip install -q huggingface_hub

# ── 2. Download V-JEPA 2 ViT-L encoder checkpoint ───────────────────────────
echo "Downloading V-JEPA 2 ViT-L checkpoint..."
wget -q https://dl.fbaipublicfiles.com/vjepa2/vitl.pt \
     -O /tmp/vjepa2_vitl.pt
cp /tmp/vjepa2_vitl.pt /mnt/home/chunquancheng/vjepa2_vitl.pt
echo "Encoder checkpoint saved: /mnt/home/chunquancheng/vjepa2_vitl.pt"

# ── 3. Download BEHAVIOR-1K dataset (5 tasks) from HuggingFace ───────────────
# Adjust --include patterns to match your 5 task folders/files if needed.
# Without --include, this downloads the full dataset (may be large).
echo "Downloading B1K dataset..."
huggingface-cli download behavior-1k/2025-challenge-demos \
    --repo-type dataset \
    --local-dir /tmp/b1k_raw

# ── 4. Archive to a single tar (keeps file count within 100k quota) ──────────
echo "Archiving to tar..."
tar -cf /tmp/b1k_data.tar -C /tmp b1k_raw
cp /tmp/b1k_data.tar /mnt/home/chunquancheng/b1k_data.tar
echo "Dataset saved: /mnt/home/chunquancheng/b1k_data.tar"
echo "Done."
