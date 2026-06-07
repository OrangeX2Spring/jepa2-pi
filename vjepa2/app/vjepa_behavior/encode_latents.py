"""
Stage 1a — Offline ViT-L latent encoding.

Reads BEHAVIOR-1K frames from BehaviorDataset, encodes frame_t and
frame_{t+H} with the frozen V-JEPA 2 ViT-L encoder, and writes
WebDataset .tar shards to an output directory.

This runs ONCE.  The shards are reused by train_predictor.py.

Output per sample (inside each tar shard):
    <key>.z_t.npy       float16  [256, 1024]  latent of frame_t
    <key>.z_tH.npy      float16  [256, 1024]  latent of frame_{t+H}
    <key>.action.npy    float32  [32,  23  ]  action chunk
    <key>.state.npy     float32  [23        ]  proprio at t

Storage estimate (ViT-L, 256px):
    256 tokens × 1024 dim × 2 bytes × 2 frames = ~1 MB / sample
    100k samples ≈ 100 GB  →  keep within cluster 250 GB quota.

Usage:
    python -m app.vjepa_behavior.encode_latents \
        --data_root  /mnt/projects/<course>/b1k_data \
        --camera_key observation.images.rgb.robot0_frontview \
        --encoder_ckpt /mnt/projects/<course>/vjepa2_vitl.pt \
        --out_dir     /mnt/projects/<course>/latents \
        --shard_size  2000 \
        --batch_size  32 \
        --num_workers 4
"""

import argparse
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

# Add vjepa2 root to path when run as __main__
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.vjepa_behavior.dataset import BehaviorDataset
from src.hub.backbones import vjepa2_vit_large


# ------------------------------------------------------------------
def load_encoder(ckpt_path: str, device: torch.device):
    """Load frozen ViT-L encoder from a local checkpoint or hub."""
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading encoder from local checkpoint: {ckpt_path}")
        # build the model architecture first, then load weights
        encoder, _ = _build_vitl_encoder(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # V-JEPA checkpoints store weights under 'target_encoder' or 'encoder'
        state_key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        state_dict = ckpt[state_key]
        # strip DistributedDataParallel 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=True)
    else:
        print("Loading ViT-L encoder from torch.hub (requires internet)…")
        encoder = vjepa2_vit_large(pretrained=True)
        encoder = encoder.to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded. Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    return encoder


def _build_vitl_encoder(device):
    """Instantiate ViT-L without pretrained weights (used when loading from ckpt)."""
    encoder = vjepa2_vit_large(pretrained=False).to(device)
    return encoder, None


# ------------------------------------------------------------------
@torch.no_grad()
def encode_frame(encoder, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encode a batch of single frames into V-JEPA latents.

    Args:
        frames: [B, 3, H, W]  normalised RGB
    Returns:
        latents: [B, num_tokens, embed_dim]  float32
    """
    B = frames.size(0)
    # Replicate the single frame into a (tubelet_size=2)-frame clip
    # Shape required by encoder: [B, C, T, H, W] where T=tubelet_size=2
    clip = frames.unsqueeze(2).repeat(1, 1, 2, 1, 1).to(device)  # [B, 3, 2, H, W]
    latents = encoder(clip)                                        # [B, num_tokens, D]
    return latents.float()


# ------------------------------------------------------------------
def write_shards(encoder, loader, out_dir: str, shard_size: int, device: torch.device):
    os.makedirs(out_dir, exist_ok=True)
    shard_pattern = os.path.join(out_dir, "shard-%06d.tar")

    sample_idx = 0
    shard_idx  = 0
    sink = None

    def open_shard():
        nonlocal shard_idx, sink
        if sink is not None:
            sink.close()
        path = shard_pattern % shard_idx
        sink = wds.TarWriter(path)
        print(f"  opened shard {path}")
        shard_idx += 1

    open_shard()

    for batch_idx, (frame_t, action_chunk, state_t, frame_tH) in enumerate(loader):
        # Encode both frames
        z_t  = encode_frame(encoder, frame_t,  device)   # [B, 256, 1024]
        z_tH = encode_frame(encoder, frame_tH, device)   # [B, 256, 1024]

        # Layer-norm (matches train_predictor.py normalise_reps=True)
        z_t  = F.layer_norm(z_t,  z_t.shape[-1:])
        z_tH = F.layer_norm(z_tH, z_tH.shape[-1:])

        B = z_t.size(0)
        for i in range(B):
            if sample_idx > 0 and sample_idx % shard_size == 0:
                open_shard()

            key = f"{sample_idx:010d}"
            sink.write({
                "__key__": key,
                "z_t.npy":     _to_bytes(z_t[i].half().cpu().numpy()),
                "z_tH.npy":    _to_bytes(z_tH[i].half().cpu().numpy()),
                "action.npy":  _to_bytes(action_chunk[i].cpu().numpy()),
                "state.npy":   _to_bytes(state_t[i].cpu().numpy()),
            })
            sample_idx += 1

        if batch_idx % 50 == 0:
            used = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  batch {batch_idx:6d} | samples {sample_idx:8d} | GPU {used:.0f} MB")

    if sink is not None:
        sink.close()
    print(f"Done. {sample_idx} samples written to {out_dir} ({shard_idx} shards).")


def _to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--camera_key",   default="observation.images.rgb.robot0_frontview")
    parser.add_argument("--task_ids",     nargs="*", type=int, default=None,
                        help="Restrict to these task indices (space-separated). Default: all.")
    parser.add_argument("--encoder_ckpt", default="",
                        help="Path to local V-JEPA 2 checkpoint. Leave empty to use torch.hub.")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--shard_size",   type=int, default=2000,
                        help="Samples per .tar shard. 2000 × ~1 MB ≈ 2 GB per shard.")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--chunk_len",    type=int, default=32)
    parser.add_argument("--img_size",     type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = load_encoder(args.encoder_ckpt, device)

    dataset = BehaviorDataset(
        data_root=args.data_root,
        camera_key=args.camera_key,
        chunk_len=args.chunk_len,
        img_size=args.img_size,
        task_ids=args.task_ids,
    )
    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,        # preserve order for reproducibility
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    write_shards(encoder, loader, args.out_dir, args.shard_size, device)


if __name__ == "__main__":
    main()
