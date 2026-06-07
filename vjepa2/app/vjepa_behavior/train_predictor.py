"""
Stage 1b — Train the BEHAVIOR-1K AC predictor on cached latents.

The encoder is NEVER loaded here; all training consumes the .tar shards
written by encode_latents.py.  The predictor (~70M params) trains alone,
so this fits easily on a single 24 GB GPU.

Kill-rule compliance (TUM CAMP 24g partition):
    GPU memory must average > 12 GB or the job is killed after 6h.
    Default batch_size=64 keeps ~14–16 GB in use; adjust if needed.

Checkpoint / resume:
    Latest checkpoint is saved after every epoch to <ckpt_dir>/latest.pt.
    Relaunch with the same command; the script resumes automatically.

Usage:
    python -m app.vjepa_behavior.train_predictor \
        --latent_dir /tmp/latents \
        --ckpt_dir   /mnt/projects/<course>/ckpt_predictor \
        --config     app/vjepa_behavior/configs/vitl-256-b1k.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.ac_predictor import vit_ac_predictor

# ------------------------------------------------------------------
# Constants matching π0.5-comet / R1Pro
ACTION_DIM   = 23
CHUNK_LEN    = 32
STATE_DIM    = 23
FLAT_ACTION  = CHUNK_LEN * ACTION_DIM   # 736  — fed as a single token per frame
# ViT-L encoder output
ENCODER_DIM  = 1024
TOKENS_PER_FRAME = 256   # (256px / 16 patch)^2


# ------------------------------------------------------------------
def build_predictor(cfg: dict, device: torch.device):
    """Instantiate a fresh AC predictor for R1Pro (no pretrained weights)."""
    pred_cfg = cfg["model"]
    predictor = vit_ac_predictor(
        img_size        = cfg["data"]["crop_size"],
        patch_size      = cfg["data"]["patch_size"],
        num_frames      = 2,            # context frame + target frame
        tubelet_size    = cfg["data"]["tubelet_size"],
        embed_dim       = ENCODER_DIM,
        predictor_embed_dim = pred_cfg["pred_embed_dim"],
        depth           = pred_cfg["pred_depth"],
        num_heads       = pred_cfg["pred_num_heads"],
        is_frame_causal = True,
        use_rope        = pred_cfg.get("use_rope", True),
        use_activation_checkpointing = pred_cfg.get("use_activation_checkpointing", False),
        action_embed_dim = FLAT_ACTION,
        state_embed_dim  = STATE_DIM,
        use_extrinsics   = False,
    )
    predictor = predictor.to(device)
    n = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Predictor parameters: {n:,}")
    return predictor


# ------------------------------------------------------------------
def make_loader(latent_dir: str, batch_size: int, shuffle_buffer: int = 5000):
    """Build a WebDataset loader over the cached latent shards."""
    shards = sorted([
        os.path.join(latent_dir, f)
        for f in os.listdir(latent_dir)
        if f.endswith(".tar")
    ])
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {latent_dir}")
    print(f"Found {len(shards)} shards in {latent_dir}")

    def decode_sample(sample):
        z_t     = torch.from_numpy(np.load(sample["z_t.npy"],    allow_pickle=False)).float()
        z_tH    = torch.from_numpy(np.load(sample["z_tH.npy"],   allow_pickle=False)).float()
        action  = torch.from_numpy(np.load(sample["action.npy"], allow_pickle=False)).float()
        state   = torch.from_numpy(np.load(sample["state.npy"],  allow_pickle=False)).float()
        return z_t, z_tH, action, state

    dataset = (
        wds.WebDataset(shards, shardshuffle=True)
        .shuffle(shuffle_buffer)
        .map(decode_sample)
        .batched(batch_size, partial=False)
    )
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)
    return loader


# ------------------------------------------------------------------
def loss_fn(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """L1 loss on latents (latents are already layer-normed by encode_latents.py)."""
    return torch.mean(torch.abs(z_pred - z_target))


def forward_step(predictor, z_t, action_chunk, state_t):
    """
    Single teacher-forced prediction step.

    Args:
        z_t:          [B, 256, 1024]  context latent
        action_chunk: [B, 32,  23  ]  action chunk
        state_t:      [B, 23       ]  proprio at t
    Returns:
        z_pred: [B, 256, 1024]  predicted target latent
    """
    # Flatten action chunk: [B, 32, 23] -> [B, 1, 736]  (T=1 context frame)
    B = z_t.size(0)
    action_flat = action_chunk.reshape(B, FLAT_ACTION).unsqueeze(1)  # [B, 1, 736]
    state_in    = state_t.unsqueeze(1)                                # [B, 1, 23]

    z_pred = predictor(z_t, action_flat, state_in)   # [B, 256, 1024]
    return z_pred


# ------------------------------------------------------------------
def save_checkpoint(path, predictor, optimizer, scheduler, epoch, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "predictor":  predictor.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "epoch":      epoch,
        "loss":       loss,
    }, path)


def load_checkpoint(path, predictor, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    predictor.load_state_dict(ckpt["predictor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from epoch {ckpt['epoch']}  (loss={ckpt['loss']:.4f})")
    return ckpt["epoch"]


# ------------------------------------------------------------------
def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    opt_cfg = cfg["optimization"]
    predictor = build_predictor(cfg, device)
    predictor.train()

    optimizer = AdamW(
        predictor.parameters(),
        lr           = opt_cfg["lr"],
        weight_decay = opt_cfg["weight_decay"],
        betas        = (0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = opt_cfg["epochs"],
        eta_min = opt_cfg.get("final_lr", 0.0),
    )

    # Auto-resume from latest checkpoint
    start_epoch = 0
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        start_epoch = load_checkpoint(latest, predictor, optimizer, scheduler)

    loader = make_loader(args.latent_dir, opt_cfg["batch_size"])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, opt_cfg["epochs"]):
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (z_t, z_tH, action, state) in enumerate(loader):
            z_t    = z_t.to(device,    non_blocking=True)    # [B, 256, 1024]
            z_tH   = z_tH.to(device,   non_blocking=True)
            action = action.to(device, non_blocking=True)    # [B, 32, 23]
            state  = state.to(device,  non_blocking=True)    # [B, 23]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z_pred = forward_step(predictor, z_t, action, state)
                loss   = loss_fn(z_pred, z_tH)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(loss.item())

            if batch_idx % 100 == 0:
                mem = torch.cuda.max_memory_allocated(device) / 1024**2
                print(
                    f"epoch {epoch+1:3d} | batch {batch_idx:6d} "
                    f"| loss {loss.item():.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| GPU {mem:.0f} MB"
                )
                # Safety check: warn if GPU memory is below the kill threshold
                if mem < 12 * 1024:
                    print(
                        f"  WARNING: GPU memory {mem:.0f} MB < 12288 MB. "
                        "Increase batch_size to avoid auto-kill on 24g partition."
                    )

        scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        elapsed  = time.time() - t0
        print(f"=== epoch {epoch+1} | avg loss {avg_loss:.4f} | {elapsed:.0f}s ===")

        # Save after every epoch (survives preemption on opportunistic QoS)
        save_checkpoint(latest, predictor, optimizer, scheduler, epoch + 1, avg_loss)

        # Also save a numbered snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            snap = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            save_checkpoint(snap, predictor, optimizer, scheduler, epoch + 1, avg_loss)
            print(f"Snapshot saved: {snap}")

    print("Training complete.")


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True,
                        help="Directory of .tar shards from encode_latents.py. "
                             "Copy to /tmp at job start for faster I/O.")
    parser.add_argument("--ckpt_dir",   required=True,
                        help="Network storage path for checkpoints (survives job end).")
    parser.add_argument("--config",     default="app/vjepa_behavior/configs/vitl-256-b1k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(args, cfg)


if __name__ == "__main__":
    main()
