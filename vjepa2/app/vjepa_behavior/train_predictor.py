"""
Stage 1b — Train the BEHAVIOR-1K AC predictor on cached latents.

Training runs until convergence (no fixed step budget). The scheduler
(ReduceLROnPlateau) decays the LR when the loss stops improving; training
stops automatically when the LR falls to min_lr.

On a SLURM cluster where jobs are killed before convergence: relaunch the
same command — the script resumes from the latest checkpoint automatically.
Resubmit as many times as needed until the loss curve flattens.

Kill-rule compliance (TUM CAMP 24g partition):
    Default batch_size=64 keeps GPU memory ~14–16 GB (above the 12 GB kill
    threshold). Adjust if needed.

Checkpoints:
    latest.pt            — overwritten every meta.save_every steps (default 500)
    step_XXXXXX.pt       — permanent snapshot every meta.snapshot_every steps

Usage:
    python -m app.vjepa_behavior.train_predictor \
        --latent_dir /tmp/latents \
        --ckpt_dir   /mnt/home/<user>/ckpt_predictor \
        --config     app/vjepa_behavior/configs/vitl-256-b1k.yaml
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.ac_predictor import vit_ac_predictor

# ------------------------------------------------------------------
# Constants matching π0.5-comet / R1Pro
ACTION_DIM       = 23
CHUNK_LEN        = 32
STATE_DIM        = 23
FLAT_ACTION      = CHUNK_LEN * ACTION_DIM   # 736 — fed as a single token per frame
ENCODER_DIM      = 1024
TOKENS_PER_FRAME = 256                      # (256px / 16 patch)^2


# ------------------------------------------------------------------
def build_predictor(cfg: dict, device: torch.device):
    pred_cfg = cfg["model"]
    predictor = vit_ac_predictor(
        img_size                     = cfg["data"]["crop_size"],
        patch_size                   = cfg["data"]["patch_size"],
        num_frames                   = 2,
        tubelet_size                 = cfg["data"]["tubelet_size"],
        embed_dim                    = ENCODER_DIM,
        predictor_embed_dim          = pred_cfg["pred_embed_dim"],
        depth                        = pred_cfg["pred_depth"],
        num_heads                    = pred_cfg["pred_num_heads"],
        is_frame_causal              = True,
        use_rope                     = pred_cfg.get("use_rope", True),
        use_activation_checkpointing = pred_cfg.get("use_activation_checkpointing", False),
        action_embed_dim             = FLAT_ACTION,
        state_embed_dim              = STATE_DIM,
        use_extrinsics               = False,
    )
    predictor = predictor.to(device)
    n = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Predictor parameters: {n:,}")
    return predictor


# ------------------------------------------------------------------
def make_loader(latent_dir: str, batch_size: int, shuffle_buffer: int = 5000):
    shards = sorted([
        os.path.join(latent_dir, f)
        for f in os.listdir(latent_dir)
        if f.endswith(".tar")
    ])
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {latent_dir}")

    def decode_sample(sample):
        z_t    = torch.from_numpy(np.load(io.BytesIO(sample["z_t.npy"]),    allow_pickle=False)).float()
        z_tH   = torch.from_numpy(np.load(io.BytesIO(sample["z_tH.npy"]),   allow_pickle=False)).float()
        action = torch.from_numpy(np.load(io.BytesIO(sample["action.npy"]), allow_pickle=False)).float()
        state  = torch.from_numpy(np.load(io.BytesIO(sample["state.npy"]),  allow_pickle=False)).float()
        return z_t, z_tH, action, state

    dataset = (
        wds.WebDataset(shards, shardshuffle=True)
        .shuffle(shuffle_buffer)
        .map(decode_sample)
        .batched(batch_size, partial=False)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)


def infinite_loader(latent_dir: str, batch_size: int):
    """Yields batches indefinitely, restarting when the dataset is exhausted."""
    while True:
        yield from make_loader(latent_dir, batch_size)


# ------------------------------------------------------------------
def loss_fn(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(z_pred - z_target))


def forward_step(predictor, z_t, action_chunk, state_t):
    B = z_t.size(0)
    action_flat = action_chunk.reshape(B, FLAT_ACTION).unsqueeze(1)  # [B, 1, 736]
    state_in    = state_t.unsqueeze(1)                                # [B, 1, 23]
    # Layer-norm before encoders: handles mixed units (velocities, angles, [0,1] gripper)
    action_flat = F.layer_norm(action_flat, action_flat.shape[-1:])
    state_in    = F.layer_norm(state_in,    state_in.shape[-1:])
    return predictor(z_t, action_flat, state_in)


# ------------------------------------------------------------------
def save_checkpoint(path, predictor, optimizer, scheduler, step, loss):
    """Full checkpoint for resuming training (model + optimizer + scheduler)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step":      step,
        "loss":      loss,
    }, path)


def save_snapshot(path, predictor, step, loss):
    """Lightweight snapshot for evaluation — model weights only (~120 MB vs ~360 MB)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "predictor": predictor.state_dict(),
        "step":      step,
        "loss":      loss,
    }, path)


def load_checkpoint(path, predictor, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    predictor.load_state_dict(ckpt["predictor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from step {ckpt['step']}  (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


# ------------------------------------------------------------------
def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    opt_cfg  = cfg["optimization"]
    meta_cfg = cfg.get("meta", {})

    save_every     = meta_cfg.get("save_every",     500)
    snapshot_every = meta_cfg.get("snapshot_every", 2000)
    log_freq       = meta_cfg.get("log_freq",        100)
    min_lr         = opt_cfg.get("min_lr",           1e-6)

    predictor = build_predictor(cfg, device)
    predictor.train()

    optimizer = AdamW(
        predictor.parameters(),
        lr           = opt_cfg["lr"],
        weight_decay = opt_cfg["weight_decay"],
        betas        = (0.9, 0.95),
    )
    # Decay LR when loss plateaus; stop when LR reaches min_lr.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = opt_cfg.get("lr_decay_factor", 0.5),
        patience  = opt_cfg.get("lr_patience",     1000),
        min_lr    = min_lr,
    )

    # Auto-resume
    start_step = 0
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        start_step = load_checkpoint(latest, predictor, optimizer, scheduler)

    scaler   = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    data_gen = infinite_loader(args.latent_dir, opt_cfg["batch_size"])
    recent_losses = []
    t0 = time.time()
    step = start_step

    print(f"Training from step {start_step} — runs until LR reaches {min_lr:.0e}")

    while True:
        z_t, z_tH, action, state = next(data_gen)

        z_t    = z_t.to(device,    non_blocking=True)
        z_tH   = z_tH.to(device,   non_blocking=True)
        action = action.to(device, non_blocking=True)
        state  = state.to(device,  non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            z_pred = forward_step(predictor, z_t, action, state)
            loss   = loss_fn(z_pred, z_tH)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        recent_losses.append(loss.item())
        step += 1

        # Log + plateau-based LR update
        if step % log_freq == 0:
            avg_loss   = np.mean(recent_losses[-log_freq:])
            current_lr = optimizer.param_groups[0]["lr"]
            mem        = torch.cuda.max_memory_allocated(device) / 1024**2
            elapsed    = time.time() - t0
            print(
                f"step {step:7d}"
                f" | loss {avg_loss:.4f}"
                f" | lr {current_lr:.2e}"
                f" | GPU {mem:.0f} MB"
                f" | {elapsed:.0f}s"
            )
            if mem < 12 * 1024:
                print(
                    f"  WARNING: GPU {mem:.0f} MB < 12288 MB. "
                    "Increase batch_size to avoid auto-kill on 24g partition."
                )
            scheduler.step(avg_loss)

        # Periodic latest checkpoint (safe to lose at most save_every steps)
        if step % save_every == 0:
            save_checkpoint(latest, predictor, optimizer, scheduler, step, loss.item())
            print(f"  [ckpt] step {step} -> {latest}")

        # Permanent snapshot — weights only (lightweight, for evaluation)
        if step % snapshot_every == 0:
            snap = os.path.join(args.ckpt_dir, f"step_{step:06d}.pt")
            save_snapshot(snap, predictor, step, loss.item())
            print(f"  [snap] step {step} -> {snap}")

        # Convergence: LR has decayed to the floor — nothing left to learn
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr <= min_lr and step > opt_cfg.get("lr_patience", 1000):
            print(f"LR reached min_lr ({current_lr:.2e}) at step {step} — converged.")
            save_checkpoint(latest, predictor, optimizer, scheduler, step, loss.item())
            snap = os.path.join(args.ckpt_dir, f"step_{step:06d}_final.pt")
            save_snapshot(snap, predictor, step, loss.item())
            print(f"Final snapshot saved: {snap}")
            break


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--ckpt_dir",   required=True)
    parser.add_argument("--config",     default="app/vjepa_behavior/configs/vitl-256-b1k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(args, cfg)


if __name__ == "__main__":
    main()
