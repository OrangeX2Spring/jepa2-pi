"""
BEHAVIOR-1K dataset for Stage-1 AC predictor training.

Expected data layout (LeRobot / HuggingFace format):
  <data_root>/
    data/
      chunk-000/episode_000000.parquet   (columns: observation.state [23],
                                                    action [23],
                                                    episode_index, frame_index)
      chunk-000/episode_000001.parquet
      ...
    videos/
      <camera_key>/
        chunk-000/episode_000000.mp4
        ...

Each parquet file covers exactly one episode.
If your data is arranged differently, adjust _discover_episodes() below.

Returns per sample:
    frame_t     FloatTensor [3, H, W]   normalised RGB at step t
    action      FloatTensor [chunk, 23] action chunk starting at t
    state       FloatTensor [23]        proprio at t
    frame_tH    FloatTensor [3, H, W]   normalised RGB at step t+chunk
"""

import glob
import os
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalisation (same as vjepa2 pre-training)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


class BehaviorDataset(Dataset):
    """
    Args:
        data_root:   path to the LeRobot-format dataset directory
        camera_key:  subfolder under videos/ that holds per-episode mp4 files
                     e.g. "observation.images.rgb.robot0_frontview"
        chunk_len:   action horizon (32 for π0.5-comet)
        img_size:    spatial crop fed to the encoder (256 for ViT-L recipe)
        task_ids:    optional list of task indices to restrict the dataset;
                     matched against the 'task_index' column if present.
                     Pass None to use all tasks.
    """

    def __init__(
        self,
        data_root: str,
        camera_key: str = "observation.images.rgb.robot0_frontview",
        chunk_len: int = 32,
        img_size: int = 256,
        task_ids=None,
    ):
        self.data_root  = Path(data_root)
        self.camera_key = camera_key
        self.chunk_len  = chunk_len
        self.img_size   = img_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

        self._episodes = self._discover_episodes(task_ids)
        # Build flat index: list of (episode_dict, frame_t_local_idx)
        self._index = self._build_index()

    # ------------------------------------------------------------------
    def _discover_episodes(self, task_ids):
        """
        Load every parquet file under data_root/data/ as one episode.
        Returns list of dicts with keys: states, actions, video_path.
        """
        parquet_files = sorted(glob.glob(
            str(self.data_root / "data" / "**" / "*.parquet"), recursive=True
        ))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.data_root}/data/. "
                "Check data_root and the directory layout described in the docstring."
            )

        episodes = []
        for pq in parquet_files:
            df = pd.read_parquet(pq)
            if task_ids is not None and "task_index" in df.columns:
                df = df[df["task_index"].isin(task_ids)]
                if df.empty:
                    continue

            # Derive expected video path from parquet path
            rel = Path(pq).relative_to(self.data_root / "data")
            video_path = self.data_root / "videos" / self.camera_key / rel.with_suffix(".mp4")

            if not video_path.exists():
                # Tolerate missing video — skip episode with a warning
                import warnings
                warnings.warn(f"Video not found, skipping episode: {video_path}")
                continue

            states  = np.stack(df["observation.state"].values).astype(np.float32)  # [T, 23]
            actions = np.stack(df["action"].values).astype(np.float32)              # [T, 23]

            episodes.append({
                "states":     states,
                "actions":    actions,
                "video_path": str(video_path),
                "length":     len(df),
            })

        if not episodes:
            raise RuntimeError("No valid episodes found — check task_ids and video paths.")
        return episodes

    def _build_index(self):
        index = []
        for ep_idx, ep in enumerate(self._episodes):
            # Valid start frames: need chunk_len steps ahead for action + frame_tH
            for t in range(ep["length"] - self.chunk_len):
                index.append((ep_idx, t))
        return index

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]

        tH = t + self.chunk_len

        # Decode only the two needed frames using decord (avoids full video load)
        vr = decord.VideoReader(ep["video_path"], ctx=decord.cpu(0))
        # clamp tH in case the video is slightly shorter than the parquet
        tH_clamped = min(tH, len(vr) - 1)
        frames = vr.get_batch([t, tH_clamped]).asnumpy()  # [2, H, W, 3] uint8

        frame_t  = self.transform(frames[0])   # [3, img_size, img_size]
        frame_tH = self.transform(frames[1])   # [3, img_size, img_size]

        action_chunk = torch.from_numpy(ep["actions"][t : t + self.chunk_len])  # [32, 23]
        state_t      = torch.from_numpy(ep["states"][t])                         # [23]

        return frame_t, action_chunk, state_t, frame_tH
