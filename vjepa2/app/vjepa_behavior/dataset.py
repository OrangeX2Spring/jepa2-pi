"""
BEHAVIOR-1K dataset for Stage-1 AC predictor training.

Expected data layout (LeRobot / HuggingFace format):
  <data_root>/
    data/
      <task>/episode_000000.parquet   (columns: observation.state [256],
                                                action [23],
                                                task_index, frame_index)
      ...
    videos/
      <task>/
        <camera_key>/
          episode_000000.mp4
          ...

Each parquet file covers exactly one episode.

Returns per sample:
    frame_t     FloatTensor [3, H, W]   normalised RGB at step t
    action      FloatTensor [chunk, 23] action chunk starting at t
    state       FloatTensor [256]       proprio at t
    frame_tH    FloatTensor [3, H, W]   normalised RGB at step t+chunk
"""

import glob
import warnings
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

_VR_CACHE_MAX = 500   # VideoReader objects per worker; each holds ~1-5 MB of video index


class BehaviorDataset(Dataset):
    """
    Args:
        data_root:             path to the LeRobot-format dataset directory
        camera_key:            subfolder under videos/<task>/ holding per-episode mp4s
        chunk_len:             action horizon (32 for π0.5-comet)
        img_size:              spatial crop fed to the encoder (256 for ViT-L recipe)
        task_ids:              optional list of task indices to restrict the dataset
        max_episodes_per_task: cap episodes per task to control dataset size and RAM;
                               None = use all episodes
    """

    def __init__(
        self,
        data_root: str,
        camera_key: str = "observation.images.rgb.head",
        chunk_len: int = 32,
        img_size: int = 256,
        task_ids=None,
        max_episodes_per_task: int = None,
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

        self._episodes = self._discover_episodes(task_ids, max_episodes_per_task)
        self._index    = self._build_index()

        # Per-worker VideoReader cache — populated after DataLoader forks workers.
        # Avoids re-opening the same MP4 file on every __getitem__ call.
        self._vr_cache: dict = {}

    # ------------------------------------------------------------------
    def _discover_episodes(self, task_ids, max_episodes_per_task):
        parquet_files = sorted(glob.glob(
            str(self.data_root / "data" / "**" / "*.parquet"), recursive=True
        ))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.data_root}/data/. "
                "Check data_root and the directory layout described in the docstring."
            )

        episodes = []
        task_counts: dict = {}   # task_key -> episode count (for per-task cap)

        for pq in parquet_files:
            df = pd.read_parquet(pq)

            if task_ids is not None and "task_index" in df.columns:
                df = df[df["task_index"].isin(task_ids)]
                if df.empty:
                    continue

            # Per-task cap: derive a stable task key from the parquet directory name
            task_key = Path(pq).parent.name
            if max_episodes_per_task is not None:
                if task_counts.get(task_key, 0) >= max_episodes_per_task:
                    continue
            task_counts[task_key] = task_counts.get(task_key, 0) + 1

            # Derive video path: data/<task>/ep.parquet → videos/<task>/<camera>/ep.mp4
            rel     = Path(pq).relative_to(self.data_root / "data")
            rel_mp4 = rel.with_suffix(".mp4")
            video_path = self.data_root / "videos" / rel_mp4.parent / self.camera_key / rel_mp4.name

            if not video_path.exists():
                warnings.warn(f"Video not found, skipping: {video_path}")
                continue

            states  = np.stack(df["observation.state"].values).astype(np.float32)  # [T, 256]
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

        try:
            vpath = ep["video_path"]
            if vpath not in self._vr_cache:
                # Evict oldest half when cache is full (dict preserves insertion order)
                if len(self._vr_cache) >= _VR_CACHE_MAX:
                    for k in list(self._vr_cache.keys())[: _VR_CACHE_MAX // 2]:
                        del self._vr_cache[k]
                self._vr_cache[vpath] = decord.VideoReader(vpath, ctx=decord.cpu(0))

            vr = self._vr_cache[vpath]
            tH_clamped = min(tH, len(vr) - 1)
            frames = vr.get_batch([t, tH_clamped]).asnumpy()
        except Exception:
            dummy  = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frames = np.stack([dummy, dummy])

        frame_t  = self.transform(frames[0])
        frame_tH = self.transform(frames[1])

        action_chunk = torch.from_numpy(ep["actions"][t : t + self.chunk_len])  # [32, 23]
        state_t      = torch.from_numpy(ep["states"][t])                         # [256]

        return frame_t, action_chunk, state_t, frame_tH
