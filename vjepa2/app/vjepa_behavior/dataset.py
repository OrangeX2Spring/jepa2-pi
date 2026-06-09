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
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalisation (same as vjepa2 pre-training)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

_VR_CACHE_MAX = 32
_EPISODE_CACHE_MAX = 32


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
        video_cache_size: int = _VR_CACHE_MAX,
        episode_cache_size: int = _EPISODE_CACHE_MAX,
    ):
        self.data_root  = Path(data_root)
        self.camera_key = camera_key
        self.chunk_len  = chunk_len
        self.img_size   = img_size
        self.video_cache_size = max(0, video_cache_size)
        self.episode_cache_size = max(0, episode_cache_size)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

        self._episodes = self._discover_episodes(task_ids, max_episodes_per_task)
        self._offsets  = self._build_offsets()
        self._num_samples = self._offsets[-1]

        # Per-worker caches populated after DataLoader forks workers. Keeping
        # these bounded is important on the 24 GB Slurm nodes.
        self._vr_cache: OrderedDict[str, decord.VideoReader] = OrderedDict()
        self._episode_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

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

        task_id_set = set(task_ids) if task_ids is not None else None

        for pq in parquet_files:
            parquet_file = parquet.ParquetFile(pq)
            length = parquet_file.metadata.num_rows
            filter_task_ids = None

            if task_id_set is not None and "task_index" in parquet_file.schema_arrow.names:
                task_index = pd.read_parquet(pq, columns=["task_index"])["task_index"]
                mask = task_index.isin(task_id_set)
                if not mask.any():
                    continue
                if not mask.all():
                    length = int(mask.sum())
                    filter_task_ids = tuple(task_id_set)

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

            episodes.append({
                "parquet_path": pq,
                "video_path": str(video_path),
                "length":     length,
                "task_ids":   filter_task_ids,
            })

        if not episodes:
            raise RuntimeError("No valid episodes found — check task_ids and video paths.")
        return episodes

    def _build_offsets(self):
        offsets = [0]
        for ep in self._episodes:
            offsets.append(offsets[-1] + max(0, ep["length"] - self.chunk_len))
        return offsets

    def _lookup_index(self, idx):
        if idx < 0:
            idx += self._num_samples
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)

        ep_idx = bisect_right(self._offsets, idx) - 1
        t = idx - self._offsets[ep_idx]
        return ep_idx, t

    def _load_episode_arrays(self, ep_idx):
        if ep_idx in self._episode_cache:
            self._episode_cache.move_to_end(ep_idx)
            return self._episode_cache[ep_idx]

        ep = self._episodes[ep_idx]
        columns = ["observation.state", "action"]
        if ep.get("task_ids") is not None:
            columns.append("task_index")
        df = pd.read_parquet(ep["parquet_path"], columns=columns)
        if ep.get("task_ids") is not None:
            df = df[df["task_index"].isin(ep["task_ids"])]
        states = np.stack(df["observation.state"].values).astype(np.float32)
        actions = np.stack(df["action"].values).astype(np.float32)

        if self.episode_cache_size > 0:
            self._episode_cache[ep_idx] = (states, actions)
            self._episode_cache.move_to_end(ep_idx)
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)

        return states, actions

    # ------------------------------------------------------------------
    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        ep_idx, t = self._lookup_index(idx)
        ep = self._episodes[ep_idx]
        tH = t + self.chunk_len

        try:
            vpath = ep["video_path"]
            if vpath not in self._vr_cache:
                vr = decord.VideoReader(vpath, ctx=decord.cpu(0))
                if self.video_cache_size > 0:
                    self._vr_cache[vpath] = vr
                    while len(self._vr_cache) > self.video_cache_size:
                        self._vr_cache.popitem(last=False)
            else:
                self._vr_cache.move_to_end(vpath)
                vr = self._vr_cache[vpath]
            tH_clamped = min(tH, len(vr) - 1)
            frames = vr.get_batch([t, tH_clamped]).asnumpy()
        except Exception:
            dummy  = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frames = np.stack([dummy, dummy])

        frame_t  = self.transform(frames[0])
        frame_tH = self.transform(frames[1])

        states, actions = self._load_episode_arrays(ep_idx)
        action_chunk = torch.from_numpy(actions[t : t + self.chunk_len])  # [32, 23]
        state_t      = torch.from_numpy(states[t])                         # [256]

        return frame_t, action_chunk, state_t, frame_tH
