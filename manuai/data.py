"""Datasets, sampling, augmentation and collate functions.

Everything here is module-level (picklable), so DataLoader workers work on macOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset, WeightedRandomSampler

from . import features as F

VIT_SIZE = 224


@dataclass(frozen=True)
class AugmentConfig:
    freq_masks: int = 2
    freq_mask_max: int = 16    # of 128 mel bins — birds are narrow-band, keep masks small
    time_masks: int = 2
    time_mask_max: int = 40    # of 498 frames (0.4 s)
    roll_prob: float = 0.5
    roll_max: int = 50         # circular time shift, frames
    mask_value: float = 0.0    # ≈ dataset mean after AST normalisation


def read_rows(feat_dir: Path, split: str) -> pd.DataFrame:
    rows = pd.read_csv(feat_dir / f"{split}_segments.csv")
    return rows[~rows["bad"]].reset_index(drop=True)


def spread_subset(rows: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Up to `per_class` rows per class, spread over as many recordings as possible."""
    parts = []
    for _, g in rows.groupby("label_id"):
        g = g.sample(frac=1.0, random_state=seed)
        g = g.assign(_r=g.groupby("recording_id").cumcount()).sort_values("_r", kind="stable")
        parts.append(g.head(per_class).drop(columns="_r"))
    return pd.concat(parts).sort_values("seg").reset_index(drop=True)


class SegmentDataset(Dataset):
    """Reads (498, 128) float16 features from the memmap cache."""

    def __init__(self, feat_dir: Path, split: str, rows: pd.DataFrame,
                 training: bool = False, augment: AugmentConfig | None = None):
        self.path = Path(feat_dir) / f"{split}.npy"
        self.segs = rows["seg"].to_numpy()
        self.labels = rows["label_id"].to_numpy().astype(np.int64)
        self.recording_ids = rows["recording_id"].to_numpy()
        self.training = training
        self.augment = augment or AugmentConfig()
        self._arr = None

    def __len__(self) -> int:
        return len(self.segs)

    def _array(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        return self._arr

    def __getitem__(self, i: int) -> dict:
        x = torch.from_numpy(np.asarray(self._array()[self.segs[i]], dtype=np.float32))
        if self.training:
            x = augment(x, self.augment)
        return {"x": x, "label": int(self.labels[i])}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arr"] = None  # never pickle the memmap into workers
        return state


def augment(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """Time roll + SpecAugment on a (T, F) tensor. Uses torch RNG (seeded per worker)."""
    x = x.clone()
    t, f = x.shape
    if cfg.roll_prob and torch.rand(()) < cfg.roll_prob:
        x = torch.roll(x, int(torch.randint(-cfg.roll_max, cfg.roll_max + 1, ())), dims=0)
    for _ in range(cfg.freq_masks):
        w = int(torch.randint(0, cfg.freq_mask_max + 1, ()))
        if w:
            s = int(torch.randint(0, f - w + 1, ()))
            x[:, s:s + w] = cfg.mask_value
    for _ in range(cfg.time_masks):
        w = int(torch.randint(0, cfg.time_mask_max + 1, ()))
        if w:
            s = int(torch.randint(0, t - w + 1, ()))
            x[s:s + w, :] = cfg.mask_value
    return x


class Collate:
    """Batch → model inputs. mode='ast' pads to `frames`; mode='vit' makes 224² images."""

    def __init__(self, mode: str, frames: int = 512, vit_mean: float = 0.0, vit_std: float = 0.5):
        if mode not in ("ast", "vit"):
            raise ValueError(mode)
        self.mode = mode
        self.frames = frames
        self.vit_mean = vit_mean   # dataset statistics of the cached features (meta.json)
        self.vit_std = vit_std

    def __call__(self, batch: list[dict]) -> dict:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        if self.mode == "ast":
            x = torch.stack([F.pad_frames(b["x"], self.frames) for b in batch])
            return {"input_values": x, "labels": labels}
        x = torch.stack([b["x"] for b in batch])            # (B, T, F)
        img = x.transpose(1, 2).flip(1).unsqueeze(1)          # (B, 1, F, T), low freq at bottom
        img = TF.interpolate(img, size=(VIT_SIZE, VIT_SIZE), mode="area")
        img = ((img - self.vit_mean) / self.vit_std).repeat(1, 3, 1, 1)  # fixed global standardisation
        return {"pixel_values": img, "labels": labels}


def balanced_sampler(rows: pd.DataFrame, num_samples: int, alpha: float = 0.5,
                     per_recording_cap: int = 20, seed: int = 21) -> WeightedRandomSampler:
    """Sample classes ∝ n_c^alpha (alpha=0.5 → square-root), and within a class
    down-weight recordings that contribute more than `per_recording_cap` segments."""
    labels = rows["label_id"].to_numpy()
    rec_n = rows.groupby("recording_id")["seg"].transform("size").to_numpy()
    v = np.minimum(1.0, per_recording_cap / rec_n)
    w = np.zeros(len(rows), dtype=np.float64)
    for c in np.unique(labels):
        m = labels == c
        n_c = m.sum()
        w[m] = (n_c ** alpha) * v[m] / v[m].sum()
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=num_samples, replacement=True, generator=gen)


def class_sampling_share(rows: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    n = rows.groupby("label")["seg"].size()
    s = n ** alpha
    return (s / s.sum()).sort_values(ascending=False)
