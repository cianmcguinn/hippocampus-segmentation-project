# src/class_imbalance.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

# ---- slice-level flags & weights ----
def slice_roi_flags(masks: List[np.ndarray]) -> list[int]:
    """Return 1 for slices containing any ROI, else 0; order matches dataset slice order."""
    flags: list[int] = []
    for m in masks:
        D = m.shape[2]
        for z in range(D):
            flags.append(int((m[:, :, z] > 0).any()))
    return flags

def oversample_weights_from_flags(flags: list[int]) -> list[float]:
    """ROI slices get higher weight: bg weight=1.0, roi weight=(#bg/#roi)."""
    roi = sum(flags)
    bg  = len(flags) - roi
    if roi == 0:
        return [1.0] * len(flags)
    pos_w = bg / roi
    return [pos_w if f == 1 else 1.0 for f in flags]

# ---- Weighted sampler builders (single source of truth) ----
def make_sampler_from_weights(
    weights: List[float],
    seed: Optional[int] = None,
    replacement: bool = True
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from per-sample weights, optionally seeded."""
    w = torch.tensor(weights, dtype=torch.float)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(w, num_samples=len(w), replacement=replacement, generator=gen)

def make_sampler_from_flags(
    flags: List[int],
    seed: Optional[int] = None,
    replacement: bool = True
) -> WeightedRandomSampler:
    """Convenience: compute weights from flags, then build sampler."""
    return make_sampler_from_weights(oversample_weights_from_flags(flags), seed, replacement)

# ---- voxel-level CE class weights ----
def compute_ce_class_weights_from_masks(
    masks: List[np.ndarray],
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean=num_classes."""
    vox = torch.cat([torch.from_numpy(m.ravel()).long() for m in masks]).to(device)
    counts = torch.bincount(vox, minlength=num_classes).float()
    total  = counts.sum()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = (total / (num_classes * counts)).to(device)
    weights = weights / weights.mean() * num_classes
    return weights

