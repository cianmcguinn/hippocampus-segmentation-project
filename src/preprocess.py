# src/preprocess.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, ToPILImage
import random

# ----------------------------
# Crop size (ROI-driven)
# ----------------------------
# src/preprocess.py

def compute_square_crop_size(masks: list[np.ndarray], margin: int = 10) -> int:
    """
    ORIGINAL (procedural notebook) behavior:
    Use the maximum image-plane size across volumes, then add 2*margin.
    This ignores the ROI bbox size.
    """
    if not masks:
        raise ValueError("No masks provided")
    Hmax = max(m.shape[0] for m in masks)
    Wmax = max(m.shape[1] for m in masks)
    return int(max(Hmax, Wmax) + 2 * margin)


# ----------------------------
# N4 + normalisation
# ----------------------------
def bias_field_correction(vol: np.ndarray) -> np.ndarray:
    sitk_vol = sitk.GetImageFromArray(vol.astype(np.float32))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_vol)
    return sitk.GetArrayFromImage(corrected)

def zscore(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = vol.mean(); sd = vol.std()
    return (vol - mu) / (sd + eps)

def correct_and_normalize(
    vols: List[np.ndarray],
    basenames: List[str],
    out_dir: str | Path,
    cache: bool = True
) -> List[np.ndarray]:
    """
    Apply N4 bias-field correction + z-score normalisation per volume.
    Cache corrected volumes to `out_dir` when `cache=True`.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    processed: List[np.ndarray] = []

    for idx, (vol, name) in enumerate(zip(vols, basenames), start=1):
        base = name.replace(".nii.gz", "")
        out_path = out_dir / f"{base}_n4.nii.gz"
        if cache and out_path.exists():
            print(f"[{idx}/{len(vols)}] Loading cached: {out_path.name}")
            v_n4 = nib.load(out_path).get_fdata()
        else:
            print(f"[{idx}/{len(vols)}] N4 correcting: {name}")
            v_n4 = bias_field_correction(vol)
            nib.save(nib.Nifti1Image(v_n4.astype(np.float32), affine=np.eye(4)), out_path)
            print(f"Saved corrected volume → {out_path.name}")
        processed.append(zscore(v_n4))
    return processed

# ----------------------------
# Augmentation (image+mask)
# ----------------------------
def joint_transform(img, msk, aug_prob: float = 0.5, angle_range = (-10, 10)):
    """
    Apply identical random flip/rotation to a PIL image and its mask.
    """
    if random.random() < aug_prob:
        if random.random() < 0.5:
            img = F.hflip(img); msk = F.hflip(msk)
        angle = random.uniform(*angle_range)
        img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        msk = F.rotate(msk, angle, interpolation=InterpolationMode.NEAREST)
    return img, msk

# ----------------------------
# 2D Slice Dataset
# ----------------------------
class SliceDataset(Dataset):
    """
    2D slice dataset with ROI-aware cropping + optional joint augmentation + resizing.
    Returns:
      img_t: torch.float32 tensor [1, H, W] in [0,1]
      msk_t: torch.int64  tensor [H, W] (class indices)
    """
    def __init__(self,
                 vols: List[np.ndarray],
                 masks: List[np.ndarray],
                 crop_size: int,
                 aug_prob: float = 0.5,
                 resize: Tuple[int, int] = (128, 128),
                 center_on_empty: bool = True):
        assert len(vols) == len(masks), "vols/masks length mismatch"
        self.vols = vols
        self.masks = masks
        self.crop = crop_size
        self.margin = 10
        self.aug_prob = aug_prob
        self.resize = resize
        self.center_on_empty = center_on_empty

        H, W, D = vols[0].shape
        self.H, self.W = H, W
        self.slice_counts = [m.shape[2] for m in self.masks]
        self.total = sum(self.slice_counts)

        # in SliceDataset.__init__ (after computing self.slice_counts, self.total)
        self.slice_to_vol = []
        for v, Dz in enumerate(self.slice_counts):
            self.slice_to_vol.extend([v] * Dz)


    def __len__(self):
        return self.total

    def _vol_and_local_idx(self, global_idx: int):
        v = 0; idx = global_idx
        while idx >= self.slice_counts[v]:
            idx -= self.slice_counts[v]; v += 1
        return v, idx

    def __getitem__(self, idx: int):
        v, z = self._vol_and_local_idx(idx)
        img = self.vols[v][:, :, z]
        msk = self.masks[v][:, :, z]

        # ROI-aware crop; fallback to center crop if slice is empty
        ys, xs = np.where(msk > 0)
        if ys.size:
            y0 = max(0, ys.min() - self.margin); y1 = min(self.H, ys.max() + self.margin)
            x0 = max(0, xs.min() - self.margin); x1 = min(self.W, xs.max() + self.margin)
        else:
            y0 = (self.H - self.crop) // 2; x0 = (self.W - self.crop) // 2
            y1, x1 = y0 + self.crop, x0 + self.crop

        crop_img = img[y0:y1, x0:x1]
        crop_msk = msk[y0:y1, x0:x1]

        # To PIL for geometric ops
        pil_img = ToPILImage()(crop_img)
        pil_msk = ToPILImage()(crop_msk.astype(np.uint8))

        # Joint augment
        pil_img, pil_msk = joint_transform(pil_img, pil_msk, self.aug_prob)

        # Resize + to tensors
        pil_img = F.resize(pil_img, self.resize, interpolation=InterpolationMode.BILINEAR)
        pil_msk = F.resize(pil_msk, self.resize, interpolation=InterpolationMode.NEAREST)
        img_t = F.to_tensor(pil_img)                       # (1,H,W) float32 [0,1]
        msk_t = torch.from_numpy(np.array(pil_msk)).long() # (H,W) int64

        return img_t, msk_t

