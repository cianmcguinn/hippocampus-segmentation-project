# src/data_io.py
import os
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple

def load_nii_volumes(images_dir: str | Path, labels_dir: str | Path
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load NIfTI image and label volumes from parallel directories and
    return (images, labels, basenames). Basenames include the '.nii.gz' filename.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])

    images, labels, basenames = [], [], []
    for i, fname in enumerate(image_files, start=1):
        img = nib.load(images_dir / fname).get_fdata()
        lbl = nib.load(labels_dir / fname).get_fdata()
        if img.shape != lbl.shape:
            raise ValueError(f"Shape mismatch for {fname}: img {img.shape} vs lbl {lbl.shape}")
        images.append(img); labels.append(lbl); basenames.append(fname)
        print(f"Loaded {i}/{len(image_files)}: {fname}")
    return images, labels, basenames
