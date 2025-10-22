# src/infer.py
from __future__ import annotations
from typing import Dict, List, Tuple
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

def predict_slices(model: torch.nn.Module,
                   dataset,
                   device: torch.device,
                   batch_size: int = 4,
                   num_workers: int = 2,
                   deterministic: bool = False,
                   seed_worker = None,
                   generator = None):
    """
    Run the model over all slices in `dataset` (no shuffle), return
    a dict vol_id -> list[(z, pred_slice, true_slice)].
    """
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0 if deterministic else num_workers,
                    worker_init_fn=seed_worker if deterministic else None,
                    generator=generator if deterministic else None)

    # Build a cumulative depth index so we can map flat slice idx -> (volume, z)
    depths = [m.shape[2] for m in dataset.masks]
    cum = np.cumsum([0] + depths)  # len = n_vols+1

    def flat_to_vz(i: int) -> Tuple[int,int]:
        # find volume such that cum[v] <= i < cum[v+1]
        v = int(np.searchsorted(cum, i, side='right') - 1)
        z = i - cum[v]
        return v, z

    vol_preds: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {v: [] for v in range(len(depths))}
    flat_idx = 0
    model.eval()
    with torch.no_grad():
        for imgs, msks in dl:
            logits = model(imgs.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            ys    = msks.numpy()
            bs = imgs.size(0)
            for b in range(bs):
                v, z = flat_to_vz(flat_idx + b)
                vol_preds[v].append((z, preds[b], ys[b]))
            flat_idx += bs
    # sort by slice index per volume
    for v in vol_preds:
        vol_preds[v].sort(key=lambda t: t[0])
    return vol_preds

def dice_iou_from_vols(pred_vol: np.ndarray, true_vol: np.ndarray) -> Tuple[float,float]:
    tp = np.logical_and(pred_vol>0, true_vol>0).sum()
    fp = np.logical_and(pred_vol>0, true_vol==0).sum()
    fn = np.logical_and(pred_vol==0, true_vol>0).sum()
    dice = 2*tp / (2*tp + fp + fn + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    return float(dice), float(iou)

def summarise_vol_metrics(vol_preds: Dict[int, List[Tuple[int,np.ndarray,np.ndarray]]]) -> Tuple[List[float], List[float]]:
    dice_scores, iou_scores = [], []
    for v, rows in vol_preds.items():
        pred_vol = np.stack([r[1] for r in rows], axis=2)
        true_vol = np.stack([r[2] for r in rows], axis=2)
        d,i = dice_iou_from_vols(pred_vol, true_vol)
        dice_scores.append(d); iou_scores.append(i)
    return dice_scores, iou_scores

def confusion_from_vol_preds(vol_preds: Dict[int, List[Tuple[int,np.ndarray,np.ndarray]]], num_classes: int = 3):
    y_true = np.concatenate([np.stack([r[2] for r in rows], axis=2).ravel() for rows in vol_preds.values()])
    y_pred = np.concatenate([np.stack([r[1] for r in rows], axis=2).ravel() for rows in vol_preds.values()])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm, cm_norm

def save_test_summary(dice_scores: List[float], iou_scores: List[float], results_dir: str):
    dice_mean, dice_std = float(np.mean(dice_scores)), float(np.std(dice_scores))
    iou_mean,  iou_std  = float(np.mean(iou_scores)),  float(np.std(iou_scores))
    print(f"Average 3D Dice: {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"Average 3D IoU:  {iou_mean:.4f} ± {iou_std:.4f}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "test_summary.json"), "w") as f:
        json.dump({"dice_mean": dice_mean, "dice_std": dice_std, "iou_mean": iou_mean, "iou_std": iou_std}, f, indent=2)
