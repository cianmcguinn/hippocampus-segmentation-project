# src/train.py
from __future__ import annotations
from typing import Dict, List, Tuple
import json, os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold, ParameterSampler

import segmentation_models_pytorch as smp

from .utils import seed_worker_factory
from .class_imbalance import make_sampler_from_weights

import segmentation_models_pytorch as smp
import torch.nn as nn
from src.config import CFG


# ---------- helpers ----------

def indices_for_volumes(slice_to_vol: List[int], allowed_vols: List[int]) -> List[int]:
    """Return the global slice indices whose volume id is in allowed_vols."""
    allowed = set(allowed_vols)
    return [i for i, v in enumerate(slice_to_vol) if v in allowed]

def make_model(num_classes: int | None = None) -> nn.Module:
    return smp.Unet(
        encoder_name=CFG.ENCODER_NAME,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=CFG.IN_CHANNELS,
        classes=num_classes or CFG.NUM_CLASSES,
    )

# ---------- train / eval ----------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    loss_fn,
                    device: torch.device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_loss(model: nn.Module,
              loader: DataLoader,
              loss_fn,
              device: torch.device) -> float:
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

# ---------- cross-validation search ----------

def cv_search(dataset,
              slice_to_vol: List[int],
              train_vols: List[int],
              weights_per_slice: List[float],
              class_weights: torch.Tensor,
              device: torch.device,
              seed: int,
              param_dist: Dict,
              n_folds: int,
              n_iter: int,
              max_epochs: int,
              deterministic: bool = False,
              verbose: int = 1  # 0=silent, 1=trial summaries, 2=fold-level
              ) -> Tuple[Dict, float]:
    """
    Returns (best_cfg, best_score). Score is -val_loss averaged across folds (larger is better).
    verbose: 0=silent, 1=trial summaries only, 2=include per-fold lines.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_score, best_cfg = float('-inf'), None

    # deterministic plumbing for DataLoader workers
    num_w = 0 if deterministic else 2
    worker_init = seed_worker_factory(seed) if deterministic else None
    PIN_MEM = torch.cuda.is_available()  # avoid CPU pin_memory warning

    for i, cfg in enumerate(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed), 1):
        if verbose >= 1:
            print(f"=== Trial {i}/{n_iter}: {cfg} ===")
        fold_scores = []

        for fold, (ti, vi) in enumerate(kf.split(train_vols), 1):
            trv = [train_vols[j] for j in ti]
            vlv = [train_vols[j] for j in vi]

            train_idxs = indices_for_volumes(slice_to_vol, trv)
            valid_idxs = indices_for_volumes(slice_to_vol, vlv)

            # sampler for class imbalance
            train_weights = [weights_per_slice[i] for i in train_idxs]
            sampler = make_sampler_from_weights(train_weights, seed)

            gen = torch.Generator().manual_seed(seed) if deterministic else None

            trL = DataLoader(
                Subset(dataset, train_idxs),
                batch_size=cfg['batch_size'],
                sampler=sampler,
                num_workers=num_w,
                worker_init_fn=worker_init,
                generator=gen,
                pin_memory=PIN_MEM,
            )
            vlL = DataLoader(
                Subset(dataset, valid_idxs),
                batch_size=cfg['batch_size'],
                shuffle=False,
                num_workers=num_w,
                worker_init_fn=worker_init,
                pin_memory=PIN_MEM,
            )

            model = make_model(num_classes=3).to(device)
            opt = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

            ce   = nn.CrossEntropyLoss(weight=class_weights)
            dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
            loss_fn = lambda p, t: ce(p, t) + dice(p, t)

            best_vl = float('inf'); best_state = None; patience = 0
            for ep in range(1, max_epochs + 1):
                _ = train_one_epoch(model, trL, opt, loss_fn, device)
                vl = eval_loss(model, vlL, loss_fn, device)
                if vl < best_vl:
                    best_vl, patience, best_state = vl, 0, model.state_dict()
                else:
                    patience += 1
                    if patience >= 3:
                        break
                sch.step()

            model.load_state_dict(best_state)
            fold_score = -best_vl
            fold_scores.append(fold_score)
            if verbose >= 2:
                print(f"Fold {fold} score: {fold_score:.4f}")

        mean_score = float(np.mean(fold_scores))
        if verbose >= 1:
            print(f"Mean CV score: {mean_score:.4f}")
        if mean_score > best_score:
            best_score, best_cfg = mean_score, cfg

    return best_cfg, best_score



# ---------- final fit on train + val loss tracking ----------

def fit_final(dataset,
              slice_to_vol: List[int],
              train_vols: List[int],
              final_val_vols: List[int],
              batch_size: int,
              weights_per_slice: List[float],
              class_weights: torch.Tensor,
              device: torch.device,
              seed: int,
              lr: float,
              weight_decay: float,
              max_epochs: int,
              results_dir: str,
              deterministic: bool = False,
              verbose: int = 1  # 0=silent, 1=epoch loss only, 2=also show checkpoint msgs
              ) -> Tuple[nn.Module, List[float], List[float], float, str]:
    """
    Train final model and return (model, train_losses, val_losses, best_val, best_ckpt_path).

    verbose:
        0 -> no prints
        1 -> 'Epoch N, Train Loss: ..., Val Loss: ...'
        2 -> epoch line + checkpoint save messages
    """
    PIN_MEM = torch.cuda.is_available()

    train_idxs = indices_for_volumes(slice_to_vol, train_vols)
    val_idxs   = indices_for_volumes(slice_to_vol, final_val_vols)

    sw = [weights_per_slice[i] for i in train_idxs]
    sampler = make_sampler_from_weights(sw, seed)

    num_w = 0 if deterministic else 2
    worker_init = seed_worker_factory(seed) if deterministic else None
    gen = torch.Generator().manual_seed(seed) if deterministic else None

    trL = DataLoader(
        Subset(dataset, train_idxs),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_w,
        worker_init_fn=worker_init,
        generator=gen,
        pin_memory=PIN_MEM,
    )
    vlL = DataLoader(
        Subset(dataset, val_idxs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_w,
        worker_init_fn=worker_init,
        pin_memory=PIN_MEM,
    )

    model = make_model(num_classes=3).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    ce   = nn.CrossEntropyLoss(weight=class_weights)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    loss_fn = lambda p, t: ce(p, t) + dice(p, t)

    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val = float('inf'); best_state = None

    for ep in range(1, max_epochs + 1):
        tr = train_one_epoch(model, trL, opt, loss_fn, device)
        vl = eval_loss(model, vlL, loss_fn, device)
        train_losses.append(tr); val_losses.append(vl)
        sch.step()

        if verbose >= 1:
            print(f"Epoch {ep}, Train Loss: {tr:.4f}, Val Loss: {vl:.4f}")

        # save epoch checkpoint (quiet unless verbose >= 2)
        ckpt = os.path.join(results_dir, f"baseline_unet_epoch_{ep}.pth")
        torch.save(model.state_dict(), ckpt)
        if verbose >= 2:
            print(f"Saved checkpoint: {ckpt}")

        if vl < best_val:
            best_val, best_state = vl, model.state_dict().copy()

    final_ckpt = os.path.join(results_dir, "baseline_unet_best.pth")
    torch.save(best_state, final_ckpt)
    if verbose >= 2:
        print(f"Saved best‐epoch model (val_loss={best_val:.4f}) to {final_ckpt}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_val, final_ckpt


