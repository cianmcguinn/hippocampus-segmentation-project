# src/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def _ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def plot_slice_hist(shapes, save_path, show: bool = True):
    _ensure_dir(save_path)
    plt.figure()
    plt.hist(shapes[:, 2], bins=20)
    plt.title('Slice Counts per Volume')
    plt.xlabel('Number of Slices'); plt.ylabel('Frequency')
    plt.savefig(save_path)
    if show: plt.show()
    else: plt.close()

def plot_roi_percent_hist(percent_roi, save_path, show: bool = True,
                          subtitle: str | None = None,
                          caption: str | None = None):
    _ensure_dir(save_path)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(percent_roi, bins=50, density=True)
    ax.set_title('Hippocampus ROI % per Slice')
    ax.set_xlabel('ROI %'); ax.set_ylabel('Density')

    # ① draw subtitle inside, top-right, with a white background
    if subtitle:
        ax.text(0.98, 0.98, subtitle,
                transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    # ② optional caption below the figure (outside axes)
    if caption:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=130)
    if show: plt.show()
    plt.close(fig)


def plot_sample_overlays(vols, masks, percent_roi, shapes, save_path, show: bool = True):
    _ensure_dir(save_path)
    thresholds = [0, np.median(percent_roi), np.percentile(percent_roi, 90)]
    labels = ['No ROI', 'Median ROI', 'High ROI']
    indices = [np.where(percent_roi >= t)[0][0] for t in thresholds]

    fig, axes = plt.subplots(3, 2, figsize=(6, 12))
    for ax_row, label, idx in zip(axes, labels, indices):
        vol_idx, slice_idx = divmod(idx, shapes[0, 2])
        img = vols[vol_idx][:, :, slice_idx]
        mask = masks[vol_idx][:, :, slice_idx]
        ax_row[0].imshow(img, cmap='gray'); ax_row[0].set_title(f'{label} Image'); ax_row[0].axis('off')
        ax_row[1].imshow(img, cmap='gray'); ax_row[1].imshow(mask, alpha=0.3, cmap='Reds')
        ax_row[1].set_title(f'{label} Overlay'); ax_row[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    if show: plt.show()
    else: plt.close()


def plot_confusion(cm_norm, save_path, title="Confusion Matrix", show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    classes = ["BG", "Ant", "Post"]
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)

    # annotate
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=130)
    if show: plt.show()
    plt.close(fig)


def plot_best_worst_overlays(
    vol_preds: dict,
    vols: list[np.ndarray],
    results_dir: str,
    palette: dict = {'ant': (1, 0.60, 0.20, 0.35), 'post': (0.86, 0.23, 0.23, 0.35)},
    *,
    k: int = 1,                 # show top-k worst and top-k best (k=1 => 2x2 grid)
    slice_mode: str = "roi_max",# 'roi_max' (recommended) or 'mid'
    figsize=(10, 10),
    save_name: str = "worst_best_overlay.png",
    show: bool = True,
    verbose: bool = False
) -> dict:
    """
    Create a figure with worst and best overlays (True vs Pred) per volume.

    Returns:
        {'worst': [(vi, dice), ...], 'best': [(vi, dice), ...], 'path': save_path}
    """

    def dice_from_entries(rows):
        pred = np.stack([r[1] for r in rows], axis=2)
        true = np.stack([r[2] for r in rows], axis=2)
        tp = np.logical_and(pred > 0, true > 0).sum()
        fp = np.logical_and(pred > 0, true == 0).sum()
        fn = np.logical_and(pred == 0, true > 0).sum()
        return (2 * tp) / (2 * tp + fp + fn + 1e-8)

    def pick_slice(rows, mode: str) -> int:
        rows = sorted(rows, key=lambda x: x[0])
        if mode == "mid":
            return rows[len(rows)//2][0]
        # roi_max: choose the slice with largest GT ROI area
        best_z, best_area = rows[0][0], -1
        for z, _, y in rows:
            area = (y > 0).sum()
            if area > best_area:
                best_z, best_area = z, area
        return best_z

    def extract_slice(vol_idx, mode: str):
        rows = sorted(vol_preds[vol_idx], key=lambda x: x[0])
        z = pick_slice(rows, mode)
        pred = next(r[1] for r in rows if r[0] == z)   # HxW (e.g., 128x128)
        true = next(r[2] for r in rows if r[0] == z)

        # normalize MRI to 0..1 and RESIZE to mask resolution for proper overlay
        mri = vols[vol_idx][:, :, z].astype(np.float32)
        mri = (mri - mri.min()) / (np.ptp(mri) + 1e-8)  # NumPy 2.0-safe
        Hm, Wm = true.shape
        mri = np.array(
            Image.fromarray((mri * 255).astype(np.uint8)).resize((Wm, Hm), resample=Image.BILINEAR)
        ) / 255.0

        return mri, true, pred

    def make_rgba(mask):
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[mask == 1] = palette['ant']
        rgba[mask == 2] = palette['post']
        return rgba

    # --- select volumes
    scores = {v: dice_from_entries(rows) for v, rows in vol_preds.items()}
    order = sorted(scores, key=scores.get)  # ascending by dice
    worst = order[:k]
    best  = order[-k:][::-1]

    # --- figure
    rows_n = 2 * k
    fig, axes = plt.subplots(rows_n, 2, figsize=figsize)
    axes = np.atleast_2d(axes)

    def render_row(axL, axR, vi, tag):
        mri, true, pred = extract_slice(vi, slice_mode)
        axL.imshow(mri, cmap='gray', interpolation='nearest')
        axL.imshow(make_rgba(true), interpolation='nearest')
        axL.set_title(f"{tag} {vi} True  (Dice={scores[vi]:.3f})", fontsize=10)
        axL.axis('off')

        axR.imshow(mri, cmap='gray', interpolation='nearest')
        axR.imshow(make_rgba(pred), interpolation='nearest')
        axR.set_title(f"{tag} {vi} Pred  (Dice={scores[vi]:.3f})", fontsize=10)
        axR.axis('off')

    # worst on top, best below
    for i, vi in enumerate(worst):
        render_row(axes[i, 0], axes[i, 1], vi, "Worst")
    for i, vi in enumerate(best):
        render_row(axes[k + i, 0], axes[k + i, 1], vi, "Best ")

    legend_patches = [
        mpatches.Patch(color=palette['ant'],  label='Anterior'),
        mpatches.Patch(color=palette['post'], label='Posterior')
    ]
    # Place legend outside the axes to avoid overlapping the title
    # legend (put BELOW the grid)
    fig.legend(
        handles=legend_patches,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02)  # below the axes
    )
    fig.suptitle("Best & Worst Predictions (overlays on MRI)", y=0.99, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # leave a bit more space for the bottom legend

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    if verbose:
        print(f"Saved overlays to {save_path}")

    return {
        "worst": [(vi, float(scores[vi])) for vi in worst],
        "best":  [(vi, float(scores[vi])) for vi in best],
        "path":  save_path
    }


