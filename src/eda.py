import numpy as np
import pandas as pd

def compute_shapes_stats(vols):
    """Return (shapes ndarray, stats DataFrame)."""
    shapes = np.array([v.shape for v in vols])
    stats = pd.DataFrame(shapes, columns=['Height','Width','Slices']).agg(['min','median','max'])
    return shapes, stats

def compute_percent_roi(masks, shapes):
    """Return (percent_roi array, empty_count)."""
    area = shapes[0, 0] * shapes[0, 1]
    roi_counts = [(m[:, :, s] > 0).sum() for m in masks for s in range(m.shape[2])]
    percent_roi = np.array(roi_counts) / area * 100
    empty = (percent_roi == 0).sum()
    return percent_roi, empty
