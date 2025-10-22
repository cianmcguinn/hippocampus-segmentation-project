# src/utils.py
import random, numpy as np, torch

def get_device():
    # Add MPS if you sometimes run on Apple Silicon locally
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def set_deterministic(deterministic: bool = True):
    # cuDNN settings for determinism (some ops may fall back / be slower)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

def seed_worker_factory(seed: int):
    # returns a worker_init_fn that deterministically seeds numpy+random+torch
    def _seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _seed_worker
