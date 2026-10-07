"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    deterministic=True additionally asks PyTorch for deterministic kernels
    (slower; some ops then raise if no deterministic implementation exists).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    """The best available device: CUDA if present, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
