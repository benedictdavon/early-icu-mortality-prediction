"""Reproducible random seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Seed Python, NumPy, and PyTorch if PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def backend_seed_params(seed: int = 42) -> dict[str, dict[str, int]]:
    """Return deterministic seed parameters for supported model backends."""
    return {
        "sklearn": {"random_state": seed},
        "xgboost": {"random_state": seed},
        "lightgbm": {"random_state": seed},
        "catboost": {"random_seed": seed},
        "torch": {"seed": seed},
    }


def apply_backend_seed(
    params: dict | None,
    backend: str,
    seed: int = 42,
) -> dict:
    """Copy params and add the backend-specific seed field when known."""
    seeded = dict(params or {})
    seeded.update(backend_seed_params(seed).get(backend.lower(), {}))
    return seeded
