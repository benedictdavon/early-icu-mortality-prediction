"""Evaluation helpers for ICU6H-MAFNet."""

from __future__ import annotations

import numpy as np
import torch

from evaluation.calibration import logits_to_probabilities
from training.train_mafnet import predict_logits


@torch.no_grad()
def predict_mafnet_logits(
    model,
    loader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and mortality logits for a loader."""
    return predict_logits(model, loader, device=device)


@torch.no_grad()
def predict_mafnet_probabilities(
    model,
    loader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and probabilities for a loader without writing row predictions."""
    labels, logits = predict_mafnet_logits(model, loader, device=device)
    return labels, logits_to_probabilities(logits)


__all__ = ["predict_mafnet_logits", "predict_mafnet_probabilities"]
