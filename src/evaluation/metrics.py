"""Reusable leakage-safe binary classification metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def _as_1d_array(values: Iterable, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Compute discrimination, probability, and threshold metrics.

    Threshold selection should happen before this function is called. This
    function only applies the provided threshold to the supplied split.
    """
    y_true_arr = _as_1d_array(y_true, "y_true").astype(int)
    y_score_arr = _as_1d_array(y_score, "y_score").astype(float)

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("y_true and y_score must have the same length")
    if len(y_true_arr) == 0:
        raise ValueError("metrics require at least one row")

    y_pred = (y_score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = float(tp / precision_denominator) if precision_denominator else 0.0
    recall = float(tp / recall_denominator) if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = float(2 * precision * recall / f1_denominator) if f1_denominator else 0.0

    return {
        "auc_roc": _safe_auc_roc(y_true_arr, y_score_arr),
        "average_precision": _safe_average_precision(y_true_arr, y_score_arr),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n": int(len(y_true_arr)),
        "positive_rate": float(np.mean(y_true_arr)),
    }


def binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Backward-compatible wrapper used by legacy model classes."""
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    return {
        **metrics,
        "f1_score": metrics["f1"],
        # Historical code used `auc_pr`; the canonical metric is
        # average precision, which is more stable for imbalanced outcomes.
        "auc_pr": metrics["average_precision"],
        "confusion_matrix": [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
    }
