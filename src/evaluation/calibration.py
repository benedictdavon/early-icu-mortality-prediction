"""Calibration summaries for binary risk predictions."""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    """Compute fixed-width expected calibration error."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if len(y) == 0:
        raise ValueError("calibration metrics require at least one row")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(p[mask]))
        bin_observed = float(np.mean(y[mask]))
        ece += (np.sum(mask) / len(y)) * abs(bin_confidence - bin_observed)
    return float(ece)


def calibration_intercept_slope(y_true, p_pred, eps: float = 1e-6) -> dict:
    """Estimate calibration intercept and slope via logistic recalibration."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if np.unique(y).size < 2:
        return {"calibration_intercept": math.nan, "calibration_slope": math.nan}

    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def calibration_summary(y_true, p_pred, n_bins: int = 10) -> dict:
    """Return calibration-ready aggregate metrics without row-level output."""
    summary = calibration_intercept_slope(y_true, p_pred)
    summary.update(
        {
            "brier_score": float(brier_score_loss(y_true, p_pred)),
            "expected_calibration_error": expected_calibration_error(
                y_true, p_pred, n_bins=n_bins
            ),
            "calibration_bins": int(n_bins),
        }
    )
    return summary
