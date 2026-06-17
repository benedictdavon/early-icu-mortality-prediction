"""Bootstrap confidence interval helpers for aggregate metrics."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def bootstrap_metric_ci(
    y_true,
    p_pred,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Estimate a percentile bootstrap confidence interval for one metric.

    Samples that cannot evaluate the metric, such as one-class AUC samples, are
    skipped rather than forcing invalid values into the interval.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if len(y) == 0:
        raise ValueError("bootstrap requires at least one row")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            value = float(metric_fn(y[idx], p[idx]))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)

    if not values:
        return {
            "mean": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "n_boot": int(n_boot),
            "n_success": 0,
            "confidence": float(confidence),
        }

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot": int(n_boot),
        "n_success": int(len(values)),
        "confidence": float(confidence),
    }
