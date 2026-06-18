"""Bootstrap confidence interval helpers for aggregate metrics."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from evaluation.metrics import compute_binary_metrics


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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UndefinedMetricWarning, module="sklearn"
                )
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


def _metric_value(y_true, p_pred, metric_name: str, *, threshold: float | None = None) -> float:
    if metric_name == "auc_roc":
        if np.unique(y_true).size < 2:
            raise ValueError("AUC requires both classes")
        return float(roc_auc_score(y_true, p_pred))
    if metric_name == "average_precision":
        if np.unique(y_true).size < 2:
            raise ValueError("Average precision requires both classes")
        return float(average_precision_score(y_true, p_pred))
    if metric_name == "brier_score":
        return float(brier_score_loss(y_true, p_pred))
    if metric_name in {
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "npv",
    }:
        if threshold is None:
            raise ValueError(f"{metric_name} requires a fixed threshold")
        return float(compute_binary_metrics(y_true, p_pred, threshold=threshold)[metric_name])
    raise ValueError(f"Unsupported bootstrap metric: {metric_name}")


def bootstrap_binary_metrics_ci(
    y_true,
    p_pred,
    *,
    threshold: float,
    metrics: tuple[str, ...] = (
        "auc_roc",
        "average_precision",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "npv",
    ),
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, dict]:
    """Bootstrap CIs for discrimination, calibration, and fixed-threshold metrics."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if len(y) == 0:
        raise ValueError("bootstrap requires at least one row")

    results = {}
    for offset, metric_name in enumerate(metrics):
        results[metric_name] = bootstrap_metric_ci(
            y,
            p,
            metric_fn=lambda yy, pp, name=metric_name: _metric_value(
                yy,
                pp,
                name,
                threshold=threshold,
            ),
            n_boot=n_boot,
            seed=seed + offset,
            confidence=confidence,
        )
    return results


def paired_bootstrap_metric_difference(
    y_true,
    p_model_a,
    p_model_b,
    *,
    metric_name: str,
    model_a_name: str,
    model_b_name: str,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap CI for metric(model_a) - metric(model_b).

    The same sampled row indices are used for both models. Samples that cannot
    evaluate the requested metric, such as one-class AUC samples, are skipped.
    """
    y = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_model_a).astype(float)
    p_b = np.asarray(p_model_b).astype(float)
    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError("y_true and both prediction arrays must have the same length")
    if len(y) == 0:
        raise ValueError("paired bootstrap requires at least one row")

    estimate_a = _metric_value(y, p_a, metric_name, threshold=threshold_a)
    estimate_b = _metric_value(y, p_b, metric_name, threshold=threshold_b)
    observed_diff = estimate_a - estimate_b

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            value_a = _metric_value(y[idx], p_a[idx], metric_name, threshold=threshold_a)
            value_b = _metric_value(y[idx], p_b[idx], metric_name, threshold=threshold_b)
        except ValueError:
            continue
        diff = value_a - value_b
        if math.isfinite(diff):
            diffs.append(float(diff))

    if diffs:
        alpha = (1.0 - confidence) / 2.0
        ci_lower, ci_upper = np.quantile(diffs, [alpha, 1.0 - alpha])
        p_value_two_sided = min(
            1.0,
            2.0 * min(
                float(np.mean(np.asarray(diffs) <= 0.0)),
                float(np.mean(np.asarray(diffs) >= 0.0)),
            ),
        )
    else:
        ci_lower = math.nan
        ci_upper = math.nan
        p_value_two_sided = math.nan

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "metric": metric_name,
        "estimate_a": float(estimate_a),
        "estimate_b": float(estimate_b),
        "difference": float(observed_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value_two_sided": float(p_value_two_sided),
        "n_boot": int(n_boot),
        "n_success": int(len(diffs)),
        "confidence": float(confidence),
    }
