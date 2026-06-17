"""Threshold selection utilities.

Thresholds must be selected on validation data and then applied unchanged to
the held-out test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def threshold_table(y_true, y_score, thresholds=None, fn_cost: float = 2.0, fp_cost: float = 1.0):
    """Evaluate a list of thresholds against validation labels."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for threshold in thresholds:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "npv": tn / (tn + fn) if (tn + fn) else 0.0,
                "cost": (fp * fp_cost) + (fn * fn_cost),
            }
        )
    return pd.DataFrame(rows)


def select_optimal_threshold(y_true, y_score, strategy: str = "f1", **kwargs):
    """Select a threshold from validation data using a named strategy."""
    results = threshold_table(y_true, y_score, **kwargs)

    if strategy == "f1":
        best_idx = results["f1"].idxmax()
    elif strategy == "cost":
        best_idx = results["cost"].idxmin()
    elif strategy == "balanced":
        best_idx = (results["precision"] - results["recall"]).abs().idxmin()
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    best_row = results.loc[best_idx].to_dict()
    return float(best_row["threshold"]), best_row, results
