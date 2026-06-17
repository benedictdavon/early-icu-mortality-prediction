"""Validation-only threshold selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import compute_binary_metrics


DEFAULT_POLICIES = ("high_sensitivity", "balanced_f1", "high_precision")


def _candidate_thresholds(y_score, thresholds=None) -> np.ndarray:
    if thresholds is not None:
        candidates = np.asarray(thresholds, dtype=float)
    else:
        scores = np.asarray(y_score, dtype=float)
        candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    candidates = candidates[np.isfinite(candidates)]
    candidates = np.clip(candidates, 0.0, 1.0)
    return np.unique(candidates)


def threshold_table(
    y_true,
    y_score,
    thresholds=None,
    fn_cost: float = 2.0,
    fp_cost: float = 1.0,
):
    """Evaluate candidate thresholds against one labeled split.

    Use this on validation data for threshold selection. On test data, use it
    only for diagnostics after thresholds are already fixed.
    """
    rows = []
    for threshold in _candidate_thresholds(y_score, thresholds):
        metrics = compute_binary_metrics(y_true, y_score, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "specificity": metrics["specificity"],
                "npv": metrics["npv"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "cost": (metrics["fp"] * fp_cost) + (metrics["fn"] * fn_cost),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _row_to_policy(policy: str, row: pd.Series, rule: str, fallback_used: bool) -> dict:
    return {
        "policy": policy,
        "threshold": float(row["threshold"]),
        "rule": rule,
        "selected_on": "validation",
        "fallback_used": bool(fallback_used),
        "validation_metrics": {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "specificity": float(row["specificity"]),
            "npv": float(row["npv"]),
        },
    }


def _select_high_sensitivity(table: pd.DataFrame, target_recall: float) -> dict:
    rule = f"highest threshold with recall >= {target_recall:.2f}"
    eligible = table[table["recall"] >= target_recall]
    if not eligible.empty:
        row = eligible.sort_values("threshold", ascending=False).iloc[0]
        return _row_to_policy("high_sensitivity", row, rule, fallback_used=False)

    positive_rate = (table["tp"].iloc[0] + table["fn"].iloc[0]) / max(
        table[["tn", "fp", "fn", "tp"]].iloc[0].sum(), 1
    )
    fallback = table[table["precision"] > positive_rate]
    if fallback.empty:
        fallback = table
    row = fallback.sort_values(
        ["recall", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return _row_to_policy(
        "high_sensitivity",
        row,
        rule + "; fallback to max recall with precision above base rate if possible",
        fallback_used=True,
    )


def _select_balanced_f1(table: pd.DataFrame) -> dict:
    row = table.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    return _row_to_policy(
        "balanced_f1",
        row,
        "threshold maximizing validation F1",
        fallback_used=False,
    )


def _select_high_precision(
    table: pd.DataFrame,
    target_precision: float,
    fallback_min_recall: float,
) -> dict:
    rule = f"highest threshold with precision >= {target_precision:.2f}"
    eligible = table[(table["precision"] >= target_precision) & ((table["tp"] + table["fp"]) > 0)]
    if not eligible.empty:
        row = eligible.sort_values("threshold", ascending=False).iloc[0]
        return _row_to_policy("high_precision", row, rule, fallback_used=False)

    fallback = table[table["recall"] >= fallback_min_recall]
    if fallback.empty:
        fallback = table
    row = fallback.sort_values(
        ["precision", "recall", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return _row_to_policy(
        "high_precision",
        row,
        rule + f"; fallback to max precision with recall >= {fallback_min_recall:.2f} if possible",
        fallback_used=True,
    )


def select_thresholds(
    y_valid,
    p_valid,
    *,
    policies=DEFAULT_POLICIES,
    high_sensitivity_recall: float = 0.85,
    high_precision_precision: float = 0.65,
    high_precision_min_recall: float = 0.20,
    thresholds=None,
) -> dict:
    """Select threshold policies from validation labels/probabilities only."""
    table = threshold_table(y_valid, p_valid, thresholds=thresholds)
    selected = {}

    for policy in policies:
        if policy == "high_sensitivity":
            selected[policy] = _select_high_sensitivity(table, high_sensitivity_recall)
        elif policy == "balanced_f1":
            selected[policy] = _select_balanced_f1(table)
        elif policy == "high_precision":
            selected[policy] = _select_high_precision(
                table,
                high_precision_precision,
                high_precision_min_recall,
            )
        else:
            raise ValueError(f"Unknown threshold policy: {policy}")

    return selected


def select_optimal_threshold(y_true, y_score, strategy: str = "f1", **kwargs):
    """Backward-compatible single-threshold selector for legacy model code."""
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
