"""Structured aggregate reporting for validation and test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.calibration import calibration_summary
from evaluation.metrics import compute_binary_metrics
from evaluation.thresholds import select_thresholds


RESULT_SCHEMA_FIELDS = [
    "model_name",
    "split",
    "auc_roc",
    "average_precision",
    "brier_score",
    "threshold_policy",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "npv",
    "n",
    "positive_rate",
]


def make_result_record(
    model_name: str,
    split: str,
    y_true,
    p_pred,
    threshold_policy: str,
    threshold: float,
) -> dict:
    """Create one aggregate result record in the project schema."""
    metrics = compute_binary_metrics(y_true, p_pred, threshold=threshold)
    record = {
        "model_name": model_name,
        "split": split,
        "threshold_policy": threshold_policy,
        **metrics,
    }
    return record


def evaluate_fixed_thresholds(
    model_name: str,
    split: str,
    y_true,
    p_pred,
    thresholds: dict,
) -> list[dict]:
    """Apply preselected thresholds unchanged to a labeled split."""
    records = []
    for policy_name, policy in thresholds.items():
        records.append(
            make_result_record(
                model_name=model_name,
                split=split,
                y_true=y_true,
                p_pred=p_pred,
                threshold_policy=policy_name,
                threshold=policy["threshold"],
            )
        )
    return records


def evaluate_validation_and_test(
    model_name: str,
    y_valid,
    p_valid,
    y_test,
    p_test,
) -> dict:
    """Select thresholds on validation and apply them unchanged to test."""
    thresholds = select_thresholds(y_valid, p_valid)
    validation_records = evaluate_fixed_thresholds(
        model_name, "validation", y_valid, p_valid, thresholds
    )
    test_records = evaluate_fixed_thresholds(
        model_name, "test", y_test, p_test, thresholds
    )
    return {
        "model_name": model_name,
        "thresholds": thresholds,
        "results": validation_records + test_records,
        "calibration": {
            "validation": calibration_summary(y_valid, p_valid),
            "test": calibration_summary(y_test, p_test),
        },
    }


def results_dataframe(report: dict) -> pd.DataFrame:
    """Return report records as a stable-column DataFrame."""
    df = pd.DataFrame(report["results"])
    ordered = [field for field in RESULT_SCHEMA_FIELDS if field in df.columns]
    remaining = [field for field in df.columns if field not in ordered]
    return df[ordered + remaining]


def save_aggregate_report(report: dict, output_dir) -> dict:
    """Save aggregate JSON and CSV reports. No row-level predictions are saved."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "threshold_policy_report.json"
    csv_path = out / "threshold_policy_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    results_dataframe(report).to_csv(csv_path, index=False)

    return {"json": str(json_path), "csv": str(csv_path)}
