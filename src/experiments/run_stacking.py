"""Run probability-level ensemble reports without saving row predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import compute_binary_metrics
from models.stacking import average_calibrated_probabilities


def evaluate_probability_ensemble(
    *,
    y_true,
    probability_frame: pd.DataFrame,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """Evaluate an ensemble from already-calibrated probability columns."""
    if probability_frame.shape[1] < 2:
        raise ValueError("At least two calibrated probability columns are required")
    ensemble_probabilities = average_calibrated_probabilities(probability_frame.to_numpy())
    metrics = compute_binary_metrics(y_true, ensemble_probabilities, threshold=threshold)
    return {
        "model_name": model_name,
        "component_models": list(probability_frame.columns),
        "threshold": float(threshold),
        **metrics,
    }


def save_ensemble_report(report: dict, output_dir) -> dict:
    """Save aggregate ensemble report. Row-level probabilities are not written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ensemble_report.json"
    csv_path = out / "ensemble_report.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=True)
    pd.DataFrame([report]).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated probability ensembles")
    parser.add_argument("--probabilities-path", required=True)
    parser.add_argument("--target-col", default="mortality")
    parser.add_argument("--model-name", default="XGB+MAFNet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "ensembles"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.probabilities_path)
    if args.target_col not in frame.columns:
        raise ValueError(f"`{args.target_col}` column is required")
    y_true = frame[args.target_col].to_numpy()
    probabilities = frame.drop(columns=[args.target_col])
    report = evaluate_probability_ensemble(
        y_true=y_true,
        probability_frame=probabilities,
        model_name=args.model_name,
        threshold=args.threshold,
    )
    paths = save_ensemble_report(report, args.output_dir)
    print(f"Saved ensemble report: {paths['json']}")


__all__ = ["evaluate_probability_ensemble", "save_ensemble_report"]


if __name__ == "__main__":
    main()
