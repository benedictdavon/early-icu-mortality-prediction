"""Run leakage-safe XGBoost ablations for baseline vs expanded features.

The script expects preprocessed, patient-level CSVs to exist locally. Those
files are restricted artifacts and are intentionally ignored by git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.reporting import results_dataframe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline vs expanded-feature XGBoost ablations"
    )
    parser.add_argument(
        "--baseline-data-path",
        default=str(ROOT / "data" / "processed" / "preprocessed_xgboost_features.csv"),
        help="Baseline preprocessed XGBoost CSV",
    )
    parser.add_argument(
        "--expanded-data-path",
        default=str(
            ROOT
            / "data"
            / "processed"
            / "preprocessed_xgboost_expanded_features.csv"
        ),
        help="Expanded-feature preprocessed XGBoost CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "xgboost_expanded_ablation"),
        help="Directory for aggregate ablation outputs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="XGBoost device. Default uses CPU for portability.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=600,
        help="XGBoost n_estimators for each ablation run",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds using validation split only",
    )
    return parser.parse_args()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required preprocessed data file not found: {path}. "
            "Run the restricted-data extraction/preprocessing pipeline first."
        )


def _xgboost_params(args) -> dict:
    from models.xgboost.tuning import xgboost_default_params

    params = xgboost_default_params()
    params.update(
        {
            "device": args.device,
            "tree_method": "hist",
            "n_estimators": args.n_estimators,
        }
    )
    return params


def run_single_ablation(
    *,
    ablation_name: str,
    data_path: Path,
    output_dir: Path,
    params: dict,
    early_stopping_rounds: int,
) -> tuple[dict, pd.DataFrame]:
    from models.xgboost import ICUMortalityXGBoost

    model_output_dir = output_dir / ablation_name
    model = ICUMortalityXGBoost(output_dir=str(model_output_dir), gpu_device=params["device"])
    model.load_data(str(data_path))
    model.train(params=params, early_stopping_rounds=early_stopping_rounds)
    report = model.evaluate_threshold_policies()

    result_df = results_dataframe(report)
    result_df.insert(0, "ablation_name", ablation_name)
    result_df.insert(1, "data_path", str(data_path))
    result_df.insert(2, "feature_count", len(model.feature_names))

    run_metadata = {
        "ablation_name": ablation_name,
        "data_path": str(data_path),
        "output_dir": str(model.output_dir),
        "feature_count": len(model.feature_names),
        "train_rows": int(len(model.split_indices["train"])),
        "train_rows_after_resampling": int(len(model.y_train)),
        "validation_rows": int(len(model.y_val)),
        "test_rows": int(len(model.y_test)),
        "threshold_report": str(Path(model.output_dir) / "threshold_policy_report.json"),
    }
    return run_metadata, result_df


def main(args=None):
    args = args or parse_args()
    baseline_path = Path(args.baseline_data_path)
    expanded_path = Path(args.expanded_data_path)
    output_dir = Path(args.output_dir)

    _require_file(baseline_path)
    _require_file(expanded_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _xgboost_params(args)
    runs = [
        ("baseline", baseline_path),
        ("expanded_features", expanded_path),
    ]

    metadata = []
    result_tables = []
    for ablation_name, data_path in runs:
        run_metadata, result_df = run_single_ablation(
            ablation_name=ablation_name,
            data_path=data_path,
            output_dir=output_dir,
            params=params,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        metadata.append(run_metadata)
        result_tables.append(result_df)

    combined = pd.concat(result_tables, ignore_index=True)
    csv_path = output_dir / "ablation_threshold_policy_results.csv"
    json_path = output_dir / "ablation_summary.json"

    combined.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": metadata,
                "params": params,
                "results": combined.to_dict(orient="records"),
            },
            f,
            indent=2,
            allow_nan=True,
        )

    print(f"Saved aggregate ablation CSV: {csv_path}")
    print(f"Saved aggregate ablation JSON: {json_path}")
    print(combined.round(4).to_string(index=False))
    return combined


if __name__ == "__main__":
    main()
