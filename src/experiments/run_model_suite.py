"""Run the tabular model suite with leakage-safe evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schema import build_feature_matrix
from data.splitting import make_train_valid_test_split
from evaluation.bootstrap import bootstrap_metric_ci
from evaluation.reporting import (
    evaluate_validation_and_test,
    results_dataframe,
    save_aggregate_report,
)
from evaluation.plots import save_calibration_curve
from models.base import OptionalDependencyUnavailable
from models.tabular import get_model_class
from utils.seed import set_global_seed


DEFAULT_MODELS = (
    "logistic",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "ebm",
)


def _parse_model_names(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(config_dir: Path, model_name: str) -> dict:
    """Load optional per-model config."""
    config = _read_yaml(config_dir / f"{model_name}.yaml")
    return config.get("model", config)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _simple_imputer(strategy: str) -> SimpleImputer:
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


def fit_feature_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit imputation/encoding on training features only."""
    numeric_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", _simple_imputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", _simple_imputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns available after leakage guards")

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(X_train)
    return preprocessor


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def prepare_tabular_splits(
    data: pd.DataFrame,
    *,
    target_col: str = "mortality",
    patient_id_col: str = "subject_id",
    valid_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Build train/validation/test arrays using train-only preprocessing."""
    if target_col not in data.columns:
        raise ValueError(f"Dataset must contain `{target_col}`")

    y = data[target_col].astype(int)
    patient_ids = data[patient_id_col] if patient_id_col in data.columns else None
    X_raw, dropped_columns = build_feature_matrix(data, target_col=target_col)

    splits = make_train_valid_test_split(
        y,
        patient_ids=patient_ids,
        test_size=test_size,
        valid_size=valid_size,
        stratify=True,
        group_by_patient=patient_ids is not None,
        random_state=random_state,
    )

    X_train_raw = X_raw.iloc[splits["train"]]
    X_valid_raw = X_raw.iloc[splits["validation"]]
    X_test_raw = X_raw.iloc[splits["test"]]

    preprocessor = fit_feature_preprocessor(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_valid = preprocessor.transform(X_valid_raw)
    X_test = preprocessor.transform(X_test_raw)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y.iloc[splits["train"]].to_numpy(),
        "y_valid": y.iloc[splits["validation"]].to_numpy(),
        "y_test": y.iloc[splits["test"]].to_numpy(),
        "feature_names": _feature_names(preprocessor),
        "dropped_columns": dropped_columns,
        "split_indices": splits,
        "preprocessor": preprocessor,
    }


def _bootstrap_summary(y_true, p_pred, n_boot: int, seed: int) -> dict:
    if n_boot <= 0:
        return {}
    return {
        "auc_roc": bootstrap_metric_ci(
            y_true,
            p_pred,
            roc_auc_score,
            n_boot=n_boot,
            seed=seed,
        ),
        "average_precision": bootstrap_metric_ci(
            y_true,
            p_pred,
            average_precision_score,
            n_boot=n_boot,
            seed=seed + 1,
        ),
        "brier_score": bootstrap_metric_ci(
            y_true,
            p_pred,
            brier_score_loss,
            n_boot=n_boot,
            seed=seed + 2,
        ),
    }


def _comparison_record(report: dict, bootstrap: dict) -> dict:
    df = results_dataframe(report)
    row = df[(df["split"] == "test") & (df["threshold_policy"] == "balanced_f1")]
    if row.empty:
        row = df[df["split"] == "test"].head(1)
    record = row.iloc[0].to_dict()
    for metric_name, interval in bootstrap.items():
        record[f"{metric_name}_ci_lower"] = interval["ci_lower"]
        record[f"{metric_name}_ci_upper"] = interval["ci_upper"]
        record[f"{metric_name}_ci_n_success"] = interval["n_success"]
    return record


def run_model_suite(
    *,
    data_path,
    output_dir,
    models=DEFAULT_MODELS,
    config_dir=ROOT / "configs" / "models",
    target_col: str = "mortality",
    patient_id_col: str = "subject_id",
    seed: int = 42,
    valid_size: float = 0.20,
    test_size: float = 0.20,
    max_rows: int | None = None,
    save_models: bool = False,
    bootstrap_iterations: int = 200,
    fail_fast: bool = False,
) -> dict:
    """Run all requested models and save aggregate model-suite artifacts."""
    set_global_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_dir = Path(config_dir)

    data = pd.read_csv(data_path)
    if max_rows is not None:
        data = data.head(max_rows).copy()

    prepared = prepare_tabular_splits(
        data,
        target_col=target_col,
        patient_id_col=patient_id_col,
        valid_size=valid_size,
        test_size=test_size,
        random_state=seed,
    )

    statuses = []
    reports = []
    comparison_records = []
    result_frames = []

    for model_name in _parse_model_names(models):
        model_out = out / model_name
        status = {"model_name": model_name, "status": "started"}
        try:
            model_config = load_model_config(config_dir, model_name)
            if model_config.get("enabled", True) is False:
                status.update({"status": "skipped", "reason": "disabled in config"})
                statuses.append(status)
                continue

            model_cls = get_model_class(model_name)
            if not model_cls.is_available():
                status.update(
                    {
                        "status": "skipped",
                        "reason": "optional dependency is not installed",
                    }
                )
                statuses.append(status)
                continue

            model = model_cls(
                params=model_config.get("params", {}),
                random_state=model_config.get("random_state", seed),
            )
            model._set_feature_names(prepared["feature_names"])
            search_config = model_config.get("search", {})
            if search_config.get("enabled", False):
                model.fit_randomized_search(
                    prepared["X_train"],
                    prepared["y_train"],
                    param_distributions=search_config["param_distributions"],
                    n_iter=search_config.get("n_iter", 20),
                    cv=search_config.get("cv", 3),
                    scoring=search_config.get("scoring", "average_precision"),
                    n_jobs=search_config.get("n_jobs", -1),
                )
            else:
                model.fit(
                    prepared["X_train"],
                    prepared["y_train"],
                    prepared["X_valid"],
                    prepared["y_valid"],
                )
            p_valid = model.predict_proba(prepared["X_valid"])
            p_test = model.predict_proba(prepared["X_test"])

            report = evaluate_validation_and_test(
                model_name=model_name,
                y_valid=prepared["y_valid"],
                p_valid=p_valid,
                y_test=prepared["y_test"],
                p_test=p_test,
            )
            save_aggregate_report(report, model_out)
            save_calibration_curve(
                prepared["y_test"],
                p_test,
                model_out / "calibration_curve.png",
                title=f"{model_name} Test Calibration Curve",
            )

            test_bootstrap = _bootstrap_summary(
                prepared["y_test"],
                p_test,
                n_boot=bootstrap_iterations,
                seed=seed,
            )
            comparison_records.append(_comparison_record(report, test_bootstrap))
            result_frames.append(results_dataframe(report))
            reports.append(report)

            if save_models:
                model.save(out / "model_artifacts" / f"{model_name}.joblib")

            status.update(
                {
                    "status": "completed",
                    "validation_rows": int(len(prepared["y_valid"])),
                    "test_rows": int(len(prepared["y_test"])),
                    "feature_count": int(prepared["X_train"].shape[1]),
                    "search_enabled": bool(search_config.get("enabled", False)),
                    "best_params": getattr(model, "best_params_", None),
                }
            )
            statuses.append(status)

        except OptionalDependencyUnavailable as exc:
            status.update({"status": "skipped", "reason": str(exc)})
            statuses.append(status)
            if fail_fast:
                raise
        except Exception as exc:
            status.update({"status": "failed", "reason": str(exc)})
            statuses.append(status)
            if fail_fast:
                raise

    if result_frames:
        all_results = pd.concat(result_frames, ignore_index=True)
        all_results.to_csv(out / "model_suite_results.csv", index=False)
    else:
        all_results = pd.DataFrame()

    comparison = pd.DataFrame(comparison_records)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["average_precision", "auc_roc"],
            ascending=[False, False],
        )
    comparison.to_csv(out / "model_comparison_table.csv", index=False)

    run_summary = {
        "data_path": str(data_path),
        "output_dir": str(out),
        "target_col": target_col,
        "patient_id_col": patient_id_col if patient_id_col in data.columns else None,
        "seed": int(seed),
        "n_rows": int(len(data)),
        "feature_count": int(prepared["X_train"].shape[1]),
        "dropped_columns": prepared["dropped_columns"],
        "statuses": statuses,
        "reports": reports,
    }
    with (out / "model_suite_status.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, allow_nan=True)

    return {
        "summary": run_summary,
        "results": all_results,
        "comparison": comparison,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the tabular model suite")
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data" / "processed" / "preprocessed_xgboost_features.csv"),
        help="Processed feature CSV containing the mortality target",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "model_suite"),
        help="Directory for aggregate model-suite outputs",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "configs" / "models"),
        help="Directory containing per-model YAML configs",
    )
    parser.add_argument("--target-col", default="mortality")
    parser.add_argument("--patient-id-col", default="subject_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_model_suite(
        data_path=args.data_path,
        output_dir=args.output_dir,
        models=args.models,
        config_dir=args.config_dir,
        target_col=args.target_col,
        patient_id_col=args.patient_id_col,
        seed=args.seed,
        valid_size=args.valid_size,
        test_size=args.test_size,
        max_rows=args.max_rows,
        save_models=args.save_models,
        bootstrap_iterations=args.bootstrap_iterations,
        fail_fast=args.fail_fast,
    )
    statuses = result["summary"]["statuses"]
    completed = sum(1 for row in statuses if row["status"] == "completed")
    skipped = sum(1 for row in statuses if row["status"] == "skipped")
    failed = sum(1 for row in statuses if row["status"] == "failed")
    print(
        f"Model suite finished: {completed} completed, "
        f"{skipped} skipped, {failed} failed"
    )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
