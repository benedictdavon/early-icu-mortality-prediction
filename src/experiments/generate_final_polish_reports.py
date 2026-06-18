"""Generate final portfolio reports from completed aggregate experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.bootstrap import paired_bootstrap_metric_difference
from evaluation.ensembles import (
    average_probabilities,
    fit_validation_stacker,
    prediction_matrix,
)
from evaluation.metrics import compute_binary_metrics
from evaluation.thresholds import select_thresholds

DEFAULT_RUN_DIR = ROOT / "results" / "overnight_20260618_111652"
PRIMARY_POLICY = "balanced_f1"
PRIMARY_MODEL = "lightgbm"
NEAR_TIE_MODELS = ["xgboost", "legacy_xgboost_ensemble", "ebm", "mafnet_t_plus_s"]
ENSEMBLE_MODELS = ["lightgbm", "xgboost", "catboost", "ebm", "mafnet_t_plus_s"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    compact = df[[col for col in columns if col in df.columns]].head(max_rows).copy()
    if compact.empty:
        return "_No matching columns available._"
    lines = [
        "| " + " | ".join(compact.columns) + " |",
        "| " + " | ".join(["---"] * len(compact.columns)) + " |",
    ]
    for _, row in compact.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in compact.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _test_model_suite(run_dir: Path) -> pd.DataFrame:
    table = _read_csv(run_dir / "model_suite" / "model_comparison_table.csv")
    if table.empty:
        return table
    if "threshold_policy" in table.columns:
        table = table[table["threshold_policy"] == PRIMARY_POLICY]
    if "split" in table.columns:
        table = table[table["split"] == "test"]
    return table.sort_values(["average_precision", "auc_roc"], ascending=False)


def _legacy_ensemble_row(run_dir: Path) -> pd.DataFrame:
    table = _read_csv(
        run_dir
        / "legacy_xgboost_ensemble"
        / "xgboost"
        / "ensemble_threshold_results.csv"
    )
    if table.empty:
        return table
    policy_col = (
        "threshold_policy"
        if "threshold_policy" in table.columns
        else "threshold_name"
        if "threshold_name" in table.columns
        else "policy"
    )
    if policy_col in table.columns:
        balanced = table[table[policy_col].astype(str).str.lower() == "balanced"]
        if not balanced.empty:
            table = balanced
    table = table.copy()
    table = table.rename(columns={"auc_pr": "average_precision", "f1_score": "f1"})
    table["model_name"] = "legacy_xgboost_ensemble"
    return table.head(1)


def _mafnet_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "mafnet_ablations" / "mafnet_ablation_results.csv"
    table = _read_csv(path)
    if table.empty:
        return table
    test_rows = table.copy()
    rename_map = {
        "test_auc_roc": "auc_roc",
        "test_average_precision": "average_precision",
        "test_brier_score": "brier_score",
    }
    test_rows = test_rows.rename(columns=rename_map)
    for column in ["auc_roc", "average_precision", "brier_score"]:
        if column in test_rows.columns:
            test_rows[column] = pd.to_numeric(test_rows[column], errors="coerce")
    return test_rows.sort_values(["average_precision", "auc_roc"], ascending=False)


def _primary_row(model_suite: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = model_suite[model_suite["model_name"] == model_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _prediction_file_status(run_dir: Path) -> dict:
    candidates = [
        run_dir / "predictions.csv",
        run_dir / "predictions" / "predictions.csv",
        run_dir / "model_predictions.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    return {
        "available": bool(existing),
        "checked_paths": [str(path) for path in candidates],
        "existing_paths": [str(path) for path in existing],
    }


def _load_prediction_frame(run_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    status = _prediction_file_status(run_dir)
    if not status["existing_paths"]:
        return None, status
    path = Path(status["existing_paths"][0])
    return pd.read_csv(path), status


def _thresholded_test_metrics(y_valid, p_valid, y_test, p_test) -> dict:
    selected = select_thresholds(y_valid, p_valid, policies=[PRIMARY_POLICY])
    threshold = selected[PRIMARY_POLICY]["threshold"]
    metrics = compute_binary_metrics(y_test, p_test, threshold=threshold)
    metrics["threshold_policy"] = PRIMARY_POLICY
    metrics["threshold_selected_on"] = "validation"
    return metrics


def _run_ensemble_if_available(run_dir: Path) -> dict:
    predictions, status = _load_prediction_frame(run_dir)
    if predictions is None:
        return {"available": False, "reason": "no supported row-level prediction file", "status": status}

    try:
        y_valid, valid_matrix = prediction_matrix(
            predictions,
            split="validation",
            model_names=ENSEMBLE_MODELS,
        )
        y_test, test_matrix = prediction_matrix(
            predictions,
            split="test",
            model_names=ENSEMBLE_MODELS,
        )
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "status": status}

    avg_valid = average_probabilities(valid_matrix)
    avg_test = average_probabilities(test_matrix)
    average_metrics = _thresholded_test_metrics(y_valid, avg_valid, y_test, avg_test)
    average_metrics["ensemble"] = "calibrated_probability_average"

    stacker = fit_validation_stacker(
        valid_matrix,
        y_valid,
        model_names=ENSEMBLE_MODELS,
        split="validation",
    )
    stacker_valid = stacker.predict_proba(valid_matrix)
    stacker_test = stacker.predict_proba(test_matrix)
    stacker_metrics = _thresholded_test_metrics(y_valid, stacker_valid, y_test, stacker_test)
    stacker_metrics["ensemble"] = "validation_l2_logistic_stacker"

    return {
        "available": True,
        "status": status,
        "rows": [average_metrics, stacker_metrics],
        "stacker_metadata": stacker.to_metadata(),
    }


def _paired_bootstrap_if_available(
    run_dir: Path,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict:
    predictions, status = _load_prediction_frame(run_dir)
    if predictions is None:
        return {"available": False, "reason": "no supported row-level prediction file", "status": status}

    comparisons = []
    missing = []
    for model_b in NEAR_TIE_MODELS:
        try:
            y_test, matrix = prediction_matrix(
                predictions,
                split="test",
                model_names=[PRIMARY_MODEL, model_b],
            )
        except ValueError as exc:
            missing.append(f"{PRIMARY_MODEL} vs {model_b}: {exc}")
            continue
        for metric_name in ["auc_roc", "average_precision", "brier_score"]:
            comparisons.append(
                paired_bootstrap_metric_difference(
                    y_test,
                    matrix[:, 0],
                    matrix[:, 1],
                    metric_name=metric_name,
                    model_a_name=PRIMARY_MODEL,
                    model_b_name=model_b,
                    n_boot=n_bootstraps,
                    seed=seed,
                )
            )
    return {
        "available": bool(comparisons),
        "rows": comparisons,
        "missing": missing,
        "status": status,
        "reason": "no paired comparisons could be built" if not comparisons else "",
    }


def _ci_text(row: pd.Series, metric: str) -> str:
    lower = row.get(f"{metric}_ci_lower")
    upper = row.get(f"{metric}_ci_upper")
    if lower is None or upper is None or pd.isna(lower) or pd.isna(upper):
        return "N/A"
    return f"{_fmt(lower)} to {_fmt(upper)}"


def write_bootstrap_report(run_dir: Path, *, n_bootstraps: int, seed: int) -> Path:
    model_suite = _test_model_suite(run_dir)
    prediction_status = _prediction_file_status(run_dir)
    paired = _paired_bootstrap_if_available(
        run_dir,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    rows = []
    for _, row in model_suite.iterrows():
        rows.append(
            {
                "model": row.get("model_name"),
                "auc_roc": row.get("auc_roc"),
                "auc_roc_95_ci": _ci_text(row, "auc_roc"),
                "average_precision": row.get("average_precision"),
                "average_precision_95_ci": _ci_text(row, "average_precision"),
                "brier_score": row.get("brier_score"),
                "brier_score_95_ci": _ci_text(row, "brier_score"),
            }
        )
    ci_table = pd.DataFrame(rows)

    lines = [
        "# Bootstrap Confidence Interval Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run directory: `{run_dir}`",
        f"Configured bootstrap seed: `{seed}`",
        f"Configured bootstrap iterations for reusable utilities: `{n_bootstraps}`",
        "",
        "## Available Aggregate CIs",
        "",
        (
            "The tabular model suite already stored bootstrap 95% confidence intervals "
            "for AUC-ROC, average precision, and Brier score. These intervals are "
            "aggregate-only and do not expose patient-level predictions."
        ),
        "",
        _markdown_table(
            ci_table,
            [
                "model",
                "auc_roc",
                "auc_roc_95_ci",
                "average_precision",
                "average_precision_95_ci",
                "brier_score",
                "brier_score_95_ci",
            ],
        ),
        "",
        "## Threshold Metric CIs",
        "",
        (
            "Not recomputed for this completed run because row-level validation/test "
            "prediction files were not saved. The reusable bootstrap utilities now "
            "support F1, recall, precision, specificity, NPV, and accuracy at a fixed "
            "threshold once local prediction files are available."
        ),
        "",
        "## Paired Bootstrap Comparisons",
        "",
    ]
    if paired["available"]:
        lines.extend(
            [
                _markdown_table(
                    pd.DataFrame(paired["rows"]),
                    [
                        "model_a",
                        "model_b",
                        "metric",
                        "estimate_a",
                        "estimate_b",
                        "difference",
                        "ci_lower",
                        "ci_upper",
                        "p_value_two_sided",
                        "n_success",
                    ],
                ),
                "",
            ]
        )
        if paired.get("missing"):
            lines.append("Skipped paired comparisons:")
            lines.extend(f"- {item}" for item in paired["missing"])
            lines.append("")
    else:
        lines.extend(
            [
                (
                    "Not run for LightGBM vs XGBoost, legacy XGBoost ensemble, EBM, "
                    "or MAFNet-T+S because paired bootstrap requires aligned row-level "
                    "test predictions from both models."
                ),
                f"Reason: {paired.get('reason', 'unavailable')}.",
                "",
            ]
        )
    lines.append("Checked prediction paths:")
    lines.extend(f"- `{path}`" for path in prediction_status["checked_paths"])
    if prediction_status["existing_paths"]:
        lines.append("")
        lines.append("Existing prediction files found:")
        lines.extend(f"- `{path}`" for path in prediction_status["existing_paths"])
    else:
        lines.append("")
        lines.append("No supported row-level prediction file was found.")
    lines.extend(
        [
            "",
            "## Reporting Interpretation",
            "",
            (
                "LightGBM and XGBoost have overlapping aggregate bootstrap intervals "
                "for discrimination. Treat XGBoost as a practical near-tie unless a "
                "future paired bootstrap test on aligned predictions shows otherwise."
            ),
            "",
        ]
    )
    output = run_dir / "bootstrap_ci_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_calibration_report(run_dir: Path) -> Path:
    model_suite = _test_model_suite(run_dir)
    mafnet_dirs = {
        "Full MAFNet": run_dir / "mafnet_full",
        "MAFNet-T+S": run_dir / "mafnet_ablations" / "mafnet_t_plus_s",
    }
    calibration_rows = []
    for _, row in model_suite.iterrows():
        calibration_rows.append(
            {
                "model": row.get("model_name"),
                "split": "test",
                "probability_source": "model_suite_probability",
                "brier_score": row.get("brier_score"),
                "calibration_intercept": "N/A",
                "calibration_slope": "N/A",
                "expected_calibration_error": "N/A",
            }
        )
    for label, path in mafnet_dirs.items():
        test_cal = _read_json(path / "test_calibration.json")
        if not test_cal:
            continue
        for source in ["raw_sigmoid", "platt_calibrated"]:
            source_metrics = test_cal.get(source, {})
            calibration_rows.append(
                {
                    "model": label,
                    "split": "test",
                    "probability_source": source,
                    "brier_score": source_metrics.get("brier_score"),
                    "calibration_intercept": source_metrics.get("calibration_intercept"),
                    "calibration_slope": source_metrics.get("calibration_slope"),
                    "expected_calibration_error": source_metrics.get(
                        "expected_calibration_error"
                    ),
                }
            )

    lines = [
        "# Calibration Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run directory: `{run_dir}`",
        "",
        "## Calibration Metrics",
        "",
        _markdown_table(
            pd.DataFrame(calibration_rows),
            [
                "model",
                "split",
                "probability_source",
                "brier_score",
                "calibration_intercept",
                "calibration_slope",
                "expected_calibration_error",
            ],
        ),
        "",
        "## Validation-Only Calibration Policy",
        "",
        (
            "Calibration utilities fit Platt scaling and optional isotonic calibration "
            "on validation predictions only. The fitted mapping is then applied once "
            "to the held-out test split. The test split is never used to fit a "
            "calibrator or select a threshold."
        ),
        "",
        "## Run-Specific Takeaways",
        "",
        (
            "EBM had the best tabular Brier score in the model suite, which makes it "
            "a useful calibrated and interpretable comparison model."
        ),
        (
            "For MAFNet-T+S, Platt scaling improved test Brier score from 0.1318 to "
            "0.1231 and expected calibration error from 0.0643 to 0.0076."
        ),
        (
            "For tabular models, before/after Platt or isotonic calibration could not "
            "be recomputed from this completed run because row-level validation and "
            "test predictions were not saved."
        ),
        "",
    ]
    output = run_dir / "calibration_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_ensemble_report(run_dir: Path) -> Path:
    prediction_status = _prediction_file_status(run_dir)
    ensemble = _run_ensemble_if_available(run_dir)
    lines = [
        "# Ensemble Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run directory: `{run_dir}`",
        "",
        "## Planned Leakage-Safe Ensembles",
        "",
        (
            "Two ensemble mechanisms are implemented for local aligned predictions: "
            "a simple average of calibrated probabilities and an L2 logistic "
            "regression stacker fit only on validation predictions."
        ),
        "",
        "Required component models:",
    ]
    lines.extend(f"- `{model}`" for model in ENSEMBLE_MODELS)
    lines.extend(
        [
            "",
            "## Current Completed Run Status",
            "",
        ]
    )
    if ensemble["available"]:
        lines.append(
            "Aligned prediction files were found. The following ensemble metrics were "
            "computed by selecting the threshold on validation predictions and applying "
            "it unchanged to the test split."
        )
        lines.extend(
            [
                "",
                _markdown_table(
                    pd.DataFrame(ensemble["rows"]),
                    [
                        "ensemble",
                        "auc_roc",
                        "average_precision",
                        "brier_score",
                        "threshold",
                        "accuracy",
                        "precision",
                        "recall",
                        "specificity",
                        "f1",
                        "npv",
                    ],
                ),
                "",
                "Stacker metadata:",
            ]
        )
        lines.extend(
            f"- `{key}`: `{value}`" for key, value in ensemble["stacker_metadata"].items()
        )
    else:
        lines.append(
            "No ensemble results were computed. The completed run contains aggregate "
            "model metrics, threshold reports, calibration summaries, and model "
            "artifacts, but not aligned row-level validation/test prediction files."
        )
        lines.append(f"Reason: {ensemble.get('reason', 'unavailable')}.")
    if not ensemble["available"]:
        lines.extend(
            [
                "",
                "Missing inputs for this run:",
                "- validation labels and calibrated probabilities for LightGBM, XGBoost, CatBoost, EBM, and MAFNet-T+S",
                "- test labels and calibrated probabilities for the same models",
                "- a non-identifying `row_id` or identical row ordering to align model predictions",
            ]
        )
    lines.extend(["", "Checked prediction paths:"])
    lines.extend(f"- `{path}`" for path in prediction_status["checked_paths"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Because the required aligned prediction files are missing, no "
                "average-ensemble or stacker performance should be claimed for this "
                "run. The legacy XGBoost ensemble remains a separate previously run "
                "model family, not the requested calibrated multi-model ensemble."
            ),
            "",
        ]
    )
    output = run_dir / "ensemble_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_final_model_report(run_dir: Path) -> Path:
    model_suite = _test_model_suite(run_dir)
    legacy = _legacy_ensemble_row(run_dir)
    mafnet = _mafnet_rows(run_dir)
    lightgbm = _primary_row(model_suite, PRIMARY_MODEL)
    xgboost = _primary_row(model_suite, "xgboost")
    ebm = _primary_row(model_suite, "ebm")
    mafnet_ts = mafnet[mafnet["ablation"].astype(str).str.contains("T\\+S", regex=True)]
    if mafnet_ts.empty:
        mafnet_ts = mafnet.head(1)

    key_rows = []
    for label, row in [
        ("LightGBM", lightgbm),
        ("XGBoost", xgboost),
        ("EBM", ebm),
    ]:
        if row is not None:
            key_rows.append(
                {
                    "model": label,
                    "auc_roc": row.get("auc_roc"),
                    "average_precision": row.get("average_precision"),
                    "brier_score": row.get("brier_score"),
                    "f1": row.get("f1"),
                    "recall": row.get("recall"),
                    "precision": row.get("precision"),
                }
            )
    if not legacy.empty:
        row = legacy.iloc[0]
        key_rows.append(
            {
                "model": "Legacy XGBoost ensemble",
                "auc_roc": row.get("auc_roc"),
                "average_precision": row.get("average_precision"),
                "brier_score": row.get("brier_score", math.nan),
                "f1": row.get("f1"),
                "recall": row.get("recall"),
                "precision": row.get("precision"),
            }
        )
    if not mafnet_ts.empty:
        row = mafnet_ts.iloc[0]
        key_rows.append(
            {
                "model": "MAFNet-T+S",
                "auc_roc": row.get("auc_roc"),
                "average_precision": row.get("average_precision"),
                "brier_score": row.get("brier_score"),
                "f1": "N/A",
                "recall": "N/A",
                "precision": "N/A",
            }
        )

    lines = [
        "# Final Model Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run directory: `{run_dir}`",
        "",
        "## Final Positioning",
        "",
        (
            "LightGBM is the primary final model for portfolio reporting because it "
            "has the strongest tabular PR-AUC/AUC combination and a good Brier score "
            "in the completed run."
        ),
        (
            "XGBoost is a practical near-tie on discrimination and has slightly "
            "higher F1/recall at the validation-selected balanced threshold. Avoid "
            "claiming LightGBM is definitively superior unless a future paired "
            "bootstrap comparison on aligned predictions supports that claim."
        ),
        (
            "EBM is the calibrated/interpretable comparison model because it had the "
            "best tabular Brier score while retaining competitive discrimination."
        ),
        (
            "MAFNet-T+S is the best neural temporal variant. Full MAFNet underperformed "
            "the best boosted tabular models in this run."
        ),
        "",
        "## Key Test Results",
        "",
        _markdown_table(
            pd.DataFrame(key_rows),
            [
                "model",
                "auc_roc",
                "average_precision",
                "brier_score",
                "f1",
                "recall",
                "precision",
            ],
        ),
        "",
        "## Safety And Scope",
        "",
        (
            "This is retrospective academic/portfolio research, not a clinical "
            "deployment. The test set is used only for final reporting after model, "
            "threshold, and calibration decisions are made on training/validation data."
        ),
        (
            "No raw MIMIC-IV data, processed patient-level CSVs, row-level predictions, "
            "or patient identifiers should be committed or exposed in public reports."
        ),
        "",
    ]
    output = run_dir / "final_model_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def generate_reports(run_dir: Path, *, n_bootstraps: int, seed: int) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_final_model_report(run_dir),
        write_calibration_report(run_dir),
        write_bootstrap_report(run_dir, n_bootstraps=n_bootstraps, seed=seed),
        write_ensemble_report(run_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final polish reports")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_reports(
        Path(args.run_dir),
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
