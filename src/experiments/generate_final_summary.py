"""Generate final-deliverable summaries from aggregate outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = ROOT / "results" / "model_suite"
DEFAULT_OUTPUT = ROOT / "report" / "FINAL_SUMMARY.md"
DEFAULT_FEATURE_DICTIONARY = ROOT / "docs" / "feature_dictionary.csv"


ASSIGNMENT_TRACEABILITY = [
    (
        "TODO 1",
        "Cohort selection and flow chart",
        "src/cohort_selection.py; report/REPORT.md cohort section",
    ),
    (
        "TODO 2",
        "Feature extraction and descriptive analysis",
        "src/feature_extraction.py; docs/feature_dictionary.csv",
    ),
    (
        "TODO 3",
        "Preprocessing strategy explanation",
        "src/data_preprocessing.py; docs/leakage_checklist.md",
    ),
    (
        "TODO 4",
        "Model development, architecture, evaluation, and conclusion",
        "src/main.py; src/experiments/run_model_suite.py; report/REPORT.md",
    ),
    (
        "Final deliverable",
        "Share code, results, TODO descriptions, and conclusion",
        "report/FINAL_SUMMARY.md",
    ),
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_name",
        "split",
        "threshold_policy",
        "auc_roc",
        "average_precision",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "npv",
        "threshold",
        "n",
    ]
    return [col for col in preferred if col in df.columns]


def _format_number(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Return a compact GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows available._"

    compact = df.head(max_rows).copy()
    headers = list(compact.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in compact.iterrows():
        lines.append("| " + " | ".join(_format_number(row[col]) for col in headers) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def summarize_statuses(statuses: list[dict]) -> dict:
    summary = {"completed": 0, "skipped": 0, "failed": 0, "started": 0}
    for status in statuses:
        state = status.get("status", "started")
        summary[state] = summary.get(state, 0) + 1
    return summary


def select_primary_results(results: pd.DataFrame, policy: str) -> pd.DataFrame:
    if results.empty:
        return results
    filtered = results.copy()
    if "split" in filtered.columns:
        filtered = filtered[filtered["split"] == "test"]
    if "threshold_policy" in filtered.columns:
        policy_rows = filtered[filtered["threshold_policy"] == policy]
        if not policy_rows.empty:
            filtered = policy_rows
    filtered = filtered[_metric_columns(filtered)]
    sort_cols = [col for col in ["average_precision", "auc_roc"] if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=False)
    return filtered


def feature_dictionary_summary(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    df = pd.read_csv(path)
    group_counts = (
        df["feature_group"].value_counts().sort_index().to_dict()
        if "feature_group" in df.columns
        else {}
    )
    return {
        "available": True,
        "path": str(path),
        "rows": int(len(df)),
        "groups": group_counts,
    }


def build_final_summary(
    *,
    model_suite_dir: Path,
    feature_dictionary_path: Path = DEFAULT_FEATURE_DICTIONARY,
    primary_policy: str = "balanced_f1",
    generated_at: datetime | None = None,
) -> str:
    suite_dir = Path(model_suite_dir)
    status = _read_json(suite_dir / "model_suite_status.json")
    comparison = _read_csv(suite_dir / "model_comparison_table.csv")
    results = _read_csv(suite_dir / "model_suite_results.csv")
    feature_summary = feature_dictionary_summary(Path(feature_dictionary_path))
    generated_at = generated_at or datetime.now(timezone.utc)

    statuses = status.get("statuses", [])
    status_counts = summarize_statuses(statuses)
    primary_results = select_primary_results(results, primary_policy)
    best_row = primary_results.head(1)

    if not comparison.empty:
        comparison_view = comparison[
            [col for col in _metric_columns(comparison) if col in comparison.columns]
        ]
    else:
        comparison_view = primary_results

    lines = [
        "# Final Deliverable Summary",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Model suite directory: `{suite_dir}`",
        "",
        "## Assignment Traceability",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                ASSIGNMENT_TRACEABILITY,
                columns=["Requirement", "Deliverable", "Repository evidence"],
            ),
            max_rows=20,
        ),
        "",
        "## Run Status",
        "",
        f"- Completed models: {status_counts.get('completed', 0)}",
        f"- Skipped models: {status_counts.get('skipped', 0)}",
        f"- Failed models: {status_counts.get('failed', 0)}",
        f"- Rows evaluated by suite: {status.get('n_rows', 'unknown')}",
        f"- Feature count after train-only preprocessing: {status.get('feature_count', 'unknown')}",
        "",
        "## Primary Test Result",
        "",
    ]

    if best_row.empty:
        lines.append("_No primary test result was available._")
    else:
        row = best_row.iloc[0]
        lines.extend(
            [
                f"Primary policy: `{primary_policy}` selected on validation data.",
                (
                    f"Best model by average precision/AUC-ROC sort: `{row.get('model_name', 'unknown')}` "
                    f"with AUC-ROC {_format_number(row.get('auc_roc'))}, "
                    f"average precision {_format_number(row.get('average_precision'))}, "
                    f"F1 {_format_number(row.get('f1'))}, and recall {_format_number(row.get('recall'))}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            dataframe_to_markdown(comparison_view, max_rows=15),
            "",
            "## Threshold Policy Results",
            "",
            dataframe_to_markdown(primary_results, max_rows=15),
            "",
            "## Feature Provenance",
            "",
        ]
    )

    if feature_summary.get("available"):
        groups = feature_summary["groups"]
        group_text = ", ".join(f"{name}: {count}" for name, count in groups.items())
        lines.extend(
            [
                f"- Feature dictionary rows: {feature_summary['rows']}",
                f"- Feature groups: {group_text or 'not available'}",
            ]
        )
    else:
        lines.append("- Feature dictionary not found.")

    lines.extend(
        [
            "",
            "## Safety And Reporting Notes",
            "",
            "- This is a retrospective academic analysis, not a clinical decision-support system.",
            "- Thresholds are selected on validation probabilities and applied unchanged to the held-out test split.",
            "- The final summary consumes aggregate model-suite outputs only; it does not write row-level predictions.",
            "- Patient-level data, trained model artifacts, and raw MIMIC-IV files must remain local and uncommitted.",
            "",
            "## Final Conclusion",
            "",
            (
                "The repository now contains an end-to-end mortality-prediction workflow covering cohort "
                "selection, first-6-hour feature extraction, preprocessing, model comparison, threshold "
                "analysis, feature provenance, and final aggregate reporting. Any clinical use would require "
                "external validation, calibration review, subgroup fairness analysis, and governance approval."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_final_summary(
    *,
    model_suite_dir: Path,
    output_path: Path,
    feature_dictionary_path: Path = DEFAULT_FEATURE_DICTIONARY,
    primary_policy: str = "balanced_f1",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_final_summary(
        model_suite_dir=Path(model_suite_dir),
        feature_dictionary_path=Path(feature_dictionary_path),
        primary_policy=primary_policy,
    )
    output.write_text(report, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a final aggregate report from model-suite outputs"
    )
    parser.add_argument("--model-suite-dir", default=str(DEFAULT_SUITE_DIR))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-dictionary", default=str(DEFAULT_FEATURE_DICTIONARY))
    parser.add_argument("--primary-policy", default="balanced_f1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_final_summary(
        model_suite_dir=Path(args.model_suite_dir),
        output_path=Path(args.output_path),
        feature_dictionary_path=Path(args.feature_dictionary),
        primary_policy=args.primary_policy,
    )
    print(f"Final summary written to {output}")


if __name__ == "__main__":
    main()
