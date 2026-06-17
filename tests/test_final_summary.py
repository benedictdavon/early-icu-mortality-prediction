from __future__ import annotations

import json

import pandas as pd

from experiments.generate_final_summary import build_final_summary, write_final_summary


def _write_suite_outputs(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "model_suite_status.json").write_text(
        json.dumps(
            {
                "n_rows": 120,
                "feature_count": 12,
                "statuses": [
                    {"model_name": "logistic", "status": "completed"},
                    {"model_name": "xgboost", "status": "completed"},
                    {"model_name": "ebm", "status": "skipped"},
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model_name": "logistic",
                "split": "test",
                "threshold_policy": "balanced_f1",
                "auc_roc": 0.71,
                "average_precision": 0.42,
                "brier_score": 0.19,
                "accuracy": 0.68,
                "precision": 0.40,
                "recall": 0.62,
                "specificity": 0.70,
                "f1": 0.49,
                "npv": 0.84,
                "threshold": 0.44,
                "n": 24,
            },
            {
                "model_name": "xgboost",
                "split": "test",
                "threshold_policy": "balanced_f1",
                "auc_roc": 0.79,
                "average_precision": 0.55,
                "brier_score": 0.16,
                "accuracy": 0.72,
                "precision": 0.48,
                "recall": 0.70,
                "specificity": 0.73,
                "f1": 0.57,
                "npv": 0.88,
                "threshold": 0.38,
                "n": 24,
            },
        ]
    ).to_csv(suite / "model_suite_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "split": "test",
                "threshold_policy": "balanced_f1",
                "auc_roc": 0.79,
                "average_precision": 0.55,
                "f1": 0.57,
            }
        ]
    ).to_csv(suite / "model_comparison_table.csv", index=False)

    feature_dictionary = tmp_path / "feature_dictionary.csv"
    pd.DataFrame(
        [
            {"feature_name": "age", "feature_group": "demographic"},
            {"feature_name": "lactate_max", "feature_group": "lab"},
        ]
    ).to_csv(feature_dictionary, index=False)
    return suite, feature_dictionary


def test_final_summary_summarizes_aggregate_model_suite_outputs(tmp_path):
    suite, feature_dictionary = _write_suite_outputs(tmp_path)

    report = build_final_summary(
        model_suite_dir=suite,
        feature_dictionary_path=feature_dictionary,
        primary_policy="balanced_f1",
    )

    assert "Final Deliverable Summary" in report
    assert "TODO 1" in report
    assert "Completed models: 2" in report
    assert "Skipped models: 1" in report
    assert "Best model by average precision/AUC-ROC sort: `xgboost`" in report
    assert "Feature dictionary rows: 2" in report
    assert "does not write row-level predictions" in report


def test_final_summary_writer_creates_parent_directory(tmp_path):
    suite, feature_dictionary = _write_suite_outputs(tmp_path)
    output_path = tmp_path / "nested" / "final_summary.md"

    written = write_final_summary(
        model_suite_dir=suite,
        output_path=output_path,
        feature_dictionary_path=feature_dictionary,
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith(
        "# Final Deliverable Summary"
    )
