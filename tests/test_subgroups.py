from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.robustness import summarize_seed_stability
from evaluation.subgroups import derive_subgroup_columns, evaluate_subgroups


def _feature_frame() -> pd.DataFrame:
    n = 16
    return pd.DataFrame(
        {
            "age": [42, 55, 70, 84] * 4,
            "sex": ["F", "M"] * 8,
            "major_diagnosis_group": ["cardiac", "respiratory", "sepsis", "cardiac"] * 4,
            "lactate_max_0_6h": [np.nan, 1.2, 2.0, np.nan] * 4,
            "panel_missing_count_0_6h": np.arange(n),
            "total_lab_count": np.tile([1, 2, 4, 8], 4),
            "total_vital_count": np.tile([3, 4, 6, 7], 4),
            "icu_unit_type": ["MICU", "SICU"] * 8,
            "ventilation_status": [0, 1, 0, 1] * 4,
        }
    )


def test_derive_subgroup_columns_covers_supported_groups():
    groups = derive_subgroup_columns(_feature_frame())

    assert set(groups["age_group"].dropna()) == {"<50", "50-64", "65-79", "80+"}
    assert "sex" in groups
    assert "major_diagnosis_group" in groups
    assert "lactate_measured" in groups
    assert "missingness_level" in groups
    assert "measurement_intensity_quartile" in groups
    assert "icu_unit_type" in groups
    assert "ventilation_status" in groups


def test_evaluate_subgroups_returns_required_metrics_and_flags_small_groups():
    y = np.asarray([0, 0, 1, 1] * 4)
    p = np.asarray([0.05, 0.30, 0.70, 0.95] * 4)

    report = evaluate_subgroups(
        y,
        p,
        _feature_frame(),
        threshold=0.5,
        min_n=5,
        n_bins=4,
    )

    required = {
        "subgroup_variable",
        "subgroup",
        "n",
        "mortality_rate",
        "auc_roc",
        "average_precision",
        "brier_score",
        "calibration_slope",
        "calibration_intercept",
        "recall_at_screening_threshold",
        "precision_at_screening_threshold",
        "specificity_at_screening_threshold",
        "small_subgroup",
    }
    assert required.issubset(report.columns)
    assert report["small_subgroup"].any()
    assert (report["n"] > 0).all()


def test_seed_stability_summary_is_aggregate():
    summary = summarize_seed_stability(
        [
            {"seed": 1, "average_precision": 0.40, "auc_roc": 0.70},
            {"seed": 2, "average_precision": 0.44, "auc_roc": 0.72},
            {"seed": 3, "average_precision": 0.42, "auc_roc": 0.71},
        ],
        ["average_precision", "auc_roc"],
    )

    assert set(summary["metric"]) == {"average_precision", "auc_roc"}
    assert (summary["n_seeds"] == 3).all()
    assert "std" in summary
