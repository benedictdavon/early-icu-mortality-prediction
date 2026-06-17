from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.interactions import compute_clinical_interaction_features
from features.instability import compute_instability_features
from features.measurement_process import compute_measurement_process_features
from features.organ_dysfunction import compute_organ_dysfunction_features
from features.time_bins import build_hourly_time_bin_features
from features.trajectory import compute_trajectory_features
from preprocessing.feature_engineering import extract_temporal_features


def _expanded_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [100, 101, 102],
            "intime": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:00",
                ]
            ),
        }
    )


def _expanded_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 2, 2],
            "charttime": pd.to_datetime(
                [
                    "2025-01-01 00:15:00",
                    "2025-01-01 01:30:00",
                    "2025-01-01 06:00:00",
                    "2025-01-01 06:00:01",
                    "2025-01-01 05:30:00",
                    "2025-01-01 00:30:00",
                    "2025-01-01 07:00:00",
                ]
            ),
            "variable": [
                "heart_rate",
                "heart_rate",
                "heart_rate",
                "heart_rate",
                "lactate",
                "heart_rate",
                "heart_rate",
            ],
            "source": ["vital", "vital", "vital", "vital", "lab", "vital", "vital"],
            "valuenum": [80.0, 100.0, 120.0, 999.0, 2.0, 70.0, 250.0],
        }
    )


def test_hourly_time_bins_exclude_post_6h_and_keep_exact_6h():
    result = build_hourly_time_bin_features(_expanded_events(), _expanded_cohort())

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]
    stay_3 = result[result["stay_id"] == 3].iloc[0]

    assert stay_1["heart_rate_bin_0_1h_mean"] == 80.0
    assert stay_1["heart_rate_bin_1_2h_last"] == 100.0
    assert stay_1["heart_rate_bin_5_6h_max"] == 120.0
    assert stay_1["heart_rate_bin_5_6h_count"] == 1
    assert 999.0 not in set(stay_1.drop(labels=["stay_id"]).dropna())

    assert stay_2["heart_rate_bin_0_1h_count"] == 1
    assert stay_2["heart_rate_bin_5_6h_observed"] == 0
    assert stay_3["heart_rate_bin_0_1h_count"] == 0


def test_trajectory_features_use_only_first_6h_events():
    result = compute_trajectory_features(_expanded_events(), _expanded_cohort())

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]

    assert stay_1["heart_rate_first"] == 80.0
    assert stay_1["heart_rate_last"] == 120.0
    assert stay_1["heart_rate_last_minus_first"] == 40.0
    assert stay_1["heart_rate_percent_change"] == 0.5
    assert stay_1["heart_rate_first_2h_mean"] == 90.0
    assert stay_1["heart_rate_last_2h_mean"] == 120.0
    assert stay_1["heart_rate_last2h_minus_first2h"] == 30.0
    assert stay_1["heart_rate_slope_0_6h"] > 0
    assert stay_1["heart_rate_deterioration_flag"] == 1
    assert stay_1["heart_rate_recovery_flag"] == 0

    assert stay_2["heart_rate_last"] == 70.0
    assert np.isnan(stay_2["heart_rate_slope_0_6h"])


def test_measurement_process_features_exclude_post_window_counts():
    result = compute_measurement_process_features(
        _expanded_events(),
        _expanded_cohort(),
        source_col="source",
        expected_variables=["heart_rate", "lactate", "creatinine"],
    )

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]
    stay_3 = result[result["stay_id"] == 3].iloc[0]

    assert stay_1["heart_rate_measured_0_6h"] == 1
    assert stay_1["heart_rate_measurement_count_0_6h"] == 3
    assert stay_1["heart_rate_time_to_first_measurement"] == pytest.approx(0.25)
    assert stay_1["heart_rate_time_since_last_measurement_at_6h"] == 0
    assert stay_1["total_measurements_0_6h"] == 4
    assert stay_1["total_chart_event_count_0_6h"] == 4
    assert stay_1["total_vital_measurements_0_6h"] == 3
    assert stay_1["total_lab_measurements_0_6h"] == 1
    assert stay_1["panel_missing_count_0_6h"] == 1

    assert stay_2["heart_rate_measurement_count_0_6h"] == 1
    assert stay_2["heart_rate_time_since_last_measurement_at_6h"] == pytest.approx(5.5)
    assert stay_3["heart_rate_measured_0_6h"] == 0
    assert np.isnan(stay_3["heart_rate_time_to_first_measurement"])


def test_instability_features_capture_variability_and_abnormal_runs():
    result = compute_instability_features(_expanded_events(), _expanded_cohort())
    stay_1 = result[result["stay_id"] == 1].iloc[0]

    assert stay_1["heart_rate_range_0_6h"] == 40.0
    assert stay_1["heart_rate_abnormal_count_0_6h"] == 1
    assert stay_1["heart_rate_longest_abnormal_run_0_6h"] == 1
    assert stay_1["heart_rate_worst_recent_value_0_6h"] == 120.0


def test_organ_dysfunction_and_interaction_features_are_deterministic():
    features = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "age": [80, 45],
            "prev_dx_count_total": [3, 0],
            "map_min": [60, 80],
            "sbp_min": [85, 115],
            "shock_index": [1.1, 0.6],
            "spo2_min": [90, 98],
            "resp_rate_max": [32, 18],
            "resp_rate_mean": [28, 16],
            "creatinine_max": [2.0, 0.9],
            "bun_max": [55, 12],
            "urine_output_total": [200, 700],
            "bilirubin_max": [1.0, 0.6],
            "inr_max": [1.6, 1.0],
            "platelets_min": [80, 220],
            "hemoglobin_min": [7.5, 13],
            "lactate_max": [4.0, 1.0],
            "lactate_mean": [3.0, 1.0],
            "critical_value_count": [3, 0],
            "panel_missing_count_0_6h": [5, 1],
            "sirs_criteria_count": [2, 0],
            "bicarbonate_min": [16, 24],
            "anion_gap_max": [18, 11],
        }
    )

    result = compute_organ_dysfunction_features(features)
    result = compute_clinical_interaction_features(result)

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]

    assert stay_1["organ_dysfunction_count"] == 6
    assert stay_2["organ_dysfunction_count"] == 0
    assert stay_1["age_x_prev_dx_count"] == 240
    assert stay_1["age_x_shock_index"] == pytest.approx(88.0)
    assert stay_1["lactate_x_hypotension"] == 4.0
    assert stay_1["resp_rate_x_spo2_deficit"] == 280
    assert stay_1["platelets_x_inr"] == pytest.approx(128.0)
    assert stay_1["sirs_x_lactate"] == 6.0
    assert stay_1["creatinine_x_urine_output"] == pytest.approx(400.0)
    assert stay_1["bilirubin_x_inr"] == pytest.approx(1.6)
    assert stay_1["critical_count_x_missing_lab_count"] == 15


def test_temporal_extraction_ignores_numeric_hourly_measurements():
    df = pd.DataFrame(
        {
            "intime": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "heart_rate_hour_0": [80.0, 95.0],
            "heart_rate_hour_1": [85.0, 100.0],
            "duration_hours": [6.0, 6.0],
        }
    )

    result = extract_temporal_features(df)

    assert "intime_hour" in result.columns
    assert "heart_rate_hour_0_hour" not in result.columns
    assert "hours_since_heart_rate_hour_0_to_heart_rate_hour_1" not in result.columns
