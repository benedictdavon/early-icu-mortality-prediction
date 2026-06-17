from __future__ import annotations

import pandas as pd

from feature_extraction.expanded_features import (
    add_expanded_derived_features,
    build_expanded_long_events,
    expanded_features_enabled,
    extract_expanded_event_features,
)


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2],
            "subject_id": [100, 101],
            "intime": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:00"]
            ),
            "window_end": pd.to_datetime(
                ["2025-01-01 06:00:00", "2025-01-01 06:00:00"]
            ),
        }
    )


def _chart_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2],
            "charttime": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 06:00:00",
                    "2025-01-01 06:00:01",
                    "2025-01-01 07:00:00",
                ]
            ),
            "itemid": [220045, 220045, 220045, 220045],
            "valuenum": [80.0, 100.0, 999.0, 250.0],
        }
    )


def _write_lab_data(tmp_path):
    hosp_path = tmp_path / "hosp"
    hosp_path.mkdir()
    labs = pd.DataFrame(
        {
            "subject_id": [100, 100, 101],
            "charttime": [
                "2025-01-01 05:30:00",
                "2025-01-01 06:30:00",
                "2025-01-01 05:00:00",
            ],
            "itemid": [50813, 50813, 50813],
            "valuenum": [2.5, 9.9, 1.0],
        }
    )
    labs.to_csv(hosp_path / "_labevents.csv", index=False)
    return hosp_path


def test_expanded_features_enabled_uses_override_before_config():
    assert expanded_features_enabled({"feature_engineering": {"enabled": False}}, True)
    assert not expanded_features_enabled(
        {"feature_engineering": {"enabled": True}},
        False,
    )
    assert expanded_features_enabled({"feature_engineering": {"enabled": True}})
    assert not expanded_features_enabled({"feature_engineering": {"enabled": False}})


def test_expanded_feature_events_convert_raw_mimic_rows(tmp_path):
    hosp_path = _write_lab_data(tmp_path)

    events = build_expanded_long_events(_cohort(), _chart_data(), str(hosp_path))

    assert {"stay_id", "charttime", "variable", "source", "valuenum"}.issubset(
        events.columns
    )
    assert set(events["variable"]) == {"heart_rate", "lactate"}
    assert set(events["source"]) == {"vital", "lab"}


def test_expanded_event_features_are_first_6h_only(tmp_path):
    hosp_path = _write_lab_data(tmp_path)

    result = extract_expanded_event_features(
        _cohort(),
        _chart_data(),
        str(hosp_path),
        config={"feature_groups": {}},
    )

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]

    assert stay_1["heart_rate_bin_5_6h_last"] == 100.0
    assert stay_1["heart_rate_last"] == 100.0
    assert stay_1["heart_rate_measurement_count_0_6h"] == 2
    assert stay_1["lactate_last"] == 2.5
    assert 999.0 not in set(stay_1.drop(labels=["stay_id"]).dropna())

    assert stay_2["heart_rate_measurement_count_0_6h"] == 0
    assert stay_2["lactate_measurement_count_0_6h"] == 1


def test_expanded_derived_features_validate_provenance():
    features = pd.DataFrame(
        {
            "stay_id": [1],
            "age": [70],
            "prev_dx_count_total": [2],
            "shock_index": [1.2],
            "map_min": [55],
            "spo2_min": [90],
            "resp_rate_max": [34],
            "resp_rate_mean": [26],
            "creatinine_max": [2.1],
            "lactate_max": [3.0],
        }
    )

    result = add_expanded_derived_features(features, config={"feature_groups": {}})

    assert result.loc[0, "cardiovascular_dysfunction"] == 1
    assert result.loc[0, "respiratory_dysfunction"] == 1
    assert result.loc[0, "renal_dysfunction"] == 1
    assert result.loc[0, "metabolic_dysfunction"] == 1
    assert result.loc[0, "age_x_prev_dx_count"] == 140
    assert result.loc[0, "age_x_shock_index"] == 84
