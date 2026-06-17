from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_extraction.time_windows import extract_early_window_values
from feature_extraction.vitals import process_vital_batch
from features.temporal_window import filter_events_to_observation_window


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_data():
    cohort = pd.read_csv(FIXTURE_DIR / "synthetic_cohort.csv")
    events = pd.read_csv(FIXTURE_DIR / "synthetic_events.csv")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=6)
    events["charttime"] = pd.to_datetime(events["charttime"])
    return cohort, events


def test_filter_events_to_observation_window_excludes_post_6h_events():
    cohort, events = _load_fixture_data()

    filtered = filter_events_to_observation_window(
        events,
        cohort,
        id_col="stay_id",
        event_time_col="charttime",
        window_hours=6,
    )

    assert set(filtered["valuenum"]) == {80, 90, 100}
    assert 200 not in set(filtered["valuenum"])
    assert 220 not in set(filtered["valuenum"])
    assert filtered["charttime"].max() == pd.Timestamp("2025-01-01 06:00:00")


def test_extract_early_window_values_allows_exact_6h_not_after_6h():
    cohort, events = _load_fixture_data()

    result = extract_early_window_values(
        events,
        cohort,
        itemids=[220045],
        var_name="heart_rate",
        time_windows=[6],
    )

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]

    assert stay_1["heart_rate_hour_6"] == 100
    assert stay_2["heart_rate_hour_6"] == 90


def test_vital_batch_excludes_post_6h_values_from_aggregates():
    cohort, events = _load_fixture_data()
    cohort_dict = cohort.set_index("stay_id")[["intime", "window_end"]].to_dict("index")

    result = process_vital_batch(
        "heart_rate",
        [220045],
        events,
        patient_batch=[1, 2],
        cohort_dict=cohort_dict,
    )

    stay_1 = result[result["stay_id"] == 1].iloc[0]
    stay_2 = result[result["stay_id"] == 2].iloc[0]

    assert stay_1["heart_rate_max"] == 100
    assert stay_1["heart_rate_measured"] == 1
    assert stay_2["heart_rate_max"] == 90
