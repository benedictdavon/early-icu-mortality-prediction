"""Shared event-table helpers for first-6-hour feature builders."""

from __future__ import annotations

import re

import pandas as pd

from features.temporal_window import filter_events_to_observation_window


def sanitize_feature_token(value) -> str:
    """Return a stable, lowercase token suitable for feature column names."""
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "feature"


def prepare_windowed_events(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str,
    event_time_col: str,
    intime_col: str = "intime",
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Filter events to the observation window and add hours-from-admit.

    The window is inclusive at the upper boundary, so an event exactly at
    ICU admission + 6 hours remains eligible.
    """
    filtered = filter_events_to_observation_window(
        events,
        cohort,
        id_col=id_col,
        event_time_col=event_time_col,
        intime_col=intime_col,
        window_hours=window_hours,
    )

    cohort_times = cohort[[id_col, intime_col]].drop_duplicates(subset=[id_col]).copy()
    cohort_times[intime_col] = pd.to_datetime(cohort_times[intime_col])

    windowed = filtered.drop(columns=[intime_col], errors="ignore").merge(
        cohort_times,
        on=id_col,
        how="inner",
    )
    windowed[event_time_col] = pd.to_datetime(windowed[event_time_col])
    windowed["_hours_from_admit"] = (
        windowed[event_time_col] - windowed[intime_col]
    ).dt.total_seconds() / 3600.0

    mask = (
        (windowed["_hours_from_admit"] >= 0)
        & (windowed["_hours_from_admit"] <= window_hours)
    )
    return windowed.loc[mask].sort_values([id_col, event_time_col]).reset_index(drop=True)
