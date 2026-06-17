"""First-6-hour timestamp filtering helpers."""

from __future__ import annotations

import pandas as pd


def filter_events_to_observation_window(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str,
    event_time_col: str,
    intime_col: str = "intime",
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Return events occurring from ICU admission through the window end.

    The upper boundary is inclusive: events exactly at ICU admission + 6 hours
    are allowed; later events are excluded.
    """
    required_event_cols = {id_col, event_time_col}
    required_cohort_cols = {id_col, intime_col}
    missing_event_cols = required_event_cols - set(events.columns)
    missing_cohort_cols = required_cohort_cols - set(cohort.columns)
    if missing_event_cols:
        raise ValueError(f"events missing columns: {sorted(missing_event_cols)}")
    if missing_cohort_cols:
        raise ValueError(f"cohort missing columns: {sorted(missing_cohort_cols)}")

    cohort_window = cohort[[id_col, intime_col]].copy()
    cohort_window[intime_col] = pd.to_datetime(cohort_window[intime_col])
    cohort_window["window_end"] = cohort_window[intime_col] + pd.to_timedelta(
        window_hours, unit="h"
    )

    merged = events.copy()
    merged[event_time_col] = pd.to_datetime(merged[event_time_col])
    merged = merged.merge(cohort_window, on=id_col, how="inner")
    mask = (
        (merged[event_time_col] >= merged[intime_col])
        & (merged[event_time_col] <= merged["window_end"])
    )
    return merged.loc[mask, events.columns].copy()
