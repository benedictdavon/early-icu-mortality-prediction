"""Instability features for first-6-hour clinical events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token


DEFAULT_NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "resp_rate": (12, 20),
    "map": (65, 110),
    "sbp": (90, 140),
    "dbp": (50, 90),
    "spo2": (92, 100),
    "temp": (36, 38),
    "lactate": (0, 2),
    "creatinine": (0, 1.5),
    "bun": (0, 25),
    "bilirubin": (0, 1.2),
    "inr": (0.8, 1.2),
    "platelets": (150, 450),
    "wbc": (4, 11),
    "hemoglobin": (12, 18),
    "bicarbonate": (22, 29),
    "anion_gap": (8, 16),
}


def _abnormal_mask(variable: str, values: pd.Series, ranges: dict) -> pd.Series:
    limits = ranges.get(variable)
    if limits is None:
        return pd.Series(False, index=values.index)
    low, high = limits
    return (values < low) | (values > high)


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def compute_instability_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    event_time_col: str = "charttime",
    variable_col: str = "variable",
    value_col: str = "valuenum",
    intime_col: str = "intime",
    window_hours: float = 6.0,
    normal_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compute variability and abnormal-run features inside the first window."""
    required_cols = {id_col, event_time_col, variable_col, value_col}
    missing_cols = required_cols - set(events.columns)
    if missing_cols:
        raise ValueError(f"events missing columns: {sorted(missing_cols)}")

    result = cohort[[id_col]].drop_duplicates().copy()
    windowed = prepare_windowed_events(
        events,
        cohort,
        id_col=id_col,
        event_time_col=event_time_col,
        intime_col=intime_col,
        window_hours=window_hours,
    ).dropna(subset=[variable_col, value_col])

    if windowed.empty:
        return result

    ranges = normal_ranges or DEFAULT_NORMAL_RANGES
    windowed["_feature_variable"] = windowed[variable_col].map(sanitize_feature_token)
    feature_rows: dict[object, dict[str, float]] = {}

    for (stay_id, variable), group in windowed.groupby(
        [id_col, "_feature_variable"], sort=False
    ):
        group = group.sort_values(event_time_col)
        values = group[value_col].astype(float)
        abnormal = _abnormal_mask(variable, values, ranges)
        recent = group[group["_hours_from_admit"] >= max(0.0, window_hours - 2.0)]
        recent_values = recent[value_col].astype(float)

        mean = values.mean()
        row = feature_rows.setdefault(stay_id, {})
        row[f"{variable}_range_0_6h"] = float(values.max() - values.min())
        row[f"{variable}_std_0_6h"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        row[f"{variable}_cv_0_6h"] = (
            float(values.std(ddof=0) / abs(mean))
            if len(values) > 1 and np.isfinite(mean) and mean != 0
            else np.nan
        )
        row[f"{variable}_abnormal_count_0_6h"] = int(abnormal.sum())
        row[f"{variable}_longest_abnormal_run_0_6h"] = _longest_true_run(abnormal.tolist())
        row[f"{variable}_worst_recent_value_0_6h"] = (
            float(recent_values.max()) if not recent_values.empty else np.nan
        )

    features = pd.DataFrame.from_dict(feature_rows, orient="index")
    features.index.name = id_col
    return result.merge(features.reset_index(), on=id_col, how="left")
