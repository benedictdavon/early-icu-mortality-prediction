"""Measurement-process features for first-6-hour clinical events."""

from __future__ import annotations

import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token


def compute_measurement_process_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    event_time_col: str = "charttime",
    variable_col: str = "variable",
    source_col: str | None = None,
    intime_col: str = "intime",
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Create measurement count, timing, and availability features."""
    required_cols = {id_col, event_time_col, variable_col}
    if source_col is not None:
        required_cols.add(source_col)
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
    ).dropna(subset=[variable_col])

    if windowed.empty:
        return result

    windowed["_feature_variable"] = windowed[variable_col].map(sanitize_feature_token)

    feature_rows: dict[object, dict[str, float]] = {}
    for (stay_id, variable), group in windowed.groupby(
        [id_col, "_feature_variable"], sort=False
    ):
        hours = group["_hours_from_admit"]
        row = feature_rows.setdefault(stay_id, {})
        row[f"{variable}_measured_0_6h"] = 1
        row[f"{variable}_measurement_count_0_6h"] = int(len(group))
        row[f"{variable}_time_to_first_measurement"] = float(hours.min())
        row[f"{variable}_time_since_last_measurement_at_6h"] = float(
            window_hours - hours.max()
        )

    for stay_id, group in windowed.groupby(id_col, sort=False):
        row = feature_rows.setdefault(stay_id, {})
        row["total_measurements_0_6h"] = int(len(group))

    if source_col is not None:
        windowed["_feature_source"] = windowed[source_col].map(sanitize_feature_token)
        for (stay_id, source), group in windowed.groupby(
            [id_col, "_feature_source"], sort=False
        ):
            row = feature_rows.setdefault(stay_id, {})
            row[f"total_{source}_measurements_0_6h"] = int(len(group))

    features = pd.DataFrame.from_dict(feature_rows, orient="index")
    features.index.name = id_col
    result = result.merge(features.reset_index(), on=id_col, how="left")

    count_cols = [
        col
        for col in result.columns
        if col.endswith("_measurement_count_0_6h")
        or col.endswith("_measurements_0_6h")
        or col.endswith("_measured_0_6h")
    ]
    for col in count_cols:
        result[col] = result[col].fillna(0).astype(int)

    return result
