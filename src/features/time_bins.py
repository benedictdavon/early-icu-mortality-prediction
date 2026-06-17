"""Hourly first-6-hour bin feature builders."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token


SUPPORTED_AGGREGATIONS = {"mean", "min", "max", "last"}


def _format_hour(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_hourly_time_bin_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    event_time_col: str = "charttime",
    variable_col: str = "variable",
    value_col: str = "valuenum",
    intime_col: str = "intime",
    window_hours: float = 6.0,
    bin_hours: float = 1.0,
    aggregations: tuple[str, ...] = ("mean", "min", "max", "last"),
    include_observed: bool = True,
    include_count: bool = True,
) -> pd.DataFrame:
    """Create per-stay hourly bins from long-form first-window events.

    Bins are left-closed and right-open except the final bin, which includes
    events exactly at `window_hours`.
    """
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if bin_hours <= 0:
        raise ValueError("bin_hours must be positive")

    unknown_aggs = set(aggregations) - SUPPORTED_AGGREGATIONS
    if unknown_aggs:
        raise ValueError(f"unsupported aggregations: {sorted(unknown_aggs)}")

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

    n_bins = int(math.ceil(window_hours / bin_hours))
    bin_index = np.floor(windowed["_hours_from_admit"] / bin_hours).astype(int)
    windowed["_bin_index"] = np.minimum(bin_index, n_bins - 1)
    windowed["_feature_variable"] = windowed[variable_col].map(sanitize_feature_token)

    feature_rows: dict[object, dict[str, float]] = {}
    group_cols = [id_col, "_feature_variable", "_bin_index"]
    for (stay_id, variable, bin_idx), group in windowed.groupby(group_cols, sort=False):
        group = group.sort_values(event_time_col)
        start_hour = bin_idx * bin_hours
        end_hour = min((bin_idx + 1) * bin_hours, window_hours)
        base = f"{variable}_bin_{_format_hour(start_hour)}_{_format_hour(end_hour)}h"
        row = feature_rows.setdefault(stay_id, {})

        values = group[value_col]
        if "mean" in aggregations:
            row[f"{base}_mean"] = float(values.mean())
        if "min" in aggregations:
            row[f"{base}_min"] = float(values.min())
        if "max" in aggregations:
            row[f"{base}_max"] = float(values.max())
        if "last" in aggregations:
            row[f"{base}_last"] = float(values.iloc[-1])
        if include_observed:
            row[f"{base}_observed"] = 1
        if include_count:
            row[f"{base}_count"] = int(values.count())

    features = pd.DataFrame.from_dict(feature_rows, orient="index")
    features.index.name = id_col
    features = features.reset_index()
    result = result.merge(features, on=id_col, how="left")

    indicator_cols = [
        col
        for col in result.columns
        if col.endswith("_observed") or col.endswith("_count")
    ]
    for col in indicator_cols:
        result[col] = result[col].fillna(0).astype(int)

    return result
