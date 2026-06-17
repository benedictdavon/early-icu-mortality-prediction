"""First-6-hour trajectory feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token


def _safe_percent_change(first: float, last: float) -> float:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return np.nan
    return float((last - first) / abs(first))


def _linear_slope(hours: pd.Series, values: pd.Series) -> float:
    x = hours.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or len(np.unique(x[valid])) < 2:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def compute_trajectory_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    event_time_col: str = "charttime",
    variable_col: str = "variable",
    value_col: str = "valuenum",
    intime_col: str = "intime",
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Summarize first/last movement for long-form first-window events."""
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

    windowed["_feature_variable"] = windowed[variable_col].map(sanitize_feature_token)

    feature_rows: dict[object, dict[str, float]] = {}
    for (stay_id, variable), group in windowed.groupby(
        [id_col, "_feature_variable"], sort=False
    ):
        group = group.sort_values(event_time_col)
        values = group[value_col]
        hours = group["_hours_from_admit"]
        first = float(values.iloc[0])
        last = float(values.iloc[-1])
        first_2h = values[hours <= 2.0].mean()
        last_2h = values[hours >= max(0.0, window_hours - 2.0)].mean()

        row = feature_rows.setdefault(stay_id, {})
        row[f"{variable}_first"] = first
        row[f"{variable}_last"] = last
        row[f"{variable}_last_minus_first"] = float(last - first)
        row[f"{variable}_percent_change"] = _safe_percent_change(first, last)
        row[f"{variable}_first_2h_mean"] = float(first_2h) if pd.notna(first_2h) else np.nan
        row[f"{variable}_last_2h_mean"] = float(last_2h) if pd.notna(last_2h) else np.nan
        row[f"{variable}_last2h_minus_first2h"] = (
            float(last_2h - first_2h)
            if pd.notna(first_2h) and pd.notna(last_2h)
            else np.nan
        )
        row[f"{variable}_slope_0_6h"] = _linear_slope(hours, values)

    features = pd.DataFrame.from_dict(feature_rows, orient="index")
    features.index.name = id_col
    return result.merge(features.reset_index(), on=id_col, how="left")
