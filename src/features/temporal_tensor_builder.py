"""Build 15-minute first-6-hour temporal tensors from long-form events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token
from features.temporal_channels import TEMPORAL_CHANNELS


def _channel_base_and_aggregation(channel: str) -> tuple[str, str]:
    if channel.endswith("_bin"):
        return channel, "last"
    for suffix in ("_last", "_min", "_max"):
        if channel.endswith(suffix):
            return channel[: -len(suffix)], suffix[1:]
    return channel, "last"


def build_15min_temporal_tensors(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    channels: tuple[str, ...] = TEMPORAL_CHANNELS,
    id_col: str = "stay_id",
    event_time_col: str = "charttime",
    variable_col: str = "variable",
    value_col: str = "valuenum",
    intime_col: str = "intime",
    window_hours: float = 6.0,
    bin_minutes: int = 15,
) -> dict:
    """Return raw temporal value, mask, delta, and count tensors.

    Shapes are `(n_stays, n_bins, n_channels)`. Missing values are stored as
    zero in `x_temporal`; the corresponding `mask_temporal` entry is zero.
    """
    required_cols = {id_col, event_time_col, variable_col, value_col}
    missing_cols = required_cols - set(events.columns)
    if missing_cols:
        raise ValueError(f"events missing columns: {sorted(missing_cols)}")

    stay_ids = cohort[id_col].drop_duplicates().tolist()
    stay_to_index = {stay_id: idx for idx, stay_id in enumerate(stay_ids)}
    bin_hours = bin_minutes / 60.0
    n_bins = int(round(window_hours / bin_hours))
    n_channels = len(channels)

    values = np.zeros((len(stay_ids), n_bins, n_channels), dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.float32)
    counts = np.zeros_like(values, dtype=np.float32)

    windowed = prepare_windowed_events(
        events,
        cohort,
        id_col=id_col,
        event_time_col=event_time_col,
        intime_col=intime_col,
        window_hours=window_hours,
    ).dropna(subset=[variable_col, value_col])

    if not windowed.empty:
        windowed["_feature_variable"] = windowed[variable_col].map(sanitize_feature_token)
        windowed["_bin_index"] = np.floor(windowed["_hours_from_admit"] / bin_hours).astype(int)
        windowed["_bin_index"] = windowed["_bin_index"].clip(0, n_bins - 1)

        for channel_index, channel in enumerate(channels):
            base, aggregation = _channel_base_and_aggregation(channel)
            base_token = sanitize_feature_token(base)
            channel_events = windowed[
                (windowed["_feature_variable"] == base_token)
                | (windowed["_feature_variable"] == sanitize_feature_token(channel))
            ]
            if channel_events.empty:
                continue
            for (stay_id, bin_index), group in channel_events.groupby(
                [id_col, "_bin_index"], sort=False
            ):
                stay_index = stay_to_index.get(stay_id)
                if stay_index is None:
                    continue
                series = group.sort_values(event_time_col)[value_col].astype(float)
                if aggregation == "min":
                    value = series.min()
                elif aggregation == "max":
                    value = series.max()
                else:
                    value = series.iloc[-1]
                values[stay_index, int(bin_index), channel_index] = float(value)
                mask[stay_index, int(bin_index), channel_index] = 1.0
                counts[stay_index, int(bin_index), channel_index] = float(len(series))

    deltas = np.zeros_like(values, dtype=np.float32)
    for stay_index in range(len(stay_ids)):
        for channel_index in range(n_channels):
            since_last = 0.0
            seen = False
            for bin_index in range(n_bins):
                if mask[stay_index, bin_index, channel_index] == 1:
                    since_last = 0.0
                    seen = True
                else:
                    since_last = since_last + bin_hours if seen else (bin_index + 1) * bin_hours
                deltas[stay_index, bin_index, channel_index] = since_last

    return {
        "stay_ids": np.asarray(stay_ids),
        "channels": list(channels),
        "x_temporal": values,
        "mask_temporal": mask,
        "delta_temporal": deltas,
        "count_temporal": counts,
        "bin_minutes": int(bin_minutes),
        "window_hours": float(window_hours),
    }


class TemporalTensorNormalizer:
    """Fit per-channel normalization on observed tensor values only."""

    def __init__(self, clip_value: float = 5.0):
        self.clip_value = float(clip_value)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x_temporal: np.ndarray, mask_temporal: np.ndarray) -> "TemporalTensorNormalizer":
        x = np.asarray(x_temporal, dtype=np.float32)
        mask = np.asarray(mask_temporal, dtype=bool)
        n_channels = x.shape[-1]
        mean = np.zeros(n_channels, dtype=np.float32)
        std = np.ones(n_channels, dtype=np.float32)
        for channel_index in range(n_channels):
            observed = x[..., channel_index][mask[..., channel_index]]
            if observed.size:
                mean[channel_index] = float(observed.mean())
                channel_std = float(observed.std())
                std[channel_index] = channel_std if channel_std > 1e-6 else 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x_temporal: np.ndarray, mask_temporal: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("TemporalTensorNormalizer must be fit before transform")
        x = np.asarray(x_temporal, dtype=np.float32)
        mask = np.asarray(mask_temporal, dtype=np.float32)
        transformed = (x - self.mean_) / self.std_
        transformed = np.clip(transformed, -self.clip_value, self.clip_value)
        return transformed * mask

    def transform_bundle(self, bundle: dict) -> dict:
        normalized = dict(bundle)
        normalized["x_temporal"] = self.transform(
            bundle["x_temporal"],
            bundle["mask_temporal"],
        )
        normalized["delta_temporal"] = np.log1p(bundle["delta_temporal"]).astype(np.float32)
        normalized["count_temporal"] = np.log1p(bundle["count_temporal"]).astype(np.float32)
        return normalized


def fit_transform_temporal_bundle(bundle: dict) -> tuple[dict, TemporalTensorNormalizer]:
    normalizer = TemporalTensorNormalizer().fit(
        bundle["x_temporal"],
        bundle["mask_temporal"],
    )
    return normalizer.transform_bundle(bundle), normalizer
