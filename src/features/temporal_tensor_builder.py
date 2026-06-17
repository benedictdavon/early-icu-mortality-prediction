"""Build 15-minute first-6-hour temporal tensors from long-form events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._events import prepare_windowed_events, sanitize_feature_token
from features.temporal_channels import TEMPORAL_CHANNELS


VARIABLE_ALIASES = {
    "resp_rate": "respiratory_rate",
    "respiratory_rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "temp": "temperature",
    "temperature": "temperature",
    "platelet": "platelets",
    "platelets": "platelets",
    "alk_phos": "alkaline_phosphatase",
    "alkaline_phosphatase": "alkaline_phosphatase",
    "ast": "ast",
    "alt": "alt",
}

DERIVED_CHANNELS = {
    "shock_index_bin",
    "hypotension_flag_bin",
    "hypoxemia_flag_bin",
    "sirs_count_bin",
    "critical_value_count_bin",
}


def _channel_base_and_aggregation(channel: str) -> tuple[str, str]:
    if channel.endswith("_bin"):
        return channel, "last"
    for suffix in ("_last", "_min", "_max"):
        if channel.endswith(suffix):
            return channel[: -len(suffix)], suffix[1:]
    return channel, "last"


def _canonical_variable(value) -> str:
    token = sanitize_feature_token(value)
    return VARIABLE_ALIASES.get(token, token)


def _channel_index(channels: tuple[str, ...] | list[str], channel: str) -> int | None:
    try:
        return list(channels).index(channel)
    except ValueError:
        return None


def _observed_value(
    values: np.ndarray,
    mask: np.ndarray,
    stay_index: int,
    bin_index: int,
    channel_index: int | None,
) -> tuple[bool, float]:
    if channel_index is None:
        return False, 0.0
    observed = mask[stay_index, bin_index, channel_index] == 1
    return bool(observed), float(values[stay_index, bin_index, channel_index])


def _set_derived(
    values: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    stay_index: int,
    bin_index: int,
    channel_index: int | None,
    value: float,
    observed: bool,
) -> None:
    if channel_index is None or not observed:
        return
    values[stay_index, bin_index, channel_index] = float(value)
    mask[stay_index, bin_index, channel_index] = 1.0
    counts[stay_index, bin_index, channel_index] = 1.0


def _compute_derived_channels(
    values: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    channels: tuple[str, ...],
) -> None:
    idx = {channel: _channel_index(channels, channel) for channel in channels}
    epsilon = 1e-6

    for stay_index in range(values.shape[0]):
        for bin_index in range(values.shape[1]):
            hr_obs, hr = _observed_value(
                values, mask, stay_index, bin_index, idx.get("heart_rate_last")
            )
            sbp_obs, sbp = _observed_value(
                values, mask, stay_index, bin_index, idx.get("sbp_last")
            )
            sbp_min_obs, sbp_min = _observed_value(
                values, mask, stay_index, bin_index, idx.get("sbp_min")
            )
            map_min_obs, map_min = _observed_value(
                values, mask, stay_index, bin_index, idx.get("map_min")
            )
            spo2_min_obs, spo2_min = _observed_value(
                values, mask, stay_index, bin_index, idx.get("spo2_min")
            )
            rr_obs, rr = _observed_value(
                values, mask, stay_index, bin_index, idx.get("respiratory_rate_last")
            )
            temp_obs, temp = _observed_value(
                values, mask, stay_index, bin_index, idx.get("temperature_last")
            )
            wbc_obs, wbc = _observed_value(
                values, mask, stay_index, bin_index, idx.get("wbc_last")
            )

            _set_derived(
                values,
                mask,
                counts,
                stay_index,
                bin_index,
                idx.get("shock_index_bin"),
                hr / max(abs(sbp), epsilon),
                hr_obs and sbp_obs,
            )
            _set_derived(
                values,
                mask,
                counts,
                stay_index,
                bin_index,
                idx.get("hypotension_flag_bin"),
                float((sbp_min_obs and sbp_min < 90) or (map_min_obs and map_min < 65)),
                sbp_min_obs or map_min_obs,
            )
            _set_derived(
                values,
                mask,
                counts,
                stay_index,
                bin_index,
                idx.get("hypoxemia_flag_bin"),
                float(spo2_min < 90),
                spo2_min_obs,
            )

            sirs_count = 0
            if temp_obs:
                sirs_count += int(temp > 38 or temp < 36)
            if hr_obs:
                sirs_count += int(hr > 90)
            if rr_obs:
                sirs_count += int(rr > 20)
            if wbc_obs:
                sirs_count += int(wbc > 12 or wbc < 4)
            _set_derived(
                values,
                mask,
                counts,
                stay_index,
                bin_index,
                idx.get("sirs_count_bin"),
                float(sirs_count),
                temp_obs or hr_obs or rr_obs or wbc_obs,
            )

            critical = 0
            any_checked = False

            def add_if(channel: str, predicate) -> None:
                nonlocal critical, any_checked
                observed, value = _observed_value(
                    values, mask, stay_index, bin_index, idx.get(channel)
                )
                if observed:
                    any_checked = True
                    critical += int(predicate(value))

            add_if("heart_rate_last", lambda value: value < 40 or value > 130)
            add_if("respiratory_rate_last", lambda value: value < 8 or value > 30)
            add_if("sbp_min", lambda value: value < 90)
            add_if("map_min", lambda value: value < 65)
            add_if("temperature_last", lambda value: value < 36 or value > 38.5)
            add_if("spo2_min", lambda value: value < 90)
            add_if("wbc_last", lambda value: value < 4 or value > 12)
            add_if("hemoglobin_last", lambda value: value < 8)
            add_if("platelets_last", lambda value: value < 100)
            add_if("sodium_last", lambda value: value < 130 or value > 150)
            add_if("potassium_last", lambda value: value < 3 or value > 5.5)
            add_if("creatinine_last", lambda value: value > 2)
            add_if("bun_last", lambda value: value > 40)
            add_if("glucose_last", lambda value: value < 70 or value > 180)
            add_if("bilirubin_last", lambda value: value > 2)
            add_if("inr_last", lambda value: value > 1.5)
            add_if("lactate_last", lambda value: value > 2)
            add_if("bicarbonate_last", lambda value: value < 18)
            add_if("anion_gap_last", lambda value: value > 16)

            _set_derived(
                values,
                mask,
                counts,
                stay_index,
                bin_index,
                idx.get("critical_value_count_bin"),
                float(critical),
                any_checked,
            )


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
        windowed["_feature_variable"] = windowed[variable_col].map(_canonical_variable)
        windowed["_bin_index"] = np.floor(windowed["_hours_from_admit"] / bin_hours).astype(int)
        windowed["_bin_index"] = windowed["_bin_index"].clip(0, n_bins - 1)

        for channel_index, channel in enumerate(channels):
            base, aggregation = _channel_base_and_aggregation(channel)
            if base in DERIVED_CHANNELS:
                continue
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

    _compute_derived_channels(values, mask, counts, tuple(channels))

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
                    since_last = since_last + bin_hours if seen else window_hours
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

    def transform_bundle(
        self,
        bundle: dict,
        *,
        delta_clip_hours: float = 6.0,
        count_clip: float = 10.0,
    ) -> dict:
        normalized = dict(bundle)
        normalized["x_temporal"] = self.transform(
            bundle["x_temporal"],
            bundle["mask_temporal"],
        )
        normalized["delta_temporal"] = (
            np.log1p(np.minimum(bundle["delta_temporal"], delta_clip_hours))
            / np.log1p(delta_clip_hours)
        ).astype(np.float32)
        normalized["count_temporal"] = (
            np.log1p(np.minimum(bundle["count_temporal"], count_clip))
            / np.log1p(count_clip)
        ).astype(np.float32)
        return normalized


def fit_transform_temporal_bundle(bundle: dict) -> tuple[dict, TemporalTensorNormalizer]:
    normalizer = TemporalTensorNormalizer().fit(
        bundle["x_temporal"],
        bundle["mask_temporal"],
    )
    return normalizer.transform_bundle(bundle), normalizer


def transform_temporal_splits(
    split_bundles: dict[str, dict],
    *,
    train_split: str = "train",
    normalizer: TemporalTensorNormalizer | None = None,
) -> tuple[dict[str, dict], TemporalTensorNormalizer]:
    """Fit on the training bundle and transform every split with one normalizer."""
    if train_split not in split_bundles:
        raise ValueError(f"split_bundles must contain `{train_split}`")
    fitted = normalizer or TemporalTensorNormalizer().fit(
        split_bundles[train_split]["x_temporal"],
        split_bundles[train_split]["mask_temporal"],
    )
    return {
        split_name: fitted.transform_bundle(bundle)
        for split_name, bundle in split_bundles.items()
    }, fitted
