"""Channel definitions for 15-minute first-6-hour temporal tensors."""

from __future__ import annotations


VITAL_CHANNEL_BASES = (
    "heart_rate",
    "resp_rate",
    "map",
    "temp",
    "sbp",
    "dbp",
    "spo2",
)

LAB_CHANNEL_BASES = (
    "bun",
    "alkaline_phosphatase",
    "albumin",
    "bilirubin",
    "creatinine",
    "glucose",
    "platelets",
    "hemoglobin",
    "wbc",
    "sodium",
    "potassium",
    "lactate",
    "hematocrit",
    "chloride",
    "bicarbonate",
    "anion_gap",
    "inr",
)

DERIVED_BIN_CHANNELS = (
    "shock_index_bin",
    "hypotension_flag_bin",
    "hypoxemia_flag_bin",
    "sirs_count_bin",
    "critical_value_count_bin",
)

TEMPORAL_CHANNELS = tuple(
    [f"{base}_{agg}" for base in VITAL_CHANNEL_BASES for agg in ("last", "min", "max")]
    + [f"{base}_last" for base in LAB_CHANNEL_BASES]
    + list(DERIVED_BIN_CHANNELS)
)

assert len(TEMPORAL_CHANNELS) == 43
