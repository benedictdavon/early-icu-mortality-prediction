"""Focused clinical interaction features for early ICU mortality models."""

from __future__ import annotations

import pandas as pd


def compute_clinical_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add a small, interpretable set of interaction features when inputs exist."""
    df = features.copy()

    if {"age", "prev_dx_count_total"}.issubset(df.columns):
        df["age_x_prev_dx_count"] = df["age"] * df["prev_dx_count_total"]

    if {"age", "shock_index"}.issubset(df.columns):
        df["age_x_shock_index"] = df["age"] * df["shock_index"]

    if {"lactate_max", "map_min"}.issubset(df.columns):
        df["lactate_x_hypotension"] = df["lactate_max"] * (df["map_min"] < 65).astype(int)
    elif {"lactate_max", "sbp_min"}.issubset(df.columns):
        df["lactate_x_hypotension"] = df["lactate_max"] * (df["sbp_min"] < 90).astype(int)

    if {"resp_rate_mean", "spo2_min"}.issubset(df.columns):
        df["resp_rate_x_spo2_deficit"] = df["resp_rate_mean"] * (100 - df["spo2_min"])

    if {"platelets_min", "inr_max"}.issubset(df.columns):
        df["platelets_x_inr"] = df["platelets_min"] * df["inr_max"]

    if {"sirs_criteria_count", "lactate_mean"}.issubset(df.columns):
        df["sirs_x_lactate"] = df["sirs_criteria_count"] * df["lactate_mean"]

    if {"creatinine_max", "urine_output_total"}.issubset(df.columns):
        df["creatinine_x_urine_output"] = df["creatinine_max"] * df["urine_output_total"]

    if {"bilirubin_max", "inr_max"}.issubset(df.columns):
        df["bilirubin_x_inr"] = df["bilirubin_max"] * df["inr_max"]

    critical_count_col = next(
        (
            col
            for col in ["critical_value_count", "critical_value_count_0_6h"]
            if col in df.columns
        ),
        None,
    )
    missing_count_col = next(
        (
            col
            for col in ["panel_missing_count_0_6h", "missing_lab_count"]
            if col in df.columns
        ),
        None,
    )
    if critical_count_col and missing_count_col:
        df["critical_count_x_missing_lab_count"] = (
            df[critical_count_col] * df[missing_count_col]
        )

    return df
