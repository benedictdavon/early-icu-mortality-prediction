"""Feature schema and leakage guards for processed tabular data."""

from __future__ import annotations

import re

import pandas as pd


TARGET_COLUMNS = {"mortality"}
IDENTIFIER_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
    "icustay_id",
}

OBVIOUS_LEAKAGE_TERMS = (
    "death",
    "deathtime",
    "dod",
    "expire",
    "expired",
    "mortality",
    "outcome",
    "survival",
    "died",
    "discharge",
    "dischtime",
    "outtime",
    "length_of_stay",
    "time_in_hospital",
    "duration_hours",
)


def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def is_identifier_column(column: str) -> bool:
    return normalize_column_name(column) in IDENTIFIER_COLUMNS


def is_target_column(column: str, target_col: str = "mortality") -> bool:
    return normalize_column_name(column) == normalize_column_name(target_col)


def has_obvious_leakage_risk(column: str, target_col: str = "mortality") -> bool:
    normalized = normalize_column_name(column)
    if is_target_column(normalized, target_col=target_col):
        return True
    if normalized == "los" or normalized.endswith("_los") or normalized.startswith("los_"):
        return True
    return any(term in normalized for term in OBVIOUS_LEAKAGE_TERMS)


def split_feature_columns(df: pd.DataFrame, target_col: str = "mortality") -> tuple[list[str], list[str]]:
    """Return safe feature columns and columns excluded from model features."""
    feature_cols = []
    dropped_cols = []
    for col in df.columns:
        if (
            is_target_column(col, target_col=target_col)
            or is_identifier_column(col)
            or has_obvious_leakage_risk(col, target_col=target_col)
        ):
            dropped_cols.append(col)
        else:
            feature_cols.append(col)
    return feature_cols, dropped_cols


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "mortality",
) -> tuple[pd.DataFrame, list[str]]:
    """Build X from processed data while dropping identifiers and leakage terms."""
    if target_col not in df.columns:
        raise ValueError(f"Dataset must contain `{target_col}` as target variable")
    feature_cols, dropped_cols = split_feature_columns(df, target_col=target_col)
    X = df.loc[:, feature_cols].copy()
    assert_no_leakage_columns(X.columns, target_col=target_col)
    return X, dropped_cols


def assert_no_leakage_columns(columns, target_col: str = "mortality") -> None:
    """Fail fast if model features include targets, IDs, or obvious outcome proxies."""
    bad_columns = [
        col
        for col in columns
        if is_target_column(col, target_col=target_col)
        or is_identifier_column(col)
        or has_obvious_leakage_risk(col, target_col=target_col)
    ]
    if bad_columns:
        raise ValueError(f"Potential leakage columns found in model features: {bad_columns}")
