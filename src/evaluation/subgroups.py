"""Subgroup evaluation helpers for aggregate clinical ML reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.calibration import calibration_summary
from evaluation.metrics import compute_binary_metrics


AGE_LABELS = ["<50", "50-64", "65-79", "80+"]


def derive_subgroup_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive supported subgroup labels from available non-identifier columns."""
    groups = pd.DataFrame(index=frame.index)
    if "age" in frame.columns:
        groups["age_group"] = pd.cut(
            frame["age"],
            bins=[-np.inf, 49, 64, 79, np.inf],
            labels=AGE_LABELS,
        ).astype("object")
    sex_col = _first_present(frame, ["sex", "gender"])
    if sex_col:
        groups["sex"] = frame[sex_col].astype("object")
    diagnosis_col = _first_present(
        frame,
        ["major_diagnosis_group", "diagnosis_group", "admission_diagnosis_group"],
    )
    if diagnosis_col:
        groups["major_diagnosis_group"] = frame[diagnosis_col].astype("object")
    groups["lactate_measured"] = _lactate_measured(frame)

    missingness = _missingness_score(frame)
    if missingness is not None:
        threshold = float(missingness.median())
        groups["missingness_level"] = np.where(
            missingness > threshold,
            "high_missingness",
            "low_missingness",
        )

    intensity = _measurement_intensity(frame)
    if intensity is not None and intensity.nunique(dropna=True) >= 2:
        groups["measurement_intensity_quartile"] = pd.qcut(
            intensity.rank(method="first"),
            q=min(4, intensity.nunique(dropna=True)),
            labels=False,
            duplicates="drop",
        ).map(lambda value: f"Q{int(value) + 1}" if pd.notna(value) else np.nan)

    unit_col = _first_present(frame, ["icu_unit_type", "first_careunit", "careunit"])
    if unit_col:
        groups["icu_unit_type"] = frame[unit_col].astype("object")
    ventilation_col = _first_present(
        frame,
        ["ventilation_status", "mechanical_ventilation", "ventilated", "ventilation_flag"],
    )
    if ventilation_col:
        groups["ventilation_status"] = frame[ventilation_col].map(
            lambda value: "ventilated" if bool(value) else "not_ventilated"
        )
    return groups


def evaluate_subgroups(
    y_true,
    p_pred,
    feature_frame: pd.DataFrame,
    *,
    threshold: float,
    min_n: int = 30,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute required aggregate metrics for every available subgroup."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) != len(p) or len(y) != len(feature_frame):
        raise ValueError("y_true, p_pred, and feature_frame must have the same row count")

    subgroup_frame = derive_subgroup_columns(feature_frame)
    records = []
    for variable in subgroup_frame.columns:
        labels = subgroup_frame[variable]
        for subgroup in sorted(labels.dropna().unique(), key=str):
            mask = labels == subgroup
            if not np.any(mask):
                continue
            y_sub = y[mask.to_numpy()]
            p_sub = p[mask.to_numpy()]
            metrics = compute_binary_metrics(y_sub, p_sub, threshold=threshold)
            cal = calibration_summary(y_sub, p_sub, n_bins=n_bins)
            records.append(
                {
                    "subgroup_variable": variable,
                    "subgroup": str(subgroup),
                    "small_subgroup": bool(len(y_sub) < min_n),
                    "interpretation_note": (
                        "small subgroup; interpret cautiously"
                        if len(y_sub) < min_n
                        else "aggregate subgroup estimate"
                    ),
                    "mortality_rate": metrics["positive_rate"],
                    "auc_roc": metrics["auc_roc"],
                    "average_precision": metrics["average_precision"],
                    "brier_score": metrics["brier_score"],
                    "calibration_slope": cal["calibration_slope"],
                    "calibration_intercept": cal["calibration_intercept"],
                    "recall_at_screening_threshold": metrics["recall"],
                    "precision_at_screening_threshold": metrics["precision"],
                    "specificity_at_screening_threshold": metrics["specificity"],
                    "threshold": float(threshold),
                    "n": metrics["n"],
                }
            )
    return pd.DataFrame(records)


def save_subgroup_report(report: pd.DataFrame, output_dir) -> dict:
    """Save aggregate subgroup report as CSV and JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "subgroup_report.csv"
    json_path = output / "subgroup_report.json"
    report.to_csv(csv_path, index=False)
    report.to_json(json_path, orient="records", indent=2)
    return {"csv": str(csv_path), "json": str(json_path)}


def _first_present(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _lactate_measured(frame: pd.DataFrame) -> pd.Series:
    if "lactate_measured" in frame.columns:
        measured = frame["lactate_measured"].astype(bool)
    else:
        lactate_cols = [col for col in frame.columns if "lactate" in col.lower()]
        if lactate_cols:
            measured = frame[lactate_cols].notna().any(axis=1)
        else:
            measured = pd.Series(False, index=frame.index)
    return measured.map(lambda value: "lactate_measured" if value else "lactate_not_measured")


def _missingness_score(frame: pd.DataFrame) -> pd.Series | None:
    columns = [
        col
        for col in frame.columns
        if "missing" in col.lower() or "panel_missing_count" in col.lower()
    ]
    if not columns:
        return None
    return frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def _measurement_intensity(frame: pd.DataFrame) -> pd.Series | None:
    columns = [
        col
        for col in frame.columns
        if "measurement_count" in col.lower()
        or "event_count" in col.lower()
        or col.lower() in {"total_lab_count", "total_vital_count"}
    ]
    if not columns:
        return None
    return frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


__all__ = ["derive_subgroup_columns", "evaluate_subgroups", "save_subgroup_report"]
