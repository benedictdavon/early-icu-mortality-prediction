"""Shared evaluation utilities for ICU mortality models."""

from evaluation.bootstrap import bootstrap_metric_ci
from evaluation.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    apply_isotonic_calibrator,
    apply_platt_scaler,
    calibration_summary,
    expected_calibration_error,
    fit_isotonic_calibrator,
    fit_platt_scaler,
    logits_to_probabilities,
)
from evaluation.metrics import binary_classification_metrics, compute_binary_metrics
from evaluation.reporting import evaluate_validation_and_test, make_result_record
from evaluation.robustness import summarize_seed_stability
from evaluation.subgroups import derive_subgroup_columns, evaluate_subgroups
from evaluation.thresholds import select_optimal_threshold, select_thresholds, threshold_table

__all__ = [
    "binary_classification_metrics",
    "compute_binary_metrics",
    "select_thresholds",
    "select_optimal_threshold",
    "threshold_table",
    "evaluate_validation_and_test",
    "make_result_record",
    "calibration_summary",
    "expected_calibration_error",
    "IsotonicCalibrator",
    "PlattScaler",
    "fit_isotonic_calibrator",
    "fit_platt_scaler",
    "apply_isotonic_calibrator",
    "apply_platt_scaler",
    "logits_to_probabilities",
    "derive_subgroup_columns",
    "evaluate_subgroups",
    "summarize_seed_stability",
    "bootstrap_metric_ci",
]
