"""Calibration utilities for binary risk predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _as_1d_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr.reshape(-1)


def _sigmoid(values) -> np.ndarray:
    logits = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def logits_to_probabilities(logits) -> np.ndarray:
    """Convert logits to positive-class probabilities with stable clipping."""
    return _sigmoid(logits).reshape(-1)


@dataclass
class PlattScaler:
    """Validation-fitted logistic calibration for model logits."""

    C: float = 1.0
    max_iter: int = 1000
    fit_split: str = "validation"

    def __post_init__(self) -> None:
        self.model_: LogisticRegression | None = None
        self.fit_n_: int | None = None
        self.fit_positive_rate_: float | None = None

    def fit(self, validation_logits, validation_labels) -> "PlattScaler":
        """Fit sigmoid calibration on validation logits and labels only."""
        logits = _as_1d_array(validation_logits, "validation_logits")
        labels = np.asarray(validation_labels, dtype=int).reshape(-1)
        if len(logits) != len(labels):
            raise ValueError("validation_logits and validation_labels must have the same length")
        if len(labels) == 0:
            raise ValueError("Platt scaling requires at least one validation row")
        if np.unique(labels).size < 2:
            raise ValueError("Platt scaling requires both outcome classes in validation labels")

        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter)
        model.fit(logits.reshape(-1, 1), labels)
        self.model_ = model
        self.fit_n_ = int(len(labels))
        self.fit_positive_rate_ = float(np.mean(labels))
        return self

    def predict_proba(self, logits) -> np.ndarray:
        """Apply the fitted calibration map to logits."""
        if self.model_ is None:
            raise ValueError("PlattScaler must be fit before predict_proba")
        arr = _as_1d_array(logits, "logits")
        return self.model_.predict_proba(arr.reshape(-1, 1))[:, 1].astype(float)

    def transform(self, logits) -> np.ndarray:
        """Alias for predict_proba to mirror sklearn transformer naming."""
        return self.predict_proba(logits)

    @property
    def coefficient_(self) -> float:
        if self.model_ is None:
            raise ValueError("PlattScaler must be fit before reading coefficient_")
        return float(self.model_.coef_[0][0])

    @property
    def intercept_(self) -> float:
        if self.model_ is None:
            raise ValueError("PlattScaler must be fit before reading intercept_")
        return float(self.model_.intercept_[0])

    def to_metadata(self) -> dict:
        """Return aggregate calibration metadata without row-level predictions."""
        if self.model_ is None:
            raise ValueError("PlattScaler must be fit before metadata is available")
        return {
            "method": "platt_scaling",
            "fit_split": self.fit_split,
            "fit_n": int(self.fit_n_ or 0),
            "fit_positive_rate": float(self.fit_positive_rate_ or 0.0),
            "coefficient": self.coefficient_,
            "intercept": self.intercept_,
            "C": float(self.C),
            "max_iter": int(self.max_iter),
        }


def fit_platt_scaler(
    validation_logits,
    validation_labels,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
) -> PlattScaler:
    """Fit Platt scaling from validation logits only."""
    return PlattScaler(C=C, max_iter=max_iter).fit(validation_logits, validation_labels)


def apply_platt_scaler(calibrator: PlattScaler, logits) -> np.ndarray:
    """Apply a pre-fitted Platt scaler to another split's logits."""
    return calibrator.predict_proba(logits)


def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    """Compute fixed-width expected calibration error."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if len(y) == 0:
        raise ValueError("calibration metrics require at least one row")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(p[mask]))
        bin_observed = float(np.mean(y[mask]))
        ece += (np.sum(mask) / len(y)) * abs(bin_confidence - bin_observed)
    return float(ece)


def calibration_intercept_slope(y_true, p_pred, eps: float = 1e-6) -> dict:
    """Estimate calibration intercept and slope via logistic recalibration."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    if len(y) != len(p):
        raise ValueError("y_true and p_pred must have the same length")
    if np.unique(y).size < 2:
        return {"calibration_intercept": math.nan, "calibration_slope": math.nan}

    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def calibration_summary(y_true, p_pred, n_bins: int = 10) -> dict:
    """Return calibration-ready aggregate metrics without row-level output."""
    summary = calibration_intercept_slope(y_true, p_pred)
    summary.update(
        {
            "brier_score": float(brier_score_loss(y_true, p_pred)),
            "expected_calibration_error": expected_calibration_error(
                y_true, p_pred, n_bins=n_bins
            ),
            "calibration_bins": int(n_bins),
        }
    )
    return summary


def platt_calibration_summary(
    y_true,
    logits,
    calibrator: PlattScaler,
    *,
    n_bins: int = 10,
) -> dict:
    """Return aggregate metrics for logits after a fitted Platt scaler."""
    probabilities = calibrator.predict_proba(logits)
    return {
        **calibration_summary(y_true, probabilities, n_bins=n_bins),
        **calibrator.to_metadata(),
    }
