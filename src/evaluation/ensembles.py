"""Leakage-safe ensemble helpers for validation-fitted probability models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


REQUIRED_PREDICTION_COLUMNS = {"split", "model_name", "y_true", "p_pred"}
FORBIDDEN_PUBLIC_REPORT_COLUMNS = {"subject_id", "hadm_id", "stay_id"}


def _as_model_list(model_names: Iterable[str]) -> list[str]:
    models = [str(model) for model in model_names]
    if not models:
        raise ValueError("at least one model is required")
    return models


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate the row-level prediction schema used for local analyses."""
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame missing required columns: {sorted(missing)}")
    overlap = FORBIDDEN_PUBLIC_REPORT_COLUMNS & set(predictions.columns)
    if overlap:
        raise ValueError(
            "prediction frame contains patient identifiers that must not be used "
            f"in public reports: {sorted(overlap)}"
        )


def prediction_matrix(
    predictions: pd.DataFrame,
    *,
    split: str,
    model_names: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build an aligned y/model probability matrix for one split.

    This requires a stable non-identifying row key when more than one model is
    present. If `row_id` is absent, the function assumes each model's rows are
    already in identical order and verifies equal lengths/labels.
    """
    validate_prediction_frame(predictions)
    models = _as_model_list(model_names)
    split_df = predictions[predictions["split"] == split].copy()
    missing_models = sorted(set(models) - set(split_df["model_name"].unique()))
    if missing_models:
        raise ValueError(f"missing prediction rows for {split}: {missing_models}")

    if "row_id" in split_df.columns:
        wide = split_df.pivot(index="row_id", columns="model_name", values="p_pred")
        labels = split_df.drop_duplicates("row_id").set_index("row_id")["y_true"]
        wide = wide.reindex(columns=models)
        if wide.isna().any().any():
            raise ValueError(f"incomplete aligned predictions for split {split}")
        labels = labels.loc[wide.index]
        return labels.to_numpy(dtype=int), wide.to_numpy(dtype=float)

    labels = None
    columns = []
    for model_name in models:
        model_rows = split_df[split_df["model_name"] == model_name].reset_index(drop=True)
        if labels is None:
            labels = model_rows["y_true"].to_numpy(dtype=int)
        elif not np.array_equal(labels, model_rows["y_true"].to_numpy(dtype=int)):
            raise ValueError(
                "prediction rows must be aligned by row order or include a non-identifying row_id"
            )
        columns.append(model_rows["p_pred"].to_numpy(dtype=float))
    return labels, np.column_stack(columns)


def average_probabilities(probability_matrix) -> np.ndarray:
    """Return a simple average of aligned model probabilities."""
    matrix = np.asarray(probability_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("probability_matrix must be two-dimensional with at least one model")
    return np.mean(matrix, axis=1)


@dataclass
class ValidationStacker:
    """L2 logistic stacker fit only on validation predictions."""

    model_names: list[str]
    C: float = 1.0
    max_iter: int = 1000
    fit_split: str = "validation"

    def __post_init__(self) -> None:
        self.model_: LogisticRegression | None = None
        self.fit_n_: int | None = None
        self.fit_positive_rate_: float | None = None

    def fit(self, probabilities, labels, *, split: str) -> "ValidationStacker":
        if split != "validation":
            raise ValueError("ValidationStacker must be fit on validation predictions only")
        matrix = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional matrix")
        if len(y) != matrix.shape[0]:
            raise ValueError("probabilities and labels must have the same number of rows")
        if matrix.shape[1] != len(self.model_names):
            raise ValueError("probability columns must match model_names")
        if np.unique(y).size < 2:
            raise ValueError("stacker validation labels must contain both classes")

        model = LogisticRegression(C=self.C, penalty="l2", solver="lbfgs", max_iter=self.max_iter)
        model.fit(matrix, y)
        self.model_ = model
        self.fit_n_ = int(len(y))
        self.fit_positive_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, probabilities) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("ValidationStacker must be fit before predict_proba")
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional matrix")
        if matrix.shape[1] != len(self.model_names):
            raise ValueError("probability columns must match fitted model_names")
        return self.model_.predict_proba(matrix)[:, 1].astype(float)

    def to_metadata(self) -> dict:
        if self.model_ is None:
            raise ValueError("ValidationStacker must be fit before metadata is available")
        return {
            "method": "l2_logistic_regression_stacker",
            "fit_split": self.fit_split,
            "fit_n": int(self.fit_n_ or 0),
            "fit_positive_rate": float(self.fit_positive_rate_ or 0.0),
            "model_names": list(self.model_names),
            "C": float(self.C),
            "max_iter": int(self.max_iter),
        }


def fit_validation_stacker(
    validation_probabilities,
    validation_labels,
    *,
    model_names: Iterable[str],
    C: float = 1.0,
    max_iter: int = 1000,
    split: str = "validation",
) -> ValidationStacker:
    """Fit an L2 logistic stacker on validation predictions only."""
    return ValidationStacker(
        model_names=_as_model_list(model_names),
        C=C,
        max_iter=max_iter,
    ).fit(validation_probabilities, validation_labels, split=split)

