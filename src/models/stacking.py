"""Leakage-safe probability ensembling and stacking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold


def _positive_class_proba(predicted) -> np.ndarray:
    arr = np.asarray(predicted, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError("predict_proba output must have two columns for binary stacking")
        return arr[:, 1]
    if arr.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    return arr


def average_calibrated_probabilities(probability_columns) -> np.ndarray:
    """Average calibrated positive-class probabilities at the same row grain."""
    matrix = np.asarray(probability_columns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probability_columns must be a two-dimensional array")
    if matrix.shape[1] < 2:
        raise ValueError("at least two probability columns are required")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    return matrix.mean(axis=1)


@dataclass
class OOFStackingResult:
    meta_model: LogisticRegression
    base_models: dict[str, object]
    model_names: list[str]
    oof_meta_features: np.ndarray

    def predict_proba_from_base_probabilities(self, probability_columns) -> np.ndarray:
        matrix = np.asarray(probability_columns, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("probability_columns must be two-dimensional")
        if matrix.shape[1] != len(self.model_names):
            raise ValueError("probability column count does not match fitted base models")
        return self.meta_model.predict_proba(matrix)[:, 1]


def fit_oof_logistic_stacker(
    X,
    y,
    model_factories: dict[str, Callable[[], object]],
    *,
    groups=None,
    n_splits: int = 5,
    random_state: int = 42,
) -> OOFStackingResult:
    """Fit a logistic meta-learner from out-of-fold base model probabilities."""
    if not model_factories:
        raise ValueError("model_factories must contain at least one base model")
    X_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=int)
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows")
    if np.unique(y_arr).size < 2:
        raise ValueError("stacking requires both outcome classes")

    model_names = list(model_factories)
    oof = np.zeros((len(y_arr), len(model_names)), dtype=float)
    splitter = _splitter(y_arr, groups=groups, n_splits=n_splits, random_state=random_state)
    split_iter = splitter.split(X_arr, y_arr, groups) if groups is not None else splitter.split(X_arr, y_arr)

    for train_idx, valid_idx in split_iter:
        for col_idx, name in enumerate(model_names):
            model = model_factories[name]()
            model.fit(X_arr[train_idx], y_arr[train_idx])
            oof[valid_idx, col_idx] = _positive_class_proba(
                model.predict_proba(X_arr[valid_idx])
            )

    meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta_model.fit(oof, y_arr)

    base_models = {}
    for name in model_names:
        model = model_factories[name]()
        model.fit(X_arr, y_arr)
        base_models[name] = model

    return OOFStackingResult(
        meta_model=meta_model,
        base_models=base_models,
        model_names=model_names,
        oof_meta_features=oof,
    )


def clone_estimator_factory(estimator) -> Callable[[], object]:
    """Create a factory that returns a fresh clone for each OOF fold."""
    return lambda: clone(estimator)


def _splitter(y, *, groups=None, n_splits: int, random_state: int):
    y_arr = np.asarray(y)
    if groups is not None:
        unique_groups = np.unique(groups)
        splits = min(int(n_splits), len(unique_groups))
        if splits < 2:
            raise ValueError("grouped stacking requires at least two groups")
        return GroupKFold(n_splits=splits)
    class_counts = np.bincount(y_arr.astype(int))
    min_class = int(np.min(class_counts[class_counts > 0]))
    splits = min(int(n_splits), min_class)
    if splits < 2:
        raise ValueError("stacking requires at least two rows per class")
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


__all__ = [
    "OOFStackingResult",
    "average_calibrated_probabilities",
    "clone_estimator_factory",
    "fit_oof_logistic_stacker",
]
