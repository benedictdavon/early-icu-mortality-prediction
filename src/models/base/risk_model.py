"""Common model interface for tabular risk models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


class OptionalDependencyUnavailable(ImportError):
    """Raised when an optional model backend is not installed."""


class BaseRiskModel(ABC):
    """Small interface shared by tabular model wrappers."""

    model_name = "base"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42):
        self.params = dict(params or {})
        self.random_state = int(random_state)
        self.estimator = None
        self.feature_names: list[str] | None = None
        self.best_params_: dict[str, Any] | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this model can run in the current environment."""
        return True

    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """Fit the model using training data and optional validation data."""

    def _build_estimator(self, y_train=None):
        """Build an unfitted estimator for direct fitting or randomized search."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _build_estimator")

    def fit_randomized_search(
        self,
        X_train,
        y_train,
        *,
        param_distributions: dict[str, Any],
        n_iter: int = 20,
        cv: int = 3,
        scoring: str = "average_precision",
        n_jobs: int = -1,
    ):
        """Fit with training-only randomized search.

        The validation split remains untouched for threshold selection and final
        model comparison after search chooses training-CV hyperparameters.
        """
        if not param_distributions:
            raise ValueError("param_distributions must not be empty")

        estimator = self._build_estimator(y_train=y_train)
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            random_state=self.random_state,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        self.estimator = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        return self

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return positive-class probabilities."""

    def save(self, path) -> str:
        """Persist the fitted wrapper without saving row-level predictions."""
        if self.estimator is None:
            raise ValueError("Cannot save an unfitted model")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "params": self.params,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "best_params": self.best_params_,
                "estimator": self.estimator,
            },
            out,
        )
        return str(out)

    @classmethod
    def load(cls, path):
        """Load a previously saved wrapper."""
        bundle = joblib.load(path)
        obj = cls(
            params=bundle.get("params"),
            random_state=bundle.get("random_state", 42),
        )
        obj.feature_names = bundle.get("feature_names")
        obj.best_params_ = bundle.get("best_params")
        obj.estimator = bundle["estimator"]
        return obj

    def _set_feature_names(self, feature_names) -> None:
        if feature_names is None:
            self.feature_names = None
        else:
            self.feature_names = [str(name) for name in feature_names]

    @staticmethod
    def _positive_class_proba(probabilities) -> np.ndarray:
        proba = np.asarray(probabilities, dtype=float)
        if proba.ndim == 2:
            if proba.shape[1] < 2:
                raise ValueError("predict_proba returned fewer than two columns")
            return proba[:, 1]
        if proba.ndim != 1:
            raise ValueError("predict_proba output must be one- or two-dimensional")
        return proba
