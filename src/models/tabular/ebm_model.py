"""Explainable Boosting Machine wrapper for the tabular model suite."""

from __future__ import annotations

import importlib.util

from models.base import BaseRiskModel, OptionalDependencyUnavailable


class EBMRiskModel(BaseRiskModel):
    model_name = "ebm"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("interpret") is not None

    def _build_estimator(self, y_train=None):
        if not self.is_available():
            raise OptionalDependencyUnavailable("Install interpret to run EBMRiskModel")

        from interpret.glassbox import ExplainableBoostingClassifier

        params = {
            "interactions": 10,
            "learning_rate": 0.01,
            "max_rounds": 5000,
            "n_jobs": -1,
            **self.params,
        }
        params.setdefault("random_state", self.random_state)
        return ExplainableBoostingClassifier(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        self.estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
