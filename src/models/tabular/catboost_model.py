"""CatBoost wrapper for the tabular model suite."""

from __future__ import annotations

import importlib.util

from models.base import BaseRiskModel, OptionalDependencyUnavailable


class CatBoostRiskModel(BaseRiskModel):
    model_name = "catboost"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("catboost") is not None

    def _build_estimator(self, y_train=None):
        if not self.is_available():
            raise OptionalDependencyUnavailable("Install catboost to run CatBoostRiskModel")

        from catboost import CatBoostClassifier

        params = {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 5,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "auto_class_weights": "Balanced",
            "verbose": False,
            "allow_writing_files": False,
            **self.params,
        }
        params.setdefault("random_seed", self.random_state)
        return CatBoostClassifier(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        fit_kwargs = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = (X_valid, y_valid)
            fit_kwargs["use_best_model"] = True
        self.estimator.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
