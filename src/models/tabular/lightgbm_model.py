"""LightGBM wrapper for the tabular model suite."""

from __future__ import annotations

import importlib.util

from models.base import BaseRiskModel, OptionalDependencyUnavailable


class LightGBMRiskModel(BaseRiskModel):
    model_name = "lightgbm"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("lightgbm") is not None

    def _build_estimator(self, y_train=None):
        if not self.is_available():
            raise OptionalDependencyUnavailable("Install lightgbm to run LightGBMRiskModel")

        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbosity": -1,
            **self.params,
        }
        params.setdefault("random_state", self.random_state)
        return LGBMClassifier(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        fit_kwargs = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["eval_metric"] = "average_precision"
        self.estimator.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
