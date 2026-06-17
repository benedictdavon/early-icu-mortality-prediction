"""XGBoost wrapper for the tabular model suite."""

from __future__ import annotations

import importlib.util

import numpy as np

from models.base import BaseRiskModel, OptionalDependencyUnavailable


class XGBoostRiskModel(BaseRiskModel):
    model_name = "xgboost"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("xgboost") is not None

    def _build_estimator(self, y_train=None):
        if not self.is_available():
            raise OptionalDependencyUnavailable("Install xgboost to run XGBoostRiskModel")

        from xgboost import XGBClassifier

        y = np.asarray(y_train) if y_train is not None else np.array([])
        neg = int(np.sum(y == 0))
        pos = int(np.sum(y == 1))
        scale_pos_weight = neg / max(pos, 1)

        params = {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "gamma": 0.0,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            **self.params,
        }
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        params.setdefault("random_state", self.random_state)

        return XGBClassifier(
            **params,
            early_stopping_rounds=early_stopping_rounds,
        )

    def fit_randomized_search(self, *args, **kwargs):
        # RandomizedSearchCV does not pass a fixed validation set, so early
        # stopping must be disabled during training-CV hyperparameter search.
        original_params = dict(self.params)
        self.params["early_stopping_rounds"] = None
        try:
            return super().fit_randomized_search(*args, **kwargs)
        finally:
            self.params = original_params

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        fit_kwargs = {"verbose": False}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        self.estimator.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
