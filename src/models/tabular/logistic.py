"""Logistic-regression wrapper for the tabular model suite."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseRiskModel


class LogisticRiskModel(BaseRiskModel):
    model_name = "logistic"

    def _build_estimator(self, y_train=None):
        params = {
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            **self.params,
        }
        params.setdefault("random_state", self.random_state)
        return Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(**params))]
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        self.estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
