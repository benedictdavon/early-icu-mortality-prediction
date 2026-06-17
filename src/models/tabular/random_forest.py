"""Random-forest wrapper for the tabular model suite."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from models.base import BaseRiskModel


class RandomForestRiskModel(BaseRiskModel):
    model_name = "random_forest"

    def _build_estimator(self, y_train=None):
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            **self.params,
        }
        params.setdefault("random_state", self.random_state)
        return RandomForestClassifier(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.estimator = self._build_estimator(y_train=y_train)
        self.estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self._positive_class_proba(self.estimator.predict_proba(X))
