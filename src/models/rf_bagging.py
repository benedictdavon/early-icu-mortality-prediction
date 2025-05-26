import os
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

# Add the parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.random_forest import ICUMortalityRandomForest


class ICUMortalityRandomForestBagging(ICUMortalityRandomForest):
    """
    Bagging(RandomForest) model for ICU mortality prediction.
    Wraps a RandomForestClassifier inside a BaggingClassifier for improved stability.
    """

    def __init__(self, output_dir="../../results"):
        super().__init__(output_dir=output_dir)
        self.model_name = "random_forest_bagging"

    def tune_hyperparameters(self, cv=5, n_iter=50):
        """Tune hyperparameters for the base RandomForest used inside BaggingClassifier."""
        print("Tuning base Random Forest hyperparameters for BaggingClassifier...")

        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be loaded first using load_data()")

        # Define hyperparameter search space for base RandomForest
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced", "balanced_subsample"],
            "criterion": ["gini", "entropy"],
        }

        base_rf = RandomForestClassifier(
            random_state=42,
            bootstrap=True,
            n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        X_resampled, y_resampled = self.handle_class_imbalance()

        # Run search
        random_search.fit(X_resampled, y_resampled)

        # Save best params
        self.best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"Best hyperparameters for base Random Forest:")
        print(self.best_params)
        print(f"Best cross-validation AUC: {best_score:.4f}")

        return self.best_params

    def train(self, params=None, bagging_estimators=10, max_samples=0.8):
        """Train the Random Forest with BaggingClassifier wrapper"""
        print("Training Bagging(RandomForest) model...")

        if self.X_train is None:
            raise ValueError("Data must be loaded first using load_data()")

        # Use tuned parameters if available
        if params is None:
            params = self.best_params or {
                "n_estimators": 300,
                "max_depth": 30,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": "balanced",
            }

        # Base model
        base_rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params
        )

        # Bagging wrapper
        self.model = BaggingClassifier(
            base_estimator=base_rf,
            n_estimators=bagging_estimators,
            max_samples=max_samples,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # Resample training data
        X_resampled, y_resampled = self.handle_class_imbalance()

        # Fit model
        start_time = time.time()
        self.model.fit(X_resampled, y_resampled)
        end_time = time.time()
        self.train_time = end_time - start_time
        print(f"Bagging model trained in {self.train_time:.2f} seconds")

        # Feature importance is averaged over base estimators
        try:
            importances = np.mean([
                tree.feature_importances_ for tree in self.model.estimators_
            ], axis=0)
        except AttributeError:
            print("WARNING: Base estimators do not expose feature_importances_. Skipping importance calculation.")
            importances = np.zeros(X_resampled.shape[1])

        if len(self.feature_names) != len(importances):
            print("WARNING: Feature names length doesn't match importances length. Using generic feature names.")
            feature_names_to_use = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names_to_use = self.feature_names

        self.feature_importance = pd.DataFrame({
            "feature": feature_names_to_use,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return self.model
