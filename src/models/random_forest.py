import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import shap
import time
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add the parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import ICUMortalityBaseModel


class ICUMortalityRandomForest(ICUMortalityBaseModel):
    """
    Random Forest model for ICU mortality prediction, inheriting from base model class.
    Implements Random Forest specific functionality while reusing common methods.
    """

    def __init__(self, output_dir="../../results"):
        """Initialize RF model with appropriate name"""
        super().__init__(model_name="random_forest", output_dir=output_dir)

    def tune_hyperparameters(self, cv=5, n_iter=50):
        """Tune Random Forest hyperparameters with expanded search space"""
        print("Tuning Random Forest hyperparameters...")

        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be loaded first using load_data()")

        # Define expanded parameter grid
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 15, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3],
            "class_weight": ["balanced", "balanced_subsample"],
            "bootstrap": [True],
            "criterion": ["gini", "entropy"],
        }
        # Initialize RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV

        model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        # Fit model
        random_search.fit(self.X_train, self.y_train)

        # Get best parameters and score
        self.best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {best_score:.4f}")

        return self.best_params

    def train(self, params=None):
        """Train the Random Forest model"""
        print("Training Random Forest model...")

        if self.X_train is None:
            raise ValueError("Data must be loaded first using load_data()")

        # Use tuned parameters if available, otherwise use default
        if params is None:
            if self.best_params is not None:
                params = self.best_params
            else:
                params = {
                    "n_estimators": 300,
                    "max_depth": 30,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                }

        # Initialize and train model
        start_time = time.time()
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        X_resampled, y_resampled = self.handle_class_imbalance(sampling_strategy="auto")
        self.model.fit(X_resampled, y_resampled)
        end_time = time.time()

        self.train_time = end_time - start_time
        print(f"Model trained in {self.train_time:.2f} seconds")

        # Calculate feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return self.model

    def train_with_early_stopping(
        self, params=None, max_iters=500, patience=10, eval_every=10
    ):
        """Train Random Forest with manual early stopping based on validation score"""
        print("Training Random Forest with early stopping...")

        if self.X_train is None or self.X_val is None:
            raise ValueError("Data must be loaded and split with a validation set")

        # Start with base parameters
        if params is None:
            params = {
                "max_depth": 30,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": "balanced",
                "warm_start": True,  # Enable warm start for iterative training
            }

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=0, random_state=42, n_jobs=-1, **params
        )

        # Resample training data
        X_resampled, y_resampled = self.handle_class_imbalance(sampling_strategy="auto")

        # Initial estimators batch size
        n_est_batch = eval_every

        # Track scores
        best_score = 0
        best_n_est = 0
        no_improve_count = 0
        scores = []

        # Training loop
        for i in range(0, max_iters, n_est_batch):
            # Increase number of trees
            self.model.n_estimators += n_est_batch

            # Fit model (will only fit n_est_batch new trees due to warm_start)
            self.model.fit(X_resampled, y_resampled)

            # Evaluate on validation set
            y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
            score = roc_auc_score(self.y_val, y_pred_proba)
            scores.append(score)

            print(f"Iteration {i+n_est_batch}: AUC = {score:.4f}")

            # Check for improvement
            if score > best_score:
                best_score = score
                best_n_est = self.model.n_estimators
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stop if no improvement for 'patience' iterations
            if no_improve_count >= patience:
                print(
                    f"Early stopping at {self.model.n_estimators} trees: "
                    f"No improvement for {patience} iterations"
                )
                break

        # Reset to best number of estimators
        if best_n_est < self.model.n_estimators:
            print(f"Rolling back to best model with {best_n_est} trees")
            self.model.n_estimators = best_n_est
            self.model.fit(X_resampled, y_resampled)

        # Calculate feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_est_batch, len(scores) * n_est_batch + 1, n_est_batch), scores)
        plt.axvline(
            x=best_n_est,
            color="r",
            linestyle="--",
            label=f"Best estimators: {best_n_est}",
        )
        plt.xlabel("Number of Trees")
        plt.ylabel("Validation AUC")
        plt.title("Random Forest Training Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, "rf_training_curve.png"), dpi=300)
        plt.close()

        print(
            f"Final model has {best_n_est} trees with validation AUC: {best_score:.4f}"
        )
        return self.model

    def analyze_shap_values(self, max_display=20, n_samples=200):
        """Analyze SHAP values for feature interpretation"""
        print("Analyzing SHAP values for feature interpretation...")

        try:
            import shap

            # Sample data to make SHAP analysis faster
            if self.X_test.shape[0] > n_samples:
                # Get a random sample from test data
                np.random.seed(42)
                indices = np.random.choice(
                    self.X_test.shape[0], n_samples, replace=False
                )
                X_sample = self.X_test[indices]
            else:
                X_sample = self.X_test

            # Convert back to DataFrame for better visualization
            # This is critical for matching feature names
            if not isinstance(X_sample, pd.DataFrame):
                X_sample = pd.DataFrame(X_sample, columns=self.feature_names)

            # Create TreeExplainer
            explainer = shap.TreeExplainer(self.model)

            # Get correct SHAP values format based on model type
            # For random forest classifier with binary output
            if hasattr(self.model, "classes_") and len(self.model.classes_) == 2:
                # For binary classification
                shap_values = explainer.shap_values(X_sample)

                # Check if shap_values is a list (happens with some models)
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        # We want the positive class (index 1)
                        shap_values_for_plot = shap_values[1]
                    else:
                        # Use the only available set
                        shap_values_for_plot = shap_values[0]
                else:
                    # Direct array
                    shap_values_for_plot = shap_values

                # Summary plot (beeswarm)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values_for_plot, X_sample, max_display=max_display, show=False
                )
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=300)
                plt.close()

                # Bar plot
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values_for_plot,
                    X_sample,
                    plot_type="bar",
                    max_display=max_display,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_bar.png"), dpi=300)
                plt.close()

                # Dependence plots for top 3 features
                # Get feature importance based on SHAP
                feature_importance = np.abs(shap_values_for_plot).mean(0)
                top_indices = feature_importance.argsort()[-3:][::-1]

                for idx in top_indices:
                    feature_name = self.feature_names[int(idx)] if isinstance(idx, np.ndarray) else self.feature_names[idx]
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        idx,
                        shap_values_for_plot,
                        X_sample,
                        feature_names=self.feature_names,
                        show=False,
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            self.output_dir, f"shap_dependence_{feature_name}.png"
                        ),
                        dpi=300,
                    )
                    plt.close()

                return shap_values
            else:
                # For regression or multi-class
                # (less likely for your use case, but included for completeness)
                shap_values = explainer.shap_values(X_sample)

                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values, X_sample, max_display=max_display, show=False
                )
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=300)
                plt.close()

                return shap_values

        except Exception as e:
            print(f"Error in SHAP analysis: {str(e)}")
            print("Check if SHAP is installed: pip install shap")
            import traceback

            traceback.print_exc()
            return None
