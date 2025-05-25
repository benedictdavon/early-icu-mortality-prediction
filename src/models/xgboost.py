# models/xg_boost.py
import os, time, json, joblib, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from xgboost import XGBClassifier


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.inspection import permutation_importance

# Add parent directory so "from models.base_model import ..." works
from pathlib import Path, PurePath
import sys

sys.path.append(str(PurePath(__file__).parent.parent))

from models.base_model import ICUMortalityBaseModel  # ← your shared superclass


class ICUMortalityXGBoost(ICUMortalityBaseModel):
    """
    Gradient-boosted tree model inspired by
    Hou et al., *J Transl Med* 2020 (doi:10.1186/s12967-020-02620-5).
    """

    def __init__(self, output_dir="../../results", gpu_device="cuda:0"):
        super().__init__(model_name="xgboost", output_dir=output_dir)
        self.gpu_device = gpu_device

        if XGBClassifier is None:
            warnings.warn(
                "XGBoost is not installed. Install with 'pip install xgboost'"
            )

    def tune_hyperparameters(self, cv=5, n_iter=50):
        """
        Randomised search over parameter space based on Hou et al. 2020.
        """
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed. Install with 'pip install xgboost'"
            )

        if self.X_train is None:
            raise ValueError("Call load_data() first")

        print("Tuning XGBoost hyper-parameters...")
        # Compute class weight for scale_pos_weight = #neg / #pos
        pos = np.sum(self.y_train == 1)
        neg = np.sum(self.y_train == 0)
        base_pos_w = neg / max(pos, 1)

        param_dist = {
            # Settings based on the paper
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate": [0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 1, 7],
            "max_depth": [3, 5, 7, 9],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_lambda": [1, 2, 5, 10],  # L2 regularisation
            "scale_pos_weight": [base_pos_w * f for f in (0.5, 1, 2, 4)],
        }

        # Using partial fit approach instead of RandomizedSearchCV
        # to avoid sklearn compatibility issues
        best_score = 0
        best_params = None

        # Convert pandas Series to numpy arrays if needed
        X_train = (
            self.X_train.values if hasattr(self.X_train, "values") else self.X_train
        )
        y_train = (
            self.y_train.values if hasattr(self.y_train, "values") else self.y_train
        )

        # Create cross-validation splitter
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Generate parameter combinations
        from itertools import product
        import random

        param_keys = list(param_dist.keys())
        param_values = [
            param_dist[k] if isinstance(param_dist[k], list) else param_dist[k].tolist()
            for k in param_keys
        ]
        all_combinations = list(product(*param_values))

        # Randomly sample n_iter combinations
        if len(all_combinations) > n_iter:
            sampled_combinations = random.sample(all_combinations, n_iter)
        else:
            sampled_combinations = all_combinations

        print(f"Testing {len(sampled_combinations)} parameter combinations")

        for i, combination in enumerate(sampled_combinations):
            params = {param_keys[j]: combination[j] for j in range(len(param_keys))}
            params.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "tree_method": "hist",
                    "device": "cuda",
                }
            )
            model = XGBClassifier(**params, random_state=42, use_label_encoder=False)

            # Cross-validate
            cv_scores = []
            for train_idx, val_idx in cv_splitter.split(X_train, y_train):
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                model = XGBClassifier(
                    **params,
                    random_state=42,
                    use_label_encoder=False,
                    early_stopping_rounds=30,
                )

                model.fit(
                    X_train_cv,
                    y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    verbose=False,
                )

                y_pred = model.predict_proba(X_val_cv)[:, 1]
                cv_scores.append(roc_auc_score(y_val_cv, y_pred))

            mean_score = np.mean(cv_scores)
            print(
                f"Combination {i+1}/{len(sampled_combinations)}: AUC = {mean_score:.4f}"
            )

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params = best_params
        print(f"Best AUC (average CV): {best_score:.4f}")
        print(
            f"Best parameters: {json.dumps({k: str(v) for k, v in self.best_params.items()}, indent=2)}"
        )
        return self.best_params

    def cross_validate(self, cv=5, scoring=None, n_jobs=-1):
        """
        Perform cross-validation with the trained model.
        """
        from sklearn.model_selection import cross_validate as sk_cross_validate
        from sklearn.base import BaseEstimator, ClassifierMixin

        class ModelWrapper(BaseEstimator, ClassifierMixin):
            """Wrapper to make a custom model compatible with sklearn's cross_validate"""

            _estimator_type = (
                "classifier"  # Class-level attribute for scikit-learn detection
            )

            def __init__(self, model_instance, best_params=None):
                self.model_instance = model_instance
                self.model = None
                self.best_params = best_params
                self.classes_ = np.array([0, 1])  # Required for classifier

            def fit(self, X, y):
                # Create a new XGBoost model for each fold
                if XGBClassifier is None:
                    raise ImportError("XGBoost not installed")

                params = (
                    self.best_params
                    if self.best_params
                    else {
                        "n_estimators": 1000,
                        "learning_rate": 0.02,
                        "max_depth": 4,
                        "min_child_weight": 1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "gamma": 0,
                        "reg_lambda": 1,
                        # class imbalance
                        "scale_pos_weight": sum(y == 0) / max(sum(y == 1), 1),
                        "objective": "binary:logistic",
                        "eval_metric": "aucpr",
                        "tree_method": "hist",
                        "device": "cuda",
                    }
                )

                # Create a new model for each fold
                self.model = XGBClassifier(
                    **params, random_state=42, use_label_encoder=False
                )
                self.model.fit(X, y)
                return self

            def predict(self, X):
                return self.model.predict(X)

            def predict_proba(self, X):
                return self.model.predict_proba(X)

            def get_params(self, deep=True):
                return {
                    "model_instance": self.model_instance,
                    "best_params": self.best_params,
                }

            def set_params(self, **parameters):
                for parameter, value in parameters.items():
                    setattr(self, parameter, value)
                return self

        print("Performing cross-validation...")
        print(f"Performing {cv}-fold cross-validation...")

        X = self.X_train.values if hasattr(self.X_train, "values") else self.X_train
        y = self.y_train.values if hasattr(self.y_train, "values") else self.y_train

        if scoring is None:
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        # Create a fresh instance for cross-validation
        wrapped_model = ModelWrapper(self, self.best_params)

        try:
            cv_results = sk_cross_validate(
                wrapped_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
            )

            # Process results
            for metric in scoring:
                score_key = f"test_{metric}"
                if score_key in cv_results:
                    scores = cv_results[score_key]
                    mean_score = (
                        np.mean(scores[~np.isnan(scores)])
                        if np.any(~np.isnan(scores))
                        else float("nan")
                    )
                    std_score = (
                        np.std(scores[~np.isnan(scores)])
                        if np.any(~np.isnan(scores))
                        else float("nan")
                    )
                    print(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")
        except Exception as e:
            print(f"Cross-validation error: {e}")
            cv_results = {}

        return cv_results

    def train(self, params=None, early_stopping_rounds=30):
        """
        Train XGBClassifier with early stopping on validation set
        """
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed. Install with 'pip install xgboost'"
            )

        if self.X_train is None or self.X_val is None:
            raise ValueError("Run load_data() first")

        if params is None:
            params = (
                self.best_params
                if self.best_params
                else {
                    "n_estimators": 1000,
                    "learning_rate": 0.01,
                    "max_depth": 5,
                    "min_child_weight": 3,  
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0.2,
                    "reg_lambda": 5,
                    "scale_pos_weight": 1.9,  
                    "objective": "binary:logistic",
                    "tree_method": "hist", 
                    "device": "cuda",  
                    "eval_metric": "aucpr",  
                }
            )


        # Optional: still resample with SMOTE
        X_res, y_res = self.handle_class_imbalance(sampling_strategy="auto")

        # Convert to numpy arrays if needed
        X_val = self.X_val.values if hasattr(self.X_val, "values") else self.X_val
        y_val = self.y_val.values if hasattr(self.y_val, "values") else self.y_val

        # Set early_stopping_rounds in constructor instead of fit method
        self.model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            early_stopping_rounds=early_stopping_rounds,
            **{k: v for k, v in params.items() if k != "eval_metric"},
            eval_metric=params.get("eval_metric", "logloss"),
            n_jobs=-1,
        )

        print("Fitting XGBoost...")
        t0 = time.time()
        # Create eval set
        eval_set = [(X_val, y_val)]

        # Remove early_stopping_rounds from fit
        self.model.fit(
            X_res,
            y_res,
            eval_set=eval_set,
            verbose=False,
        )
        self.train_time = time.time() - t0
        print(
            f"Training done in {self.train_time:.1f} s — best iter = {self.model.best_iteration}"
        )

        # Store feature importances with length check
        feature_importances = self.model.feature_importances_
        if len(self.feature_names) != len(feature_importances):
            print(
                f"Warning: Feature names ({len(self.feature_names)}) and importances ({len(feature_importances)}) have different lengths"
            )
            # Check if a subset of features could match
            if len(self.feature_names) > len(feature_importances):
                # Use only the needed number of feature names
                feature_names = self.feature_names[: len(feature_importances)]
                print(f"Using first {len(feature_names)} feature names")
            else:
                # Use generic feature names
                feature_names = [
                    f"feature_{i}" for i in range(len(feature_importances))
                ]
                print("Using generic feature names")

            self.feature_importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": feature_importances,
                }
            ).sort_values("importance", ascending=False)
        else:
            self.feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": feature_importances,
                }
            ).sort_values("importance", ascending=False)

        return self.model

    def train_ensemble(self, seeds=[42, 123, 2021, 777, 888]):
        """
        Train an ensemble of XGBoost models with different random seeds
        """
        if self.X_train_smote is None or self.y_train_smote is None:
            raise ValueError("Apply SMOTE first by calling apply_smote() method")
        
        print(f"Training XGBoost ensemble with {len(seeds)} models...")
        
        # Use best params if available, otherwise use defaults
        if self.best_params:
            params = self.best_params.copy()
            # Convert string values to appropriate types if needed
            if 'n_estimators' in params:
                params['n_estimators'] = int(params['n_estimators'])
            if 'learning_rate' in params:
                params['learning_rate'] = float(params['learning_rate'])
            if 'max_depth' in params:
                params['max_depth'] = int(params['max_depth'])
            if 'min_child_weight' in params:
                params['min_child_weight'] = int(params['min_child_weight'])
            if 'subsample' in params:
                params['subsample'] = float(params['subsample'])
            if 'colsample_bytree' in params:
                params['colsample_bytree'] = float(params['colsample_bytree'])
            if 'gamma' in params:
                params['gamma'] = float(params['gamma'])
            if 'reg_lambda' in params:
                params['reg_lambda'] = float(params['reg_lambda'])
            if 'scale_pos_weight' in params:
                params['scale_pos_weight'] = float(params['scale_pos_weight'])
        else:
            params = {
                "n_estimators": 600,
                "learning_rate": 0.01,
                "max_depth": 7,
                "min_child_weight": 1,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "gamma": 0.2,
                "reg_lambda": 1,
                "scale_pos_weight": sum(self.y_train == 0) / max(sum(self.y_train == 1), 1),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "tree_method": "hist",
                "device": "cuda"
            }
        
        # Remove eval_metric from params to avoid duplicate keyword argument
        eval_metric = params.pop("eval_metric", "logloss")
        
        ensemble_models = []
        ensemble_probas = []
        
        for i, seed in enumerate(seeds):
            print(f"\nTraining model {i+1}/{len(seeds)} with random seed {seed}")
            
            # Create model with unique random seed
            model = XGBClassifier(
                random_state=seed,
                use_label_encoder=False,
                eval_metric=eval_metric,  # Pass eval_metric separately
                **params  # Unpack remaining parameters
            )
            
            # Train the model
            model.fit(
                self.X_train_smote, 
                self.y_train_smote,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            
            # Get predictions on test set
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            ensemble_models.append(model)
            ensemble_probas.append(y_proba)
            
            print(f"Model {i+1} trained successfully")
        
        # Store ensemble models and predictions
        self.ensemble_models = ensemble_models
        self.ensemble_probas = np.array(ensemble_probas)
        
        # Calculate ensemble average predictions
        self.ensemble_avg_proba = np.mean(self.ensemble_probas, axis=0)
        
        print(f"\nEnsemble training completed with {len(ensemble_models)} models")
        
        return ensemble_models, ensemble_probas


    def analyze_shap_values(self, max_display=20, n_samples=300):
        """
        Produces beeswarm & bar plots for feature interpretation
        """
        try:
            import shap
        except ImportError:
            warnings.warn("Install SHAP first: pip install shap")
            return None

        if self.model is None:
            raise ValueError("Train the model first")

        # Sample to keep SHAP computation quick
        idx = np.random.choice(
            self.X_test.shape[0], min(n_samples, self.X_test.shape[0]), replace=False
        )

        # Convert to DataFrame if necessary
        if isinstance(self.X_test, np.ndarray):
            X_sample = pd.DataFrame(self.X_test[idx], columns=self.feature_names)
        else:
            X_sample = self.X_test.iloc[idx]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_sample, plot_type="dot", show=False, max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=300)
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_sample, plot_type="bar", show=False, max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "shap_bar.png"), dpi=300)
        plt.close()

        return shap_values

    def calculate_permutation_importance(self, n_repeats=5):
        """
        Uses sklearn's permutation_importance to complement SHAP.
        """
        if self.model is None:
            raise ValueError("Train the model first")

        print("Computing permutation importance...")
        result = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        # Handle potential length mismatch
        if len(self.feature_names) != len(result.importances_mean):
            print(f"Warning: Feature names ({len(self.feature_names)}) and permutation importances ({len(result.importances_mean)}) have different lengths")
            
            # Check if a subset of features could match
            if len(self.feature_names) > len(result.importances_mean):
                # Use only the needed number of feature names
                feature_names = self.feature_names[:len(result.importances_mean)]
                print(f"Using first {len(feature_names)} feature names for permutation importance")
            else:
                # Use generic feature names
                feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]
                print("Using generic feature names for permutation importance")
        else:
            feature_names = self.feature_names
            
        perm_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances.std,
            }
        ).sort_values("importance_mean", ascending=False)

        plt.figure(figsize=(11, 9))
        sns.barplot(y="feature", x="importance_mean", data=perm_df.head(25))
        plt.title(f"{self.model_name} - permutation importance (top 25)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "permutation_importance.png"), dpi=300)
        plt.close()

        perm_df.to_csv(os.path.join(self.output_dir, "permutation_importance.csv"), index=False)
        return perm_df

    def evaluate(
        self,
        threshold=0.5,
        optimize_threshold=True,
        optimization_metric="clinical_utility",
        run_comprehensive=True,
    ):
        """
        Evaluate the model on the test set with option to optimize threshold

        Parameters:
        -----------
        threshold : float
            Default threshold to use if optimize_threshold is False
        optimize_threshold : bool
            Whether to perform threshold optimization
        optimization_metric : str
            Metric to optimize for: 'f1', 'clinical_utility', or 'balanced'

        Returns:
        --------
        dict : Evaluation results
        """
        if self.model is None:
            raise ValueError("You must train the model first!")

        if optimize_threshold:
            # Use the enhanced evaluation with threshold optimization
            print("Performing threshold-optimized evaluation...")
            return self.evaluate_with_threshold_optimization(metric=optimization_metric)

        # Original evaluation with fixed threshold
        print(f"Evaluating model performance with threshold {threshold}...")

        # Get predictions
        y_pred_proba = self.predict_proba(self.X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate standard metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auroc = roc_auc_score(self.y_test, y_pred_proba)

        # Calculate precision-recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(
            self.y_test, y_pred_proba
        )
        auc_pr = auc(recall_vals, precision_vals)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auroc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Create visualization plots
        self._plot_roc_curve(self.y_test, y_pred_proba)
        self._plot_pr_curve(self.y_test, y_pred_proba)
        self._plot_confusion_matrix(self.y_test, y_pred)

        if run_comprehensive:
            comparison_results = self.evaluate_comprehensive()
            print("\nComparison of Different Threshold Approaches:")
            print(comparison_results)

        # Create evaluation results dict with converted values
        evaluation_results = {
            "model": self.model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auroc),
            "auc_pr": float(auc_pr),
            "threshold": float(threshold),
            "train_time": (
                float(self.train_time) if hasattr(self, "train_time") else None
            ),
            "feature_importance": (
                [
                    {
                        "feature": self._convert_to_python_type(row["feature"]),
                        "importance": float(row["importance"]),
                    }
                    for _, row in self.feature_importance.iterrows()
                ]
                if hasattr(self, "feature_importance")
                and self.feature_importance is not None
                else None
            ),
        }

        # Save to JSON
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)

        return evaluation_results

    def evaluate_ensemble(self, threshold=0.45):
        """
        Evaluate ensemble by averaging predictions from all ensemble models.
        """
        if not hasattr(self, "ensemble_probas"):
            raise ValueError("Run train_ensemble() first")

        avg_proba = np.mean(np.array(self.ensemble_probas), axis=0)
        y_pred = (avg_proba >= threshold).astype(int)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, avg_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, avg_proba)
        auc_pr = auc(recall_vals, precision_vals)

        print("\n=== ENSEMBLE EVALUATION ===")
        print(f"Threshold: {threshold:.2f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        print(f"AUC-PR:    {auc_pr:.4f}")


    def predict(self, X):
        """
        Make binary predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first")

        # Convert pandas Series/DataFrame to numpy arrays if needed
        X_data = X.values if hasattr(X, "values") else X

        return self.model.predict(X_data)

    def predict_proba(self, X):
        """
        Get probability predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first")

        # Convert pandas Series/DataFrame to numpy arrays if needed
        X_data = X.values if hasattr(X, "values") else X

        # Get positive class probability
        return self.model.predict_proba(X_data)[:, 1]

    def evaluate_with_threshold_optimization(
        self, metric="clinical_utility", recall_weight=1.5, beta=1.5
    ):
        """
        Evaluate the model with threshold optimization to balance precision and recall
        based on clinical priorities.

        Parameters:
        -----------
        metric : str
            Metric to optimize for: 'f1', 'clinical_utility' (weights recall higher), or 'balanced'

        Returns:
        --------
        dict : Evaluation results with the optimal threshold
        """
        if self.model is None:
            raise ValueError("You must train the model first")

        print("Evaluating model with threshold optimization...")

        # Get probability predictions
        y_pred_proba = self.predict_proba(self.X_test)

        # Try different thresholds to find the optimal one
        thresholds = np.linspace(0.1, 0.9, 17)
        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate metrics at this threshold
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            # Clinical utility score - prioritizing recall over precision
            # Higher weight on recall = fewer missed mortality cases
            if metric == "clinical_utility":
                utility = ((recall * recall_weight) + precision) / (recall_weight + 1)
            elif metric == "balanced":
                utility = (recall + precision) / 2  # Equal weight
            # Inside evaluate_with_threshold_optimization method:
            elif metric == "balanced_f_beta":
                utility = (
                    ((1 + beta**2) * precision * recall)
                    / ((beta**2 * precision) + recall)
                    if precision + recall > 0
                    else 0
                )
            else:  # Default to F1
                utility = f1

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "utility": utility,
                }
            )

        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)

        # Find threshold that maximizes the chosen metric
        best_idx = results_df["utility"].idxmax()
        optimal_threshold = results_df.loc[best_idx, "threshold"]

        print(f"Optimal threshold: {optimal_threshold:.2f} (optimized for {metric})")
        print(
            f"At optimal threshold - Precision: {results_df.loc[best_idx, 'precision']:.4f}, "
            f"Recall: {results_df.loc[best_idx, 'recall']:.4f}, "
            f"F1: {results_df.loc[best_idx, 'f1']:.4f}"
        )

        # Calculate AUC-ROC and AUC-PR
        auroc = roc_auc_score(self.y_test, y_pred_proba)

        # Calculate precision-recall curve and AUC
        precision_vals, recall_vals, _ = precision_recall_curve(
            self.y_test, y_pred_proba
        )
        auc_pr = auc(recall_vals, precision_vals)

        print(f"AUC-ROC: {auroc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")

        # Create visualization plots
        self._plot_threshold_optimization(results_df, optimal_threshold)
        self._plot_precision_recall_tradeoff(results_df, optimal_threshold)
        self._plot_roc_curve(self.y_pred, y_pred_proba)
        self._plot_pr_curve(self.y_pred, y_pred_proba)

        # Get predictions with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

        # Print detailed classification report with optimal threshold
        print("\nClassification Report (optimal threshold):")
        print(classification_report(self.y_test, y_pred_optimal))

        # Create confusion matrix visualization
        self._plot_confusion_matrix(self.y_test, y_pred_optimal)

        # Prepare evaluation results with optimal threshold
        best_result = results_df.loc[best_idx].to_dict()

        # Create evaluation results dict with default Python types
        evaluation_results = {
            "model": self.model_name,
            "accuracy": float(best_result["accuracy"]),
            "precision": float(best_result["precision"]),
            "recall": float(best_result["recall"]),
            "f1_score": float(best_result["f1"]),
            "auc_roc": float(auroc),
            "auc_pr": float(auc_pr),
            "threshold": {
                "default": 0.5,
                "optimal": float(optimal_threshold),
                "optimization_metric": metric,
            },
            "train_time": (
                float(self.train_time) if hasattr(self, "train_time") else None
            ),
            "feature_importance": (
                [
                    {
                        "feature": self._convert_to_python_type(row["feature"]),
                        "importance": float(row["importance"]),
                    }
                    for _, row in self.feature_importance.iterrows()
                ]
                if hasattr(self, "feature_importance")
                and self.feature_importance is not None
                else None
            ),
        }

        # Save to JSON
        results_path = os.path.join(self.output_dir, "enhanced_evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)

        # Save threshold results for reference
        threshold_path = os.path.join(self.output_dir, "threshold_optimization.csv")
        results_df.to_csv(threshold_path, index=False)
        print(f"Threshold optimization results saved to {threshold_path}")

        # Calculate additional clinically relevant metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_optimal).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        print(f"Specificity: {specificity:.4f}")
        print(f"Negative Predictive Value: {npv:.4f}")

        # Add these to your evaluation results
        evaluation_results["specificity"] = float(specificity)
        evaluation_results["npv"] = float(npv)
        evaluation_results["pos_neg_ratio"] = (
            float((tp + fp) / (tn + fn)) if (tn + fn) > 0 else float("inf")
        )

        return evaluation_results

    def evaluate_comprehensive(self):
        """Evaluate model with multiple threshold optimization approaches for comparison"""

        # Standard evaluation (0.5 threshold)
        std_results = self.evaluate(
            threshold=0.5, optimize_threshold=False, run_comprehensive=False
        )

        # Clinical utility (high recall)
        clinical_results = self.evaluate_with_threshold_optimization(
            metric="clinical_utility", recall_weight=1.5
        )

        # F1 optimized (balanced precision-recall)
        f1_results = self.evaluate_with_threshold_optimization(metric="f1")

        # Balanced approach (custom formula)
        balanced_results = self.evaluate_with_threshold_optimization(
            metric="balanced_f_beta"
        )

        # Compile comparison table
        comparison = pd.DataFrame(
            {
                "Metric": [
                    "Threshold",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Specificity",
                    "NPV",
                    "AUC-ROC",
                ],
                "Standard (0.5)": [
                    0.5,
                    std_results["accuracy"],
                    std_results["precision"],
                    std_results["recall"],
                    std_results["f1_score"],
                    "N/A",
                    "N/A",
                    std_results["auc_roc"],
                ],
                "Clinical": [
                    clinical_results["threshold"]["optimal"],
                    clinical_results["accuracy"],
                    clinical_results["precision"],
                    clinical_results["recall"],
                    clinical_results["f1_score"],
                    clinical_results["specificity"],
                    clinical_results["npv"],
                    clinical_results["auc_roc"],
                ],
                "F1 Optimized": [
                    f1_results["threshold"]["optimal"],
                    f1_results["accuracy"],
                    f1_results["precision"],
                    f1_results["recall"],
                    f1_results["f1_score"],
                    f1_results["specificity"],
                    f1_results["npv"],
                    f1_results["auc_roc"],
                ],
                "Balanced": [
                    balanced_results["threshold"]["optimal"],
                    balanced_results["accuracy"],
                    balanced_results["precision"],
                    balanced_results["recall"],
                    balanced_results["f1_score"],
                    balanced_results["specificity"],
                    balanced_results["npv"],
                    balanced_results["auc_roc"],
                ],
            }
        )

        # Save comparison
        comparison.to_csv(
            os.path.join(self.output_dir, "threshold_comparison.csv"), index=False
        )

        # Create visualization of different operating points
        self._plot_threshold_comparison(comparison)

        return comparison

    def _plot_clinical_cost(self, results_df):
        """Plot clinical cost based on different misclassification penalties"""
        plt.figure(figsize=(12, 8))

        # Define costs
        fn_costs = [
            1,
            2,
            4,
            8,
            16,
        ]  # Range of potential costs for missing a mortality case

        for fn_cost in fn_costs:
            costs = []
            for _, row in results_df.iterrows():
                threshold = row["threshold"]
                y_pred = (self.predict_proba(self.X_test) >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

                # Cost function: FP cost = 1, FN cost = variable
                total_cost = fp + (fn * fn_cost)
                costs.append(total_cost)

            plt.plot(
                results_df["threshold"], costs, label=f"FN Cost = {fn_cost}x FP Cost"
            )

        plt.xlabel("Threshold")
        plt.ylabel("Total Misclassification Cost")
        plt.title("Clinical Cost by Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.output_dir, "clinical_cost_analysis.png"), dpi=300
        )
        plt.close()

    def _plot_threshold_optimization(self, results_df, optimal_threshold):
        """Plot threshold optimization results"""
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_df["threshold"], results_df["precision"], "b-", label="Precision"
        )
        plt.plot(results_df["threshold"], results_df["recall"], "g-", label="Recall")
        plt.plot(results_df["threshold"], results_df["f1"], "r-", label="F1 Score")
        plt.plot(
            results_df["threshold"],
            results_df["utility"],
            "m-",
            label="Clinical Utility",
        )
        plt.axvline(
            x=optimal_threshold,
            color="k",
            linestyle="--",
            label=f"Optimal Threshold = {optimal_threshold:.2f}",
        )
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Optimization")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(
            os.path.join(self.output_dir, "threshold_optimization.png"), dpi=300
        )
        plt.close()

    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"), dpi=300)
        plt.close()

    def _plot_pr_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 8))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "pr_curve.png"), dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Survived", "Died"],
            yticklabels=["Survived", "Died"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    # Add this method to your class
    def _plot_precision_recall_tradeoff(self, results_df, optimal_threshold):
        """Plot precision-recall tradeoff curve with threshold values"""
        plt.figure(figsize=(10, 8))

        # Create twin axis for displaying thresholds
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Plot precision and recall curves
        ax1.plot(
            results_df["threshold"],
            results_df["precision"],
            "b-",
            linewidth=2,
            label="Precision",
        )
        ax1.plot(
            results_df["threshold"],
            results_df["recall"],
            "g-",
            linewidth=2,
            label="Recall",
        )

        # Find intersection point (if it exists)
        idx = np.argmin(np.abs(results_df["precision"] - results_df["recall"]))
        balanced_threshold = results_df.loc[idx, "threshold"]

        # Highlight points
        ax1.axvline(
            x=optimal_threshold,
            color="r",
            linestyle="--",
            label=f"Optimal: {optimal_threshold:.2f}",
        )
        ax1.axvline(
            x=balanced_threshold,
            color="k",
            linestyle=":",
            label=f"Balanced: {balanced_threshold:.2f}",
        )

        # Add threshold values as text annotations
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if threshold in results_df["threshold"].values:
                idx = results_df[results_df["threshold"] == threshold].index[0]
                prec = results_df.loc[idx, "precision"]
                rec = results_df.loc[idx, "recall"]
                ax1.annotate(
                    f"{threshold:.1f}",
                    (threshold, (prec + rec) / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                )

        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Score")
        ax1.set_title("Precision-Recall Tradeoff by Threshold")
        ax1.legend(loc="best")
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "precision_recall_tradeoff.png"), dpi=300
        )
        plt.close()

    def _plot_threshold_comparison(self, comparison_df):
        """Plot comparison of different threshold optimization approaches"""
        # Extract the threshold values and performance metrics
        thresholds = comparison_df.iloc[0, 1:].values.astype(float)
        precision_values = comparison_df.iloc[2, 1:].values.astype(float)
        recall_values = comparison_df.iloc[3, 1:].values.astype(float)

        # Create plot for different operating points
        plt.figure(figsize=(10, 8))

        # Plot PR curve if we have the data
        if hasattr(self, "X_test") and hasattr(self, "y_test"):
            y_pred_proba = self.predict_proba(self.X_test)
            precision_curve, recall_curve, _ = precision_recall_curve(
                self.y_test, y_pred_proba
            )
            plt.plot(recall_curve, precision_curve, "b-", alpha=0.2, linewidth=1)

        # Plot the operating points
        strategies = comparison_df.columns[1:]
        colors = ["red", "green", "blue", "purple"]
        markers = ["o", "s", "D", "^"]

        for i, strategy in enumerate(strategies):
            plt.scatter(
                recall_values[i],
                precision_values[i],
                s=100,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f"{strategy} (t={thresholds[i]:.2f})",
            )

        # Add baseline
        no_skill = np.sum(self.y_test) / len(self.y_test)
        plt.plot([0, 1], [no_skill, no_skill], "k--", label="No Skill")

        # Format the plot
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Operating Points")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")

        # Add annotations with F1 scores
        f1_values = comparison_df.iloc[4, 1:].values.astype(float)
        for i, strategy in enumerate(strategies):
            plt.annotate(
                f"F1={f1_values[i]:.2f}",
                (recall_values[i], precision_values[i]),
                xytext=(7, -5),
                textcoords="offset points",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "threshold_comparison_pr.png"), dpi=300
        )

        # Create a second plot showing the metrics comparison as a bar chart
        metrics_to_plot = ["Precision", "Recall", "F1 Score", "Specificity"]
        rows = [
            comparison_df.index[comparison_df["Metric"] == m].tolist()[0]
            for m in metrics_to_plot
        ]

        plt.figure(figsize=(12, 8))

        # Create grouped bar chart
        x = np.arange(len(metrics_to_plot))
        width = 0.2

        for i, strategy in enumerate(strategies):
            offset = width * (i - len(strategies) / 2 + 0.5)
            values = comparison_df.loc[rows, strategy].values
            values = [
                float(v) if isinstance(v, str) and v != "N/A" else np.nan
                for v in values
            ]
            plt.bar(
                x + offset, values, width, label=f"{strategy} (t={thresholds[i]:.2f})"
            )

        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("Performance Metrics by Threshold Strategy")
        plt.xticks(x, metrics_to_plot)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "threshold_comparison_metrics.png"), dpi=300
        )
        plt.close()

    def _convert_to_python_type(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def apply_smote(self):
        """Apply SMOTE to training data"""
        print("Applying SMOTE for class imbalance...")
        
        # Use the existing handle_class_imbalance method from base class
        self.X_train_smote, self.y_train_smote = self.handle_class_imbalance(sampling_strategy="auto")
        
        print("Class distribution after SMOTE: ")
        print(pd.Series(self.y_train_smote).value_counts().sort_index())
        
        return self.X_train_smote, self.y_train_smote
    
    def evaluate_ensemble_comprehensive(self):
        """
        Evaluate ensemble with multiple thresholds and save results
        """
        if not hasattr(self, 'ensemble_avg_proba'):
            raise ValueError("Train ensemble first")
        
        # Define different thresholds to test
        thresholds = {
            'standard': 0.50,
            'f1_optimized': 0.45,
            'balanced': 0.30,
            'high_sensitivity': 0.20,
            'clinical_utility': 0.10
        }
        
        results = []
        
        print("\n=== COMPREHENSIVE ENSEMBLE EVALUATION ===")
        
        for threshold_name, threshold in thresholds.items():
            print(f"\nEvaluating ensemble with {threshold_name} threshold ({threshold})...")
            
            # Calculate predictions with current threshold
            y_pred = (self.ensemble_avg_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Calculate AUC scores
            auc_roc = roc_auc_score(self.y_test, self.ensemble_avg_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.ensemble_avg_proba)
            auc_pr = auc(recall_curve, precision_curve)
            
            # Calculate specificity and NPV
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            result = {
                'threshold_name': threshold_name,
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'specificity': specificity,
                'npv': npv
            }
            
            results.append(result)
            
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1 Score:    {f1:.4f}")
            print(f"AUC-ROC:     {auc_roc:.4f}")
            print(f"AUC-PR:      {auc_pr:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"NPV:         {npv:.4f}")
            
            # Classification report
            print(f"\nClassification Report ({threshold_name} threshold):")
            print(classification_report(self.y_test, y_pred))
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        ensemble_results_path = os.path.join(self.output_dir, "ensemble_threshold_results.csv")
        results_df.to_csv(ensemble_results_path, index=False)
        print(f"\nEnsemble threshold results saved to: {ensemble_results_path}")
        
        # Create comparison table
        print("\n=== ENSEMBLE THRESHOLD COMPARISON ===")
        comparison_df = results_df[['threshold_name', 'threshold', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']]
        print(comparison_df.round(4).to_string(index=False))
        
        # Save comparison table
        comparison_path = os.path.join(self.output_dir, "ensemble_comparison_table.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        return results_df

    def plot_ensemble_threshold_analysis(self):
        """Plot ensemble performance across different thresholds"""
        if not os.path.exists(os.path.join(self.output_dir, "ensemble_threshold_results.csv")):
            print("No ensemble threshold results found. Run evaluate_ensemble_comprehensive first.")
            return
        
        results_df = pd.read_csv(os.path.join(self.output_dir, "ensemble_threshold_results.csv"))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Precision vs Recall
        ax1.plot(results_df['threshold'], results_df['precision'], 'o-', label='Precision', color='blue')
        ax1.plot(results_df['threshold'], results_df['recall'], 's-', label='Recall', color='red')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision vs Recall by Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score by Threshold
        ax2.plot(results_df['threshold'], results_df['f1_score'], 'o-', color='green')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score by Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs Specificity
        ax3.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy', color='purple')
        ax3.plot(results_df['threshold'], results_df['specificity'], 's-', label='Specificity', color='orange')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Accuracy vs Specificity by Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: AUC Scores
        ax4.plot(results_df['threshold'], results_df['auc_roc'], 'o-', label='AUC-ROC', color='brown')
        ax4.plot(results_df['threshold'], results_df['auc_pr'], 's-', label='AUC-PR', color='pink')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('AUC Score')
        ax4.set_title('AUC Scores by Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ensemble_threshold_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ensemble threshold analysis plot saved to: {os.path.join(self.output_dir, 'ensemble_threshold_analysis.png')}")
