import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import time
from datetime import datetime

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

from data.schema import assert_no_leakage_columns, build_feature_matrix
from data.splitting import make_train_valid_test_split
from evaluation.metrics import binary_classification_metrics
from evaluation.plots import (
    evaluation_plot_paths,
    save_calibration_curve,
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
)
from evaluation.reporting import evaluate_validation_and_test, save_aggregate_report
from evaluation.thresholds import select_optimal_threshold
from models.base.persistence import load_model_bundle, save_model_bundle


def _simple_imputer(strategy):
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


class ICUMortalityBaseModel:
    """
    Base class for ICU mortality prediction models with shared functionality:
    - Data loading and preparation
    - Class imbalance handling
    - Cross-validation
    - Feature importance analysis
    - Comprehensive evaluation metrics
    - Model persistence
    - Visualization
    """
    
    def __init__(self, model_name='base_model', output_dir='../../results/base_model'):
        """Initialize the model with default parameters"""
        self.model = None
        self.model_name = model_name
        self.best_params = None
        self.feature_importance = None
        self.output_dir = os.path.join(output_dir, model_name)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_time = None
        self.feature_names = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self, data_path):
        """Load preprocessed data with train/validation/test split"""
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Check if mortality column exists
        if 'mortality' not in data.columns:
            raise ValueError("Dataset must contain 'mortality' column as target variable")
        
        # Extract target and features
        print("Columns before dropping:", data.columns.tolist())
        
        y = data['mortality'].astype(int)
        patient_ids = data['subject_id'] if 'subject_id' in data.columns else None

        # Remove target, identifiers, and obvious outcome proxies before any
        # preprocessing fit step. Identifiers are retained only in memory for
        # patient-level split separation.
        X, dropped_feature_columns = build_feature_matrix(data, target_col='mortality')
        self.dropped_feature_columns = dropped_feature_columns
        if dropped_feature_columns:
            print(
                "Dropped non-feature/leakage-guard columns:",
                dropped_feature_columns,
            )
        
        # Detect and handle datetime columns
        datetime_pattern = r'\d{4}-\d{2}-\d{2}.*|.*\d{2}:\d{2}:\d{2}'
        potential_datetime_cols = []
        
        for col in X.columns:
            # Check a sample of non-null values to see if they look like dates
            sample = X[col].dropna().astype(str).sample(min(5, len(X[col].dropna())))
            if any(sample.str.match(datetime_pattern)):
                potential_datetime_cols.append(col)
        
        if potential_datetime_cols:
            print(f"Detected potential datetime columns: {potential_datetime_cols}")
            print("These columns will be dropped to avoid imputation issues")
            X = X.drop(columns=potential_datetime_cols)
        
        # NEW: Identify and handle categorical columns
        categorical_cols = []
        for col in X.columns:
            # Check if column has string values that aren't numbers
            if X[col].dtype == 'object':
                sample = X[col].dropna().astype(str).sample(min(5, len(X[col].dropna())))
                if any(not val.replace('.', '', 1).isdigit() for val in sample if val):
                    categorical_cols.append(col)
        
        if categorical_cols:
            print(f"Detected categorical columns: {categorical_cols}")
            print("Converting categorical columns to numeric using one-hot encoding")
            
            # Option 1: One-hot encode categorical columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            # Option 2: Or drop them if you prefer not to expand dimensions
            # print("These columns will be dropped to avoid imputation issues")
            # X = X.drop(columns=categorical_cols)
    
        print("\nColumns after preprocessing:", X.columns.tolist())
        
        # Report class distribution
        class_counts = y.value_counts()
        print(f"Class distribution: \n{class_counts}")
        print(f"Mortality rate: {class_counts[1] / len(y):.2%}")
        
        assert_no_leakage_columns(X.columns, target_col='mortality')

        split_indices = make_train_valid_test_split(
            y,
            patient_ids=patient_ids,
            test_size=0.20,
            valid_size=0.20,
            stratify=True,
            group_by_patient=patient_ids is not None,
            random_state=42,
        )
        self.split_indices = split_indices

        X_train = X.iloc[split_indices['train']]
        X_val = X.iloc[split_indices['validation']]
        X_test = X.iloc[split_indices['test']]
        y_train = y.iloc[split_indices['train']]
        y_val = y.iloc[split_indices['validation']]
        y_test = y.iloc[split_indices['test']]
        
        # Handle missing values
        imputer = _simple_imputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
        self.imputer = imputer

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        print(f"Data loaded successfully with shapes:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_val: {self.X_val.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_val: {self.y_val.shape}")
        print(f"  y_test: {self.y_test.shape}")
        
        return self.X_train, self.y_train
    
    def handle_class_imbalance(self, sampling_strategy='auto'):
        """Apply SMOTE to handle class imbalance"""
        print("Applying SMOTE for class imbalance...")

        if SMOTE is None:
            raise ImportError(
                "SMOTE requires imbalanced-learn. Install it or use a non-resampling imbalance strategy."
            )
        
        if self.X_train is None:
            raise ValueError("Data must be loaded first using load_data()")
            
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        # Update training data
        self.X_train = X_resampled
        self.y_train = y_resampled
        
        # Report new class distribution
        class_counts = pd.Series(y_resampled).value_counts()
        print(f"Class distribution after SMOTE: \n{class_counts}")
        
        return X_resampled, y_resampled
    
    def tune_hyperparameters(self, cv=5, n_iter=50):
        """Tune hyperparameters - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement hyperparameter tuning")
    
    def train(self, params=None):
        """Train the model - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement model training")
    
    def evaluate(self, threshold=0.5):
        """Evaluate model performance on test set"""
        print("Evaluating model performance...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Get predictions
        y_pred_proba = self._predict_positive_proba(self.X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        evaluation_results = binary_classification_metrics(
            self.y_test, y_pred_proba, threshold=threshold
        )
        
        # Print metrics
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall: {evaluation_results['recall']:.4f}")
        print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
        print(f"AUC-ROC: {evaluation_results['auc_roc']:.4f}")
        print(f"Average Precision: {evaluation_results['average_precision']:.4f}")
        print(f"Brier Score: {evaluation_results['brier_score']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        plot_paths = evaluation_plot_paths(self.output_dir)
        save_confusion_matrix(
            self.y_test,
            y_pred,
            plot_paths["confusion_matrix"],
            title=f"{self.model_name} Confusion Matrix",
        )
        save_roc_curve(
            self.y_test,
            y_pred_proba,
            plot_paths["roc_curve"],
            title=f"{self.model_name} ROC Curve",
        )
        save_precision_recall_curve(
            self.y_test,
            y_pred_proba,
            plot_paths["precision_recall_curve"],
            title=f"{self.model_name} Precision-Recall Curve",
        )
        save_calibration_curve(
            self.y_test,
            y_pred_proba,
            plot_paths["calibration_curve"],
            title=f"{self.model_name} Calibration Curve",
        )
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        return evaluation_results
    
    def evaluate_validation(self, threshold=0.5):
        """Evaluate model performance on validation set"""
        print("Evaluating model on validation set...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
                
        # Get predictions
        y_pred_proba = self._predict_positive_proba(self.X_val)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        results = binary_classification_metrics(
            self.y_val, y_pred_proba, threshold=threshold
        )
        
        # Print metrics
        print(f"Validation Accuracy: {results['accuracy']:.4f}")
        print(f"Validation Precision: {results['precision']:.4f}")
        print(f"Validation Recall: {results['recall']:.4f}")
        print(f"Validation F1 Score: {results['f1_score']:.4f}")
        print(f"Validation AUC-ROC: {results['auc_roc']:.4f}")
        print(f"Validation Average Precision: {results['average_precision']:.4f}")
        print(f"Validation Brier Score: {results['brier_score']:.4f}")
        
        return {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'auc_roc': results['auc_roc'],
            'average_precision': results['average_precision'],
            'brier_score': results['brier_score'],
        }

    def evaluate_threshold_policies(self):
        """Select threshold policies on validation and apply them to test."""
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
        if self.X_val is None or self.X_test is None:
            raise ValueError("Validation and test data are required")

        p_valid = self._predict_positive_proba(self.X_val)
        p_test = self._predict_positive_proba(self.X_test)

        report = evaluate_validation_and_test(
            model_name=self.model_name,
            y_valid=self.y_val,
            p_valid=p_valid,
            y_test=self.y_test,
            p_test=p_test,
        )
        saved_paths = save_aggregate_report(report, self.output_dir)
        print("Saved threshold-policy aggregate report:")
        print(f"  JSON: {saved_paths['json']}")
        print(f"  CSV: {saved_paths['csv']}")
        return report

    def _predict_positive_proba(self, X):
        """Return positive-class probabilities using model-specific preprocessing."""
        custom_predict_proba = getattr(type(self), "predict_proba", None)
        if custom_predict_proba is not None:
            proba = custom_predict_proba(self, X)
        elif hasattr(self, "predict") and getattr(type(self), "predict", None) is not ICUMortalityBaseModel.__dict__.get("predict"):
            prediction = self.predict(X)
            if isinstance(prediction, tuple) and len(prediction) == 2:
                proba = prediction[1]
            else:
                proba = prediction
        else:
            proba = self.model.predict_proba(X)

        proba = np.asarray(proba)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba.astype(float)

    def cross_validate(self, cv=5):
        """Perform cross-validation with multiple metrics at once"""
        print(f"Performing {cv}-fold cross-validation...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Define scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Use cross_validate instead of cross_val_score for efficiency
        from sklearn.model_selection import cross_validate, StratifiedKFold
        
        # Set up the cross-validation splitter with stratification
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation for all metrics at once
        cv_results = cross_validate(
            self.model, self.X_train, self.y_train,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        # Extract and report results
        results_dict = {}
        for metric in scoring:
            score_key = f'test_{metric}'
            mean_score = cv_results[score_key].mean()
            std_score = cv_results[score_key].std()
            
            print(f"CV {metric}: {mean_score:.4f} +/- {std_score:.4f}")
            
            results_dict[metric] = {
                'mean': float(mean_score),
                'std': float(std_score),
                'scores': cv_results[score_key].tolist()
            }
        
        # Save cross-validation results
        import json
        with open(os.path.join(self.output_dir, 'cross_validation_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        return results_dict
    
    def calculate_permutation_importance(self, n_repeats=10, random_state=42):
        """Calculate permutation importance for features"""
        print("Calculating permutation feature importance...")
        
        if self.model is None or self.X_test is None:
            raise ValueError("Model must be trained and test data must be available")
        
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_result = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        
        # DEBUG: Check array lengths before creating DataFrame
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of permutation importances: {len(perm_result.importances_mean)}")
        print(f"X_test shape: {self.X_test.shape}")
        
        # Make sure feature names match the actual features used by the model
        if len(self.feature_names) != len(perm_result.importances_mean):
            print("WARNING: Feature names length doesn't match permutation importance length")
            print("Using generic feature names...")
            feature_names_to_use = [f"feature_{i}" for i in range(len(perm_result.importances_mean))]
        else:
            feature_names_to_use = self.feature_names
    
        # Create DataFrame with proper length checking
        perm_importance = pd.DataFrame({
            'feature': feature_names_to_use,
            'importance_mean': perm_result.importances_mean,
            'importance_std': perm_result.importances_std
        }).sort_values('importance_mean', ascending=False)
    
        # Save permutation importance
        perm_importance.to_csv(
            os.path.join(self.output_dir, f"{self.model_name}_permutation_importance.csv"),
            index=False
        )
    
        # Plot permutation importance (top 20 features)
        plt.figure(figsize=(10, 8))
        top_features = perm_importance.head(20)
    
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                 xerr=top_features['importance_std'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'{self.model_name.replace("_", " ").title()} - Permutation Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{self.model_name}_permutation_importance.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
        print(f"Permutation importance saved to: {os.path.join(self.output_dir, f'{self.model_name}_permutation_importance.csv')}")
    
        return perm_importance
    
    def find_optimal_threshold(self):
        """Find the optimal classification threshold on the validation set."""
        print("Finding optimal classification threshold on validation set...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data is required for threshold optimization")

        # Tune thresholds on validation data only. The test set is reserved for
        # final evaluation after the threshold has been selected.
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        best_threshold, best_row, threshold_results = select_optimal_threshold(
            self.y_val, y_pred_proba, strategy="f1"
        )
        best_f1 = best_row["f1"]
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"F1 score at optimal threshold: {best_f1:.4f}")
        
        # Plot F1 score vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_results["threshold"], threshold_results["f1"], label='F1 Score')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'Optimal threshold: {best_threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'{self.model_name} F1 Score vs Classification Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_optimization.png'), dpi=300)
        plt.close()
        
        return best_threshold, best_f1
    
    # Adjust classification threshold based on costs
    def find_optimal_threshold_with_costs(self, fn_cost=2.0, fp_cost=1.0):
        """Find optimal classification threshold considering different costs for FP and FN"""
        print(f"Finding optimal threshold with FN cost={fn_cost}, FP cost={fp_cost}...")
        
        y_true = self.y_val
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        optimal_threshold, best_row, _ = select_optimal_threshold(
            y_true, y_pred_proba, strategy="cost", fn_cost=fn_cost, fp_cost=fp_cost
        )
        best_f1 = best_row["f1"]
        
        print(f"Optimal cost-sensitive threshold: {optimal_threshold:.4f}")
        print(f"F1 score at optimal threshold: {best_f1:.4f}")
        
        return optimal_threshold, best_f1
    
    def save_model(self, filename=None):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.joblib"
        
        model_path = os.path.join(self.output_dir, filename)
        
        save_model_bundle(self, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model"""
        print(f"Loading model from {model_path}")
        
        model_data = load_model_bundle(model_path)
        self.model = model_data['model']
        
        # Extract metadata
        metadata = model_data['metadata']
        self.feature_names = metadata['feature_names']
        self.train_time = metadata['train_time']
        self.best_params = metadata['best_params']
        self.model_name = metadata['model_name']
        
        print(f"Loaded {self.model_name} model trained on {metadata['train_samples']} samples with {metadata['feature_count']} features")
        
        return self.model

    def analyze_shap_values(self, max_display=20):
        """Placeholder for SHAP analysis - implement in subclasses as needed"""
        print("SHAP analysis to be implemented per model type")
        pass
