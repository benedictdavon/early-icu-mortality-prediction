import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# Rest of your imports
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import time
from datetime import datetime

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
        
        y = data['mortality']
        # First remove any ID columns or timestamp columns that shouldn't be features
        X = data.drop(columns=['mortality', 'subject_id', 'hadm_id', 'stay_id'], errors='ignore')
        
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
        
        # First split: separate test set (20% of data)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Second split: separate validation set (25% of train_val, which is 20% of original data)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

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
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        # Print metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{self.model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        # Generate precision-recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {pr_auc:.4f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()
        
        # Save evaluation results
        evaluation_results = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc_roc': auc_score,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'threshold': threshold
        }
        
        # Save as JSON
        import json
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        return evaluation_results
    
    def evaluate_validation(self, threshold=0.5):
        """Evaluate model performance on validation set"""
        print("Evaluating model on validation set...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
                
        # Get predictions
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(self.y_val, y_pred)
        prec = precision_score(self.y_val, y_pred)
        rec = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        auc_score = roc_auc_score(self.y_val, y_pred_proba)
        
        # Print metrics
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation Precision: {prec:.4f}")
        print(f"Validation Recall: {rec:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation AUC-ROC: {auc_score:.4f}")
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc_roc': auc_score
        }

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
            
            print(f"CV {metric}: {mean_score:.4f} ± {std_score:.4f}")
            
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
        """Find the optimal threshold for classification"""
        print("Finding optimal classification threshold...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Get predictions
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)  # Avoid div by 0
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"F1 score at optimal threshold: {best_f1:.4f}")
        
        # Plot F1 score vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
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
        
        # Calculate weighted cost at different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        costs = []
        f1_scores = []  # Track F1 scores for each threshold
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate total cost: FP*fp_cost + FN*fn_cost
            total_cost = (fp * fp_cost) + (fn * fn_cost)
            costs.append(total_cost)
            
            # Also calculate F1 score for reporting
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        # Find threshold with minimum cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        best_f1 = f1_scores[optimal_idx]
        
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
        
        # Create metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'train_time': self.train_time,
            'train_samples': self.X_train.shape[0],
            'feature_count': self.X_train.shape[1],
            'best_params': self.best_params,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model and metadata together
        joblib.dump({'model': self.model, 'metadata': metadata}, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model"""
        print(f"Loading model from {model_path}")
        
        # Load model and metadata
        model_data = joblib.load(model_path)
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