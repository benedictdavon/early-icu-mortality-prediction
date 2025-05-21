import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import sys
import time
from pathlib import Path
from datetime import datetime
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import ICUMortalityBaseModel

# Filter specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C")
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty")


class ICUMortalityLogisticRegression(ICUMortalityBaseModel):
    """
    Logistic Regression model for ICU mortality prediction.
    Inherits from base model class to leverage common functionality.
    
    Features:
    - L1/L2 regularization
    - Class weight balancing
    - Feature selection
    - Probability calibration
    - Automated feature scaling
    """

    def __init__(self, output_dir="../../results"):
        """Initialize the Logistic Regression model"""
        super().__init__(model_name="logistic_regression", output_dir=output_dir)
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.coef_df = None

    def _preprocess_data(self, X):
        """Helper method to apply scaling and feature selection"""
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        return X_scaled

    def tune_hyperparameters(self, cv=5, n_iter=50):
        """Tune Logistic Regression hyperparameters using RandomizedSearchCV"""
        print("Tuning Logistic Regression hyperparameters...")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        try:
            # Define separate parameter distributions for different solvers
            param_distributions = [
                # For saga (supports all penalties)
                {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['saga'],
                    'class_weight': ['balanced', None],
                    'max_iter': [2000, 5000, 10000, 20000, 50000],  # Increase max_iter
                    'l1_ratio': [0.1, 0.5, 0.9]  # Only used with elasticnet
                },
                # For liblinear (l1 or l2 only)
                {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'class_weight': ['balanced', None],
                    'max_iter': [2000, 5000, 10000, 20000, 50000]
                },
                # For newton-cg, lbfgs (l2 or None)
                {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l2', None],
                    'solver': ['newton-cg', 'lbfgs'],
                    'class_weight': ['balanced', None],
                    'max_iter': [2000, 5000, 10000, 20000, 50000]
                }
            ]
            
            # Create base model
            log_reg = LogisticRegression(random_state=42)
            
            # Scale features for logistic regression
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            
            # Apply feature selection if needed
            if self.feature_selector is not None:
                X_train_scaled = self.feature_selector.transform(X_train_scaled)
            
            # Run search for each parameter distribution
            best_score = 0
            best_params = None
            
            for i, params in enumerate(param_distributions):
                print(f"\nTrying parameter distribution set {i+1}/{len(param_distributions)}...")
                
                try:
                    search = RandomizedSearchCV(
                        estimator=log_reg,
                        param_distributions=params,
                        n_iter=min(n_iter // len(param_distributions) + 1, 20),
                        cv=cv,
                        scoring='roc_auc',
                        random_state=42 + i,
                        n_jobs=-1,
                        verbose=0,
                        error_score=0.5  # Return 0.5 for failed fits
                    )
                    
                    # Fit on scaled data
                    search.fit(X_train_scaled, self.y_train)
                    
                    print(f"Best score for distribution {i+1}: {search.best_score_:.4f}")
                    print(f"Best params for distribution {i+1}: {search.best_params_}")
                    
                    # Track global best parameters
                    if search.best_score_ > best_score:
                        best_score = search.best_score_
                        best_params = search.best_params_
                        
                except Exception as e:
                    print(f"Error during hyperparameter search {i+1}: {str(e)}")
                    print("Continuing with next parameter set...")
            
            if best_params is None:
                print("All parameter searches failed. Using default parameters.")
                best_params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'saga',
                    'class_weight': 'balanced',
                    'max_iter': 10000
                }
                best_score = 0.0
            
            # Save and print best parameters
            self.best_params = best_params
            print(f"\nOverall best parameters: {self.best_params}")
            print(f"Best CV score: {best_score:.4f}")
            
            return self.best_params
        
        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")
            print("Using default parameters instead")
            self.best_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'saga',
                'class_weight': 'balanced',
                'max_iter': 10000
            }
            return self.best_params

    def select_important_features(self, threshold=0.01):
        """Select important features using L1 regularization"""
        print(f"Selecting important features using L1 regularization (threshold={threshold})...")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Use L1 penalty for feature selection
        selector = LogisticRegression(
            penalty='l1',
            C=0.1,  # Strong regularization
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Fit the selector
        selector.fit(X_train_scaled, self.y_train)
        
        # Get features with non-zero coefficients
        feature_importance = np.abs(selector.coef_[0])
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features_mask = feature_importance > threshold
        selected_indices = np.where(selected_features_mask)[0]
        
        # Get names of selected features
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} out of {len(self.feature_names)} features")
        print("Top 10 selected features:")
        for i, (feature, importance) in enumerate(zip(
            feature_importance_df['feature'][:10],
            feature_importance_df['importance'][:10]
        )):
            print(f"{i+1}. {feature}: {importance:.6f}")
        
        # Create and store feature selector
        self.feature_selector = SelectFromModel(
            selector, 
            prefit=True, 
            threshold=threshold
        )
        
        return self.selected_features

    def train(self, params=None):
        """Train the Logistic Regression model"""
        print("Training Logistic Regression model...")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        # Use tuned parameters if available, otherwise use default
        if params is None:
            if self.best_params is not None:
                params = self.best_params
            else:
                params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'saga',
                    'class_weight': 'balanced',
                    'max_iter': 1000
                }
        
        # Initialize model with clean parameters (remove l1_ratio if not elasticnet)
        if params.get('penalty') != 'elasticnet' and 'l1_ratio' in params:
            del params['l1_ratio']
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            print(f"Using {len(self.selected_features)} selected features")
            X_train_scaled = self.feature_selector.transform(X_train_scaled)
        
        # Rebalance training data
        X_resampled, y_resampled = self.handle_class_imbalance(sampling_strategy="auto")
        
        # Apply scaling to resampled data
        X_resampled_scaled = self.scaler.transform(X_resampled)
        
        # Apply feature selection to resampled data if needed
        if self.feature_selector is not None:
            X_resampled_scaled = self.feature_selector.transform(X_resampled_scaled)
        
        # Train model
        start_time = time.time()
        self.model = LogisticRegression(random_state=42, n_jobs=-1, **params)
        self.model.fit(X_resampled_scaled, y_resampled)
        end_time = time.time()
        
        self.train_time = end_time - start_time
        print(f"Model trained in {self.train_time:.2f} seconds")
        
        # Store coefficients
        if self.feature_selector is not None:
            feature_names = self.selected_features
        else:
            feature_names = self.feature_names
            
        self.coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        # Print top positive and negative coefficients
        print("\nTop 5 positive coefficients (increasing mortality risk):")
        top_pos = self.coef_df[self.coef_df['coefficient'] > 0].head(5)
        for _, row in top_pos.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
            
        print("\nTop 5 negative coefficients (decreasing mortality risk):")
        top_neg = self.coef_df[self.coef_df['coefficient'] < 0].head(5)
        for _, row in top_neg.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        # Store as feature importance for consistency with other models
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        return self.model

    def evaluate_validation(self, threshold=0.5):
        """Evaluate model on validation data"""
        print("Evaluating model on validation set...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Scale validation data
        X_val_scaled = self.scaler.transform(self.X_val)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {}
        metrics['accuracy'] = accuracy_score(self.y_val, y_pred)
        metrics['precision'] = precision_score(self.y_val, y_pred)
        metrics['recall'] = recall_score(self.y_val, y_pred)
        metrics['f1_score'] = f1_score(self.y_val, y_pred)
        metrics['auc_roc'] = roc_auc_score(self.y_val, y_pred_proba)
        
        # Print key metrics
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}")
        print(f"Validation Recall: {metrics['recall']:.4f}")
        print(f"Validation F1 Score: {metrics['f1_score']:.4f}")
        print(f"Validation AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Validation AUC: {metrics['auc_roc']:.4f}")
        
        return metrics

    def evaluate(self, threshold=0.5):
        """Override to handle scaling before prediction"""
        print("Evaluating model performance...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Scale test data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics using the correct method name from parent class
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        metrics['precision'] = precision_score(self.y_test, y_pred)
        metrics['recall'] = recall_score(self.y_test, y_pred)
        metrics['f1_score'] = f1_score(self.y_test, y_pred)
        metrics['auc_roc'] = roc_auc_score(self.y_test, y_pred_proba)
        
        # Print metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Generate classification report
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['0.0', '1.0']))
        
        return metrics
    
    def cross_validate(self, cv=5):
        """Override cross-validation to handle scaling"""
        print(f"Performing {cv}-fold cross-validation...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
        
        # Scale training data
        X_train_scaled = self.scaler.transform(self.X_train)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_train_scaled = self.feature_selector.transform(X_train_scaled)
        
        # Use parent implementation but with scaled data
        from sklearn.model_selection import cross_validate, StratifiedKFold
        
        # Define scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Set up cross-validation splitter with stratification
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model, X_train_scaled, self.y_train,
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
    
    def find_optimal_threshold(self):
        """Find optimal classification threshold with scaled data"""
        print("Finding optimal classification threshold...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        # Scale validation data
        X_val_scaled = self.scaler.transform(self.X_val)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Find optimal threshold using validation set
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.y_val, y_pred_proba)
        
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
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_optimization.png'), dpi=300)
        plt.close()
        
        return best_threshold, best_f1
    
    def find_optimal_threshold_with_costs(self, fn_cost=2.0, fp_cost=1.0):
        """Find optimal classification threshold considering different costs for FP and FN"""
        print(f"Finding optimal threshold with FN cost={fn_cost}, FP cost={fp_cost}...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
        
        # Use preprocessing helper
        X_val_scaled = self._preprocess_data(self.X_val)
        
        # Get probability predictions
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate weighted cost at different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        costs = []
        f1_scores = []  # Track F1 scores for each threshold
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            from sklearn.metrics import confusion_matrix, f1_score
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            
            # Calculate total cost: FP*fp_cost + FN*fn_cost
            total_cost = (fp * fp_cost) + (fn * fn_cost)
            costs.append(total_cost)
            
            # Also calculate F1 score for reporting
            f1 = f1_score(self.y_val, y_pred)
            f1_scores.append(f1)
        
        # Find threshold with minimum cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        best_f1 = f1_scores[optimal_idx]
        
        print(f"Optimal cost-sensitive threshold: {optimal_threshold:.4f}")
        print(f"F1 score at optimal threshold: {best_f1:.4f}")
        
        return optimal_threshold, best_f1
    
    def plot_coefficient_analysis(self):
        """Visualize model coefficients"""
        if self.coef_df is None:
            raise ValueError("Model must be trained first")
            
        # Plot top 20 coefficients
        plt.figure(figsize=(12, 10))
        
        # Sort by absolute coefficient but keep sign for coloring
        top_coef = self.coef_df.sort_values('abs_coefficient', ascending=False).head(20)
        
        # Create color map based on coefficient sign
        colors = ['red' if x < 0 else 'green' for x in top_coef['coefficient']]
        
        # Plot horizontal bar chart
        bars = plt.barh(top_coef['feature'], top_coef['coefficient'], color=colors)
        
        # Add vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Label the plot
        plt.xlabel('Coefficient Value')
        plt.title(f'{self.model_name} Top 20 Feature Coefficients')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Increases mortality risk'),
            Patch(facecolor='red', label='Decreases mortality risk')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'coefficient_analysis.png'), dpi=300)
        plt.close()
        
        # Save coefficients to CSV
        self.coef_df.to_csv(os.path.join(self.output_dir, 'model_coefficients.csv'), index=False)
    
    def calculate_odds_ratios(self):
        """Calculate and visualize odds ratios for easier interpretation"""
        if self.coef_df is None:
            raise ValueError("Model must be trained first")
            
        # Calculate odds ratios (e^coefficient)
        odds_df = self.coef_df.copy()
        odds_df['odds_ratio'] = np.exp(odds_df['coefficient'])
        odds_df['percent_change'] = (odds_df['odds_ratio'] - 1) * 100
        
        # Sort by absolute percent change
        odds_df['abs_percent_change'] = np.abs(odds_df['percent_change'])
        odds_df = odds_df.sort_values('abs_percent_change', ascending=False)
        
        # Plot top 20 odds ratios
        plt.figure(figsize=(12, 10))
        
        top_odds = odds_df.head(20).copy()
        
        # Use log scale for better visualization
        plt.barh(top_odds['feature'], top_odds['percent_change'])
        
        # Add vertical line at 0% change
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Label the plot
        plt.xlabel('Effect on Odds of Mortality (%)')
        plt.title(f'{self.model_name} Feature Effects on Mortality Risk')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'odds_ratio_analysis.png'), dpi=300)
        plt.close()
        
        # Print and save top odds ratios
        print("\nTop 5 features increasing mortality risk:")
        top_increase = odds_df[odds_df['percent_change'] > 0].head(5)
        for _, row in top_increase.iterrows():
            print(f"  {row['feature']}: +{row['percent_change']:.1f}% odds")
            
        print("\nTop 5 features decreasing mortality risk:")
        top_decrease = odds_df[odds_df['percent_change'] < 0].sort_values('percent_change').head(5)
        for _, row in top_decrease.iterrows():
            print(f"  {row['feature']}: {row['percent_change']:.1f}% odds")
        
        # Save odds ratios to CSV
        odds_df.to_csv(os.path.join(self.output_dir, 'odds_ratios.csv'), index=False)
        
        return odds_df
    
    def save_model(self, filename=None):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logreg_model_{timestamp}.joblib"
        
        model_path = os.path.join(self.output_dir, filename)
        
        # Create comprehensive metadata
        metadata = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'train_time': self.train_time,
            'train_samples': self.X_train.shape[0],
            'feature_count': self.X_train.shape[1],
            'best_params': self.best_params,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model, scaler, feature selector, and metadata together
        save_dict = {
            'model': self.model, 
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'metadata': metadata
        }
        
        joblib.dump(save_dict, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model"""
        print(f"Loading model from {model_path}")
        
        # Load model and metadata
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        
        # Extract metadata
        metadata = model_data['metadata']
        self.feature_names = metadata['feature_names']
        self.selected_features = metadata['selected_features']
        self.train_time = metadata['train_time']
        self.best_params = metadata['best_params']
        
        print(f"Loaded model trained on {metadata['train_samples']} samples")
        if self.selected_features:
            print(f"Using {len(self.selected_features)} selected features")
        
        return self.model
    
    def calculate_permutation_importance(self, n_repeats=10):
        """Calculate permutation feature importance"""
        print("Calculating permutation feature importance...")
        
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
        
        # Use the preprocessing helper method    
        X_test_scaled = self._preprocess_data(self.X_test)
        
        # Determine which feature names to use
        feature_names = self.selected_features if self.feature_selector is not None else self.feature_names
        
        # Calculate permutation importance
        try:
            result = permutation_importance(
                self.model, X_test_scaled, self.y_test, 
                n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            
            # Create DataFrame with results
            perm_importance = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Plot top 20 features (or all if less than 20)
            plt.figure(figsize=(12, 10))
            num_to_plot = min(20, len(perm_importance))
            sns.barplot(x='importance_mean', y='feature', data=perm_importance.head(num_to_plot))
            plt.title(f'{self.model_name} Permutation Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'permutation_importance.png'), dpi=300)
            plt.close()
            
            # Save importance values
            perm_importance.to_csv(os.path.join(self.output_dir, 'permutation_importance.csv'), index=False)
            
            return perm_importance
        
        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}")
            print("Skipping permutation importance calculation")
            return None
    
    def predict(self, X, threshold=0.5):
        """Predict on new data with proper preprocessing"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Check feature count
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, but got {X.shape[1]}")
            
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        # Get probability predictions
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return y_pred, y_pred_proba