import os
import sys
import argparse
from pathlib import Path

# Add the src directory to path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent))
from models.random_forest import ICUMortalityRandomForest
from models.logistic_regression import ICUMortalityLogisticRegression
from models.xgboost import ICUMortalityXGBoost
from models.rf_bagging import ICUMortalityRandomForestBagging


# Update the argument parser to include xgboost-ensemble
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a model for ICU mortality prediction')
    parser.add_argument('--model', type=str, default='random_forest', 
        choices=['random_forest', 'logistic_regression', 'xgboost', 'xgboost_ensemble', 'rf_bagging'],
        help='Model type to train (default: random_forest)')

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to preprocessed data (default: data/processed/preprocessed_features.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: results/<model_type>)')
    parser.add_argument('--no-tune', dest='tune', action='store_false',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false',
                        help='Disable early stopping (for tree-based models)')
    parser.add_argument('--no-shap', dest='shap', action='store_false',
                        help='Skip SHAP value analysis')
    parser.add_argument('--ensemble-size', type=int, default=5, 
                        help='Number of models in ensemble (default: 5)')
    parser.set_defaults(tune=True, early_stopping=True, shap=True)
    return parser.parse_args()

def get_model_class(model_type):
    """Return the appropriate model class based on model_type"""
    if model_type == 'random_forest':
        return ICUMortalityRandomForest
    elif model_type == 'logistic_regression':
        return ICUMortalityLogisticRegression
    elif model_type == 'xgboost' or model_type == 'xgboost-ensemble':
        return ICUMortalityXGBoost
    elif model_type == 'rf_bagging':
        return ICUMortalityRandomForestBagging
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_and_evaluate_model(model, data_path, tune=True, early_stopping=True, use_shap=True):
    """Train and evaluate a model"""
    # Load data
    print("Loading data...")
    model.load_data(data_path)
    
    # For logistic regression, select important features first
    if isinstance(model, ICUMortalityLogisticRegression):
        print("Selecting important features...")
        model.select_important_features()

    # Tune hyperparameters if requested
    if tune:
        print("Tuning hyperparameters...")
        model.tune_hyperparameters(cv=5, n_iter=50)
    
    # Train model with early stopping if applicable and requested
    print("Training model...")
    if hasattr(model, 'train_with_early_stopping') and early_stopping:
        model.train_with_early_stopping()
    else:
        model.train()
    
    # Perform cross-validation
    print("Performing cross-validation...")
    model.cross_validate(cv=5)
    
    # Find optimal threshold
    print("Finding optimal classification threshold...")
    best_threshold, best_f1 = model.find_optimal_threshold()
    print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    
    # Evaluate model with optimal threshold
    print("Evaluating model...")
    
    # Special handling for XGBoost to use comprehensive evaluation
    if isinstance(model, ICUMortalityXGBoost):
        print("Performing comprehensive evaluation for XGBoost model...")
        evaluation = model.evaluate(
            threshold=best_threshold, 
            optimize_threshold=True,
            optimization_metric='clinical_utility', 
            run_comprehensive=True
        )
        
        # After standard evaluation, specifically run comprehensive evaluation
        print("\n=== Comprehensive Threshold Analysis ===")
        comparison_results = model.evaluate_comprehensive()
        print("Comparison of different threshold approaches:")
        print(comparison_results)
        
        # Also run clinical cost analysis if available
        if hasattr(model, 'analyze_clinical_cost'):
            print("\n=== Clinical Cost Analysis ===")
            model.analyze_clinical_cost()
    else:
        # Standard evaluation for other models
        evaluation = model.evaluate(threshold=best_threshold)
    
    # Evaluate on validation set too
    print("Evaluating on validation set...")
    val_evaluation = model.evaluate_validation(threshold=best_threshold)
    
    # Calculate feature importance
    print("Calculating feature importance...")
    if hasattr(model, 'calculate_permutation_importance'):
        model.calculate_permutation_importance()
    
    # Calculate SHAP values if applicable and requested
    if use_shap and hasattr(model, 'analyze_shap_values'):
        print("Analyzing SHAP values...")
        try:
            model.analyze_shap_values()
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
    
    # Model-specific analysis
    if isinstance(model, ICUMortalityLogisticRegression):
        print("Generating coefficient analysis...")
        model.plot_coefficient_analysis()
        model.calculate_odds_ratios()
    
    # Save model
    print("Saving model...")
    model_path = model.save_model()
    print(f"Model saved to: {model_path}")
    
    return evaluation

def train_and_evaluate_ensemble(model, data_path, tune=False, ensemble_size=5):
    """Train and evaluate an ensemble of XGBoost models"""
    # Load data
    print("Loading data...")
    model.load_data(data_path)
    
    # Skip hyperparameter tuning for ensemble to save time
    if tune:
        print("Tuning hyperparameters...")
        model.tune_hyperparameters(cv=5, n_iter=50)
    else:
        print("Skipping hyperparameter tuning for ensemble (using default params)")
    
    # Define random seeds for ensemble diversity
    import random
    random.seed(42)
    seeds = [random.randint(1, 10000) for _ in range(ensemble_size)]
    
    # Apply SMOTE for class imbalance
    print("Applying SMOTE for class imbalance...")
    model.apply_smote()
    
    # Train ensemble
    print(f"Training XGBoost ensemble with {ensemble_size} models...")
    ensemble_models, ensemble_probas = model.train_ensemble(seeds=seeds)
    
    # Evaluate ensemble with comprehensive threshold analysis
    print("Evaluating ensemble model with multiple thresholds...")
    results_df = model.evaluate_ensemble_comprehensive()
    
    # Save ensemble models
    print("Saving ensemble models...")
    import joblib
    ensemble_dir = os.path.join(model.output_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for i, ensemble_model in enumerate(ensemble_models):
        model_path = os.path.join(ensemble_dir, f"xgboost_ensemble_{i}.joblib")
        joblib.dump(ensemble_model, model_path)
    
    # Save the ensemble probabilities for later use
    import numpy as np
    np.save(os.path.join(ensemble_dir, "ensemble_probabilities.npy"), np.array(ensemble_probas))
    
    print(f"Ensemble of {len(ensemble_models)} models saved to {ensemble_dir}")
    
    # Return the best performing metrics (F1-optimized)
    f1_optimized = results_df[results_df['threshold_name'] == 'f1_optimized'].iloc[0]
    
    return {
        "accuracy": f1_optimized['accuracy'],
        "precision": f1_optimized['precision'], 
        "recall": f1_optimized['recall'],
        "f1_score": f1_optimized['f1_score'],
        "auc_roc": f1_optimized['auc_roc']
    }

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Get model-specific display name
    model_display_names = {
        'random_forest': 'RANDOM FOREST',
        'logistic_regression': 'LOGISTIC REGRESSION',
        'xgboost': 'XGBOOST',
        'xgboost_ensemble': 'XGBOOST ENSEMBLE',
        'rf_bagging': 'RF + BAGGING',
    }

    model_display = model_display_names.get(args.model, args.model.upper())
    
    # Set up paths - for ensemble, use the xgboost data path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model == 'xgboost_ensemble':
        data_path_model = 'xgboost'
    elif args.model == 'rf_bagging':
        data_path_model = 'rf_bagging'
    else:
        data_path_model = args.model
    
    # Default paths if not provided
    data_path = args.data_path or os.path.join(base_dir, 'data', 'processed', f'preprocessed_{data_path_model}_features.csv')
    output_dir = args.output_dir or os.path.join(base_dir, 'results', args.model)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"TRAINING {model_display} MODEL FOR ICU MORTALITY PREDICTION")
    print("=" * 80)
    print(f"Model type: {args.model}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Hyperparameter tuning: {'Enabled' if args.tune else 'Disabled'}")
    if args.model == 'xgboost-ensemble':
        print(f"Ensemble size: {args.ensemble_size}")
    else:
        print(f"Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
        print(f"SHAP analysis: {'Enabled' if args.shap else 'Disabled'}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Make sure you've run the data preprocessing step first.")
        sys.exit(1)
    
    # Run the pipeline
    try:
        # Get the appropriate model class
        ModelClass = get_model_class(args.model)
        
        # Initialize model
        model = ModelClass(output_dir=output_dir)
        
        # Train and evaluate model based on type
        if args.model == 'xgboost-ensemble':
            evaluation = train_and_evaluate_ensemble(
                model=model, 
                data_path=data_path,
                tune=args.tune,
                ensemble_size=args.ensemble_size
            )
            
            print("\n" + "=" * 80)
            print("XGBOOST ENSEMBLE TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            # Display ensemble performance summary
            if evaluation:
                print("\nEnsemble Performance (F1-Optimized Threshold):")
                print(f"Accuracy:  {evaluation['accuracy']:.4f}")
                print(f"Precision: {evaluation['precision']:.4f}")
                print(f"Recall:    {evaluation['recall']:.4f}")
                print(f"F1 Score:  {evaluation['f1_score']:.4f}")
                print(f"AUC-ROC:   {evaluation['auc_roc']:.4f}")
        else:
            evaluation = train_and_evaluate_model(
                model=model, 
                data_path=data_path,
                tune=args.tune,
                early_stopping=args.early_stopping,
                use_shap=args.shap
            )
            
            print("\n" + "=" * 80)
            print("MODEL TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            # Display top features
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                top_features = model.feature_importance.head(10)
                print("\nTop 10 Most Important Features:")
                for i, (feature, importance) in enumerate(
                    zip(top_features['feature'], top_features['importance']), 1
                ):
                    print(f"{i}. {feature}: {importance:.4f}")
        
        # Print evaluation metrics (except for ensemble which handles its own display)
        if evaluation and args.model != 'xgboost-ensemble':
            print("\nModel Performance:")
            print(f"Accuracy:  {evaluation['accuracy']:.4f}")
            print(f"Precision: {evaluation['precision']:.4f}")
            print(f"Recall:    {evaluation['recall']:.4f}")
            print(f"F1 Score:  {evaluation['f1_score']:.4f}")
            print(f"AUC-ROC:   {evaluation['auc_roc']:.4f}")
                
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        print("\nStacktrace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)