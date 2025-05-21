import os
import sys
import argparse
from pathlib import Path

# Add the src directory to path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent))
from models.random_forest import ICUMortalityRandomForest

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a Random Forest model for ICU mortality prediction')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to preprocessed data (default: data/processed/preprocessed_features.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: results/random_forest)')
    parser.add_argument('--no-tune', dest='tune', action='store_false',
                        help='Skip hyperparameter tuning')
    parser.set_defaults(tune=True)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default paths if not provided
    data_path = args.data_path or os.path.join(base_dir, 'data', 'processed', 'preprocessed_features.csv')
    output_dir = args.output_dir or os.path.join(base_dir, 'results', 'random_forest')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TRAINING RANDOM FOREST MODEL FOR ICU MORTALITY PREDICTION")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Hyperparameter tuning: {'Enabled' if args.tune else 'Disabled'}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Make sure you've run the data preprocessing step first.")
        sys.exit(1)
    
    # Run the pipeline
    try:
        # Initialize model
        rf_model = ICUMortalityRandomForest(output_dir=output_dir)
        
        # Load data
        print("Loading data...")
        rf_model.load_data(data_path)
        

        # Tune hyperparameters if requested
        if args.tune:  # Fixed: Using args.tune instead of undefined tune variable
            print("Tuning hyperparameters...")
            rf_model.tune_hyperparameters(cv=5, n_iter=50)
        
        # Train model
        print("Training model...")
        rf_model.train()
        
        # Perform cross-validation
        print("Performing cross-validation...")
        rf_model.cross_validate(cv=5)
        
        # Find optimal threshold
        print("Finding optimal classification threshold...")
        best_threshold, best_f1 = rf_model.find_optimal_threshold()
        print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        
        # Evaluate model with optimal threshold
        print("Evaluating model...")
        evaluation = rf_model.evaluate(threshold=best_threshold)
        
        # Calculate permutation importance
        print("Calculating feature importance...")
        rf_model.calculate_permutation_importance()
        
        # Calculate SHAP values
        print("Analyzing SHAP values...")
        rf_model.analyze_shap_values()
        
        # Save model
        print("Saving model...")
        model_path = rf_model.save_model()
        print(f"Model saved to: {model_path}")
        
        print("\n" + "=" * 80)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Display top features
        if hasattr(rf_model, 'feature_importance') and rf_model.feature_importance is not None:
            top_features = rf_model.feature_importance.head(10)
            print("\nTop 10 Most Important Features:")
            for i, (feature, importance) in enumerate(
                zip(top_features['feature'], top_features['importance']), 1
            ):
                print(f"{i}. {feature}: {importance:.4f}")
        
        # Print evaluation metrics
        if evaluation:
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