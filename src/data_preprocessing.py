import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from preprocessing import *

def enhanced_preprocess_pipeline(
    input_path, 
    output_path, 
    report_dir="./figures", 
    target_col=None, 
    keep_clinical=True,
    model_type=None,
    n_features=None
):
    """
    Execute the enhanced preprocessing pipeline with focus on predictive clinical features
    
    Parameters:
    -----------
    input_path : str
        Path to the input data file
    output_path : str
        Path where preprocessed data will be saved
    report_dir : str, default="./figures"
        Directory where preprocessing reports will be saved
    target_col : str, default=None
        Name of the target column for prediction
    keep_clinical : bool, default=True
        Whether to prioritize keeping clinical features
    model_type : str, default=None
        Type of model to optimize features for ('logistic', 'xgboost', 'random_forest')
    n_features : int, default=None
        Maximum number of features to select (if None, uses model-specific defaults)
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    print("Starting enhanced data preprocessing pipeline...")
    print(f"Target column: {target_col}")
    print(f"Model type: {model_type}")

    # Step 1: Load data
    df = load_data(input_path)
    print(f"Loaded data with shape: {df.shape}")

    # Keep a copy of the original data for reporting and feature restoration
    original_df = df.copy()
    
    # Step 2: Extract temporal features BEFORE removing time columns
    df = extract_temporal_features(df)
    print("Added temporal features from date/time columns")

    # Step 3: Remove obviously irrelevant columns first (IDs, timestamps, etc.)
    df, dropped_cols = remove_irrelevant_columns(df, target_col=target_col)
    print(f"Removed {len(dropped_cols)} irrelevant columns")

    # Step 4: Analyze missingness
    missingness_data = analyze_missingness(df)

    # Step 5: Handle gender conversion
    df = label_gender(df)

    # Step 6: Handle missing data
    df = handle_missing_data(df, missingness_data)
    
    # Step 7: Clean clinical measurements with domain knowledge (with unit detection)
    df = clean_clinical_measurements(df)
    
    # Step 8: Identify critical values
    df = identify_critical_values(df)
    print("Added clinical critical value flags")

    # Step 9: Handle outliers
    df = identify_and_handle_outliers(df)

    # Step 10: Transform skewed features
    df = transform_skewed_features(df)

    # Step 11: Create derived clinical features with focus on predictive vitals
    df = create_clinical_derived_features(df)
    
    # Step 12: Add advanced clinical features
    df = add_advanced_clinical_features(df)
    print("Added advanced clinical feature combinations")

    # Step 13: Add temporal trends with focus on key changes
    df = add_temporal_trends(df)

    # Step 14: Add polynomial features for key vitals
    df = add_polynomial_features(df)

    # Step 15: Verify feature scaling
    df = verify_feature_scaling(df)

    # Step 16: Remove redundant features
    df = remove_redundant_features(df)

    # Step 17: Remove low variance features
    df, low_var_features = remove_low_variance_features(df, threshold=0.005)
    print(f"Removed {len(low_var_features)} low variance features")

    # Step 18: If target is available, evaluate vital sign predictive power
    vital_predictors = None
    if target_col is not None and target_col in df.columns:
        vital_predictors = evaluate_vital_sign_predictive_power(df, target_col)

    # Step 19: Model-specific feature selection
    if model_type is not None and target_col is not None:
        print(f"Performing feature selection optimized for {model_type}...")
        
        if model_type.lower() == 'logistic':
            # For logistic regression: focus on non-collinear, statistically significant features
            max_features = n_features if n_features else 50
            print(f"Optimizing for logistic regression (max {max_features} features)")
            df, selected_features = optimize_features_for_logistic_regression(
                df, target_col=target_col, max_features=max_features
            )
            print(f"Selected {len(selected_features)} features for logistic regression")
            
        elif model_type.lower() == 'xgboost':
            # For XGBoost: can handle more features and interactions
            max_features = n_features if n_features else 150
            print(f"Optimizing for XGBoost (max {max_features} features)")
            df, selected_features = optimize_features_for_tree_models(
                df, target_col=target_col, model_type='xgboost', max_features=max_features
            )
            print(f"Selected {len(selected_features)} features for XGBoost")
            
        elif model_type.lower() == 'random_forest':
            # For Random Forest: balanced feature selection
            max_features = n_features if n_features else 100
            print(f"Optimizing for Random Forest (max {max_features} features)")
            df, selected_features = optimize_features_for_tree_models(
                df, target_col=target_col, model_type='random_forest', max_features=max_features
            )
            print(f"Selected {len(selected_features)} features for Random Forest")
            
        else:
            # General clinical feature selection
            print("Model type not recognized, using clinical feature selection")
            df, selected_features = select_features_for_clinical_model(
                df, target_col=target_col, clinical_focus=keep_clinical
            )
            print(f"Selected {len(selected_features)} features with clinical focus")
    else:
        # Use traditional feature selection
        print("Using traditional feature selection method")
        df = select_features(
            df, target_col=target_col, method="importance", keep_clinical=keep_clinical
        )

    # Step 20: Final check for critical features
    df = restore_critical_features(df, original_df)

    # Step 21: Handle date columns
    df = handle_date_columns(df)

    # Step 22: Standardize feature names for compatibility
    df = standardize_features(df)

    # Step 23: Save the processed data
    save_processed_data(df, output_path)

    # Step 24: Generate report
    generate_preprocessing_report(original_df, df, report_dir)

    print(f"Enhanced data preprocessing pipeline complete! Final shape: {df.shape}")
    print(f"Processed data saved to {output_path}")
    return df


if __name__ == "__main__":
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "processed", "extracted_features.csv")
    report_dir = os.path.join(base_dir, "figures")
    
    # Target column for prediction
    target_column = "mortality"
    
    # Process data for different models
    model_types = ['logistic', 'xgboost', 'random_forest']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Processing data for {model_type.upper()} model")
        print(f"{'='*50}")
        
        # Create model-specific output path
        output_path = os.path.join(
            base_dir, "data", "processed", f"preprocessed_{model_type}_features.csv"
        )
        
        # Run the pipeline with model-specific settings
        preprocessed_df = enhanced_preprocess_pipeline(
            input_path=input_path,
            output_path=output_path,
            report_dir=os.path.join(report_dir, model_type),
            target_col=target_column,
            keep_clinical=True,
            model_type=model_type
        )
        
        print(f"Completed preprocessing for {model_type} model")
    
    print("\nAll preprocessing tasks completed!")
