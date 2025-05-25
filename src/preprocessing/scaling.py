#  verify_feature_scaling, standardize_features
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def verify_feature_scaling(df, excluded_cols=None):
    """Verify and fix feature scaling issues"""
    print("Verifying feature scaling...")
    
    if excluded_cols is None:
        excluded_cols = []
    
    # Add standard excluded columns
    excluded_cols.extend(['subject_id', 'hadm_id', 'stay_id'])
    
    # Identify numeric columns that should be scaled
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols 
                   if not (col in excluded_cols or 
                          col.endswith('_missing') or 
                          col.endswith('_flag') or
                          col.startswith('has_'))]
    
    # Check scale of features
    scale_issues = []
    for col in cols_to_scale:
        col_range = df[col].max() - df[col].min()
        col_std = df[col].std()
        
        if col_range > 100 or col_std > 10:
            scale_issues.append((col, col_range, col_std))
    
    if scale_issues:
        print(f"  - Found {len(scale_issues)} features with scaling issues")
        print("  - Top 5 scaling issues:")
        for col, col_range, col_std in sorted(scale_issues, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {col}: range={col_range:.2f}, std={col_std:.2f}")
        
        # Apply robust scaling to these columns
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"  - Applied RobustScaler to {len(cols_to_scale)} numeric features")
    else:
        print("  - All features are properly scaled")
    
    return df


def standardize_features(df):
    """Standardize numeric features"""
    print("Standardizing numeric features...")
    
    processed_df = df.copy()
    
    # Identify numeric features to standardize (exclude binary/categorical)
    numeric_features = [col for col in processed_df.select_dtypes(include=[np.number]).columns 
                      if not (col.startswith('has_') or col.endswith('_missing') or
                             col in ['gender_numeric', 'sirs_criteria_count'])]
    
    # Use RobustScaler which is less sensitive to outliers
    scaler = RobustScaler()
    
    # Apply scaling only to numeric columns
    if numeric_features:
        processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])
        print(f"  - Standardized {len(numeric_features)} numeric features using RobustScaler")
    
    return processed_df
