import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def handle_class_imbalance(df, target_col, method='smote', sampling_strategy='auto'):
    """Address class imbalance for classification problems"""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in the dataframe")
        return df
    
    # Check if this is a classification problem
    n_classes = len(df[target_col].unique())
    if n_classes > 10:
        print(f"Target has {n_classes} unique values, not applying class balancing (likely regression)")
        return df
    
    # Calculate class distribution
    class_counts = df[target_col].value_counts()
    print(f"Original class distribution:\n{class_counts}")
    
    # Calculate imbalance ratio (majority/minority)
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    if imbalance_ratio < 1.5:
        print(f"Class imbalance ratio is {imbalance_ratio:.2f}, no need for balancing")
        return df
    
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}, applying {method}...")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if method == 'smote':
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    elif method == 'adasyn':
        # Apply ADASYN
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
    elif method == 'random_over':  
        # Apply random oversampling
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif method == 'random_under':
        # Apply random undersampling
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    
    else:
        print(f"Unknown balancing method: {method}, returning original data")
        return df
    
    # Create new balanced dataframe
    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df[target_col] = y_resampled
    
    # Print new class distribution
    new_class_counts = balanced_df[target_col].value_counts()
    print(f"New class distribution:\n{new_class_counts}")
    
    return balanced_df
