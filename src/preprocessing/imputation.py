# MICE, fallback methods
import pandas as pd
import numpy as np

def analyze_missingness(df):
    """Analyze and visualize missing data patterns"""
    print("Analyzing missing data patterns...")
    
    # Calculate missing percentages
    missing_pct = df.isnull().mean().sort_values(ascending=False) * 100
    missing_df = pd.DataFrame({'Feature': missing_pct.index, 'Missing Percentage': missing_pct.values})
    
    # Group features by missing percentage
    extreme_missing = missing_df[missing_df['Missing Percentage'] > 80]['Feature'].tolist()
    high_missing = missing_df[(missing_df['Missing Percentage'] <= 80) & 
                             (missing_df['Missing Percentage'] > 50)]['Feature'].tolist()
    moderate_missing = missing_df[(missing_df['Missing Percentage'] <= 50) & 
                                (missing_df['Missing Percentage'] > 20)]['Feature'].tolist()
    low_missing = missing_df[missing_df['Missing Percentage'] <= 20]['Feature'].tolist()
    
    print(f"Features with >80% missing: {len(extreme_missing)}")
    print(f"Features with 50-80% missing: {len(high_missing)}")
    print(f"Features with 20-50% missing: {len(moderate_missing)}")
    print(f"Features with <20% missing: {len(low_missing)}")
    
    return {
        'extreme_missing': extreme_missing,
        'high_missing': high_missing,
        'moderate_missing': moderate_missing,
        'low_missing': low_missing,
        'missing_df': missing_df
    }



def handle_missing_data(df, missingness_data):
    """Handle missing data using improved MICE implementation."""
    print("Handling missing data with advanced MICE implementation...")
    
    # Step 1: Drop features with extreme missingness (>80%)
    extreme_missing = missingness_data['extreme_missing']
    processed_df = df.drop(columns=extreme_missing)
    print(f"Dropped {len(extreme_missing)} features with >80% missing data")
    
    # Step 2: Create missing indicators for important features
    critical_features = ['age', 'gender_numeric', 'bmi', 'lactate_mean', 'resp_rate_mean']
    for feature in critical_features:
        if feature in processed_df.columns and processed_df[feature].isnull().any():
            processed_df[f'{feature}_missing'] = processed_df[feature].isnull().astype(int)
    
    # Step 3: Apply MICE imputation to remaining missing values
    # Exclude high-missing features (50-80%)
    exclude_from_mice = missingness_data['high_missing'] + missingness_data['moderate_missing']
    
    # Call our new MICE implementation
    processed_df, imputation_stats = impute_with_mice(
        processed_df,
        max_iter=10,
        n_estimators=50,
        exclude_features=exclude_from_mice,
        verbose=True
    )
    
    # Step 4: Handle any remaining missing values with simpler methods
    remaining_missing = processed_df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"MICE did not impute all values. Handling {remaining_missing} remaining missing values...")
        
        # Apply simpler imputation for any remaining missing values
        for col in processed_df.columns:
            if processed_df[col].isnull().any():
                if np.issubdtype(processed_df[col].dtype, np.number):
                    # For numeric columns, use median
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    # For categorical columns, use mode
                    mode_val = processed_df[col].mode().iloc[0] if not processed_df[col].empty else None
                    processed_df[col] = processed_df[col].fillna(mode_val)
    
    print(f"Final dataset shape after handling missing data: {processed_df.shape}")
    return processed_df

def impute_with_mice(df, max_iter=10, n_estimators=50, categorical_features=None, 
                     exclude_features=None, random_state=42, verbose=True):
    """
    Perform multiple imputation by chained equations (MICE) on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with missing values
    max_iter : int, default=10
        Maximum number of imputation iterations
    n_estimators : int, default=50
        Number of estimators for the RandomForestRegressor/Classifier used in MICE
    categorical_features : list, default=None
        List of categorical features, if None will auto-detect
    exclude_features : list, default=None
        List of features to exclude from imputation
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values
    dict
        Statistics about the imputation process
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    import time
    
    if verbose:
        print("Starting MICE imputation process...")
        start_time = time.time()
    
    # Create a copy to avoid modifying the original
    imputed_df = df.copy()
    
    # Track imputation statistics
    stats = {
        'total_missing_before': df.isnull().sum().sum(),
        'features_imputed': [],
        'missing_counts_before': {},
        'imputation_time': 0
    }
    
    # Identify features to exclude
    if exclude_features is None:
        exclude_features = []
    
    # Add ID columns and columns with no missing values to exclude list
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    exclude_features.extend(id_columns)
    exclude_features.extend([col for col in df.columns if df[col].isnull().sum() == 0])
    exclude_features = list(set(exclude_features))  # Remove duplicates
    
    if verbose:
        print(f"Excluding {len(exclude_features)} features from imputation")
        if exclude_features:
            print(f"  - Sample excluded: {exclude_features[:5]}" + 
                  ("..." if len(exclude_features) > 5 else ""))
    
    # Separate features for imputation
    features_to_impute = [col for col in df.columns if col not in exclude_features]
    
    # Count missing values before imputation
    for col in features_to_impute:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            stats['missing_counts_before'][col] = missing_count
            stats['features_imputed'].append(col)
    
    if len(stats['features_imputed']) == 0:
        if verbose:
            print("No features need imputation")
        return imputed_df, stats
    
    if verbose:
        print(f"Imputing {len(stats['features_imputed'])} features with MICE")
    
    # Auto-detect categorical features if not provided
    if categorical_features is None:
        categorical_features = []
        for col in features_to_impute:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_features.append(col)
            # Also detect binary numeric features
            elif df[col].dtype.kind in 'ifu':  # integer, float, unsigned integer
                if len(df[col].dropna().unique()) <= 2:
                    categorical_features.append(col)
    
    # Handle categorical and numeric features separately
    numeric_features = [f for f in features_to_impute if f not in categorical_features]
    
    if verbose:
        print(f"  - Numeric features: {len(numeric_features)}")
        print(f"  - Categorical features: {len(categorical_features)}")
    
    # First handle categorical features with mode imputation or one-hot encoding + MICE
    if categorical_features:
        if verbose:
            print("  - Imputing categorical features...")
        
        # For simplicity, use mode imputation for categorical features
        # Could be extended with more advanced methods in the future
        for col in categorical_features:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0]
                imputed_df[col] = df[col].fillna(mode_val)
                if verbose:
                    print(f"    - Imputed {col} with mode value: {mode_val}")
    
    # Then handle numeric features with MICE
    if numeric_features:
        if verbose:
            print("  - Imputing numeric features with MICE...")
        
        # Prepare numeric dataframe
        numeric_df = imputed_df[numeric_features].copy()
        
        # Only select columns that actually have missing values
        cols_with_missing = [col for col in numeric_features if df[col].isnull().sum() > 0]
        
        if cols_with_missing:
            try:
                # Use Random Forest as the estimator for more robust imputation
                regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                
                # Initialize and fit MICE imputer
                imputer = IterativeImputer(
                    estimator=regressor,
                    max_iter=max_iter,
                    random_state=random_state,
                    verbose=2 if verbose else 0
                )
                
                # Apply imputation to all numeric columns
                imputed_values = imputer.fit_transform(numeric_df)
                
                # Update the dataframe
                imputed_df[numeric_features] = imputed_values
                
                if verbose:
                    print(f"    - Successfully imputed {len(cols_with_missing)} numeric features with MICE")
            
            except Exception as e:
                if verbose:
                    print(f"    - MICE imputation failed: {str(e)}")
                    print("    - Falling back to median imputation for numeric features")
                
                # Fallback to median imputation
                for col in cols_with_missing:
                    imputed_df[col] = imputed_df[col].fillna(df[col].median())
    
    # Calculate statistics
    stats['total_missing_after'] = imputed_df[features_to_impute].isnull().sum().sum()
    stats['imputation_time'] = time.time() - start_time
    
    if verbose:
        print(f"MICE imputation completed in {stats['imputation_time']:.2f} seconds")
        print(f"Missing values before: {stats['total_missing_before']}")
        print(f"Missing values after: {stats['total_missing_after']}")
    
    return imputed_df, stats


def label_gender(df):
    """Convert categorical features to numeric representations"""
    print("Converting categorical features to numeric...")
    
    processed_df = df.copy()
    

    # Handle gender specifically

    # Create numeric gender column: 1 for male, 0 for female
    processed_df['gender_numeric'] = processed_df['gender'].map({
        'M': 1, 'm': 1, 'Male': 1, 'male': 1, '1': 1,
        'F': 0, 'f': 0, 'Female': 0, 'female': 0, '0': 0
    })
    
    # Fill missing with most common value
    if processed_df['gender_numeric'].isnull().any():
        most_common = processed_df['gender_numeric'].mode()[0]
        processed_df['gender_numeric'] = processed_df['gender_numeric'].fillna(most_common)
        print(f"  - Filled {processed_df['gender'].isnull().sum()} missing gender values with '{most_common}'")
        
    # Drop original gender column
    processed_df.drop(columns=['gender'], inplace=True)
    print("  - Converted gender to numeric (1=male, 0=female)")

    return processed_df
