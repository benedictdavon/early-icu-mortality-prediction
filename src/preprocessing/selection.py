# variance, importance, PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def remove_irrelevant_columns(df, target_col=None):
    """
    Remove irrelevant columns like IDs, dates, and timestamps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str, default=None
        Target column name to preserve
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with irrelevant columns removed
    list
        List of removed columns
    """
    print("Removing irrelevant columns...")
    
    # Identify columns to drop
    drop_patterns = ['id', 'time', 'date', 'minute', 'hour', 'day']
    drop_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in drop_patterns)]
    
    # Don't drop the target column
    if target_col and target_col in drop_cols:
        drop_cols.remove(target_col)
    
    # Don't drop specific useful time-based features
    preserve_cols = [
        col for col in drop_cols if 
        any(pattern in col.lower() for pattern in ['los', 'time_in_hospital', 'time_delta', 'duration'])
    ]
    drop_cols = [col for col in drop_cols if col not in preserve_cols]
    
    print(f"  - Identified {len(drop_cols)} columns to remove")
    print(f"  - Preserved {len(preserve_cols)} time-related features")
    
    # Remove the columns
    df_clean = df.drop(columns=drop_cols)
    
    print(f"  - Final shape after removing irrelevant columns: {df_clean.shape}")
    return df_clean, drop_cols

def remove_low_variance_features(df, threshold=0.01):
    """
    Remove features with variance below a threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    threshold : float, default=0.01
        Variance threshold
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with low-variance features removed
    list
        List of removed features
    """
    print(f"Removing low variance features (threshold={threshold})...")
    
    # Separate numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("  - No numeric columns to analyze")
        return df, []
    
    # Get non-numeric columns
    non_numeric = [col for col in df.columns if col not in numeric_cols]
    
    # Apply variance threshold to numeric features
    selector = VarianceThreshold(threshold=threshold)
    
    # Handle missing values for variance calculation
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    
    try:
        selector.fit(X)
        high_var_cols = X.columns[selector.get_support()].tolist()
        low_var_cols = [col for col in numeric_cols if col not in high_var_cols]
        
        print(f"  - Removed {len(low_var_cols)} low-variance features")
        
        # Keep non-numeric columns and high-variance numeric columns
        df_filtered = df[non_numeric + high_var_cols]
        return df_filtered, low_var_cols
    
    except Exception as e:
        print(f"  - Error in variance filtering: {str(e)}")
        print("  - Returning original dataframe")
        return df, []

def univariate_feature_selection(df, target_col, k=50, method='anova'):
    """
    Select features based on univariate statistical tests.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    k : int, default=50
        Number of features to select
    method : str, default='anova'
        Method for feature selection ('anova' or 'mutual_info')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of selected features
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found in dataframe")
        return df, df.columns.tolist()
    
    print(f"Performing univariate feature selection using {method}...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose score function
    if method == 'anova':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        print(f"Unknown method: {method}, using ANOVA")
        score_func = f_classif
    
    # Handle non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < X.shape[1]:
        print(f"  - Warning: {X.shape[1] - len(numeric_cols)} non-numeric columns will be excluded")
        X = X[numeric_cols]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Apply feature selection
    k = min(k, X.shape[1])  # Ensure k is not larger than number of features
    selector = SelectKBest(score_func=score_func, k=k)
    
    try:
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"  - Selected {len(selected_features)} features")
        
        # Create output dataframe with selected features and target
        result_df = df[selected_features + [target_col]]
        return result_df, selected_features
    
    except Exception as e:
        print(f"  - Error in univariate selection: {str(e)}")
        print("  - Returning original dataframe")
        return df, X.columns.tolist()

def model_based_feature_selection(df, target_col, model_type='xgboost', k=50, max_iter=3):
    """
    Select features using model-based importance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    model_type : str, default='xgboost'
        Model to use ('xgboost', 'random_forest')
    k : int, default=50
        Number of features to select
    max_iter : int, default=3
        Number of iterations for recursive feature elimination
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of selected features
    dict
        Feature importance scores
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found in dataframe")
        return df, df.columns.tolist(), {}
    
    print(f"Performing {model_type}-based feature selection...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        print(f"  - Converting {len(categorical_cols)} categorical features to numeric")
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Choose model
    if model_type.lower() == 'xgboost':
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        )
    elif model_type.lower() == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
    else:
        print(f"Unknown model type: {model_type}, using XGBoost")
        model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        )
    
    # Limit k to number of features
    k = min(k, X.shape[1])
    
    # Simple feature importance (single fit)
    try:
        model.fit(X, y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("  - Model doesn't have feature_importances_ attribute")
            importances = np.ones(X.shape[1])
            
        # Create feature importance dictionary
        feature_importance = dict(zip(X.columns, importances))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top k features
        selected_features = [f[0] for f in sorted_features[:k]]
        
        print(f"  - Selected {len(selected_features)} features based on importance")
        
        # Create output dataframe with selected features and target
        result_df = df[selected_features + [target_col]]
        return result_df, selected_features, feature_importance
    
    except Exception as e:
        print(f"  - Error in model-based selection: {str(e)}")
        print("  - Returning original dataframe")
        return df, X.columns.tolist(), {}

def optimize_features_for_logistic_regression(df, target_col, max_features=50):
    """
    Optimize feature selection specifically for logistic regression.
    Focuses on removing multicollinearity and selecting statistically significant features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    max_features : int, default=50
        Maximum number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of selected features
    """
    print("Optimizing features for Logistic Regression...")
    
    # Step 1: Remove low variance features
    df_filtered, _ = remove_low_variance_features(df, threshold=0.01)
    
    # Step 2: Remove multicollinearity
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    if numeric_cols:
        # Calculate correlation matrix
        corr_matrix = df_filtered[numeric_cols].corr().abs()
        
        # Create upper triangle mask
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.8
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        print(f"  - Removing {len(to_drop)} features with high correlation")
        
        # Drop highly correlated features
        df_filtered = df_filtered.drop(columns=to_drop)
        
    # Step 3: Apply univariate selection
    df_selected, selected_features = univariate_feature_selection(
        df_filtered, target_col, k=max_features, method='anova'
    )
    
    return df_selected, selected_features

def optimize_features_for_tree_models(df, target_col, model_type='xgboost', max_features=100):
    """
    Optimize feature selection specifically for tree-based models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    model_type : str, default='xgboost'
        Model type ('xgboost' or 'random_forest')
    max_features : int, default=100
        Maximum number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of selected features
    """
    print(f"Optimizing features for {model_type}...")
    
    # Step 1: Remove irrelevant columns
    df_clean, _ = remove_irrelevant_columns(df, target_col)
    
    # Step 2: Remove low variance features
    df_filtered, _ = remove_low_variance_features(df_clean, threshold=0.001)  # Tree models can handle low variance
    
    # Step 3: Use model-based feature importance
    df_selected, selected_features, _ = model_based_feature_selection(
        df_filtered, target_col, model_type=model_type, k=max_features
    )
    
    return df_selected, selected_features

def select_features_for_clinical_model(df, target_col, clinical_focus=True):
    """
    Select features with special attention to clinical importance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    clinical_focus : bool, default=True
        Whether to prioritize clinically relevant features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of selected features
    """
    print("Selecting features with clinical focus...")
    
    # Define clinically important feature patterns
    clinical_patterns = [
        'heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'temp', 
        'glucose', 'sodium', 'potassium', 'creatinine', 'bun', 'wbc', 
        'hemoglobin', 'platelets', 'lactate', 'bilirubin', 'inr',
        'age', 'gender', 'bmi', 'los', 'urine'
    ]
    
    # First pass: identify clinically relevant features
    clinical_features = []
    if clinical_focus:
        for pattern in clinical_patterns:
            matches = [col for col in df.columns if pattern.lower() in col.lower()]
            clinical_features.extend(matches)
        
        # Remove duplicates
        clinical_features = list(dict.fromkeys(clinical_features))
        print(f"  - Identified {len(clinical_features)} clinically relevant features")
        
    # Second pass: use statistical measures for remaining features
    df_filtered, _ = remove_low_variance_features(df, threshold=0.005)
    
    # Third pass: use model-based selection
    df_selected, selected_features, importances = model_based_feature_selection(
        df_filtered, target_col, model_type='random_forest', k=100
    )
    
    # Combine with clinical features
    final_features = list(set(selected_features + clinical_features))
    
    # Ensure target column is included
    if target_col not in final_features:
        final_features.append(target_col)
    
    final_df = df[final_features]
    print(f"  - Final feature count: {len(final_features) - (1 if target_col in final_features else 0)}")
    
    return final_df, [f for f in final_features if f != target_col]


def remove_redundant_features(df, correlation_threshold=0.85, keep_transformed=True, keep_clinical=True,
                             clinical_threshold=0.95, verbose=True):
    """
    Advanced redundancy reduction optimized for Random Forest models
    
    Parameters:
    -----------
    df: DataFrame
        Input features dataframe
    correlation_threshold: float, default=0.85
        Correlation threshold for general features
    keep_transformed: bool, default=True
        Whether to keep both original and transformed versions of important features
    keep_clinical: bool, default=True
        Whether to prioritize keeping clinical features
    clinical_threshold: float, default=0.95
        Higher correlation threshold for clinical features (only drop if extremely correlated)
    verbose: bool, default=True
        Whether to print detailed information
    """
    print("Reducing feature redundancy with RF-optimized approach...")
    
    processed_df = df.copy()
    
    # Define clinical feature patterns - these are treated with higher importance
    clinical_patterns = [
        'heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'temp',
        'lactate', 'creatinine', 'bun', 'glucose', 'wbc', 'platelets',
        'hemoglobin', 'sodium', 'potassium', 'age', 'shock_index', 'bmi',
        'duration_hours', 'anion_gap'
    ]
    
    # Define transform suffixes that indicate derived features
    transform_patterns = ['_log', '_min', '_max', '_mean', '_delta']
    
    # Get numeric features without missing indicators
    numeric_features = [col for col in processed_df.select_dtypes(include=[np.number]).columns
                      if not col.endswith('_missing') and not col.startswith('has_')]
    
    # Categorize features
    clinical_features = []
    derived_features = []
    other_features = []
    
    for col in numeric_features:
        # Check if it's a clinical feature
        if any(pattern in col for pattern in clinical_patterns):
            clinical_features.append(col)
        
        # Check if it's a derived/transformed feature
        elif any(pattern in col for pattern in transform_patterns):
            derived_features.append(col)
            
        # Otherwise it's a general feature
        else:
            other_features.append(col)
    
    if verbose:
        print(f"  - Found {len(clinical_features)} clinical features")
        print(f"  - Found {len(derived_features)} derived features")
        print(f"  - Found {len(other_features)} other numeric features")
    
    # Calculate full correlation matrix
    corr_matrix = processed_df[numeric_features].corr().abs()
    
    # Create a mapping from feature to its base feature name
    feature_groups = {}
    
    # Group features that measure the same underlying quantity
    for feature in numeric_features:
        # Find base feature name by removing min/max/mean/etc.
        base_name = feature
        for suffix in ['_min', '_max', '_mean', '_log', '_delta']:
            if feature.endswith(suffix):
                base_name = feature.replace(suffix, '')
                break
                
        if base_name not in feature_groups:
            feature_groups[base_name] = []
        feature_groups[base_name].append(feature)
    
    # Identify and handle highly correlated groups
    to_drop = []
    preserved = []
    
    # 1. First, filter for correlation matrix upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 2. Find pairs of highly correlated features
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx, value in upper_tri[col].items():
            if pd.notnull(value) and value > correlation_threshold:
                high_corr_pairs.append((idx, col, value))
    
    # Sort by correlation (highest first)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if verbose and high_corr_pairs:
        print(f"  - Found {len(high_corr_pairs)} highly correlated feature pairs")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"    - {feat1} ↔ {feat2}: {corr:.4f}")
        if len(high_corr_pairs) > 5:
            print(f"    - ... and {len(high_corr_pairs) - 5} more pairs")
    
    # 3. Process correlation pairs and make decisions
    for feat1, feat2, corr_val in high_corr_pairs:
        # Skip if either feature is already marked for dropping
        if feat1 in to_drop or feat2 in to_drop:
            continue
            
        # Different handling based on feature types
        is_clinical1 = any(pattern in feat1 for pattern in clinical_patterns)
        is_clinical2 = any(pattern in feat2 for pattern in clinical_patterns)
        
        # Special case: if comparing original and transformed version of same clinical feature
        if keep_transformed and is_clinical1 and is_clinical2:
            same_base = False
            for pattern in clinical_patterns:
                if pattern in feat1 and pattern in feat2:
                    same_base = True
                    break
                    
            if same_base and any(feat1.endswith(suffix) for suffix in transform_patterns) != any(feat2.endswith(suffix) for suffix in transform_patterns):
                # Keep both the original and transformed versions of important clinical features
                preserved.append((feat1, feat2, f"kept both clinical variants"))
                continue
                
        # Case: Two clinical features - use higher threshold
        if keep_clinical and is_clinical1 and is_clinical2:
            if corr_val <= clinical_threshold:
                # Not high enough correlation to drop clinical features
                continue
                
        # Decision logic for which feature to drop
        feat_to_drop = None
        
        # Calculate feature scores based on variance and missingness
        var1 = processed_df[feat1].var()
        var2 = processed_df[feat2].var()
        missing1 = processed_df[feat1].isnull().mean()
        missing2 = processed_df[feat2].isnull().mean()
        
        # Higher score = better feature
        score1 = var1 * (1 - missing1)
        score2 = var2 * (1 - missing2)
        
        # Adjust scores based on feature type 
        if keep_clinical:
            if is_clinical1:
                score1 *= 1.5  # Boost clinical features
            if is_clinical2:
                score2 *= 1.5
                
        # Prefer non-derived features for interpretability
        has_suffix1 = any(feat1.endswith(suffix) for suffix in transform_patterns)
        has_suffix2 = any(feat2.endswith(suffix) for suffix in transform_patterns)
        
        if has_suffix1 and not has_suffix2:
            score1 *= 0.9  # Slightly penalize derived features
        elif has_suffix2 and not has_suffix1:
            score2 *= 0.9
            
        # Decide which feature to drop based on scores
        reason = ""
        if score1 < score2:
            feat_to_drop = feat1
            reason = f"lower score ({score1:.2f} vs {score2:.2f})"
        else:
            feat_to_drop = feat2
            reason = f"lower score ({score2:.2f} vs {score1:.2f})"
            
        # Add to drop list with reason
        if feat_to_drop not in to_drop:
            to_drop.append(feat_to_drop)
            if verbose:
                preserved_feature = feat1 if feat_to_drop == feat2 else feat2
                preserved.append((preserved_feature, feat_to_drop, reason))
    
    # Drop the features
    if to_drop:
        if verbose:
            print(f"\n  - Dropping {len(to_drop)} redundant features:")
            for i, (kept, dropped, reason) in enumerate(preserved[:10]):
                print(f"    - Keeping {kept} over {dropped}: {reason}")
            if len(preserved) > 10:
                print(f"    - ... and {len(preserved) - 10} more decisions")
                
        processed_df = processed_df.drop(columns=to_drop)
        print(f"  - Final feature count: {processed_df.shape[1]} (removed {len(to_drop)} redundant features)")
    else:
        print("  - No redundant features to drop")
    
    return processed_df


def select_features(df, target_col=None, n_components=None, method='variance', keep_clinical=True):
    """Select most informative features using various methods"""
    print("Selecting features using method:", method)
    
    # List of key clinical features to prioritize keeping based on analysis
    critical_clinical_features = [
        # Most predictive vital signs based on analysis
        'resp_rate_mean', 'resp_rate_0', 'resp_rate_1', 'resp_rate_2', 'resp_rate_3',
        'lactate_mean', 'lactate_0', 'lactate_1', 'lactate_2',
        'heart_rate_mean', 'heart_rate_0', 'heart_rate_1', 'heart_rate_2', 'heart_rate_6',
        'anion_gap_mean', 'anion_gap_0', 'anion_gap_1', 'anion_gap_2',
        
        # Key change features identified in analysis
        'spo2_delta_1to2', 'lactate_delta_4to6', 'dbp_delta_0to1', 
        'sodium_delta_3to4', 'wbc_change_0to6',
        
        # Derived features from these vital signs
        'has_tachypnea', 'has_bradypnea', 'has_elevated_lactate', 
        'has_tachycardia', 'has_bradycardia', 'shock_index', 'has_shock',
        'has_hypoxemia', 'has_high_anion_gap',
        
        # Standard demographic features still worth keeping
        'age', 'gender_numeric', 'bmi'
    ]
    
    # Filter to only include columns that exist in the dataframe
    if keep_clinical:
        clinical_features = [f for f in critical_clinical_features if f in df.columns]
        print(f"  - Will prioritize keeping {len(clinical_features)} high-value clinical features")
    

    if method == 'variance':
        # Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        
        # Convert to numeric if target is included
        if target_col is not None and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Filter out non-numeric columns before computing variance
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        print(f"  - Using {len(numeric_cols)} numeric features for variance analysis")
        
        # Skip if there are no numeric features
        if len(numeric_cols) == 0:
            print("  - No numeric features for variance analysis")
            return df
            
        # Apply variance thresholding only to numeric columns
        selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
        X_numeric_selected = selector.fit_transform(X[numeric_cols])
        
        # Get selected feature names
        selected_numeric_features = numeric_cols[selector.get_support()]
        print(f"  - Selected {len(selected_numeric_features)} out of {len(numeric_cols)} numeric features based on variance")
        
        # Keep all non-numeric columns
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        # Reconstruct dataframe with selected numeric features and all non-numeric features
        result_df = pd.DataFrame(X_numeric_selected, columns=selected_numeric_features, index=X.index)
        
        # Add non-numeric columns back
        if non_numeric_cols:
            result_df = pd.concat([result_df, X[non_numeric_cols]], axis=1)
            
        # Add target back if provided
        if y is not None:
            result_df[target_col] = y
        
        return result_df
    
    elif method == 'pca':
        # Use PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        
        # Separate target if provided
        if target_col is not None and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Determine number of components
        if n_components is None:
            # Find number of components to explain 80% variance
            pca = PCA().fit(X)
            n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create dataframe with PCA components
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
        
        print(f"  - Reduced dimensions from {X.shape[1]} to {n_components} using PCA")
        print(f"  - Components explain {pca.explained_variance_ratio_.sum()*100:.2f}% of variance")
        
        # Add target back if provided
        if y is not None:
            return X_pca_df.join(pd.Series(y, name=target_col))
        else:
            return X_pca_df
    
    elif method == 'importance':
        # Use feature importance from Random Forest
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        from sklearn.preprocessing import LabelEncoder
        
        if target_col is None or target_col not in df.columns:
            print("  - Target column is required for importance-based feature selection")
            return df
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables - we need to encode them
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"  - Found {len(categorical_cols)} categorical columns to encode")
            X_processed = X.copy()
            
            # Apply label encoding to categorical columns
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"    - Encoded '{col}' with values: {list(le.classes_)}")
        else:
            X_processed = X
            
        # Choose classifier or regressor based on the target type
        if len(np.unique(y)) < 10:  # Classification task
            print("  - Using RandomForestClassifier for feature selection")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression task
            print("  - Using RandomForestRegressor for feature selection")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X_processed, y)
        
        # Select features above mean importance
        selector = SelectFromModel(model, threshold='mean')
        X_selected = selector.fit_transform(X_processed, y)
        
        # Get selected feature names
        selected_features = X_processed.columns[selector.get_support()].tolist()
        
        print(f"  - Selected {len(selected_features)} out of {X_processed.shape[1]} features based on importance")
        
        # Make sure to keep clinical features if specified
        if keep_clinical:
            for feat in clinical_features:
                if feat in X.columns and feat not in selected_features:
                    selected_features.append(feat)
                    print(f"  - Added clinical feature '{feat}' back to selection")
        
        # Get the top 10 most important features
        importances = pd.Series(model.feature_importances_, index=X_processed.columns).sort_values(ascending=False)
        print("  - Top 10 most important features:")
        for i, (feat, imp) in enumerate(importances.head(10).items()):
            print(f"    {i+1}. {feat}: {imp:.4f}")
        
        # Return dataframe with selected features and target
        # Note: we use X not X_processed to keep original categorical values
        selected_df = pd.concat([X[selected_features], pd.Series(y, name=target_col)], axis=1)
        return selected_df
    elif method == 'correlation':
        # Remove highly correlated features
        print("  - Finding and removing highly correlated features...")
        
        # Separate target if provided
        if target_col is not None and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Filter out non-numeric columns before computing correlations
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        print(f"  - Using {len(numeric_cols)} numeric features for correlation analysis")
        
        # Skip if there are too few numeric features
        if len(numeric_cols) < 2:
            print("  - Not enough numeric features for correlation analysis")
            return df
        
        # Calculate correlation matrix only for numeric columns
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Extract upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > threshold
        threshold = 0.85
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Process correlated features
        if len(to_drop) > 0:
            print(f"  - Found {len(to_drop)} features with correlation > {threshold}")
            
            # When deciding which of the correlated features to drop, prioritize keeping clinical features
            if keep_clinical:
                original_drop_count = len(to_drop)
                to_drop = [col for col in to_drop if col not in clinical_features]
                protected_count = original_drop_count - len(to_drop)
                print(f"  - Protected {protected_count} clinical features from correlation-based removal")
            
            # Keep features with less missingness or higher variance
            final_drop = []
            for i, feat1 in enumerate(to_drop):
                if feat1 not in final_drop:
                    # Find all highly correlated features
                    correlated = [feat2 for feat2 in upper.index 
                                if upper.loc[feat2, feat1] > threshold]
                    
                    # Add all but the "best" feature to final_drop
                    correlated = [feat1] + correlated
                    
                    # Compute a score based on variance and missingness
                    scores = {}
                    for feat in correlated:
                        if feat in X.columns:
                            var_score = X[feat].var()
                            missing_score = 1 - X[feat].isnull().mean()
                            scores[feat] = var_score * missing_score
                    
                    # Sort by score (higher is better)
                    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Keep the highest scoring feature, drop others
                    to_remove = [f for f, _ in sorted_feats[1:]]
                    final_drop.extend(to_remove)
            
            # Remove duplicates
            final_drop = list(set(final_drop))
            print(f"  - Dropping {len(final_drop)} correlated features")
            
            # Create new dataframe without dropped features
            X_reduced = X.drop(columns=final_drop)
            
            # Add target back if provided
            if y is not None:
                return X_reduced.join(pd.Series(y, name=target_col))
            else:
                return X_reduced
        else:
            print("  - No highly correlated features found")
            return df
    
    else:
        print(f"  - Unknown feature selection method: {method}")
        return df


def evaluate_vital_sign_predictive_power(df, target_col='mortality'):
    """Evaluate and rank vital signs based on their predictive power for mortality"""
    if target_col not in df.columns:
        print("Target column not found, skipping vital sign evaluation")
        return None
        
    print("Evaluating vital sign predictive power...")
    
    # Vital signs of interest based on analysis
    vital_signs = ['resp_rate', 'lactate', 'heart_rate', 'anion_gap', 
                   'bun', 'spo2', 'sbp', 'dbp', 'glucose', 'temp']
    
    results = {}
    
    # Check correlation of each vital sign with target
    for vital in vital_signs:
        # Find all columns related to this vital sign
        vital_cols = [col for col in df.columns if vital in col.lower() and 
                      not col.endswith('_missing') and
                      not 'delta' in col.lower()]
        
        if not vital_cols:
            continue
            
        # Calculate correlation with target for each column
        vital_corrs = []
        for col in vital_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                corr = abs(df[col].corr(df[target_col]))
                if not np.isnan(corr):
                    vital_corrs.append((col, corr))
        
        if vital_corrs:
            # Sort by correlation strength
            vital_corrs.sort(key=lambda x: x[1], reverse=True)
            avg_corr = sum([c[1] for c in vital_corrs]) / len(vital_corrs)
            best_col, best_corr = vital_corrs[0]
            
            results[vital] = {
                'avg_correlation': avg_corr,
                'best_column': best_col,
                'best_correlation': best_corr,
                'all_correlations': vital_corrs
            }
    
    # Print results sorted by average correlation
    if results:
        print("\nVital signs ranked by predictive power:")
        for vital, data in sorted(results.items(), key=lambda x: x[1]['avg_correlation'], reverse=True):
            print(f"  - {vital}: avg |corr|={data['avg_correlation']:.4f}, "
                  f"best={data['best_column']} (|corr|={data['best_correlation']:.4f})")
    
    return results


def handle_date_columns(df):
    """Detect and handle date/time columns that can cause model errors"""
    print("Handling date/time columns...")

    date_columns = []

    # 1. Detect by column name patterns
    for col in df.columns:
        if any(
            pattern in col.lower() for pattern in ["time", "date", "dt_", "_dt", "dob"]
        ):
            date_columns.append(col)

    # 2. Detect string columns with date patterns
    for col in df.select_dtypes(include=["object"]).columns:
        # If column isn't already identified as a date column
        if col not in date_columns:
            # Check if it looks like a datetime (contains both - and :)
            sample_values = df[col].dropna().astype(str).iloc[:5].tolist()
            for val in sample_values:
                if "-" in val and ":" in val:
                    date_columns.append(col)
                    break

    # Remove duplicates and actually drop the columns
    date_columns = list(set(date_columns))

    if date_columns:
        print(f"  - Found {len(date_columns)} date/time columns:")
        for col in date_columns[:10]:
            print(f"    - {col}")
        if len(date_columns) > 10:
            print(f"    - ... and {len(date_columns)-10} more")

        df.drop(columns=date_columns, inplace=True, errors="ignore")
        print(f"  - Dropped {len(date_columns)} date/time columns")
    else:
        print("  - No date/time columns found")

    return df
