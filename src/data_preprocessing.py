import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(path):
    """Load the extracted features dataset"""
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

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
    """Handle missing data using different strategies based on missingness level"""
    print("Handling missing data...")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # 1. Drop extremely missing features (>80%)
    extreme_missing = missingness_data['extreme_missing']
    if 'bmi' in extreme_missing:  # Keep BMI as it's clinically important
        extreme_missing.remove('bmi')
    processed_df = processed_df.drop(columns=extreme_missing)
    print(f"Dropped {len(extreme_missing)} features with >80% missing data")
    
    # 2. Create missing indicators for high and moderate missing features
    high_moderate_missing = missingness_data['high_missing'] + missingness_data['moderate_missing']
    for feature in high_moderate_missing:
        if feature in processed_df.columns:
            processed_df[f'{feature}_missing'] = processed_df[feature].isnull().astype(int)
    
    # 3. Use appropriate imputation for remaining features
    # For clinical features: use median imputation as it's more robust
    clinical_features = [col for col in processed_df.columns if any(x in col for x in 
                        ['mean', 'max', 'min', 'rate', 'bmi', 'age', 'count', 'delta'])]
    
    for feature in clinical_features:
        if feature in processed_df.columns and processed_df[feature].isnull().any():
            # For delta features, impute with 0 (no change)
            if 'delta' in feature:
                processed_df[feature] = processed_df[feature].fillna(0)
            # For other clinical features, use median
            else:
                median_value = processed_df[feature].median()
                processed_df[feature] = processed_df[feature].fillna(median_value)
    
    # 4. For low missingness features, use KNN imputation for better accuracy
    low_missing_numeric = [col for col in missingness_data['low_missing'] 
                          if col in processed_df.columns 
                          and np.issubdtype(processed_df[col].dtype, np.number)
                          and processed_df[col].isnull().any()
                          and processed_df[col].isnull().mean() < 0.05]  # Only for very low missingness
    
    if low_missing_numeric:
        # Create a subset for KNN imputation
        subset_for_knn = processed_df[low_missing_numeric].copy()
        
        # Scale the data before KNN imputation
        scaler = StandardScaler()
        subset_scaled = pd.DataFrame(
            scaler.fit_transform(subset_for_knn.fillna(subset_for_knn.median())),
            columns=subset_for_knn.columns
        )
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        imputed_values = imputer.fit_transform(subset_scaled)
        
        # Transform back to original scale
        imputed_values = scaler.inverse_transform(imputed_values)
        
        # Replace the values in the processed dataframe
        for i, col in enumerate(low_missing_numeric):
            processed_df[col] = processed_df[col].fillna(pd.Series(imputed_values[:, i], index=processed_df.index))
    
    # Check if any missing values remain
    remaining_missing = processed_df.isnull().sum().sum()
    missing_by_column = processed_df.isnull().sum()
    problematic_cols = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
    
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after imputation")
        print(f"Top 5 columns with most missing values:")
        for col, missing in problematic_cols[:5].items():
            print(f"  - {col}: {missing} missing values ({missing/len(processed_df)*100:.2f}%)")
        
        # CRITICAL FIX: Instead of dropping rows, drop columns with >50% missing values that weren't handled yet
        cols_to_drop = [col for col in problematic_cols.index 
                        if processed_df[col].isnull().mean() > 0.5]
        
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with >50% remaining missing values:")
            for col in cols_to_drop[:5]:
                print(f"  - {col}")
            if len(cols_to_drop) > 5:
                print(f"  - ... and {len(cols_to_drop) - 5} more")
            processed_df = processed_df.drop(columns=cols_to_drop)
        
        # Handle remaining columns differently based on data type
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        
        # For numeric columns, use median
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if processed_df[col].isnull().any():
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # For categorical columns, use most frequent value or a special "Unknown" category
        for col in categorical_cols:
            if processed_df[col].isnull().any():
                if processed_df[col].nunique() > 0:
                    most_freq = processed_df[col].mode().iloc[0]
                    processed_df[col] = processed_df[col].fillna(most_freq)
                else:
                    processed_df[col] = processed_df[col].fillna("Unknown")
    
    # Final check - should have no missing values now
    final_missing = processed_df.isnull().sum().sum()
    if final_missing > 0:
        print(f"WARNING: Still have {final_missing} missing values. Using simple imputation for remaining.")
        
        # Use simple imputation as a last resort - forward fill then backward fill
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # If still have missing (should be very rare), use zeros for numeric and "Unknown" for others
        for col in processed_df.columns:
            if processed_df[col].isnull().any():
                if np.issubdtype(processed_df[col].dtype, np.number):
                    processed_df[col] = processed_df[col].fillna(0)
                else:
                    processed_df[col] = processed_df[col].fillna("Unknown")
    
    print(f"Final dataset shape after handling missing data: {processed_df.shape}")
    return processed_df




def identify_and_handle_outliers(df):
    """Identify and handle outliers using clinical knowledge and statistical methods"""
    print("Handling outliers...")
    
    processed_df = df.copy()
    
    # 1. Clinical range constraints - adjust physiologically impossible values
    clinical_ranges = {
        'heart_rate_mean': (30, 200),
        'resp_rate_mean': (5, 60),
        'map_mean': (40, 180),
        'sbp_mean': (60, 220),
        'dbp_mean': (30, 120),
        'temp_mean': (33, 42),
        'spo2_mean': (60, 100),
        'glucose_mean': (30, 600),
        'creatinine_mean': (0.1, 15),
        'lactate_mean': (0.1, 30)
    }
    
    # Apply clinical range constraints
    for feature, (lower, upper) in clinical_ranges.items():
        if feature in processed_df.columns:
            # Calculate how many outliers were found
            outliers_count = ((processed_df[feature] < lower) | (processed_df[feature] > upper)).sum()
            
            # Apply capping
            processed_df[feature] = processed_df[feature].clip(lower=lower, upper=upper)
            
            if outliers_count > 0:
                print(f"  - Capped {outliers_count} outliers in {feature} to range [{lower}, {upper}]")
    
    # 2. Statistical outlier handling for other numeric features
    # Use Winsorizing (capping at percentiles) for features with high skew
    numeric_features = processed_df.select_dtypes(include=[np.number]).columns
    
    for feature in numeric_features:
        # Skip features already handled and ID columns
        if feature in clinical_ranges.keys() or 'id' in feature.lower() or '_missing' in feature:
            continue
            
        # Calculate skewness
        skew = processed_df[feature].skew()
        
        # For highly skewed data (abs(skew) > 2), apply winsorizing at 1% and 99%
        if abs(skew) > 2:
            lower_bound = processed_df[feature].quantile(0.01)
            upper_bound = processed_df[feature].quantile(0.99)
            
            # Count outliers
            outliers_count = ((processed_df[feature] < lower_bound) | 
                             (processed_df[feature] > upper_bound)).sum()
            
            # Apply winsorizing
            processed_df[feature] = processed_df[feature].clip(lower=lower_bound, upper=upper_bound)
            
            if outliers_count > 0:
                print(f"  - Winsorized {outliers_count} outliers in {feature} (skew={skew:.2f})")
    
    return processed_df

def transform_skewed_features(df):
    """Apply appropriate transformations to handle skewed distributions"""
    print("Transforming skewed features...")
    
    processed_df = df.copy()
    
    # Identify numeric features with high skewness
    numeric_features = processed_df.select_dtypes(include=[np.number]).columns
    skew_data = processed_df[numeric_features].skew()
    high_skew_features = skew_data[abs(skew_data) > 2].index.tolist()
    
    # Filter out features where log transform doesn't make sense
    features_to_transform = [f for f in high_skew_features if 
                           not any(x in f.lower() for x in ['_missing', 'count', 'binary', 'has_'])]
    
    # Apply log transformation (log1p to handle zeros)
    for feature in features_to_transform:
        # Skip if the feature already has a log version
        if f"{feature}_log" in processed_df.columns:
            continue
            
        # Skip features with negative values
        if processed_df[feature].min() < 0:
            # For features with negative values, use other transformations
            print(f"  - Skipping log transform for {feature} due to negative values")
            continue
        
        # Calculate skewness before
        skew_before = processed_df[feature].skew()
        
        # Apply log transformation
        processed_df[f"{feature}_log"] = np.log1p(processed_df[feature])
        
        # Calculate skewness after
        skew_after = processed_df[f"{feature}_log"].skew()
        
        print(f"  - Log-transformed {feature}: skew reduced from {skew_before:.2f} to {skew_after:.2f}")
    
    return processed_df

def create_clinical_derived_features(df):
    """Create clinically meaningful derived features"""
    print("Creating derived clinical features...")
    
    processed_df = df.copy()
    
    # 1. SIRS (Systemic Inflammatory Response Syndrome) Criteria
    # - Heart rate > 90 bpm
    # - Respiratory rate > 20 breaths/min
    # - Temperature < 36°C or > 38°C
    # - WBC < 4,000/mm³ or > 12,000/mm³
    
    if 'heart_rate_mean' in processed_df.columns:
        processed_df['has_tachycardia'] = (processed_df['heart_rate_mean'] > 90).astype(int)
        
    if 'resp_rate_mean' in processed_df.columns:
        processed_df['has_tachypnea'] = (processed_df['resp_rate_mean'] > 20).astype(int)
        
    if 'temp_mean' in processed_df.columns:
        processed_df['has_fever_or_hypothermia'] = (
            (processed_df['temp_mean'] < 36) | 
            (processed_df['temp_mean'] > 38)
        ).astype(int)
        
    if 'wbc_mean' in processed_df.columns:
        processed_df['has_abnormal_wbc'] = (
            (processed_df['wbc_mean'] < 4) | 
            (processed_df['wbc_mean'] > 12)
        ).astype(int)
    
    # Calculate SIRS score if we have all criteria
    sirs_criteria = ['has_tachycardia', 'has_tachypnea', 
                     'has_fever_or_hypothermia', 'has_abnormal_wbc']
                     
    if all(criterion in processed_df.columns for criterion in sirs_criteria):
        processed_df['sirs_criteria_count'] = processed_df[sirs_criteria].sum(axis=1)
        print("  - Created SIRS criteria count")
    
    # 2. Shock index (heart rate / systolic blood pressure)
    if 'heart_rate_mean' in processed_df.columns and 'sbp_mean' in processed_df.columns:
        # Avoid division by zero
        sbp_safe = processed_df['sbp_mean'].copy()
        sbp_safe[sbp_safe == 0] = np.nan
        
        processed_df['shock_index'] = processed_df['heart_rate_mean'] / sbp_safe
        print("  - Created shock index (HR/SBP)")
    
    # 3. BUN:Creatinine ratio (kidney function indicator)
    if 'bun_mean' in processed_df.columns and 'creatinine_mean' in processed_df.columns:
        # Avoid division by zero
        creatinine_safe = processed_df['creatinine_mean'].copy()
        creatinine_safe[creatinine_safe == 0] = np.nan
        
        processed_df['bun_creatinine_ratio'] = processed_df['bun_mean'] / creatinine_safe
        print("  - Created BUN:Creatinine ratio")
    
    # 4. Hypoxemia flag (low oxygen)
    if 'spo2_mean' in processed_df.columns:
        processed_df['has_hypoxemia'] = (processed_df['spo2_mean'] < 92).astype(int)
        print("  - Created hypoxemia flag")
    
    return processed_df

def reduce_feature_redundancy(df, correlation_threshold=0.85):
    """Reduce redundancy by removing highly correlated features"""
    print("Reducing feature redundancy...")
    
    processed_df = df.copy()
    
    # Get numeric features without missing indicators
    numeric_features = [col for col in processed_df.select_dtypes(include=[np.number]).columns
                       if not col.endswith('_missing')]
    
    # Calculate correlation matrix
    corr_matrix = processed_df[numeric_features].corr().abs()
    
    # Find highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    if to_drop:
        print(f"  - Found {len(to_drop)} redundant features to drop")
        print("  - Redundant features:", to_drop)
        
        # Check if any log-transformed features are in the drop list
        log_features_to_drop = [f for f in to_drop if f.endswith('_log')]
        if log_features_to_drop:
            print("  - Among redundant features, preferring to drop log-transforms and keeping originals")
            # If both original and log are in drop list, keep the one with less skew
            for log_feat in log_features_to_drop:
                orig_feat = log_feat.replace('_log', '')
                if orig_feat in to_drop:
                    # Keep the less skewed one
                    if abs(processed_df[log_feat].skew()) < abs(processed_df[orig_feat].skew()):
                        to_drop.remove(orig_feat)  # Keep log version
                        print(f"    - Keeping {log_feat} instead of {orig_feat} due to lower skew")
                    else:
                        to_drop.remove(log_feat)  # Keep original version
                        print(f"    - Keeping {orig_feat} instead of {log_feat} due to lower skew")
        
        # Remove redundant features
        processed_df = processed_df.drop(columns=to_drop)
        print(f"  - Dropped {len(to_drop)} redundant features")
    else:
        print("  - No highly correlated features found")
    
    return processed_df
def restore_critical_features(processed_df, original_df):
    """Restore critical clinical features that may have been dropped"""
    print("Restoring critical clinical features...")
    
    # Expanded list of clinically important features to restore
    critical_features = {
        # Highest priority (must have)
        'demographic': ['age', 'gender', 'bmi'],
        
        # Vital signs (extremely important)
        'vitals': ['heart_rate_mean', 'heart_rate_min', 'sbp_mean', 'dbp_mean', 
                  'map_mean', 'resp_rate_mean', 'temp_mean', 'spo2_mean'],
        
        # Key lab values (very important)
        'labs': ['lactate_max', 'wbc_mean', 'creatinine_mean', 'bun_mean', 
                'glucose_mean', 'platelets_mean', 'hemoglobin_mean',
                'sodium_mean', 'potassium_mean', 'bicarbonate_mean'],
                
        # Other useful clinical variables (important)
        'clinical': ['has_prior_diagnoses', 'duration_hours']
    }
    
    # Flatten the list
    all_critical_features = []
    for category, features in critical_features.items():
        all_critical_features.extend(features)
    
    # Check which ones were dropped and restore them
    restored_count = 0
    for feature in all_critical_features:
        if feature not in processed_df.columns and feature in original_df.columns:
            # Use sophisticated imputation for the most critical features
            if feature == 'bmi':
                # BMI can be estimated from age, gender and other vitals
                # Create a simple model to predict BMI if we have enough data
                if 'gender_numeric' in processed_df.columns:
                    median_by_gender = original_df.groupby('gender_numeric')[feature].median()
                    processed_df[feature] = processed_df['gender_numeric'].map(median_by_gender)
                    print(f"  - Restored {feature} using gender-specific median")
                    restored_count += 1
                else:
                    # Fall back to global median
                    processed_df[feature] = original_df[feature].median()
                    print(f"  - Restored {feature} using global median")
                    restored_count += 1
            
            elif feature == 'age':
                # Age is absolutely critical - restore with minimal processing
                if original_df[feature].isnull().mean() < 0.3:  # If less than 30% missing
                    processed_df[feature] = original_df[feature]
                    # Fill remaining missing with median
                    if processed_df[feature].isnull().any():
                        processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())
                    print(f"  - Restored {feature}")
                    restored_count += 1
                else:
                    print(f"  - Couldn't restore {feature} - too much missing data")
                
            elif feature in critical_features['vitals']:
                # For vital signs, try to preserve the original values
                if original_df[feature].isnull().mean() < 0.5:  # If less than 50% missing
                    processed_df[feature] = original_df[feature]
                    # Fill remaining missing with median
                    if processed_df[feature].isnull().any():
                        processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())
                    print(f"  - Restored vital sign: {feature}")
                    restored_count += 1
                else:
                    print(f"  - Couldn't restore {feature} - too much missing data")
                    
            elif feature in critical_features['labs']:
                # For lab values, use median imputation if not too much is missing
                if original_df[feature].isnull().mean() < 0.6:  # If less than 60% missing
                    processed_df[feature] = original_df[feature]
                    # Fill missing with median
                    processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())
                    print(f"  - Restored lab value: {feature}")
                    restored_count += 1
                else:
                    print(f"  - Couldn't restore {feature} - too much missing data")
                    
            else:
                # For other features, use simple imputation
                processed_df[feature] = original_df[feature].fillna(original_df[feature].median())
                print(f"  - Restored {feature}")
                restored_count += 1
    
    if restored_count == 0:
        print("  - No critical features needed restoration")
    else:
        print(f"  - Successfully restored {restored_count} critical clinical features")
        
    return processed_df


def select_features(df, target_col=None, n_components=None, method='variance', keep_clinical=True):
    """Select most informative features using various methods"""
    print("Selecting features using method:", method)
    
    # List of key clinical features to prioritize keeping
    clinical_features = [
        'age', 'bmi', 'heart_rate_mean', 'sbp_mean', 'map_mean', 
        'resp_rate_mean', 'temp_mean', 'spo2_mean', 'lactate_max',
        'wbc_mean', 'creatinine_mean', 'bun_mean', 'glucose_mean',
        'platelets_mean', 'hemoglobin_mean', 'shock_index',
        'sirs_criteria_count', 'has_hypoxemia'
    ]
    
    # Filter to only include columns that exist in the dataframe
    if keep_clinical:
        clinical_features = [f for f in clinical_features if f in df.columns]
        print(f"  - Will prioritize keeping {len(clinical_features)} clinical features")
    

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
        
        if target_col is None or target_col not in df.columns:
            print("  - Target column is required for importance-based feature selection")
            return df
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Choose classifier or regressor based on the target type
        if len(np.unique(y)) < 10:  # Classification task
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression task
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X, y)
        
        # Select features above mean importance
        selector = SelectFromModel(model, threshold='mean')
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        
        print(f"  - Selected {len(selected_features)} out of {X.shape[1]} features based on importance")
        
        # Get the top 10 most important features
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("  - Top 10 most important features:")
        for i, (feat, imp) in enumerate(importances.head(10).items()):
            print(f"    {i+1}. {feat}: {imp:.4f}")
        
        # Return dataframe with selected features and target
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index).join(pd.Series(y, name=target_col))
    
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
        from imblearn.over_sampling import SMOTE
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    elif method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        
        # Apply ADASYN
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
    elif method == 'random_over':
        from imblearn.over_sampling import RandomOverSampler
        
        # Apply random oversampling
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif method == 'random_under':
        from imblearn.under_sampling import RandomUnderSampler
        
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

def enhanced_preprocess_pipeline(input_path, output_path, report_dir='./figures', target_col=None, 
                                keep_clinical=True):
    """Execute the enhanced preprocessing pipeline with all improvements"""
    print("Starting enhanced data preprocessing pipeline...")
    
    # Load data
    df = load_data(input_path)
    
    # Keep a copy of the original data for reporting and feature restoration
    original_df = df.copy()
    
    # Analyze missingness
    missingness_data = analyze_missingness(df)
    
    # Handle missing data
    df = handle_missing_data(df, missingness_data)
    
    # Restore critical features that may have been dropped
    df = restore_critical_features(df, original_df)
    
    # Handle outliers
    df = identify_and_handle_outliers(df)
    
    # Transform skewed features
    df = transform_skewed_features(df)
    
    # Create derived clinical features
    df = create_clinical_derived_features(df)
    
    # Verify feature scaling
    df = verify_feature_scaling(df)
    
    # Reduce redundancy using correlation method, but preserve clinical features
    df = select_features(df, target_col=target_col, method='correlation', keep_clinical=keep_clinical)
    
    # Final check for critical features
    df = restore_critical_features(df, original_df)
    
    # Additional feature selection if needed
    if target_col is not None and target_col in df.columns:
        print("\nPerforming feature selection with target variable...")
        df = select_features(df, target_col=target_col, method='importance', keep_clinical=keep_clinical)
        
        # Handle class imbalance for classification problems
        n_classes = len(df[target_col].unique())
        if n_classes <= 10:  # Classification task
            df = handle_class_imbalance(df, target_col, method='smote')
    else:
        print("\nNo target variable provided, skipping importance-based feature selection")
    
    # Save the processed data
    save_processed_data(df, output_path)
    
    # Generate report
    generate_preprocessing_report(original_df, df, report_dir)
    
    print("Enhanced data preprocessing pipeline complete!")
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

def save_processed_data(df, output_path):
    """Save the preprocessed dataframe"""
    print(f"Saving preprocessed data to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data with shape {df.shape}")

def generate_preprocessing_report(original_df, processed_df, output_dir):
    """Generate a report on the preprocessing steps and their impact"""
    print("Generating preprocessing report...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    report = {
        'original_shape': original_df.shape,
        'processed_shape': processed_df.shape,
        'dropped_features': set(original_df.columns) - set(processed_df.columns),
        'new_features': set(processed_df.columns) - set(original_df.columns),
        'missing_before': original_df.isnull().sum().sum(),
        'missing_after': processed_df.isnull().sum().sum(),
    }
    
    # Generate report text
    report_text = [
        "=== Data Preprocessing Report ===",
        f"Original data shape: {report['original_shape'][0]} rows, {report['original_shape'][1]} columns",
        f"Processed data shape: {report['processed_shape'][0]} rows, {report['processed_shape'][1]} columns",
        f"Features dropped: {len(report['dropped_features'])}",
        f"New features created: {len(report['new_features'])}",
        f"Missing values before: {report['missing_before']}",
        f"Missing values after: {report['missing_after']}",
        "\nNew features created:",
        "\n".join([f"- {feat}" for feat in sorted(list(report['new_features']))]),
        "\nFeatures dropped:",
        "\n".join([f"- {feat}" for feat in sorted(list(report['dropped_features']))])
    ]
    
    # Write report to file
    with open(os.path.join(output_dir, 'preprocessing_report.txt'), 'w') as f:
        f.write('\n'.join(report_text))
    
    # Generate some visualizations
    # 1. Missing data comparison
    plt.figure(figsize=(12, 6))
    missing_before = (original_df.isnull().mean() * 100).sort_values(ascending=False)[:20]
    missing_after = (processed_df.isnull().mean() * 100).sort_values(ascending=False)[:20]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(missing_before)), missing_before.values)
    plt.title('Missing Data Before (Top 20)')
    plt.ylabel('Missing Percentage')
    plt.xticks([])
    
    plt.subplot(1, 2, 2)
    if len(missing_after) > 0 and missing_after.values[0] > 0:
        plt.bar(range(len(missing_after)), missing_after.values)
        plt.title('Missing Data After (Top 20)')
        plt.xticks([])
    else:
        plt.text(0.5, 0.5, 'No Missing Data', horizontalalignment='center', 
                verticalalignment='center', fontsize=16)
        plt.title('Missing Data After')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_data_comparison.png'))
    
    # 2. Feature count by type
    plt.figure(figsize=(10, 6))
    
    orig_numeric = len(original_df.select_dtypes(include=[np.number]).columns)
    orig_categorical = len(original_df.select_dtypes(exclude=[np.number]).columns)
    
    proc_numeric = len(processed_df.select_dtypes(include=[np.number]).columns)
    proc_categorical = len(processed_df.select_dtypes(exclude=[np.number]).columns)
    
    categories = ['Numeric', 'Categorical']
    before_counts = [orig_numeric, orig_categorical]
    after_counts = [proc_numeric, proc_categorical]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, before_counts, width, label='Before')
    plt.bar(x + width/2, after_counts, width, label='After')
    
    plt.ylabel('Feature Count')
    plt.title('Feature Count by Type')
    plt.xticks(x, categories)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_count_comparison.png'))
    
    return report

def preprocess_pipeline(input_path, output_path, report_dir='./figures'):
    """Execute the entire preprocessing pipeline"""
    print("Starting data preprocessing pipeline...")
    
    # Load data
    df = load_data(input_path)
    
    # Keep a copy of the original data for reporting
    original_df = df.copy()
    
    # Analyze missingness
    missingness_data = analyze_missingness(df)
    
    # Handle missing data
    df = handle_missing_data(df, missingness_data)
    
    # Handle outliers
    df = identify_and_handle_outliers(df)
    
    # Transform skewed features
    df = transform_skewed_features(df)
    
    # Create derived clinical features
    df = create_clinical_derived_features(df)
    
    # Reduce redundancy
    df = reduce_feature_redundancy(df)
    
    # Standardize features
    df = standardize_features(df)
    
    # Save the processed data
    save_processed_data(df, output_path)
    
    # Generate report
    generate_preprocessing_report(original_df, df, report_dir)
    
    print("Data preprocessing pipeline complete!")
    return df

if __name__ == "__main__":
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'processed', 'extracted_features.csv')
    output_path = os.path.join(base_dir, 'data', 'processed', 'preprocessed_features.csv')
    report_dir = os.path.join(base_dir, 'figures')
    
    # If you know your target column, specify it here - otherwise set to None
    target_column = 'mortality'  # For general preprocessing
    # target_column = 'mortality'  # For mortality prediction
    
    # Run enhanced preprocessing with clinical feature preservation
    preprocessed_df = enhanced_preprocess_pipeline(
        input_path, 
        output_path, 
        report_dir,
        target_col=target_column,
        keep_clinical=True  # Ensure clinical features are preserved
    )