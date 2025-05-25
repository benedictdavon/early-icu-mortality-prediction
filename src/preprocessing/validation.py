# value ranges, unit handling
import numpy as np
import pandas as pd

def restore_critical_features(processed_df, original_df):
    """Restore critical clinical features that may have been dropped"""
    print("Restoring critical clinical features...")
    
    # Expanded list of clinically important features to restore
    critical_features = {
        # Highest priority (must have)
        'demographic': ['age', 'gender_numeric', 'bmi'],
        
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
            # Check data type to decide imputation strategy
            if feature == 'gender' or original_df[feature].dtype == 'object' or original_df[feature].dtype.name == 'category':
                # For categorical features like gender, use mode (most frequent value)
                mode_value = original_df[feature].mode().iloc[0] if not original_df[feature].empty else "Unknown"
                processed_df[feature] = original_df[feature].fillna(mode_value)
                print(f"  - Restored categorical feature {feature} using mode")
                restored_count += 1
            
            elif feature == 'bmi':
                # BMI can be estimated from age, gender and other vitals
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
                # For numeric features, use median imputation
                try:
                    processed_df[feature] = original_df[feature].fillna(original_df[feature].median())
                    print(f"  - Restored numeric feature {feature}")
                    restored_count += 1
                except (TypeError, ValueError) as e:
                    # If median fails, try mode instead
                    mode_value = original_df[feature].mode().iloc[0] if not original_df[feature].empty else None
                    processed_df[feature] = original_df[feature].fillna(mode_value)
                    print(f"  - Restored {feature} using mode (median failed)")
                    restored_count += 1
    
    if restored_count == 0:
        print("  - No critical features needed restoration")
    else:
        print(f"  - Successfully restored {restored_count} critical clinical features")
        
    return processed_df


def remove_redundant_features(df):
    """Remove specific redundant features that dilute feature importance"""
    print("Removing specific redundant features...")
    
    redundant_pairs = {
        'age': 'anchor_age',  # Keep age, remove anchor_age
        'duration_hours': 'duration_hours_log'  # Keep duration_hours, remove log version
    }
    
    features_to_remove = []
    for keep_feature, remove_feature in redundant_pairs.items():
        if keep_feature in df.columns and remove_feature in df.columns:
            features_to_remove.append(remove_feature)
            print(f"  - Keeping '{keep_feature}' and removing redundant '{remove_feature}'")
    
    if features_to_remove:
        df = df.drop(columns=features_to_remove)
        print(f"  - Removed {len(features_to_remove)} redundant features")
    else:
        print("  - No redundant features to remove")
    
    return df


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



def clean_clinical_measurements(features_df):
    """Clean and validate clinical measurements to ensure physiological plausibility."""
    print("Validating and cleaning clinical measurements...")
    df = features_df.copy()
    
    # 1. BMI validation and correction
    if 'bmi' in df.columns:
        # Cap at physiologically plausible values
        outliers_before = sum((df['bmi'] < 10) | (df['bmi'] > 80))
        df['bmi'] = df['bmi'].clip(10, 80)
        print(f"BMI: capped {outliers_before} implausible values outside 10-80 range")
    
    # 2. Temperature validation (handle both F and C)
    for temp_col in [col for col in df.columns if 'temp' in col.lower() and '_mean' in col.lower()]:
        if temp_col in df.columns and not df[temp_col].empty:
            # Check if values seem to be in Fahrenheit (median > 90)
            if df[temp_col].median() > 90:
                print(f"{temp_col} appears to be in Fahrenheit, converting to Celsius")
                # Convert F to C but only for values that appear to be in F
                f_mask = df[temp_col] > 90
                df.loc[f_mask, temp_col] = (df.loc[f_mask, temp_col] - 32) * 5/9
                
            # Cap at plausible values
            outliers_before = sum((df[temp_col] < 35) | (df[temp_col] > 42))
            df[temp_col] = df[temp_col].clip(35, 42)
            if outliers_before > 0:
                print(f"{temp_col}: capped {outliers_before} values outside 35-42°C")
    
    # 3. Validate clinical lab values
    lab_limits = {
        'glucose_mean': (20, 600),      # mg/dL
        'creatinine_mean': (0.1, 15),   # mg/dL
        'sodium_mean': (110, 175),      # mEq/L
        'potassium_mean': (1.5, 9),     # mEq/L
        'bun_mean': (1, 200),           # mg/dL
        'wbc_mean': (0, 100),           # K/uL
        'hemoglobin_mean': (1, 25),     # g/dL
        'platelets_mean': (1, 1000),    # K/uL
        'lactate_mean': (0.1, 30),      # mmol/L
        'bilirubin_mean': (0.1, 30),    # mg/dL
        'alkaline_phosphatase_mean': (1, 1000),  # U/L
        'hematocrit_mean': (5, 65),     # %
        'chloride_mean': (70, 150),     # mEq/L
        'bicarbonate_mean': (5, 50),    # mEq/L
        'anion_gap_mean': (1, 40),      # mEq/L
        'inr_mean': (0.5, 12),          # INR values
        'inr_max': (0.5, 20)            # INR max values
    }
    
    for lab, (min_val, max_val) in lab_limits.items():
        if lab in df.columns:
            outliers_before = sum((df[lab] < min_val) | (df[lab] > max_val))
            if outliers_before > 0:
                df[lab] = df[lab].clip(min_val, max_val)
                print(f"{lab}: capped {outliers_before} values outside range {min_val}-{max_val}")
    
    # 4. Validate vital signs
    vital_limits = {
        'heart_rate_mean': (20, 220),   # bpm
        'resp_rate_mean': (4, 60),      # breaths/min
        'sbp_mean': (30, 250),          # mmHg
        'dbp_mean': (20, 180),          # mmHg
        'map_mean': (20, 200),          # mmHg
        'spo2_mean': (50, 100)          # percent
    }
    
    for vital, (min_val, max_val) in vital_limits.items():
        if vital in df.columns:
            outliers_before = sum((df[vital] < min_val) | (df[vital] > max_val))
            if outliers_before > 0:
                df[vital] = df[vital].clip(min_val, max_val)
                print(f"{vital}: capped {outliers_before} values outside range {min_val}-{max_val}")
    
    # 5. Handle min/max/first/last/delta variants too
    for base_feature in list(lab_limits.keys()) + list(vital_limits.keys()):
        base_name = base_feature.replace('_mean', '')
        for variant in ['_min', '_max', '_first', '_last']:
            feature = f"{base_name}{variant}"
            if feature in df.columns:
                # Use the same limits as the mean values
                if f"{base_name}_mean" in lab_limits:
                    min_val, max_val = lab_limits[f"{base_name}_mean"]
                elif f"{base_name}_mean" in vital_limits:
                    min_val, max_val = vital_limits[f"{base_name}_mean"]
                else:
                    continue
                    
                outliers_before = sum((df[feature] < min_val) | (df[feature] > max_val))
                if outliers_before > 0:
                    df[feature] = df[feature].clip(min_val, max_val)
                    print(f"{feature}: capped {outliers_before} values outside range {min_val}-{max_val}")
    
    # 6. Validate urine output
    if 'urine_output_total' in df.columns:
        # Cap at physiologically plausible values (0 to 10,000 mL for 6-hour window)
        outliers_before = sum(df['urine_output_total'] > 10000)
        df['urine_output_total'] = df['urine_output_total'].clip(0, 10000)
        if outliers_before > 0:
            print(f"urine_output_total: capped {outliers_before} values above 10000 mL")
    
    # For glucose: detect mmol/L vs mg/dL
    for glucose_col in [col for col in df.columns if 'glucose' in col.lower()]:
        if glucose_col in df.columns and not df[glucose_col].empty:
            # Check if values seem to be in mmol/L (typically <20)
            if df[glucose_col].median() < 20 and df[glucose_col].median() > 0:
                print(f"{glucose_col} appears to be in mmol/L, converting to mg/dL")
                # Convert mmol/L to mg/dL (multiply by 18)
                df[glucose_col] = df[glucose_col] * 18
    
    # For creatinine: detect μmol/L vs mg/dL
    for creat_col in [col for col in df.columns if 'creatinine' in col.lower()]:
        if creat_col in df.columns and not df[creat_col].empty:
            # Check if values seem to be in μmol/L (typically >50)
            if df[creat_col].median() > 50:
                print(f"{creat_col} appears to be in μmol/L, converting to mg/dL")
                # Convert μmol/L to mg/dL (divide by 88.4)
                df[creat_col] = df[creat_col] / 88.4
    
    # For bilirubin: detect μmol/L vs mg/dL
    for bili_col in [col for col in df.columns if 'bilirubin' in col.lower()]:
        if bili_col in df.columns and not df[bili_col].empty:
            # Check if values seem to be in μmol/L (typically >20)
            if df[bili_col].median() > 20:
                print(f"{bili_col} appears to be in μmol/L, converting to mg/dL")
                # Convert μmol/L to mg/dL (divide by 17.1)
                df[bili_col] = df[bili_col] / 17.1
    
    # For hemoglobin: detect g/L vs g/dL
    for hgb_col in [col for col in df.columns if 'hemoglobin' in col.lower() or 'hgb' in col.lower()]:
        if hgb_col in df.columns and not df[hgb_col].empty:
            # Check if values seem to be in g/L (typically >30)
            if df[hgb_col].median() > 30:
                print(f"{hgb_col} appears to be in g/L, converting to g/dL")
                # Convert g/L to g/dL (divide by 10)
                df[hgb_col] = df[hgb_col] / 10


    print("Clinical measurement validation complete")
    return df

