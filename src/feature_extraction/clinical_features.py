
import pandas as pd
import numpy as np

def add_clinical_derived_features(features_df):
    """Add clinically meaningful derived features based on raw measurements."""
    print("Adding clinically meaningful derived features...")
    df = features_df.copy()
    
    # SIRS criteria (Systemic Inflammatory Response Syndrome)
    if 'heart_rate_mean' in df.columns:
        df['has_tachycardia'] = (df['heart_rate_mean'] > 90).astype(int)
    
    if 'resp_rate_mean' in df.columns:
        df['has_tachypnea'] = (df['resp_rate_mean'] > 20).astype(int)
        
        # Enhanced respiratory features (from SHAP analysis showing tachypnea as a top predictor)
        # Calculate respiratory pattern variability across available time points
        resp_cols = [col for col in df.columns if col.startswith('resp_rate_hour_')]
        if len(resp_cols) >= 3:  # Need at least 3 measurements for variability
            # Calculate variability (standard deviation) across time points
            df['resp_variability'] = df[resp_cols].std(axis=1)
            
            # Calculate respiratory trend using polynomial fit
            try:
                # Get time points (hour numbers) from column names
                time_points = [int(col.split('_')[-1]) for col in resp_cols]
                
                # For each patient, calculate respiratory trend slope
                slopes = []
                for _, row in df.iterrows():
                    values = [row[col] for col in resp_cols]
                    # Filter out NaN values
                    valid_points = [(t, v) for t, v in zip(time_points, values) if not pd.isna(v)]
                    
                    if len(valid_points) >= 2:  # Need at least 2 points for trend
                        x = [p[0] for p in valid_points]
                        y = [p[1] for p in valid_points]
                        if len(set(x)) >= 2:  # Need at least 2 unique x values
                            try:
                                # Get slope coefficient from polynomial fit
                                slope = np.polyfit(x, y, 1)[0]
                                slopes.append(slope)
                            except:
                                slopes.append(np.nan)
                        else:
                            slopes.append(np.nan)
                    else:
                        slopes.append(np.nan)
                
                df['resp_trend'] = slopes
                print(f"Added respiratory trend feature - {df['resp_trend'].notna().sum()} non-null values")
            except Exception as e:
                print(f"Error calculating respiratory trend: {e}")
    
    if 'temp_mean' in df.columns:
        df['has_fever_or_hypothermia'] = ((df['temp_mean'] > 38) | (df['temp_mean'] < 36)).astype(int)
    
    if 'wbc_mean' in df.columns:
        df['has_abnormal_wbc'] = ((df['wbc_mean'] > 12) | (df['wbc_mean'] < 4)).astype(int)
    
    # Count SIRS criteria
    sirs_cols = ['has_tachycardia', 'has_tachypnea', 'has_fever_or_hypothermia', 'has_abnormal_wbc']
    sirs_cols_present = [col for col in sirs_cols if col in df.columns]
    
    if len(sirs_cols_present) > 0:
        df['sirs_criteria_count'] = df[sirs_cols_present].sum(axis=1)
        df['sirs_criteria_count'].fillna(0, inplace=True)
    
    # Shock index (Heart Rate / Systolic BP) - predictor of mortality
    if all(col in df.columns for col in ['heart_rate_mean', 'sbp_mean']):
        df['shock_index'] = df['heart_rate_mean'] / df['sbp_mean']
        # Cap extreme values
        df.loc[df['shock_index'] > 2, 'shock_index'] = 2
        
        # Enhanced shock features - create interaction with lactate (from SHAP analysis)
        if 'lactate_max' in df.columns:
            df['shock_lactate_interaction'] = df['shock_index'] * df['lactate_max']
            print(f"Added shock-lactate interaction - {df['shock_lactate_interaction'].notna().sum()} non-null values")
    
    # Renal dysfunction marker
    if all(col in df.columns for col in ['creatinine_mean', 'bun_mean']):
        df['bun_creatinine_ratio'] = df['bun_mean'] / df['creatinine_mean']
        # Cap extreme values
        df.loc[df['bun_creatinine_ratio'] > 30, 'bun_creatinine_ratio'] = 30
    
    # Oxygenation marker
    if 'spo2_mean' in df.columns:
        df['has_hypoxemia'] = (df['spo2_mean'] < 92).astype(int)
        
        # Enhanced respiratory distress score (SpO2 + respiratory rate)
        if 'resp_rate_mean' in df.columns:
            # Low SpO2 with high respiratory rate indicates respiratory distress
            df['resp_distress_score'] = df['resp_rate_mean'] / (df['spo2_mean'].clip(80, 100) / 100)
            print(f"Added respiratory distress score - {df['resp_distress_score'].notna().sum()} non-null values")
    
    # Group age into clinically meaningful bins - Enhanced age features from SHAP analysis
    if 'age' in df.columns:
        # Create age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 40, 60, 80, 120], 
            labels=['young_adult', 'middle_aged', 'elderly', 'very_elderly']
        )
        
        # One-hot encode age groups for easier model interpretation
        try:
            age_dummies = pd.get_dummies(df['age_group'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
            print(f"Added {len(age_dummies.columns)} age group dummy variables")
        except Exception as e:
            print(f"Error creating age group dummies: {e}")
            
        # Create age-comorbidity interaction (often significant in mortality models)
        if 'prev_dx_count_total' in df.columns:
            df['age_comorbidity'] = df['age'] * df['prev_dx_count_total']
            print(f"Added age-comorbidity interaction - {df['age_comorbidity'].notna().sum()} non-null values")
    
    # Create organ dysfunction composite scores (enhanced feature from SHAP analysis)
    organ_systems = {
        'renal': ['creatinine_max', 'bun_max', 'urine_output_total'],
        'liver': ['bilirubin_max', 'alkaline_phosphatase_max'],
        'cardiac': ['heart_rate_max', 'sbp_min', 'shock_index'],
        'respiratory': ['resp_rate_max', 'spo2_min'],
        'hematologic': ['platelets_min', 'inr_max', 'hemoglobin_min']
    }
    
    # Only create organ system scores if we have enough data
    for system, markers in organ_systems.items():
        # Check which markers are available
        available_markers = [col for col in markers if col in df.columns]
        
        if len(available_markers) >= 2:  # Only create score if we have at least 2 markers
            print(f"Creating {system} dysfunction score from: {', '.join(available_markers)}")
            
            # Normalize each marker to 0-1 scale and combine
            normalized_markers = []
            for marker in available_markers:
                if marker in df.columns and df[marker].notna().sum() > 0:
                    # For markers where higher is worse
                    if any(bad_marker in marker for bad_marker in ['creatinine', 'bun', 'bilirubin', 
                                                                  'heart_rate', 'resp_rate', 'inr']):
                        # Min-max normalization between expected clinical ranges
                        normalized = (df[marker] - df[marker].min()) / (df[marker].max() - df[marker].min())
                        normalized_markers.append(normalized)
                        
                    # For markers where lower is worse
                    elif any(bad_marker in marker for bad_marker in ['sbp', 'platelets', 'spo2', 
                                                                   'hemoglobin', 'urine_output']):
                        # Inverse normalization so higher score = worse condition
                        normalized = 1 - (df[marker] - df[marker].min()) / (df[marker].max() - df[marker].min())
                        normalized_markers.append(normalized)
            
            if normalized_markers:
                # Combine normalized scores (higher = worse organ function)
                df[f'{system}_dysfunction'] = pd.concat(normalized_markers, axis=1).mean(axis=1)
    
    # Create total organ dysfunction score
    dysfunction_cols = [col for col in df.columns if '_dysfunction' in col]
    if dysfunction_cols:
        print(f"Creating total organ dysfunction score from {len(dysfunction_cols)} systems")
        df['total_organ_dysfunction'] = df[dysfunction_cols].mean(axis=1)
    
    # Add vital sign trend features for key markers
    vital_prefixes = ['heart_rate', 'resp_rate', 'sbp', 'map']
    for vital in vital_prefixes:
        hour_cols = [col for col in df.columns if col.startswith(f"{vital}_hour_")]
        if len(hour_cols) >= 3:  # Need at least 3 time points for a meaningful trend
            print(f"Creating {vital} trend feature from {len(hour_cols)} time points")
            hour_cols.sort()  # Ensure time points are in order
            
            # Extract time points from column names
            time_points = [int(col.split('_')[-1]) for col in hour_cols]
            
            # Calculate trend for each patient
            vital_trends = []
            for _, row in df.iterrows():
                values = [row[col] for col in hour_cols]
                valid_points = [(t, v) for t, v in zip(time_points, values) if not pd.isna(v)]
                
                if len(valid_points) >= 3:  # Need at least 3 valid measurements
                    try:
                        x = [p[0] for p in valid_points]
                        y = [p[1] for p in valid_points]
                        slope = np.polyfit(x, y, 1)[0]
                        vital_trends.append(slope)
                    except:
                        vital_trends.append(np.nan)
                else:
                    vital_trends.append(np.nan)
                    
            df[f'{vital}_trend'] = vital_trends
            
    print(f"Added {len(df.columns) - len(features_df.columns)} new clinical features")
    return df



def create_clinical_interaction_features(df):
    """
    Create interaction features between clinically relevant variables 
    that may improve model predictive performance.
    """
    print("Creating clinical interaction features...")
    df = df.copy()
    
    # Track created features
    created_features = []
    
    # 1. Shock Index × Age - represents age-adjusted risk from shock
    if all(col in df.columns for col in ['shock_index', 'age']):
        df['shock_index_x_age'] = df['shock_index'] * df['age']
        created_features.append('shock_index_x_age')
    
    # 2. BUN/Creatinine ratio × Age - kidney function deteriorates with age
    if all(col in df.columns for col in ['bun_creatinine_ratio', 'age']):
        df['bun_creat_ratio_x_age'] = df['bun_creatinine_ratio'] * df['age']
        created_features.append('bun_creat_ratio_x_age')
    
    # 3. Lactate × Anion Gap - combined metabolic derangement
    if all(col in df.columns for col in ['lactate_mean', 'anion_gap_max']):
        df['lactate_x_anion_gap'] = df['lactate_mean'] * df['anion_gap_max']
        created_features.append('lactate_x_anion_gap')
    
    # 4. Previous diagnosis count × Age - comorbidity burden increases with age
    if all(col in df.columns for col in ['prev_dx_count_total', 'age']):
        df['prev_dx_count_x_age'] = df['prev_dx_count_total'] * df['age']
        created_features.append('prev_dx_count_x_age')
    
    # 5. Respiratory rate × SpO2 indicator - breathing effort vs oxygenation
    if all(col in df.columns for col in ['resp_rate_mean', 'spo2_min']):
        # Low SpO2 with high resp rate indicates respiratory distress
        df['resp_distress'] = df['resp_rate_mean'] * (100 - df['spo2_min'])
        created_features.append('resp_distress')
    
    # 6. MAP variance × Lactate - hemodynamic instability with tissue perfusion
    if all(col in df.columns for col in ['sbp_min', 'sbp_max', 'lactate_mean']):
        map_variance = df['sbp_max'] - df['sbp_min']
        df['map_variance_x_lactate'] = map_variance * df['lactate_mean']
        created_features.append('map_variance_x_lactate')
    
    # 7. Heart rate / systolic BP - another shock measure
    if all(col in df.columns for col in ['heart_rate_max', 'sbp_min']):
        df['hr_sbp_ratio'] = df['heart_rate_max'] / df['sbp_min'].replace(0, np.nan)
        # Replace infinities with NaNs to be handled by imputation later
        df['hr_sbp_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        created_features.append('hr_sbp_ratio')
    
    # 8. SIRS criteria × Lactate - sepsis risk indicator
    if all(col in df.columns for col in ['sirs_criteria_count', 'lactate_mean']):
        df['sirs_x_lactate'] = df['sirs_criteria_count'] * df['lactate_mean']
        created_features.append('sirs_x_lactate')
    
    print(f"Created {len(created_features)} interaction features: {', '.join(created_features)}")
    return df



def add_log_transformations(features_df):
    """Add log transformations for highly skewed variables."""
    print("Adding log transformations for skewed variables...")
    df = features_df.copy()
    
    # Define skewed lab values (from your analysis showing abs(skew) > 2)
    skewed_labs = [
        'bilirubin', 'creatinine', 'lactate', 'wbc', 
        'platelets', 'alkaline_phosphatase', 'bun'
    ]
    
    # Add log transformations for these values
    transform_count = 0
    for lab in skewed_labs:
        mean_col = f'{lab}_mean'
        if mean_col in df.columns and df[mean_col].notna().sum() > 0:
            # Add small constant to avoid log(0) and maintain monotonicity
            df[f'{lab}_log'] = np.log1p(df[mean_col])
            transform_count += 1
    
    print(f"Added log transformations for {transform_count} skewed variables")
    return df


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
    
    print("Clinical measurement validation complete")
    return df
