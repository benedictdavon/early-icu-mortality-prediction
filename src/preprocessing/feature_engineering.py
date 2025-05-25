# clinical, polynomial, temporal
import numpy as np
import pandas as pd

def create_clinical_derived_features(df):
    """Create clinically meaningful derived features"""
    print("Creating derived clinical features...")
    
    processed_df = df.copy()
    
    # Add BMI calculation code here
    # Check if height and weight columns exist to calculate BMI
    weight_cols = [col for col in processed_df.columns if 'weight' in col.lower()]
    height_cols = [col for col in processed_df.columns if 'height' in col.lower()]
    
    # Calculate BMI if height and weight are available
    if weight_cols and height_cols:
        print("  - Found height and weight columns, calculating BMI")
        
        # Use the first available weight and height columns
        weight_col = weight_cols[0]
        height_col = height_cols[0]
        
        # Handle unit conversions if needed (assuming weight in kg and height in cm)
        # Convert height from cm to meters if needed
        height_values = processed_df[height_col].copy()
        if height_values.median() > 3:  # Height likely in cm if median > 3
            height_values = height_values / 100.0  # Convert to meters
            
        # Calculate BMI: weight(kg) / height(m)²
        processed_df['bmi'] = processed_df[weight_col] / (height_values ** 2)
        
        # Apply clinical bounds to BMI (10-100 is a very wide but safe range)
        processed_df['bmi'] = processed_df['bmi'].clip(10, 100)
        print(f"  - Calculated BMI from {weight_col} and {height_col}")
    else:
        print("  - Height or weight columns not found, estimating BMI")
        
        # BMI estimate based on demographics
        if 'age' in processed_df.columns and 'gender' in processed_df.columns:
            # Create estimated BMI based on age and gender
            # These are approximate average values by age group and gender
            def estimate_bmi(row):
                age = row['age']
                gender = row['gender'] if 'gender' in row else 'M'
                
                if isinstance(gender, str):
                    gender = gender.upper()[0]  # Get first letter (M or F)
                elif pd.isna(gender) or gender is None:
                    gender = 'M'  # Default to male if missing
                
                # Simplified BMI estimation by age and gender
                if gender == 'M' or gender == 1:
                    if age < 30: return 24.5
                    elif age < 45: return 26.0
                    elif age < 60: return 27.5
                    else: return 26.8
                else:  # Female
                    if age < 30: return 23.8
                    elif age < 45: return 25.5
                    elif age < 60: return 26.8
                    else: return 26.2
            
            processed_df['bmi'] = processed_df.apply(estimate_bmi, axis=1)
            print("  - Estimated BMI based on age and gender")
        else:
            # Default fallback - assign median BMI for general population
            processed_df['bmi'] = 26.0  # Average adult BMI
            print("  - No demographic data for BMI estimation, using population average")

    
    # 1. SIRS (Systemic Inflammatory Response Syndrome) Criteria
    # - Heart rate > 90 bpm
    # - Respiratory rate > 20 breaths/min
    # - Temperature < 36°C or > 38°C
    # - WBC < 4,000/mm³ or > 12,000/mm³
    
    # 1. Prioritize the most predictive vital signs based on analysis
    high_value_vitals = ['resp_rate', 'lactate', 'heart_rate', 'anion_gap']
    print(f"  - Prioritizing high-value clinical features: {', '.join(high_value_vitals)}")
    
    # 2. Create focused clinical indicators based on the most important vitals
    # Respiratory rate abnormalities (high predictor of mortality)
    if 'resp_rate_mean' in processed_df.columns:
        processed_df['has_tachypnea'] = (processed_df['resp_rate_mean'] > 20).astype(int)
        processed_df['has_bradypnea'] = (processed_df['resp_rate_mean'] < 12).astype(int)
        print("  - Created respiratory rate indicators")
    
    # Lactate (strong predictor of mortality)
    if 'lactate_mean' in processed_df.columns:
        processed_df['has_elevated_lactate'] = (processed_df['lactate_mean'] > 2).astype(int)
        processed_df['has_severe_lactate'] = (processed_df['lactate_mean'] > 4).astype(int)
        print("  - Created lactate elevation indicators")
    
    # Heart rate abnormalities
    if 'heart_rate_mean' in processed_df.columns:
        processed_df['has_tachycardia'] = (processed_df['heart_rate_mean'] > 90).astype(int)
        processed_df['has_bradycardia'] = (processed_df['heart_rate_mean'] < 60).astype(int)
        print("  - Created heart rate indicators")
        
    # Anion gap (high predictor of mortality)
    if 'anion_gap_mean' in processed_df.columns:
        processed_df['has_high_anion_gap'] = (processed_df['anion_gap_mean'] > 12).astype(int)
        print("  - Created anion gap indicator")
    
    # 3. Create shock index (heart rate / systolic blood pressure)
    if 'heart_rate_mean' in processed_df.columns and 'sbp_mean' in processed_df.columns:
        # Avoid division by zero
        sbp_safe = processed_df['sbp_mean'].copy()
        sbp_safe[sbp_safe == 0] = np.nan
        
        processed_df['shock_index'] = processed_df['heart_rate_mean'] / sbp_safe
        processed_df['has_shock'] = (processed_df['shock_index'] > 0.9).astype(int)
        print("  - Created shock index (HR/SBP) and shock indicator")
    
    # 4. Check for hypoxemia (identified as valuable predictor)
    if 'spo2_mean' in processed_df.columns:
        processed_df['has_hypoxemia'] = (processed_df['spo2_mean'] < 92).astype(int)
        processed_df['has_severe_hypoxemia'] = (processed_df['spo2_mean'] < 88).astype(int)
        print("  - Created hypoxemia indicators")
        
    # 5. Create delta features for the most important changes identified in the analysis
    delta_features = [
        ('spo2', '1to2'),
        ('lactate', '4to6'),
        ('dbp', '0to1'),
        ('sodium', '3to4'),
        ('wbc', '0to6'),
        ('heart_rate', '2to3'),
        ('resp_rate', '4to6')
    ]
    
    for vital, change_period in delta_features:
        # For example: spo2_1 and spo2_2 for '1to2'
        start_hour, end_hour = change_period.split('to')
        start_col = f"{vital}_{start_hour}"
        end_col = f"{vital}_{end_hour}"
        
        if start_col in processed_df.columns and end_col in processed_df.columns:
            delta_col = f"{vital}_delta_{change_period}"
            processed_df[delta_col] = processed_df[end_col] - processed_df[start_col]
            print(f"  - Created high-value change feature: {delta_col}")
    
    return processed_df


def add_temporal_trends(df):
    """Add temporal trends by comparing first vs last measurements when available"""
    print("Adding temporal trends...")
    
    # Column groups where temporal trends might exist
    temporal_groups = {
        'lactate': ['lactate_min', 'lactate_max'],
        'heart_rate': ['heart_rate_min', 'heart_rate_max'],
        'sbp': ['sbp_min', 'sbp_max'],
        'dbp': ['dbp_min', 'dbp_max'],
        'resp_rate': ['resp_rate_min', 'resp_rate_max'],
        'temp': ['temp_min', 'temp_max'],
        'spo2': ['spo2_min', 'spo2_max']
    }
    
    trends_created = 0
    
    # For each group, create delta and trending features
    for feature_group, columns in temporal_groups.items():
        if len(columns) >= 2 and all(col in df.columns for col in columns):
            min_col = columns[0]
            max_col = columns[1]
            
            # Create delta (absolute change)
            delta_name = f"{feature_group}_delta"
            df[delta_name] = df[max_col] - df[min_col]
            
            # Create relative percent change
            pct_change_name = f"{feature_group}_pct_change"
            # Avoid division by zero
            df[pct_change_name] = ((df[max_col] - df[min_col]) / 
                                   df[min_col].replace(0, np.nan)) * 100
            df[pct_change_name] = df[pct_change_name].fillna(0)
            
            # Create trend direction (-1 decreasing, 0 stable, 1 increasing)
            # Using a threshold of 5% change to indicate meaningful trend
            trend_name = f"{feature_group}_trend"
            df[trend_name] = np.sign(df[pct_change_name])
            df.loc[abs(df[pct_change_name]) < 5, trend_name] = 0
            
            trends_created += 3
            print(f"  - Created temporal features for {feature_group}")
    
    if trends_created == 0:
        print("  - No temporal features created (min/max pairs not available)")
    else:
        print(f"  - Added {trends_created} temporal trend features")
    
    return df


def add_polynomial_features(df):
    """Add polynomial features for key vital signs to capture non-linear relationships"""
    print("Adding polynomial features for key vitals...")
    
    # Key clinical features that may have non-linear relationships with mortality
    key_vitals = [
        'temp_mean', 'heart_rate_mean', 'resp_rate_mean', 
        'lactate_mean', 'sbp_mean', 'spo2_mean'
    ]
    
    # Filter to only include available columns
    available_vitals = [col for col in key_vitals if col in df.columns]
    
    if not available_vitals:
        print("  - No key vitals available for polynomial features")
        return df
        
    features_created = 0
    
    # For each vital, create quadratic term
    for vital in available_vitals:
        # Create squared term to capture U-shaped or inverted-U relationships
        poly_name = f"{vital}_squared"
        df[poly_name] = df[vital] ** 2
        features_created += 1
        
        # Create normalized distance from clinical normal
        # Define normal ranges for key vitals
        normal_ranges = {
            'temp_mean': 36.5,      # Normal body temperature in Celsius
            'heart_rate_mean': 75,  # Normal resting heart rate
            'resp_rate_mean': 15,   # Normal respiratory rate
            'lactate_mean': 1.0,    # Normal lactate level
            'sbp_mean': 120,        # Normal systolic BP
            'spo2_mean': 97         # Normal oxygen saturation
        }
        
        if vital in normal_ranges:
            dist_name = f"{vital}_dist_from_normal"
            df[dist_name] = (df[vital] - normal_ranges[vital]) ** 2
            features_created += 1
    
    print(f"  - Created {features_created} polynomial features for key vitals")
    return df


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

def extract_temporal_features(df):
    """
    Extract useful temporal features from date/time columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with timestamp columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted temporal features
    """
    print("Extracting temporal features from timestamps...")
    
    result_df = df.copy()
    
    # Identify potential timestamp columns
    time_cols = [col for col in df.columns if any(term in col.lower() for term in 
                ['time', 'date', 'day', 'hour', 'admission', 'discharge'])]
    
    date_cols = []
    features_added = 0
    
    for col in time_cols:
        # Try to convert to datetime
        try:
            result_df[col] = pd.to_datetime(result_df[col])
            date_cols.append(col)
            
            # Extract hour of day
            result_df[f'{col}_hour'] = result_df[col].dt.hour
            features_added += 1
            
            # Extract day of week (0=Monday, 6=Sunday)
            result_df[f'{col}_dayofweek'] = result_df[col].dt.dayofweek
            features_added += 1
            
            # Extract is_weekend
            result_df[f'{col}_is_weekend'] = (result_df[col].dt.dayofweek >= 5).astype(int)
            features_added += 1
            
            print(f"  - Extracted temporal features from {col}")
        except Exception as e:
            print(f"  - Could not convert {col} to datetime: {str(e)}")
    
    # Calculate time since first record if multiple timestamps exist
    if len(date_cols) >= 2:
        # Find earliest timestamp column (typically admission)
        earliest_col = min(date_cols, key=lambda col: result_df[col].min())
        
        # Calculate hours since earliest event for each timestamp
        for col in date_cols:
            if col != earliest_col:
                try:
                    result_df[f'hours_since_{earliest_col}_to_{col}'] = (
                        (result_df[col] - result_df[earliest_col]).dt.total_seconds() / 3600
                    )
                    print(f"  - Added hours_since_{earliest_col}_to_{col}")
                    features_added += 1
                except Exception as e:
                    print(f"  - Error calculating time difference: {str(e)}")
    
    print(f"Added {features_added} temporal features")
    return result_df

def add_advanced_clinical_features(df):
    """
    Add advanced clinical features based on medical domain knowledge
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with clinical measurements
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added advanced clinical features
    """
    print("Adding advanced clinical features...")
    
    result_df = df.copy()
    features_added = 0
    
    # Shock index (HR/SBP ratio) - critical for hypoperfusion detection
    if all(col in result_df.columns for col in ['heart_rate_mean', 'sbp_mean']):
        result_df['shock_index_mean'] = result_df['heart_rate_mean'] / result_df['sbp_mean']
        print("  - Added shock_index_mean")
        features_added += 1
    
    # Worst case shock index - captures peak instability
    if all(col in result_df.columns for col in ['heart_rate_max', 'sbp_min']):
        result_df['shock_index_max'] = result_df['heart_rate_max'] / result_df['sbp_min']
        print("  - Added shock_index_max (worst case scenario)")
        features_added += 1
    
    # MAP thresholds - clinically significant cutoffs for intervention
    if 'map_mean' in result_df.columns:
        result_df['map_below_65'] = (result_df['map_mean'] < 65).astype(int)
        print("  - Added map_below_65 flag (sepsis treatment threshold)")
        features_added += 1
    
    if 'map_min' in result_df.columns:
        result_df['map_min_below_60'] = (result_df['map_min'] < 60).astype(int)
        print("  - Added map_min_below_60 flag (severe hypotension)")
        features_added += 1
    
    # Oxygen metrics
    if all(col in result_df.columns for col in ['pao2_mean', 'fio2_mean']):
        result_df['pf_ratio'] = result_df['pao2_mean'] / (result_df['fio2_mean'])
        print("  - Added P/F ratio (lung function metric)")
        
        # Add ARDS severity categories based on Berlin definition
        result_df['mild_ards'] = ((result_df['pf_ratio'] <= 300) & 
                                 (result_df['pf_ratio'] > 200)).astype(int)
        result_df['moderate_ards'] = ((result_df['pf_ratio'] <= 200) & 
                                     (result_df['pf_ratio'] > 100)).astype(int)
        result_df['severe_ards'] = (result_df['pf_ratio'] <= 100).astype(int)
        print("  - Added ARDS severity categories")
        features_added += 4
    
    # Renal dysfunction metrics
    if all(col in result_df.columns for col in ['bun_mean', 'creatinine_mean']):
        result_df['bun_creatinine_ratio'] = result_df['bun_mean'] / result_df['creatinine_mean']
        print("  - Added BUN/Creatinine ratio (pre-renal AKI indicator)")
        features_added += 1
    
    # Liver function composite
    if all(col in result_df.columns for col in ['bilirubin_mean', 'inr_mean']):
        # Normalize values to typical upper limits of normal
        bili_norm = result_df['bilirubin_mean'] / 1.2  # ULN for bilirubin
        inr_norm = result_df['inr_mean'] / 1.1         # ULN for INR
        
        # Create composite score
        result_df['liver_dysfunction_score'] = bili_norm + inr_norm
        print("  - Added liver dysfunction composite score")
        features_added += 1
    
    # Glucose variability (if available)
    if all(col in result_df.columns for col in ['glucose_max', 'glucose_min']):
        result_df['glucose_variability'] = result_df['glucose_max'] - result_df['glucose_min']
        print("  - Added glucose variability")
        features_added += 1
    
    # Add sepsis indicators based on SOFA components if available
    sepsis_indicators = []
    
    # Respiratory: P/F ratio < 300
    if 'pf_ratio' in result_df.columns:
        result_df['resp_dysfunction'] = (result_df['pf_ratio'] < 300).astype(int)
        sepsis_indicators.append('resp_dysfunction')
        
    # Cardiovascular: MAP < 70
    if 'map_min' in result_df.columns:
        result_df['cv_dysfunction'] = (result_df['map_min'] < 70).astype(int)
        sepsis_indicators.append('cv_dysfunction')
        
    # Renal: Creatinine > 1.2
    if 'creatinine_max' in result_df.columns:
        result_df['renal_dysfunction'] = (result_df['creatinine_max'] > 1.2).astype(int)
        sepsis_indicators.append('renal_dysfunction')
        
    # Hepatic: Bilirubin > 1.2
    if 'bilirubin_max' in result_df.columns:
        result_df['hepatic_dysfunction'] = (result_df['bilirubin_max'] > 1.2).astype(int)
        sepsis_indicators.append('hepatic_dysfunction')
        
    # Coagulation: Platelets < 150
    if 'platelets_min' in result_df.columns:
        result_df['coag_dysfunction'] = (result_df['platelets_min'] < 150).astype(int)
        sepsis_indicators.append('coag_dysfunction')
    
    # Create organ dysfunction count
    if sepsis_indicators:
        result_df['organ_dysfunction_count'] = result_df[sepsis_indicators].sum(axis=1)
        print(f"  - Added {len(sepsis_indicators)} organ dysfunction indicators and count")
        features_added += len(sepsis_indicators) + 1
    
    print(f"Total of {features_added} advanced clinical features added")
    return result_df


def identify_critical_values(df):
    """
    Identify and flag critical clinical values indicating severe illness
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with clinical measurements
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added critical value flags
    """
    print("Identifying critical clinical values...")
    
    result_df = df.copy()
    critical_flags_added = 0
    
    # Define critical thresholds with clinical significance
    critical_thresholds = {
        # Metabolic/Electrolyte Abnormalities
        'lactate_max': (4.0, '>'),           # Severe tissue hypoperfusion
        'potassium_max': (6.0, '>'),         # Critical hyperkalemia
        'potassium_min': (2.8, '<'),         # Critical hypokalemia
        'sodium_max': (155, '>'),            # Severe hypernatremia
        'sodium_min': (125, '<'),            # Severe hyponatremia
        'bicarbonate_min': (15, '<'),        # Severe metabolic acidosis
        'glucose_max': (400, '>'),           # Severe hyperglycemia
        'glucose_min': (50, '<'),            # Severe hypoglycemia
        'calcium_min': (7.0, '<'),           # Severe hypocalcemia
        'phosphate_max': (7.0, '>'),         # Severe hyperphosphatemia
        
        # Hematologic Abnormalities
        'wbc_max': (25, '>'),                # Severe leukocytosis
        'wbc_min': (2, '<'),                 # Severe leukopenia
        'platelets_min': (50, '<'),          # Critical thrombocytopenia
        'hemoglobin_min': (7, '<'),          # Severe anemia
        'inr_max': (2.0, '>'),               # Significant coagulopathy
        
        # Vital Signs
        'heart_rate_max': (140, '>'),        # Severe tachycardia
        'sbp_min': (80, '<'),                # Severe hypotension
        'resp_rate_max': (30, '>'),          # Severe tachypnea
        'temp_max': (39.5, '>'),             # High fever
        'temp_min': (35.0, '<'),             # Hypothermia
        'spo2_min': (88, '<'),               # Severe hypoxemia
        
        # Organ Dysfunction
        'bilirubin_max': (4.0, '>'),         # Severe liver dysfunction
        'creatinine_max': (3.0, '>'),        # Severe kidney injury
        'bun_max': (50, '>'),                # Severe azotemia
        'troponin_max': (0.5, '>'),          # Significant cardiac injury
        'ph_min': (7.25, '<'),               # Severe acidosis
        'ph_max': (7.55, '>'),               # Severe alkalosis

        'fibrinogen_min': (100, '<'),         # Critical hypofibrinogenemia
        'albumin_min': (2.0, '<'),            # Severe hypoalbuminemia
        'amylase_max': (500, '>'),            # Severe pancreatitis
        'lipase_max': (1000, '>'),            # Severe pancreatitis
        'co2_min': (15, '<'),                 # Severe metabolic acidosis
        'pco2_max': (60, '>'),                # Severe respiratory acidosis
        'pco2_min': (25, '<'),                # Severe respiratory alkalosis
        'gcs_min': (9, '<'),                  # Severe neurological dysfunction
        'neutrophil_min': (1.0, '<'),         # Severe neutropenia
        'procalcitonin_max': (10.0, '>'),     # Severe bacterial infection
        'alt_max': (1000, '>'),               # Severe hepatocellular injury
        'ast_max': (1000, '>'),               # Severe hepatocellular injury
        'magnesium_min': (1.2, '<'),          # Severe hypomagnesemia
        'ddimer_max': (5000, '>'),            # Severe coagulopathy
        'ferritin_max': (2000, '>'),          # Severe inflammation/infection
        'stroke_volume_min': (50, '<'),       # Critical decrease in cardiac output
        'cardiac_index_min': (2.0, '<'),      # Critical decrease in cardiac output
        'base_excess_min': (-8, '<'),         # Severe metabolic acidosis
        'pao2_min': (60, '<'),                # Severe hypoxemia
        'fio2_max': (0.5, '>'),               # High oxygen requirement
    }
    
    # Create flags for critical values
    for feature, (threshold, direction) in critical_thresholds.items():
        if feature in result_df.columns:
            if direction == '<':
                result_df[f'{feature}_critical'] = (result_df[feature] < threshold).astype(int)
            else:  # '>'
                result_df[f'{feature}_critical'] = (result_df[feature] > threshold).astype(int)
            
            critical_flags_added += 1
            
    # Count total critical values per patient
    critical_cols = [col for col in result_df.columns if col.endswith('_critical')]
    if critical_cols:
        result_df['critical_value_count'] = result_df[critical_cols].sum(axis=1)
        critical_patients = (result_df['critical_value_count'] > 0).sum()
        print(f"  - Identified {len(critical_cols)} types of critical values")
        print(f"  - {critical_patients} patients ({critical_patients/len(result_df)*100:.1f}%) have at least one critical value")
        
        # Create severe illness flag (multiple critical values)
        result_df['multiple_critical_values'] = (result_df['critical_value_count'] >= 3).astype(int)
        print(f"  - {result_df['multiple_critical_values'].sum()} patients have 3+ critical values")
        
        critical_flags_added += 2  # Count for critical_value_count and multiple_critical_values
    
    print(f"Added {critical_flags_added} critical value indicators")
    return result_df