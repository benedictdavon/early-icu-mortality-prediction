import pandas as pd
import numpy as np
import os
import re
from datetime import timedelta
import multiprocessing as mp
from functools import partial
from tqdm import tqdm  # For progress bars

from config import hosp_path, icu_path, label_path


def setup_directories():
    """Create output directory if it doesn't exist."""
    output_dir = os.path.join(os.path.dirname(icu_path), 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def load_cohort(cohort_path):
    """Load and preprocess the cohort data."""
    cohort = pd.read_csv(cohort_path)
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    cohort['earliest_record'] = pd.to_datetime(cohort['earliest_record'])
    
    # Define observation window endpoints
    cohort['window_end'] = cohort['intime'] + pd.Timedelta(hours=6)
    cohort['window_end_bmi'] = cohort['intime'] + pd.Timedelta(hours=24)  # 24 hour window for BMI
    
    print(f"Loaded cohort with {len(cohort)} patients")
    return cohort


def extract_demographics(cohort, hosp_path):
    """Extract demographic features including age and gender."""
    print("Extracting demographic features...")
    
    # Load patients table
    patients = pd.read_csv(os.path.join(hosp_path, '_patients.csv'), 
                          usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])

    # Load admissions for timestamp data
    admissions = pd.read_csv(os.path.join(hosp_path, '_admissions.csv'),
                           usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['admittime_year'] = admissions['admittime'].dt.year
    
    # Convert gender to numeric (1 for M, 0 for F)
    patients['gender_numeric'] = patients['gender'].map({'M': 1, 'F': 0})
    
    # Merge patients with cohort
    features = cohort.merge(patients, on='subject_id', how='left')
    
    # Merge with admissions to get admission time
    features = features.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'admittime_year']], 
        on=['subject_id', 'hadm_id'], 
        how='left'
    )
    
    # Calculate age at admission time
    features['age'] = features['anchor_age'] + (features['admittime_year'] - features['anchor_year'])
    
    # Cap ages above 89 at 90 (for de-identification)
    features.loc[features['age'] > 89, 'age'] = 90
    
    # Keep track of feature extraction progress
    print(f"Extracted demographic features for {features['subject_id'].nunique()} unique subjects")
    
    return features

def calculate_bmi(cohort, chart_data):
    """Calculate BMI from height and weight measurements with improved coverage."""
    # Extract unique stay_ids for faster filtering
    stay_ids = set(cohort['stay_id'])
    
    # Expanded set of height and weight itemids
    height_ids = [226730, 226707, 1394, 216235, 3486, 3485]  # Added more height itemids
    weight_ids = [226512, 226531, 224639, 763, 3580, 3693, 3581]  # Added more weight itemids
    
    # Create masks for faster filtering
    ht_mask = chart_data['itemid'].isin(height_ids) & chart_data['stay_id'].isin(stay_ids)
    wt_mask = chart_data['itemid'].isin(weight_ids) & chart_data['stay_id'].isin(stay_ids)
    
    # Filter the data once
    height_data = chart_data[ht_mask].copy()
    weight_data = chart_data[wt_mask].copy()
    
    # Create dictionaries for O(1) lookup time
    cohort_dict = cohort.set_index('stay_id')[['intime', 'window_end_bmi']].to_dict('index')
    
    # Expanded time window for better coverage
    window_hours = 48  # Extended from 24 to 48 hours
    
    # Extract heights with vectorized operations where possible
    height_data = height_data.sort_values('charttime')
    height_valid = []
    
    for stay_id, group in height_data.groupby('stay_id'):
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]['intime']
            window_end = intime + pd.Timedelta(hours=window_hours)
            
            # Filter by expanded time window
            valid_heights = group[(group['charttime'] >= intime) & 
                                 (group['charttime'] <= window_end)]
            
            if not valid_heights.empty:
                first_ht = valid_heights.iloc[0]
                # Convert different height units to cm
                if first_ht['itemid'] in [226730, 1394, 3486]:  # inches
                    height_cm = first_ht['valuenum'] * 2.54
                else:
                    height_cm = first_ht['valuenum']
                    
                height_valid.append((stay_id, height_cm))
    
    heights_df = pd.DataFrame(height_valid, columns=['stay_id', 'height_cm'])
    heights_df = heights_df[(heights_df['height_cm'] >= 120) & (heights_df['height_cm'] <= 220)]  # Valid range
    
    # Extract weights similarly with expanded window
    weight_data = weight_data.sort_values('charttime')
    weight_valid = []
    
    for stay_id, group in weight_data.groupby('stay_id'):
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]['intime']
            window_end = intime + pd.Timedelta(hours=window_hours)
            
            valid_weights = group[(group['charttime'] >= intime) & 
                                 (group['charttime'] <= window_end)]
            
            if not valid_weights.empty:
                first_wt = valid_weights.iloc[0]
                # Convert to kg if needed
                weight_kg = first_wt['valuenum'] * 0.453592 if first_wt['itemid'] in [226531, 763, 3693] else first_wt['valuenum']
                
                weight_valid.append((stay_id, weight_kg))
    
    weights_df = pd.DataFrame(weight_valid, columns=['stay_id', 'weight_kg'])
    weights_df = weights_df[(weights_df['weight_kg'] >= 30) & (weights_df['weight_kg'] <= 300)]  # Valid range
    
    # Calculate BMI
    bmi_df = heights_df.merge(weights_df, on='stay_id', how='inner')
    bmi_df['height_m'] = bmi_df['height_cm'] / 100
    bmi_df['bmi'] = bmi_df['weight_kg'] / (bmi_df['height_m'] ** 2)
    
    # Add missingness indicator for all stays
    result_df = pd.DataFrame({'stay_id': list(stay_ids)})
    result_df = result_df.merge(bmi_df[['stay_id', 'bmi']], on='stay_id', how='left')
    result_df['bmi_measured'] = result_df['bmi'].notna().astype(int)
    
    print(f"Calculated BMI for {len(bmi_df)} patients ({len(bmi_df)/len(stay_ids)*100:.1f}% of cohort)")
    return result_df[['stay_id', 'bmi', 'bmi_measured']]


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

def add_clinical_derived_features(features_df):
    """Add clinically meaningful derived features based on raw measurements."""
    print("Adding clinically meaningful derived features...")
    df = features_df.copy()
    
    # SIRS criteria (Systemic Inflammatory Response Syndrome)
    if 'heart_rate_mean' in df.columns:
        df['has_tachycardia'] = (df['heart_rate_mean'] > 90).astype(int)
    
    if 'resp_rate_mean' in df.columns:
        df['has_tachypnea'] = (df['resp_rate_mean'] > 20).astype(int)
    
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
    
    # Renal dysfunction marker
    if all(col in df.columns for col in ['creatinine_mean', 'bun_mean']):
        df['bun_creatinine_ratio'] = df['bun_mean'] / df['creatinine_mean']
        # Cap extreme values
        df.loc[df['bun_creatinine_ratio'] > 30, 'bun_creatinine_ratio'] = 30
    
    # Oxygenation marker
    if 'spo2_mean' in df.columns:
        df['has_hypoxemia'] = (df['spo2_mean'] < 92).astype(int)
    
    print(f"Added {len(df.columns) - len(features_df.columns)} new clinical features")
    return df

def time_weighted_vital_avg(values, times):
    """Calculate time-weighted average for vital sign measurements."""
    if len(values) <= 1:
        return values[0] if len(values) == 1 else np.nan
        
    # Sort by time
    sorted_data = sorted(zip(times, values), key=lambda x: x[0])
    times_sorted = [t for t, v in sorted_data]
    values_sorted = [v for t, v in sorted_data]
    
    # Calculate time differences between measurements
    # Handle numpy.timedelta64 objects correctly
    time_diffs = []
    for i in range(len(times_sorted)-1):
        delta = times_sorted[i+1] - times_sorted[i]
        # Convert to hours, handling numpy.timedelta64
        hours = delta / np.timedelta64(1, 'h')
        time_diffs.append(hours)
    
    total_time = sum(time_diffs)
    
    # If all measurements at the same time, use regular average
    if total_time == 0:
        return np.mean(values_sorted)
    
    # Calculate weighted average
    weighted_sum = sum(values_sorted[i] * time_diffs[i] for i in range(len(time_diffs)))
    weighted_avg = weighted_sum / total_time
    
    return weighted_avg


def process_vital_batch(vital_name, itemids, vital_data, patient_batch, cohort_dict):
    """Process a batch of patients for one vital sign with time-weighted averaging."""
    # Initialize storage for aggregated values
    result_data = {
        f'{vital_name}_mean': [],      # Will use time-weighted average
        f'{vital_name}_min': [],
        f'{vital_name}_max': [],
        f'{vital_name}_measured': [],  # Added missingness indicator
        'stay_id': []
    }
    
    # Filter for specific vital
    this_vital_data = vital_data[vital_data['itemid'].isin(itemids)]
    
    for stay_id in patient_batch:
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]['intime']
            window_end = cohort_dict[stay_id]['window_end']
            
            # Get vital measurements within time window
            pt_vitals = this_vital_data[
                (this_vital_data['stay_id'] == stay_id) & 
                (this_vital_data['charttime'] >= intime) & 
                (this_vital_data['charttime'] <= window_end)
            ]
            
            if len(pt_vitals) > 0:
                values = pt_vitals['valuenum'].values
                times = pt_vitals['charttime'].values
                
                # Calculate time-weighted average if multiple measurements
                if len(values) > 1:
                    tw_mean = time_weighted_vital_avg(values, times)
                else:
                    tw_mean = values[0]
                
                result_data[f'{vital_name}_mean'].append(tw_mean)
                result_data[f'{vital_name}_min'].append(min(values))
                result_data[f'{vital_name}_max'].append(max(values))
                result_data[f'{vital_name}_measured'].append(1)
            else:
                result_data[f'{vital_name}_mean'].append(np.nan)
                result_data[f'{vital_name}_min'].append(np.nan)
                result_data[f'{vital_name}_max'].append(np.nan)
                result_data[f'{vital_name}_measured'].append(0)
                
            result_data['stay_id'].append(stay_id)
    
    return pd.DataFrame(result_data)

def extract_vital_signs_parallel(cohort, chart_data):
    """Extract vital sign features using parallel processing."""
    # Define vital sign item IDs
    vital_items = {
        'heart_rate': [220045],   # Heart Rate
        'resp_rate': [220210],    # Respiratory Rate
        'map': [220052],          # Mean Arterial Pressure
        'temp': [223761, 223762], # Temperature in C and F
        'sbp': [220179],          # Systolic BP
        'dbp': [220180],          # Diastolic BP
        'spo2': [220277]          # SpO2
    }
    
    # Create dictionary for lookup
    cohort_dict = cohort.set_index('stay_id')[['intime', 'window_end']].to_dict('index')
    stay_ids = list(cohort['stay_id'])
    
    # Filter chart data to only include vitals and cohort patients
    all_vital_ids = [id for ids in vital_items.values() for id in ids]
    vital_data = chart_data[chart_data['itemid'].isin(all_vital_ids) & 
                            chart_data['stay_id'].isin(stay_ids)]
    print(f"Filtered to {len(vital_data)} vital sign records")
    
    # Split the patients into batches for parallel processing
    num_cores = mp.cpu_count()
    batch_size = max(1, len(stay_ids) // num_cores)
    patient_batches = [stay_ids[i:i+batch_size] for i in range(0, len(stay_ids), batch_size)]
    
    # Process each vital sign in parallel
    all_vital_dfs = []
    
    with mp.Pool(num_cores) as pool:
        for vital_name, itemids in vital_items.items():
            print(f"Processing {vital_name}...")
            vital_func = partial(process_vital_batch, vital_name, itemids, vital_data, cohort_dict=cohort_dict)
            vital_results = pool.map(vital_func, patient_batches)
            
            # Combine results
            combined_vital_df = pd.concat(vital_results, ignore_index=True)
            all_vital_dfs.append(combined_vital_df)
    
    # Merge all vital sign dataframes
    merged_df = all_vital_dfs[0]
    for df in all_vital_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='stay_id', how='outer')
    
    print(f"Extracted vital sign features for {len(merged_df)} patients")
    return merged_df


def process_lab_batch(lab_name, itemids, lab_data, patient_batch, cohort_dict):
    """Process a batch of patients for one lab test with missingness indicators and focused aggregates."""
    # Initialize storage for more focused aggregated values
    result_data = {
        f'{lab_name}_mean': [],
        f'{lab_name}_max': [], 
        f'{lab_name}_delta': [],  # Reduced from 7 to 3 key aggregates
        f'{lab_name}_measured': [],  # Added missingness indicator
        'subject_id': []
    }
    
    # Filter for specific lab
    this_lab_data = lab_data[lab_data['itemid'].isin(itemids)]
    
    for subject_id in patient_batch:
        if subject_id in cohort_dict:
            intime = cohort_dict[subject_id]['intime']
            window_end = cohort_dict[subject_id]['window_end']
            
            # Get lab measurements within time window
            pt_labs = this_lab_data[
                (this_lab_data['subject_id'] == subject_id) & 
                (this_lab_data['charttime'] >= intime) & 
                (this_lab_data['charttime'] <= window_end)
            ]
            
            if len(pt_labs) > 0:
                values = pt_labs['valuenum']
                
                # Sort by time for delta calculation
                sorted_labs = pt_labs.sort_values('charttime')
                first_val = sorted_labs.iloc[0]['valuenum']
                last_val = sorted_labs.iloc[-1]['valuenum']
                
                result_data[f'{lab_name}_mean'].append(values.mean())
                result_data[f'{lab_name}_max'].append(values.max())
                result_data[f'{lab_name}_delta'].append(last_val - first_val)
                result_data[f'{lab_name}_measured'].append(1)  # Lab was measured
            else:
                result_data[f'{lab_name}_mean'].append(np.nan)
                result_data[f'{lab_name}_max'].append(np.nan)
                result_data[f'{lab_name}_delta'].append(np.nan)
                result_data[f'{lab_name}_measured'].append(0)  # Lab was not measured
            
            result_data['subject_id'].append(subject_id)
    
    return pd.DataFrame(result_data)


def extract_lab_results_parallel(cohort, hospital_path):
    """Extract laboratory results using parallel processing."""
    # Load labs - this is a large file, so we'll use efficient loading
    print("Loading lab events data...")
    labs = pd.read_csv(os.path.join(hospital_path, '_labevents.csv'),
                      usecols=['subject_id', 'charttime', 'itemid', 'valuenum'])
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    
    # Define lab test item IDs
    lab_items = {
        'bun': [51006],                   # Blood Urea Nitrogen
        'alkaline_phosphatase': [50863],  # Alkaline Phosphatase
        'bilirubin': [50885],             # Total Bilirubin
        'creatinine': [50912],            # Creatinine
        'glucose': [50931],               # Glucose
        'platelets': [51265],             # Platelet Count
        'hemoglobin': [51222],            # Hemoglobin
        'wbc': [51301],                   # White Blood Cell Count
        'sodium': [50983],                # Sodium
        'potassium': [50971],             # Potassium
        'lactate': [50813],               # Lactate
        'hematocrit': [51221],            # Hematocrit
        'chloride': [50902],              # Chloride
        'bicarbonate': [50882],           # Bicarbonate
        'anion_gap': [50868]              # Anion Gap
    }
    
    # Create lookup dictionaries
    subject_ids = list(cohort['subject_id'].unique())
    cohort_dict = cohort.set_index('subject_id')[['intime', 'window_end']].to_dict('index')
    
    # Pre-filter lab data to only include relevant lab tests and cohort patients
    all_lab_ids = [id for ids in lab_items.values() for id in ids]
    lab_data = labs[labs['itemid'].isin(all_lab_ids) & labs['subject_id'].isin(subject_ids)]
    print(f"Filtered to {len(lab_data)} lab records")
    
    # Split patients into batches for parallel processing
    num_cores = mp.cpu_count()
    batch_size = max(1, len(subject_ids) // num_cores)
    patient_batches = [subject_ids[i:i+batch_size] for i in range(0, len(subject_ids), batch_size)]
    
    # Process each lab test in parallel
    all_lab_dfs = []
    
    print("Extracting lab features in parallel...")
    with mp.Pool(num_cores) as pool:
        for lab_name, itemids in lab_items.items():
            print(f"Processing {lab_name}...")
            lab_func = partial(process_lab_batch, lab_name, itemids, lab_data, cohort_dict=cohort_dict)
            lab_results = list(tqdm(pool.imap(lab_func, patient_batches), total=len(patient_batches)))
            
            # Combine results
            combined_lab_df = pd.concat(lab_results, ignore_index=True)
            all_lab_dfs.append(combined_lab_df)
    
    # Merge all lab dataframes
    merged_df = all_lab_dfs[0]
    for df in all_lab_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject_id', how='outer')
    
    # Join with stay_id for consistency with other features
    merged_df = merged_df.merge(cohort[['subject_id', 'stay_id']], on='subject_id', how='left')
    
    print(f"Extracted laboratory result features for {len(merged_df)} patients")
    return merged_df

def extract_prior_diagnoses(features, hosp_path):
    """Extract information about prior diagnoses with ICD chapter categorization."""
    # Load admissions and diagnoses tables
    admissions = pd.read_csv(os.path.join(hosp_path, '_admissions.csv'),
                            usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])

    diagnoses = pd.read_csv(os.path.join(hosp_path, '_diagnoses_icd.csv'),
                           usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
    
    # Create lookup dictionary for ICU admission times
    icu_admit_dict = features.set_index('subject_id')['intime'].to_dict()
    
    # ICD code categorization functions
    def map_icd9_to_chapter(icd_code_str):
        """Map ICD-9 code to clinical chapter"""
        if pd.isna(icd_code_str): return 'unknown'
        icd_code_str = str(icd_code_str).upper()
        
        # Handle V and E codes
        if any(prefix in icd_code_str for prefix in ['V', 'E']): 
            return 'other_icd9'
            
        # Process numeric codes
        try:
            numeric_part = int(re.match(r"([0-9]+)", icd_code_str).group(1))
            if 1 <= numeric_part <= 139: return 'infectious_parasitic'
            if 140 <= numeric_part <= 239: return 'neoplasms'
            if 240 <= numeric_part <= 279: return 'endocrine_metabolic'
            if 280 <= numeric_part <= 289: return 'blood_disorders'
            if 290 <= numeric_part <= 319: return 'mental_disorders'
            if 320 <= numeric_part <= 389: return 'nervous_sensory'
            if 390 <= numeric_part <= 459: return 'circulatory'
            if 460 <= numeric_part <= 519: return 'respiratory'
            if 520 <= numeric_part <= 579: return 'digestive'
            if 580 <= numeric_part <= 629: return 'genitourinary'
            if 630 <= numeric_part <= 679: return 'pregnancy_childbirth'
            if 680 <= numeric_part <= 709: return 'skin_subcutaneous'
            if 710 <= numeric_part <= 739: return 'musculoskeletal_connective'
            if 740 <= numeric_part <= 759: return 'congenital_anomalies'
            if 760 <= numeric_part <= 779: return 'perinatal_conditions'
            if 780 <= numeric_part <= 799: return 'symptoms_signs_illdefined'
            if 800 <= numeric_part <= 999: return 'injury_poisoning'
            return 'other_icd9'
        except:
            return 'other_icd9'

    def map_icd10_to_chapter(icd_code_str):
        """Map ICD-10 code to clinical chapter"""
        if pd.isna(icd_code_str): return 'unknown'
        icd_code_str = str(icd_code_str).upper()
        
        if 'A' <= icd_code_str[0] <= 'B': return 'infectious_parasitic'
        if icd_code_str[0] == 'C' or (icd_code_str[0] == 'D' and '0' <= icd_code_str[1] <= '4'): return 'neoplasms'
        if icd_code_str[0] == 'D' and '5' <= icd_code_str[1] <= '9': return 'blood_disorders'
        if icd_code_str[0] == 'E': return 'endocrine_metabolic'
        if icd_code_str[0] == 'F': return 'mental_disorders'
        if icd_code_str[0] == 'G': return 'nervous_sensory'
        if icd_code_str[0] == 'H' and '0' <= icd_code_str[1] <= '5': return 'nervous_sensory' # eye
        if icd_code_str[0] == 'H' and '6' <= icd_code_str[1] <= '9': return 'nervous_sensory' # ear
        if icd_code_str[0] == 'I': return 'circulatory'
        if icd_code_str[0] == 'J': return 'respiratory'
        if icd_code_str[0] == 'K': return 'digestive'
        if icd_code_str[0] == 'L': return 'skin_subcutaneous'
        if icd_code_str[0] == 'M': return 'musculoskeletal_connective'
        if icd_code_str[0] == 'N': return 'genitourinary'
        if icd_code_str[0] == 'O': return 'pregnancy_childbirth'
        if icd_code_str[0] == 'P': return 'perinatal_conditions'
        if icd_code_str[0] == 'Q': return 'congenital_anomalies'
        if icd_code_str[0] == 'R': return 'symptoms_signs_illdefined'
        if 'S' <= icd_code_str[0] <= 'T': return 'injury_poisoning'
        if 'V' <= icd_code_str[0] <= 'Y': return 'external_causes'
        if icd_code_str[0] == 'Z': return 'factors_health_status'
        return 'other_icd10'
    
    # Create a dataframe with unique subject IDs
    subject_ids = features['subject_id'].unique()
    prior_dx_df = pd.DataFrame({'subject_id': subject_ids})
    prior_dx_df['has_prior_diagnoses'] = 0  # Default value
    
    # Initialize columns for ICD chapters
    icd_chapters = [
        'infectious_parasitic', 'neoplasms', 'endocrine_metabolic', 'blood_disorders',
        'mental_disorders', 'nervous_sensory', 'circulatory', 'respiratory',
        'digestive', 'genitourinary', 'pregnancy_childbirth', 'skin_subcutaneous',
        'musculoskeletal_connective', 'congenital_anomalies', 'perinatal_conditions',
        'symptoms_signs_illdefined', 'injury_poisoning', 'external_causes',
        'factors_health_status', 'other_icd9', 'other_icd10', 'unknown'
    ]
    
    for chapter in icd_chapters:
        prior_dx_df[f'prev_dx_{chapter}_count'] = 0
    
    # Store all prior diagnoses
    all_prior_dx = []
    
    # Process each patient
    for subject_id in tqdm(subject_ids, desc="Processing prior diagnoses"):
        # Get all admissions for this patient
        pt_admissions = admissions[admissions['subject_id'] == subject_id]
        
        if not pt_admissions.empty and subject_id in icu_admit_dict:
            # Get the ICU admission time for this patient
            icu_admit = icu_admit_dict[subject_id]
            
            # Check for prior admissions
            prior_admits = pt_admissions[pt_admissions['admittime'] < icu_admit]
            
            if len(prior_admits) > 0:
                # Check if prior admissions have diagnoses
                prior_hadm_ids = prior_admits['hadm_id'].tolist()
                prior_diagnoses = diagnoses[diagnoses['hadm_id'].isin(prior_hadm_ids)].copy()
                
                if not prior_diagnoses.empty:
                    # Update has_prior_diagnoses value for this subject
                    prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'has_prior_diagnoses'] = 1
                    
                    # Categorize diagnoses by ICD chapters
                    prior_diagnoses['icd_chapter'] = prior_diagnoses.apply(
                        lambda r: map_icd9_to_chapter(r['icd_code']) if r['icd_version'] == 9
                        else map_icd10_to_chapter(r['icd_code']) if r['icd_version'] == 10
                        else 'unknown', axis=1
                    )
                    
                    # Count diagnoses by chapter
                    chapter_counts = prior_diagnoses['icd_chapter'].value_counts().to_dict()
                    
                    # Update counts in the dataframe
                    for chapter, count in chapter_counts.items():
                        col_name = f'prev_dx_{chapter}_count'
                        if col_name in prior_dx_df.columns:
                            prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, col_name] = count
                    
                    # Add total count
                    prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'prev_dx_count_total'] = len(prior_diagnoses)
                else:
                    prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'prev_dx_count_total'] = 0
            else:
                prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'prev_dx_count_total'] = 0
        else:
            prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'prev_dx_count_total'] = 0
    
    print(f"Extracted prior diagnosis information for {len(prior_dx_df)} patients")
    return prior_dx_df

def generate_table_one(features, output_dir, group_col=None):
    """
    Generate a Table One summarizing patient characteristics.
    
    Args:
        features: DataFrame with extracted features
        output_dir: Directory to save the output
        group_col: Optional column name to stratify by (e.g., 'mortality')
    
    Returns:
        DataFrame with Table One statistics
    """
    from scipy import stats
    import numpy as np
    
    print("Generating Table One...")
    
    # Determine columns to include (exclude administrative IDs)
    cols_for_table = [col for col in features.columns if not any(x in col.lower() for x in 
                     ['id', 'time', 'date', 'window', 'earliest_record'])]
    
    # Basic structure for results
    table_one_rows = []
    n_total = len(features)
    
    # Add header row
    if group_col and group_col in features.columns:
        group0 = features[features[group_col] == 0]
        group1 = features[features[group_col] == 1]
        n_group0 = len(group0)
        n_group1 = len(group1)
        
        table_one_rows.append(['Characteristic', f'Overall (N={n_total})',
                              f'Group 0 (N={n_group0})', f'Group 1 (N={n_group1})', 'p-value'])
    else:
        table_one_rows.append(['Characteristic', f'Overall (N={n_total})'])
    
    # Process each feature
    for col in cols_for_table:
        if features[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            # Handle numeric features
            is_binary = (features[col].min() == 0 and features[col].max() == 1 and features[col].nunique() <= 2)
            
            if features[col].nunique() > 2 and not is_binary:
                # Continuous variable
                mean_overall = features[col].mean()
                std_overall = features[col].std()
                
                if group_col and group_col in features.columns:
                    mean_g0 = group0[col].mean()
                    std_g0 = group0[col].std()
                    mean_g1 = group1[col].mean()
                    std_g1 = group1[col].std()
                    
                    # Calculate p-value
                    stat, pval = stats.ttest_ind(
                        group0[col].dropna(), 
                        group1[col].dropna(),
                        equal_var=False, 
                        nan_policy='omit'
                    )
                    
                    table_one_rows.append([
                        col, 
                        f"{mean_overall:.2f} ± {std_overall:.2f}",
                        f"{mean_g0:.2f} ± {std_g0:.2f}",
                        f"{mean_g1:.2f} ± {std_g1:.2f}", 
                        f"{pval:.3g}" if not pd.isna(pval) else "N/A"
                    ])
                else:
                    table_one_rows.append([
                        col, 
                        f"{mean_overall:.2f} ± {std_overall:.2f}"
                    ])
            else:
                # Binary variable
                count_overall = features[col].sum()
                pct_overall = count_overall / n_total * 100
                
                if group_col and group_col in features.columns:
                    count_g0 = group0[col].sum()
                    pct_g0 = count_g0 / n_group0 * 100
                    count_g1 = group1[col].sum()
                    pct_g1 = count_g1 / n_group1 * 100
                    
                    # Chi-square test
                    try:
                        contingency = pd.crosstab(features[col], features[group_col])
                        chi2, pval, _, _ = stats.chi2_contingency(contingency)
                        
                        table_one_rows.append([
                            f"{col} (N, %)", 
                            f"{count_overall} ({pct_overall:.1f}%)",
                            f"{count_g0} ({pct_g0:.1f}%)",
                            f"{count_g1} ({pct_g1:.1f}%)",
                            f"{pval:.3g}" if not pd.isna(pval) else "N/A"
                        ])
                    except:
                        table_one_rows.append([
                            f"{col} (N, %)", 
                            f"{count_overall} ({pct_overall:.1f}%)",
                            f"{count_g0} ({pct_g0:.1f}%)",
                            f"{count_g1} ({pct_g1:.1f}%)",
                            "N/A"
                        ])
                else:
                    table_one_rows.append([
                        f"{col} (N, %)", 
                        f"{count_overall} ({pct_overall:.1f}%)"
                    ])
    
    # Convert to DataFrame
    table_one_df = pd.DataFrame(table_one_rows[1:], columns=table_one_rows[0])
    
    # Save to file
    output_path = os.path.join(output_dir, 'table_one.csv')
    table_one_df.to_csv(output_path, index=False)
    print(f"Table One saved to {output_path}")
    
    return table_one_df

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
                print(f"{temp_col}: capped {outliers_before} values outside range 35-42°C")
    
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
        'anion_gap_mean': (1, 40)       # mEq/L
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
    
    print("Clinical measurement validation complete")
    return df

def save_features(features, output_dir):
    """Save features and generate descriptive statistics."""
    # Clean up and save
    feature_cols = [col for col in features.columns if col not in 
                   ['intime', 'outtime', 'earliest_record', 'window_end', 'window_end_bmi']]
    final_features = features[feature_cols]

    # Save features to CSV
    features_path = os.path.join(output_dir, 'extracted_features.csv')
    final_features.to_csv(features_path, index=False)
    print(f"Saved {len(final_features)} patient records with {len(feature_cols)} features to {features_path}")

    # Basic descriptive analysis for Table 1
    stats = final_features.describe().T
    stats['missing_pct'] = 100 * final_features.isnull().sum() / len(final_features)
    stats_path = os.path.join(output_dir, 'feature_statistics.csv')
    stats.to_csv(stats_path, index=True)
    print(f"Saved descriptive statistics to {stats_path}")
    
    return final_features, stats


def main():
    # Setup
    output_dir = setup_directories()
    
    # Load cohort
    cohort_path = os.path.join(output_dir, 'final_cohort.csv')
    cohort = load_cohort(cohort_path)
    
    # Extract features
    features = extract_demographics(cohort, hosp_path)
    
    # Store subject_id to stay_id mapping for later use
    subject_stay_mapping = cohort[['subject_id', 'stay_id']].copy()
    
    # Load chartevents once for both BMI and vitals
    print("Loading chart events data (used for both BMI and vitals)...")
    chart_data = pd.read_csv(os.path.join(icu_path, '_chartevents.csv'),
                             usecols=['stay_id', 'charttime', 'itemid', 'valuenum'])
    chart_data['charttime'] = pd.to_datetime(chart_data['charttime'])
    
    # Calculate BMI using the improved function with expanded IDs and window
    bmi_df = calculate_bmi(cohort, chart_data)
    features = features.merge(bmi_df, on='stay_id', how='left')
    
    # Extract vital signs using time-weighted averaging
    vital_df = extract_vital_signs_parallel(cohort, chart_data)
    features = features.merge(vital_df, on='stay_id', how='left')
    
    # Free up memory
    del chart_data
    import gc
    gc.collect()
    
    # Extract lab results with reduced aggregates and missingness indicators
    lab_df = extract_lab_results_parallel(cohort, hosp_path)
    features = features.merge(lab_df, on='stay_id', how='left')
    
    # Make sure subject_id is still in features
    if 'subject_id' not in features.columns:
        print("Restoring subject_id column which was lost during merges")
        features = features.merge(subject_stay_mapping, on='stay_id', how='left')
    
    # Extract prior diagnoses
    prior_dx_df = extract_prior_diagnoses(features, hosp_path)
    features = features.merge(prior_dx_df, on='subject_id', how='left')
 
    # Clean and validate clinical measurements
    features = clean_clinical_measurements(features)
    
    # Add log transformations for skewed variables
    features = add_log_transformations(features)
    
    # Add clinically meaningful derived features
    features = add_clinical_derived_features(features)
    
    # Save features and generate statistics
    final_features, stats = save_features(features, output_dir)
    
    # Generate Table One if outcome data is available
    if 'mortality' in final_features.columns:
        table_one = generate_table_one(final_features, output_dir, group_col='mortality')
    
    print("Enhanced feature extraction completed successfully")
    return final_features


if __name__ == "__main__":
    main()