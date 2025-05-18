import pandas as pd
import numpy as np
import os
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
    # Load patients table
    patients = pd.read_csv(os.path.join(hosp_path, '_patients.csv'), 
                           usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])

    # Convert gender to numeric (1 for M, 0 for F)
    patients['gender_numeric'] = patients['gender'].map({'M': 1, 'F': 0})

    # Merge with cohort
    features = cohort.merge(patients, on='subject_id', how='left')
    print(f"Extracted demographic features")
    return features


def calculate_bmi(cohort, chart_data):
    """Calculate BMI from height and weight measurements within 24 hours.
       Optimized to use pre-loaded chart data."""
    # Extract unique stay_ids for faster filtering
    stay_ids = set(cohort['stay_id'])
    
    # Filter for height and weight records for cohort patients only
    height_ids = [226730, 226707]  # Height: inches, cm
    weight_ids = [226512, 226531, 224639]  # Weight: admit, daily, kg
    
    # Create masks for faster filtering
    ht_mask = chart_data['itemid'].isin(height_ids) & chart_data['stay_id'].isin(stay_ids)
    wt_mask = chart_data['itemid'].isin(weight_ids) & chart_data['stay_id'].isin(stay_ids)
    
    # Filter the data once
    height_data = chart_data[ht_mask].copy()
    weight_data = chart_data[wt_mask].copy()
    
    # Create dictionaries for O(1) lookup time
    cohort_dict = cohort.set_index('stay_id')[['intime', 'window_end_bmi']].to_dict('index')
    
    # Extract heights with vectorized operations where possible
    height_data = height_data.sort_values('charttime')  # Take earliest reading if multiple
    height_valid = []
    
    for stay_id, group in height_data.groupby('stay_id'):
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]['intime']
            window_end = cohort_dict[stay_id]['window_end_bmi']
            
            # Filter by time window
            valid_heights = group[(group['charttime'] >= intime) & 
                                  (group['charttime'] <= window_end)]
            
            if not valid_heights.empty:
                first_ht = valid_heights.iloc[0]
                # Convert to cm if needed
                height_cm = first_ht['valuenum'] * 2.54 if first_ht['itemid'] == 226730 else first_ht['valuenum']
                height_valid.append((stay_id, height_cm))
    
    heights_df = pd.DataFrame(height_valid, columns=['stay_id', 'height_cm'])
    
    # Extract weights similarly
    weight_data = weight_data.sort_values('charttime')
    weight_valid = []
    
    for stay_id, group in weight_data.groupby('stay_id'):
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]['intime']
            window_end = cohort_dict[stay_id]['window_end_bmi']
            
            # Filter by time window
            valid_weights = group[(group['charttime'] >= intime) & 
                                  (group['charttime'] <= window_end)]
            
            if not valid_weights.empty:
                weight_kg = valid_weights.iloc[0]['valuenum']
                weight_valid.append((stay_id, weight_kg))
    
    weights_df = pd.DataFrame(weight_valid, columns=['stay_id', 'weight_kg'])
    
    # Calculate BMI
    bmi_df = heights_df.merge(weights_df, on='stay_id', how='inner')
    bmi_df['height_m'] = bmi_df['height_cm'] / 100
    bmi_df['bmi'] = bmi_df['weight_kg'] / (bmi_df['height_m'] ** 2)
    
    print(f"Calculated BMI for {len(bmi_df)} patients")
    return bmi_df[['stay_id', 'bmi']]


def process_vital_batch(vital_name, itemids, vital_data, patient_batch, cohort_dict):
    """Process a batch of patients for one vital sign (for parallel processing)"""
    # Initialize storage for aggregated values
    result_data = {
        f'{vital_name}_mean': [],
        f'{vital_name}_min': [],
        f'{vital_name}_max': [],
        f'{vital_name}_std': [],
        f'{vital_name}_count': [],
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
            ]['valuenum']
            
            if len(pt_vitals) > 0:
                result_data[f'{vital_name}_mean'].append(pt_vitals.mean())
                result_data[f'{vital_name}_min'].append(pt_vitals.min())
                result_data[f'{vital_name}_max'].append(pt_vitals.max())
                result_data[f'{vital_name}_std'].append(pt_vitals.std() if len(pt_vitals) > 1 else 0)
                result_data[f'{vital_name}_count'].append(len(pt_vitals))
            else:
                result_data[f'{vital_name}_mean'].append(np.nan)
                result_data[f'{vital_name}_min'].append(np.nan)
                result_data[f'{vital_name}_max'].append(np.nan)
                result_data[f'{vital_name}_std'].append(np.nan)
                result_data[f'{vital_name}_count'].append(0)
                
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
    """Process a batch of patients for one lab test (for parallel processing)"""
    # Initialize storage for aggregated values
    result_data = {
        f'{lab_name}_mean': [],
        f'{lab_name}_min': [],
        f'{lab_name}_max': [],
        f'{lab_name}_std': [],
        f'{lab_name}_count': [],
        f'{lab_name}_first': [],
        f'{lab_name}_last': [],
        f'{lab_name}_delta': [],
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
                
                # Sort by time for first/last calculations
                sorted_labs = pt_labs.sort_values('charttime')
                first_val = sorted_labs.iloc[0]['valuenum']
                last_val = sorted_labs.iloc[-1]['valuenum']
                
                result_data[f'{lab_name}_mean'].append(values.mean())
                result_data[f'{lab_name}_min'].append(values.min())
                result_data[f'{lab_name}_max'].append(values.max())
                result_data[f'{lab_name}_std'].append(values.std() if len(values) > 1 else 0)
                result_data[f'{lab_name}_count'].append(len(values))
                result_data[f'{lab_name}_first'].append(first_val)
                result_data[f'{lab_name}_last'].append(last_val)
                result_data[f'{lab_name}_delta'].append(last_val - first_val)
            else:
                result_data[f'{lab_name}_mean'].append(np.nan)
                result_data[f'{lab_name}_min'].append(np.nan)
                result_data[f'{lab_name}_max'].append(np.nan)
                result_data[f'{lab_name}_std'].append(np.nan)
                result_data[f'{lab_name}_count'].append(0)
                result_data[f'{lab_name}_first'].append(np.nan)
                result_data[f'{lab_name}_last'].append(np.nan)
                result_data[f'{lab_name}_delta'].append(np.nan)
            
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
    """Extract information about prior diagnoses (optimized)."""
    # Load admissions and diagnoses tables
    admissions = pd.read_csv(os.path.join(hosp_path, '_admissions.csv'),
                            usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])

    diagnoses = pd.read_csv(os.path.join(hosp_path, '_diagnoses_icd.csv'),
                           usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
    
    # Create lookup dictionary for ICU admission times
    icu_admit_dict = features.set_index('subject_id')['intime'].to_dict()
    
    # Create a dataframe with unique subject IDs
    subject_ids = features['subject_id'].unique()
    prior_dx_df = pd.DataFrame({'subject_id': subject_ids})
    prior_dx_df['has_prior_diagnoses'] = 0  # Default value
    
    # Process each patient more efficiently
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
                prior_diagnoses = diagnoses[diagnoses['hadm_id'].isin(prior_hadm_ids)]
                
                if len(prior_diagnoses) > 0:
                    # Update has_prior_diagnoses value for this subject
                    prior_dx_df.loc[prior_dx_df['subject_id'] == subject_id, 'has_prior_diagnoses'] = 1
    
    print(f"Extracted prior diagnosis information")
    return prior_dx_df


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
    """Main function to orchestrate the feature extraction pipeline."""
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
    
    # Calculate BMI using the pre-loaded chart data
    bmi_df = calculate_bmi(cohort, chart_data)
    features = features.merge(bmi_df, on='stay_id', how='left')
    
    # Extract vital signs using parallel processing
    vital_df = extract_vital_signs_parallel(cohort, chart_data)
    features = features.merge(vital_df, on='stay_id', how='left')
    
    # Free up memory
    del chart_data
    import gc
    gc.collect()
    
    # Extract lab results using parallel processing
    lab_df = extract_lab_results_parallel(cohort, hosp_path)
    features = features.merge(lab_df, on='stay_id', how='left')
    
    # Make sure subject_id is still in features
    if 'subject_id' not in features.columns:
        print("Restoring subject_id column which was lost during merges")
        features = features.merge(subject_stay_mapping, on='stay_id', how='left')
    
    # Extract prior diagnoses
    prior_dx_df = extract_prior_diagnoses(features, hosp_path)
    features = features.merge(prior_dx_df, on='subject_id', how='left')
    
    # Save features and generate statistics
    final_features, stats = save_features(features, output_dir)
    
    print("Feature extraction completed successfully")
    return final_features


if __name__ == "__main__":
    main()