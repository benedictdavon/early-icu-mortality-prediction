import pandas as pd
import os

from feature_extraction import *
from config import hosp_path, icu_path, label_path


def main():
    # Setup
    output_dir = setup_directories()
    
    # Load cohort
    cohort_path = os.path.join(output_dir, 'final_cohort.csv')
    cohort = load_cohort(cohort_path)
    
    # Extract features
    features = extract_demographics(cohort, hosp_path)

    # Extract early time window features (first 6 hours) - critical change here
    time_window_df = extract_early_window_features(cohort, icu_path, hosp_path)
    
    # Print debug information about time_window_df
    print(f"Time window features dataframe shape: {time_window_df.shape}")
    print(f"Sample columns: {time_window_df.columns[:10]}")
    print(f"Non-null counts for first few columns:")
    for col in time_window_df.columns[:5]:
        print(f"  - {col}: {time_window_df[col].notna().sum()}")
    
    # Verify merge after features are extracted
    features = features.merge(time_window_df, on='stay_id', how='left')
    print(f"After merging time window features, shape: {features.shape}")
    
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
    
    # NEW FEATURES
    # Extract urine output
    urine_df = extract_urine_output(cohort, icu_path)
    features = features.merge(urine_df, on='stay_id', how='left')
    
    # Extract INR values
    inr_df = extract_inr(cohort, hosp_path)
    features = features.merge(inr_df, on='subject_id', how='left')
    
    # Extract metastatic cancer flag
    metastatic_df = extract_metastatic_cancer_flag(cohort, hosp_path)
    features = features.merge(metastatic_df, on='subject_id', how='left')
 
    # # Clean and validate clinical measurements
    # features = clean_clinical_measurements(features)
    
    # # Add log transformations for skewed variables
    # features = add_log_transformations(features)
    
    # Add clinically meaningful derived features
    features = add_clinical_derived_features(features)
    
    # Add mortality outcome labels
    features = add_mortality_labels(features, label_path)


    # Save features and generate statistics
    final_features, stats = save_features(features, output_dir)
    

    # Generate Table One if outcome data is available
    if 'mortality' in final_features.columns:
        table_one = generate_table_one(final_features, output_dir, group_col='mortality')
    
    print("Enhanced feature extraction completed successfully")
    return final_features

if __name__ == "__main__":
    main()
