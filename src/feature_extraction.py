import pandas as pd
import os
import argparse

from feature_extraction import *
from config import hosp_path, icu_path, label_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract first-6-hour ICU mortality features"
    )
    parser.add_argument(
        "--feature-config",
        default=str(DEFAULT_PHASE3_CONFIG),
        help="Path to Phase 3 expanded feature config",
    )
    phase3_group = parser.add_mutually_exclusive_group()
    phase3_group.add_argument(
        "--enable-phase3-features",
        action="store_true",
        help="Enable expanded Phase 3 first-6-hour feature builders",
    )
    phase3_group.add_argument(
        "--disable-phase3-features",
        action="store_true",
        help="Disable expanded Phase 3 first-6-hour feature builders",
    )
    return parser.parse_args()


def _phase3_override_from_args(args):
    if args.enable_phase3_features:
        return True
    if args.disable_phase3_features:
        return False
    return None


def main(feature_config_path=None, enable_phase3_features=None):
    phase3_config = load_phase3_feature_config(feature_config_path)
    use_phase3_features = phase3_features_enabled(
        phase3_config,
        override=enable_phase3_features,
    )
    print(f"Phase 3 expanded features: {'enabled' if use_phase3_features else 'disabled'}")

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

    if use_phase3_features:
        print("Extracting expanded Phase 3 event-derived features...")
        phase3_event_df = extract_expanded_event_features(
            cohort,
            chart_data=chart_data,
            hospital_path=hosp_path,
            config=phase3_config,
        )
        features = features.merge(phase3_event_df, on='stay_id', how='left')
        print(f"After merging Phase 3 event features, shape: {features.shape}")
    
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

    if use_phase3_features:
        print("Adding expanded Phase 3 derived features...")
        features = add_expanded_derived_features(features, config=phase3_config)
        print(f"After adding Phase 3 derived features, shape: {features.shape}")
    
    # Add mortality outcome labels
    features = add_mortality_labels(features, label_path)


    # Save features and generate statistics
    features_filename = (
        'extracted_features_phase3.csv'
        if use_phase3_features
        else 'extracted_features.csv'
    )
    stats_filename = (
        'feature_statistics_phase3.csv'
        if use_phase3_features
        else 'feature_statistics.csv'
    )
    final_features, stats = save_features(
        features,
        output_dir,
        features_filename=features_filename,
        stats_filename=stats_filename,
    )
    

    # Generate Table One if outcome data is available
    if 'mortality' in final_features.columns:
        table_one = generate_table_one(final_features, output_dir, group_col='mortality')
    
    print("Enhanced feature extraction completed successfully")
    return final_features

if __name__ == "__main__":
    parsed_args = parse_args()
    main(
        feature_config_path=parsed_args.feature_config,
        enable_phase3_features=_phase3_override_from_args(parsed_args),
    )
