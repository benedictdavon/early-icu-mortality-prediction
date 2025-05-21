import pandas as pd
from datetime import timedelta
import os

from config import hosp_path, icu_path, label_path


def setup_directories():
    """Create output directory if it doesn't exist."""
    output_dir = os.path.join(os.path.dirname(icu_path), 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def load_patients(hosp_path):
    """Load all patients in the database."""
    patients = pd.read_csv(os.path.join(hosp_path, '_patients.csv'), 
                          usecols=['subject_id'])
    total_patients = len(patients['subject_id'].unique())
    print(f"Total patients in MIMIC database: {total_patients}")
    return total_patients


def load_icustays(icu_path):
    """Load and preprocess ICU stays data."""
    # Load only the columns we need
    icustays = pd.read_csv(os.path.join(icu_path, '_icustays.csv'), 
                        usecols=['stay_id', 'subject_id', 'hadm_id', 'intime', 'outtime'])

    print(f"Loaded {len(icustays)} ICU stays")
    
    # Count unique ICU patients
    icu_patients = len(icustays['subject_id'].unique())
    print(f"Patients with ICU stays: {icu_patients}")
    
    # Convert datetime columns
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    return icustays, icu_patients


def filter_first_stays(icustays, icu_patients):
    """Filter to only first ICU stay for each patient."""
    # Step 1: Mark first ICU stay for each patient
    icustays['icu_stay_seq'] = icustays.sort_values(['subject_id', 'intime'])\
                                       .groupby('subject_id').cumcount() + 1
    
    # Step 2: Filter to first ICU stays only
    first_stays = icustays[icustays['icu_stay_seq'] == 1]
    first_stays_count = len(first_stays)
    multiple_stays_excluded = icu_patients - first_stays_count
    print(f"Patients with first ICU stay only: {first_stays_count}")
    print(f"Excluded {multiple_stays_excluded} repeat ICU stays")
    
    return first_stays, first_stays_count, multiple_stays_excluded


def check_record_duration(first_stays, icu_path):
    """Check for sufficient record duration before discharge."""
    # Step 3: Check for sufficient records before discharge - VECTORIZED APPROACH
    # Load only necessary columns from chartevents (optimization 2)
    chartevents = pd.read_csv(os.path.join(icu_path, '_chartevents.csv'), 
                              usecols=['stay_id', 'charttime'])
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    print(f"Loaded chart events data")
    
    # Get first record time for each stay_id (optimization 3 - replaces the loop)
    earliest_records = chartevents.groupby('stay_id')['charttime'].min().reset_index()
    earliest_records.rename(columns={'charttime': 'earliest_record'}, inplace=True)
    
    # Merge with first_stays
    first_stays_with_records = first_stays.merge(earliest_records, on='stay_id', how='inner')
    print(f"Stays with records: {len(first_stays_with_records)}")
    
    # Calculate duration
    first_stays_with_records['duration_hours'] = (
        first_stays_with_records['outtime'] - first_stays_with_records['earliest_record']
    ).dt.total_seconds() / 3600
    
    # Filter for sufficient duration
    final_cohort = first_stays_with_records[first_stays_with_records['duration_hours'] >= 6]
    final_cohort_count = len(final_cohort)
    insufficient_records_count = len(first_stays_with_records) - final_cohort_count
    print(f"Final cohort after filtering for ≥6 hours of records: {final_cohort_count}")
    print(f"Excluded {insufficient_records_count} patients with <6 hours of records")
    
    return final_cohort, final_cohort_count, insufficient_records_count


def save_cohort(final_cohort, output_dir):
    """Save the final cohort to CSV files."""
    # Export the final cohort to CSV
    cohort_path = os.path.join(output_dir, 'final_cohort.csv')
    final_cohort.to_csv(cohort_path, index=False)
    print(f"Final cohort saved to: {cohort_path}")
    
    # Save just the patient IDs or stay IDs for easy reference
    cohort_ids_path = os.path.join(output_dir, 'cohort_stay_ids.csv')
    pd.DataFrame({'stay_id': final_cohort['stay_id']}).to_csv(cohort_ids_path, index=False)
    print(f"Cohort stay IDs saved to: {cohort_ids_path}")
    
    return cohort_path, cohort_ids_path


def print_statistics(total_patients, icu_patients, first_stays_count, 
                     multiple_stays_excluded, final_cohort_count, insufficient_records_count):
    """Print cohort selection statistics for flow diagram."""
    print("\n=== COHORT SELECTION FLOW DIAGRAM STATISTICS ===")
    print(f"Total patients in MIMIC database: {total_patients}")
    print(f"Patients with ICU stays: {icu_patients}")
    print(f"Filtered to first ICU stay only: {first_stays_count}")
    print(f"  Excluded: {multiple_stays_excluded} repeat ICU stays")
    print(f"Final cohort with ≥6 hours of records: {final_cohort_count}")
    print(f"  Excluded: {insufficient_records_count} patients with <6 hours of records")


def main():
    """Main function to orchestrate the cohort selection pipeline."""
    # Setup
    output_dir = setup_directories()
    
    # Load all patients
    total_patients = load_patients(hosp_path)
    
    # Load ICU stays
    icustays, icu_patients = load_icustays(icu_path)
    
    # Filter to first ICU stays
    first_stays, first_stays_count, multiple_stays_excluded = filter_first_stays(icustays, icu_patients)
    
    # Check for sufficient record duration
    final_cohort, final_cohort_count, insufficient_records_count = check_record_duration(first_stays, icu_path)
    
    # Save cohort
    cohort_path, cohort_ids_path = save_cohort(final_cohort, output_dir)
    
    # Print statistics
    print_statistics(total_patients, icu_patients, first_stays_count, 
                     multiple_stays_excluded, final_cohort_count, insufficient_records_count)
    
    print("Cohort selection completed successfully")
    return final_cohort


if __name__ == "__main__":
    main()