import os
import pandas as pd
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

