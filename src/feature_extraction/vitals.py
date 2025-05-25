import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

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


def process_vital_batch(vital_name, itemids, vital_data, patient_batch, cohort_dict):
    """Process a batch of patients for one vital sign with time-weighted averaging."""
    # Initialize storage for aggregated values
    result_data = {
        f"{vital_name}_mean": [],  # Will use time-weighted average
        f"{vital_name}_min": [],
        f"{vital_name}_max": [],
        f"{vital_name}_measured": [],  # Added missingness indicator
        "stay_id": [],
    }

    # Filter for specific vital
    this_vital_data = vital_data[vital_data["itemid"].isin(itemids)]

    for stay_id in patient_batch:
        if stay_id in cohort_dict:
            intime = cohort_dict[stay_id]["intime"]
            window_end = cohort_dict[stay_id]["window_end"]

            # Get vital measurements within time window
            pt_vitals = this_vital_data[
                (this_vital_data["stay_id"] == stay_id)
                & (this_vital_data["charttime"] >= intime)
                & (this_vital_data["charttime"] <= window_end)
            ]

            if len(pt_vitals) > 0:
                values = pt_vitals["valuenum"].values
                times = pt_vitals["charttime"].values

                # Calculate time-weighted average if multiple measurements
                if len(values) > 1:
                    tw_mean = time_weighted_vital_avg(values, times)
                else:
                    tw_mean = values[0]

                result_data[f"{vital_name}_mean"].append(tw_mean)
                result_data[f"{vital_name}_min"].append(min(values))
                result_data[f"{vital_name}_max"].append(max(values))
                result_data[f"{vital_name}_measured"].append(1)
            else:
                result_data[f"{vital_name}_mean"].append(np.nan)
                result_data[f"{vital_name}_min"].append(np.nan)
                result_data[f"{vital_name}_max"].append(np.nan)
                result_data[f"{vital_name}_measured"].append(0)

            result_data["stay_id"].append(stay_id)

    return pd.DataFrame(result_data)

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

