
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def extract_early_window_features(cohort, icu_path, hosp_path):
    """Memory-optimized extraction of time window features with better performance."""
    print("Extracting early time window features with improved debugging...")
    
    # Define time windows and variables up front
    time_windows = [0, 1, 2, 3, 4, 6]
    
    vital_itemids = {
        'heart_rate': [220045],   # Heart Rate
        'resp_rate': [220210],    # Respiratory Rate
        'map': [220052],          # Mean Arterial Pressure
        'temp': [223761, 223762], # Temperature in C and F
        'sbp': [220179],          # Systolic BP
        'dbp': [220180],          # Diastolic BP
        'spo2': [220277]          # SpO2
        
    }
    
    lab_itemids = {
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
    
    # Create lookup dictionaries for fast access
    stay_ids = list(cohort['stay_id'])
    
    # Verify the cohort data has the required columns
    required_cols = ['stay_id', 'subject_id', 'intime']
    missing_cols = [col for col in required_cols if col not in cohort.columns]
    if missing_cols:
        print(f"ERROR: Cohort is missing required columns: {missing_cols}")
        return pd.DataFrame({'stay_id': stay_ids})
        
    # Print cohort data information for debugging
    print(f"Cohort contains {len(cohort)} patients with {cohort['subject_id'].nunique()} unique subject_ids")
    print(f"First few intime values: {cohort['intime'].head().tolist()}")
    
    intime_dict = dict(zip(cohort['stay_id'], cohort['intime']))
    stay_to_subject = dict(zip(cohort['stay_id'], cohort['subject_id']))
    subject_to_stay = {v: k for k, v in stay_to_subject.items()}
    
    # Initialize result dataframe - explicitly with stay_id as integer
    result_df = pd.DataFrame({'stay_id': stay_ids})
    
    ###########################################
    # VITALS EXTRACTION
    ###########################################
    print("Processing vital signs time windows...")
    all_vital_itemids = []
    for ids in vital_itemids.values():
        all_vital_itemids.extend(ids)
    
    # Define a dictionary to store all measurements 
    vital_measurements = {}
    for var in vital_itemids:
        vital_measurements[var] = {}
    
    # Load chart data in chunks to reduce memory usage
    print("Loading and filtering vital sign measurements...")
    chart_path = os.path.join(icu_path, '_chartevents.csv')
    
    if not os.path.exists(chart_path):
        print(f"ERROR: Chart events file not found at {chart_path}")
        return result_df
    
    try:
        chart_sample = pd.read_csv(chart_path, nrows=10)
        print(f"Chart events sample columns: {chart_sample.columns.tolist()}")
    except Exception as e:
        print(f"Error reading chart events sample: {str(e)}")
    
    total_filtered_rows = 0
    chunk_size = 500000
    
    # Process chart data chunks
    for chunk_idx, chunk_orig in enumerate(tqdm(pd.read_csv(
        chart_path,
        usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
        dtype={'stay_id': 'Int64', 'itemid': 'Int64', 'valuenum': float},
        chunksize=chunk_size
    ))):
        try:
            # Convert charttime to datetime
            chunk = chunk_orig.copy()
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            
            # Debugging info for first chunk
            if chunk_idx == 0:
                print(f"First chunk contains {len(chunk)} rows")
                print(f"Chart events - unique stay_ids in first chunk: {chunk['stay_id'].nunique()}")
                print(f"Chart events - unique itemids in first chunk: {chunk['itemid'].nunique()}")
                common_stay_ids = set(chunk['stay_id'].dropna().astype(int)) & set(stay_ids)
                print(f"Chart events - first chunk has {len(common_stay_ids)} stay_ids matching cohort")
                common_itemids = set(chunk['itemid'].dropna().astype(int)) & set(all_vital_itemids)
                print(f"Chart events - first chunk has {len(common_itemids)} itemids matching vital signs")
            
            # Handle NaN values
            chunk = chunk.dropna(subset=['stay_id', 'itemid', 'charttime', 'valuenum'])
            
            # Convert to correct types
            chunk['stay_id'] = chunk['stay_id'].astype(int)
            chunk['itemid'] = chunk['itemid'].astype(int)
            
            # Filter to relevant patients and itemids
            mask = (chunk['stay_id'].isin(stay_ids) & 
                    chunk['itemid'].isin(all_vital_itemids))
            
            # Debug for first chunk
            if chunk_idx == 0:
                print(f"First chunk - rows with stay_id in cohort: {sum(chunk['stay_id'].isin(stay_ids))}")
                print(f"First chunk - rows with itemids matching vitals: {sum(chunk['itemid'].isin(all_vital_itemids))}")
                print(f"First chunk - rows matching both conditions: {sum(mask)}")
            
            if not mask.any():
                if chunk_idx == 0:
                    print("WARNING: No matching vital records found in first chunk")
                continue
            
            chunk_filtered = chunk.loc[mask].copy()
            total_filtered_rows += len(chunk_filtered)
            
            if chunk_filtered.empty:
                continue
            
            # Check for missing intimes
            missing_intimes = chunk_filtered['stay_id'].isin(intime_dict.keys()) == False
            if missing_intimes.any():
                missing_count = sum(missing_intimes)
                if missing_count > 0 and chunk_idx == 0:
                    print(f"WARNING: {missing_count} rows have stay_ids that lack intime values")
            
            # Calculate hours from admission
            chunk_filtered.loc[:, 'intime'] = chunk_filtered['stay_id'].map(intime_dict)
            
            if chunk_idx == 0 and chunk_filtered['intime'].isna().any():
                print(f"WARNING: {chunk_filtered['intime'].isna().sum()} rows have NaT intime values")
                
            try:
                chunk_filtered.loc[:, 'hours_from_admit'] = (
                    pd.to_datetime(chunk_filtered['charttime']) - 
                    pd.to_datetime(chunk_filtered['intime'])
                ).dt.total_seconds() / 3600
            except Exception as e:
                print(f"ERROR calculating hours_from_admit: {str(e)}")
                if chunk_idx == 0:
                    print("Sample charttime:", chunk_filtered['charttime'].iloc[0])
                    print("Sample intime:", chunk_filtered['intime'].iloc[0])
                continue
            
            # Keep only measurements within first 6.5 hours
            chart_time_mask = chunk_filtered['hours_from_admit'] <= 6.5
            if not chart_time_mask.any():
                if chunk_idx == 0:
                    print("WARNING: No vital records within 6.5 hours of admission in first chunk")
                continue
                
            chunk_filtered = chunk_filtered[chart_time_mask].copy()
            
            if chunk_filtered.empty:
                continue
                
            # Organize by vital type - STORE IN MEMORY
            for var_name, itemids in vital_itemids.items():
                var_data = chunk_filtered[chunk_filtered['itemid'].isin(itemids)].copy()
                
                if var_data.empty:
                    continue
                    
                # For each record, store in the dictionary
                for _, row in var_data.iterrows():
                    stay_id = int(row['stay_id'])  # Make sure it's an int
                    if stay_id not in vital_measurements[var_name]:
                        vital_measurements[var_name][stay_id] = []
                    
                    vital_measurements[var_name][stay_id].append({
                        'value': row['valuenum'],
                        'hours_from_admit': row['hours_from_admit']
                    })
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            continue
    
    print(f"Total filtered vitals rows across all chunks: {total_filtered_rows}")
    
    # Report coverage
    for var_name, measurements in vital_measurements.items():
        patients_with_data = len(measurements)
        pct_coverage = patients_with_data / len(stay_ids) * 100
        print(f"Found {patients_with_data} patients with {var_name} measurements ({pct_coverage:.1f}% of cohort)")
    
    # Process vital measurements and add to result_df
    for var_name, measurements in vital_measurements.items():
        print(f"Extracting {var_name} time window values...")
        
        # Create temporary storage for each time window
        window_values = {}
        for window in time_windows:
            window_values[window] = {}  # Use a dict instead of Series
        
        # Find closest measurements for each time window
        for stay_id in stay_ids:
            if stay_id in measurements and measurements[stay_id]:
                stay_data = sorted(measurements[stay_id], key=lambda x: x['hours_from_admit'])
                
                # Process each time window
                for window in time_windows:
                    # Handle window 0 (admission time)
                    if window == 0:
                        # Find measurement closest to admission time
                        admission_data = [m for m in stay_data if m['hours_from_admit'] <= 1.0]
                        
                        if admission_data:
                            # Find closest to time 0
                            closest = min(admission_data, key=lambda x: abs(x['hours_from_admit']))
                            window_values[window][stay_id] = closest['value']
                    else:
                        # For other windows
                        window_lower = max(0, window - 0.5)
                        window_upper = window + 0.5
                        
                        window_data = [m for m in stay_data 
                                      if window_lower <= m['hours_from_admit'] <= window_upper]
                        
                        if window_data:
                            closest = min(window_data, key=lambda x: abs(x['hours_from_admit'] - window))
                            window_values[window][stay_id] = closest['value']
        
        # NOW ADD DIRECTLY TO THE RESULT DATAFRAME
        for window in time_windows:
            col_name = f'{var_name}_hour_{window}'
            
            # Add to result_df
            result_df[col_name] = result_df['stay_id'].map(window_values[window])

            non_null = result_df[col_name].notna().sum()
            pct_complete = non_null / len(result_df) * 100

            print(f"  - {col_name}: {non_null} values ({pct_complete:.1f}% complete)")
    
    ###########################################
    # LABS EXTRACTION
    ###########################################
    print("Processing lab time windows...")
    all_lab_itemids = []
    for ids in lab_itemids.values():
        all_lab_itemids.extend(ids)

    # Create dictionary for lab measurements
    lab_measurements = {}
    for var in lab_itemids:
        lab_measurements[var] = {}
    
    # Check lab events file
    lab_path = os.path.join(hosp_path, '_labevents.csv')
    if not os.path.exists(lab_path):
        print(f"ERROR: Lab events file not found at {lab_path}")
        return result_df
        
    try:
        lab_sample = pd.read_csv(lab_path, nrows=10)
        print(f"Lab events sample columns: {lab_sample.columns.tolist()}")
    except Exception as e:
        print(f"Error reading lab events sample: {str(e)}")
    
    total_lab_filtered_rows = 0
    
    # Process lab chunks
    print("Loading and filtering lab measurements...")
    for chunk_idx, chunk_orig in enumerate(tqdm(pd.read_csv(
        lab_path,
        usecols=['subject_id', 'charttime', 'itemid', 'valuenum'],
        dtype={'subject_id': 'Int64', 'itemid': 'Int64', 'valuenum': float},
        chunksize=chunk_size
    ))):
        try:
            chunk = chunk_orig.copy()
            
            if chunk_idx == 0:
                print(f"Lab events first chunk contains {len(chunk)} rows")
                print(f"Lab events - unique subject_ids in first chunk: {chunk['subject_id'].nunique()}")
                common_subject_ids = set(chunk['subject_id'].dropna().astype(int)) & set(stay_to_subject.values())
                print(f"Lab events - first chunk has {len(common_subject_ids)} subject_ids matching cohort")
            
            # Handle NaN values
            chunk = chunk.dropna(subset=['subject_id', 'itemid', 'charttime', 'valuenum'])
            
            # Convert to correct types
            chunk['subject_id'] = chunk['subject_id'].astype(int)
            chunk['itemid'] = chunk['itemid'].astype(int)
            
            # Convert charttime to datetime
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            
            # Filter to relevant patients and itemids
            mask = (chunk['subject_id'].isin(stay_to_subject.values()) & 
                    chunk['itemid'].isin(all_lab_itemids))
            
            if chunk_idx == 0:
                print(f"Lab first chunk - rows with subject_id in cohort: {sum(chunk['subject_id'].isin(stay_to_subject.values()))}")
                print(f"Lab first chunk - rows with itemids matching labs: {sum(chunk['itemid'].isin(all_lab_itemids))}")
                print(f"Lab first chunk - rows matching both conditions: {sum(mask)}")
            
            if not mask.any():
                if chunk_idx == 0:
                    print("WARNING: No matching lab records found in first chunk")
                continue
            
            chunk_filtered = chunk.loc[mask].copy()
            total_lab_filtered_rows += len(chunk_filtered)
            
            if chunk_filtered.empty:
                continue
            
            # Map subject_id to stay_id
            chunk_filtered.loc[:, 'stay_id'] = chunk_filtered['subject_id'].map(subject_to_stay)
            chunk_filtered = chunk_filtered.dropna(subset=['stay_id'])
            
            if chunk_filtered.empty:
                if chunk_idx == 0:
                    print("WARNING: No lab records matched to stay_ids in first chunk")
                continue
                
            # Convert stay_id to int
            chunk_filtered.loc[:, 'stay_id'] = chunk_filtered['stay_id'].astype(int)
            
            # Get admission times
            chunk_filtered.loc[:, 'intime'] = chunk_filtered['stay_id'].map(intime_dict)
            
            # Skip rows with missing intime
            if chunk_filtered['intime'].isna().any():
                chunk_filtered = chunk_filtered.dropna(subset=['intime'])
                if chunk_filtered.empty:
                    continue
            
            # Calculate hours from admission
            try:
                chunk_filtered.loc[:, 'hours_from_admit'] = (
                    pd.to_datetime(chunk_filtered['charttime']) - 
                    pd.to_datetime(chunk_filtered['intime'])
                ).dt.total_seconds() / 3600
            except Exception as e:
                print(f"ERROR calculating lab hours_from_admit: {str(e)}")
                continue
            
            # Keep only measurements within first 6.5 hours
            lab_time_mask = chunk_filtered['hours_from_admit'] <= 6.5
            if not lab_time_mask.any():
                if chunk_idx == 0:
                    print("WARNING: No lab records within 6.5 hours of admission in first chunk")
                continue
                
            chunk_filtered = chunk_filtered[lab_time_mask].copy()
            
            if chunk_filtered.empty:
                continue
            
            # Store lab measurements by type in memory
            for var_name, itemids in lab_itemids.items():
                var_data = chunk_filtered[chunk_filtered['itemid'].isin(itemids)].copy()
                
                if var_data.empty:
                    continue
                    
                # Store data for each patient
                for _, row in var_data.iterrows():
                    stay_id = int(row['stay_id'])
                    if stay_id not in lab_measurements[var_name]:
                        lab_measurements[var_name][stay_id] = []
                    
                    lab_measurements[var_name][stay_id].append({
                        'value': row['valuenum'],
                        'hours_from_admit': row['hours_from_admit']
                    })
        except Exception as e:
            print(f"Error processing lab chunk {chunk_idx}: {str(e)}")
            continue
    
    print(f"Total filtered lab rows across all chunks: {total_lab_filtered_rows}")
    
    # Report lab coverage
    for var_name, measurements in lab_measurements.items():
        patients_with_data = len(measurements)
        pct_coverage = patients_with_data / len(stay_ids) * 100
        print(f"Found {patients_with_data} patients with {var_name} measurements ({pct_coverage:.1f}% of cohort)")
    
    # Process lab measurements and add to result_df
    for var_name, measurements in lab_measurements.items():
        print(f"Extracting {var_name} time window values...")
        
        # Create temporary storage for each window
        window_values = {}
        for window in time_windows:
            window_values[window] = {}  # Dictionary instead of Series
        
        # Find measurements for each patient and time window
        for stay_id in stay_ids:
            if stay_id in measurements and measurements[stay_id]:
                stay_data = sorted(measurements[stay_id], key=lambda x: x['hours_from_admit'])
                
                # Process each time window
                for window in time_windows:
                    if window == 0:
                        admission_data = [m for m in stay_data if m['hours_from_admit'] <= 1.0]
                        
                        if admission_data:
                            closest = min(admission_data, key=lambda x: abs(x['hours_from_admit']))
                            window_values[window][stay_id] = closest['value']
                    else:
                        window_lower = max(0, window - 0.5)
                        window_upper = window + 0.5
                        
                        window_data = [m for m in stay_data 
                                      if window_lower <= m['hours_from_admit'] <= window_upper]
                        
                        if window_data:
                            closest = min(window_data, key=lambda x: abs(x['hours_from_admit'] - window))
                            window_values[window][stay_id] = closest['value']
        
        # Add lab values to result dataframe
        for window in time_windows:
            col_name = f'{var_name}_hour_{window}'
            
            # Add to result dataframe
            result_df[col_name] = result_df['stay_id'].map(window_values[window])

            non_null = result_df[col_name].notna().sum()
            pct_complete = non_null / len(result_df) * 100
            
            print(f"  - {col_name}: {non_null} values ({pct_complete:.1f}% complete)")
    
    ###########################################
    # ADD CHANGE FEATURES
    ###########################################
    # Add derived change features if there are time window columns
    time_window_cols = [col for col in result_df.columns if '_hour_' in col]
    if time_window_cols:
        try:
            # Use this to verify data exists
            first_var = list(vital_itemids.keys())[0]
            first_col = f'{first_var}_hour_{time_windows[0]}'
            print(f"Before adding change features: {first_col} has {result_df[first_col].notna().sum()} non-null values")
            
            # Only add change features if we have data
            if result_df[first_col].notna().sum() > 0:
                result_df = add_early_change_features(result_df, vital_itemids.keys(), lab_itemids.keys(), time_windows)
                print(f"After adding change features: {first_col} has {result_df[first_col].notna().sum()} non-null values")
            else:
                print("WARNING: Not adding change features because time window columns are empty")
                
            # Calculate completeness metrics
            non_null_counts = pd.Series({col: result_df[col].notna().sum() for col in time_window_cols})
            avg_non_null = non_null_counts.mean()
            completeness = avg_non_null / len(result_df) * 100
            print(f"Extracted time window features have average completeness of {completeness:.1f}%")
            
            # Report on specific variables
            for var in list(vital_itemids.keys()) + list(lab_itemids.keys()):
                var_cols = [col for col in time_window_cols if col.startswith(f"{var}_hour_")]
                if var_cols:
                    var_non_null = pd.Series({col: result_df[col].notna().sum() for col in var_cols})
                    var_completeness = var_non_null.mean() / len(result_df) * 100
                    print(f"  - {var}: {var_completeness:.1f}% complete")
        except Exception as e:
            print(f"ERROR adding change features: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final stats
    print(f"Extracted {result_df.shape[1]-1} early time window features for {len(result_df)} stays")
    return result_df


def extract_early_window_values(data, cohort, itemids, var_name, time_windows):
    """Extract values at specific early time windows for a variable."""
    results = []
    
    for stay_id in tqdm(cohort['stay_id'].unique(), desc=f"Processing {var_name}"):
        intime = cohort.loc[cohort['stay_id'] == stay_id, 'intime'].iloc[0]
        
        # Filter data for this stay and variable
        pts_data = data[
            (data['stay_id'] == stay_id) & 
            (data['itemid'].isin(itemids))
        ].copy()  # Explicit .copy() to prevent warning
        
        # Calculate hours from admission
        pts_data['hours_from_admit'] = (pts_data['charttime'] - intime).dt.total_seconds() / 3600
        
        # Filter to first 6 hours only
        pts_data = pts_data[pts_data['hours_from_admit'] <= 6.5].copy()  # Explicit .copy()
        
        # Extract values for each time window
        result = {'stay_id': stay_id}
        
        # Special case for hour 0 (admission)
        if 0 in time_windows:
            # Get measurement closest to admission time
            admission_data = pts_data[pts_data['hours_from_admit'] <= 1.0].copy()  # Explicit .copy()
            
            if not admission_data.empty:
                admission_data.loc[:, 'time_diff'] = admission_data['hours_from_admit']
                closest_idx = admission_data['time_diff'].idxmin()
                value = admission_data.loc[closest_idx, 'valuenum']
                result[f'{var_name}_hour_0'] = value
            else:
                result[f'{var_name}_hour_0'] = np.nan
            
            # Remove 0 from time windows to process
            time_windows_to_process = [w for w in time_windows if w > 0]
        else:
            time_windows_to_process = time_windows
        
        # Process remaining time windows
        for window in time_windows_to_process:
            # Get measurements within ±0.5 hour window
            window_lower = max(0, window - 0.5)
            window_upper = window + 0.5
            
            window_data = pts_data[
                (pts_data['hours_from_admit'] >= window_lower) & 
                (pts_data['hours_from_admit'] <= window_upper)
            ].copy()  # Explicit .copy()
            
            if not window_data.empty:
                # Using .loc to avoid the warning
                window_data.loc[:, 'time_diff'] = abs(window_data['hours_from_admit'] - window)
                closest_idx = window_data['time_diff'].idxmin()
                value = window_data.loc[closest_idx, 'valuenum']
                
                # Store the value
                result[f'{var_name}_hour_{window}'] = value
            else:
                result[f'{var_name}_hour_{window}'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)

def add_early_change_features(df, vital_vars, lab_vars, time_windows):
    """Add change and delta features between time windows."""
    # Create a true deep copy to prevent any issues
    result_df = df.copy(deep=True)
    
    # Print current non-null counts for debugging
    first_var = list(vital_vars)[0] if vital_vars else list(lab_vars)[0]
    first_col = f'{first_var}_hour_{time_windows[0]}'
    
    # Check if column exists
    if first_col in result_df.columns:
        print(f"BEFORE: {first_col} has {result_df[first_col].notna().sum()} non-null values")
    else:
        print(f"WARNING: {first_col} not found in DataFrame")
        print(f"Available columns: {result_df.columns[:10]}...")
        return result_df  # Return early if we don't have the needed columns
    
    all_vars = list(vital_vars) + list(lab_vars)
    print(f"Adding change features for {len(all_vars)} variables across {len(time_windows)} time windows...")
    
    # Create new columns for the changes without modifying existing ones
    for var in all_vars:
        # Add first-to-last change (captures overall trend)
        first_window = min(time_windows)

        last_window = max(time_windows)
        
        first_col = f'{var}_hour_{first_window}'
        last_col = f'{var}_hour_{last_window}'
        
        if first_col in result_df.columns and last_col in result_df.columns:
            # Make sure we're not overwriting the original columns
            first_values = result_df[first_col].copy()
            last_values = result_df[last_col].copy()
            
            # Overall change - use loc to safely add a new column
            change_col = f'{var}_change_0to6'
            result_df.loc[:, change_col] = last_values - first_values
            
            # Debug info for first variable
            if var == all_vars[0]:
                print(f"  Created {change_col}: {result_df[change_col].notna().sum()} non-null values")
                print(f"  DURING: {first_col} still has {result_df[first_col].notna().sum()} non-null values")
            
            # Hourly rate of change
            hours_diff = last_window - first_window
            if hours_diff > 0:
                hourly_col = f'{var}_hourly_change'
                result_df.loc[:, hourly_col] = result_df[change_col] / hours_diff
        
        # Add adjacent time window changes
        for i in range(len(time_windows) - 1):
            t1 = time_windows[i]
            t2 = time_windows[i + 1]
            
            col1 = f'{var}_hour_{t1}'
            col2 = f'{var}_hour_{t2}'
            
            if col1 in result_df.columns and col2 in result_df.columns:
                # Get copies of the values to avoid any reference issues
                values1 = result_df[col1].copy()
                values2 = result_df[col2].copy()
                
                # Change between adjacent windows
                delta_col = f'{var}_delta_{t1}to{t2}'
                result_df.loc[:, delta_col] = values2 - values1
                
                # Debug info for first iteration
                if var == all_vars[0] and i == 0:
                    print(f"  Created {delta_col}: {result_df[delta_col].notna().sum()} non-null values")
    
    # Verify the original columns are still present
    time_window_cols_after = [col for col in result_df.columns if '_hour_' in col]
    print(f"After adding change features: found {len(time_window_cols_after)} time window columns")
    
    # Verify data wasn't lost
    if len(time_window_cols_after) > 0:
        print(f"  AFTER: {first_col} has {result_df[first_col].notna().sum()} non-null values")
    
    return result_df
