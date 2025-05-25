
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

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


def process_lab_batch(lab_name, itemids, lab_data, patient_batch, cohort_dict):
    """Process a batch of patients for one lab test with missingness indicators and focused aggregates."""
    # Initialize storage for more focused aggregated values
    result_data = {
        f'{lab_name}_mean': [],
        f'{lab_name}_min': [],        # Add min value for labs
        f'{lab_name}_max': [], 
        f'{lab_name}_delta': [],  
        f'{lab_name}_measured': [],
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
                result_data[f'{lab_name}_min'].append(values.min())  # Add minimum value
                result_data[f'{lab_name}_max'].append(values.max())
                result_data[f'{lab_name}_delta'].append(last_val - first_val)
                result_data[f'{lab_name}_measured'].append(1)  # Lab was measured
            else:
                result_data[f'{lab_name}_mean'].append(np.nan)
                result_data[f'{lab_name}_min'].append(np.nan)  # Add NaN for min
                result_data[f'{lab_name}_max'].append(np.nan)
                result_data[f'{lab_name}_delta'].append(np.nan)
                result_data[f'{lab_name}_measured'].append(0)  # Lab was not measured
            
            result_data['subject_id'].append(subject_id)
    
    return pd.DataFrame(result_data)


def extract_inr(cohort, hosp_path):
    """Extract INR (International Normalized Ratio) values."""
    print("Extracting INR values...")
    
    try:
        # Load lab events data
        labs = pd.read_csv(os.path.join(hosp_path, '_labevents.csv'),
                         usecols=['subject_id', 'charttime', 'itemid', 'valuenum'])
        labs['charttime'] = pd.to_datetime(labs['charttime'])
        
        # Load lab items dictionary to verify INR itemids
        try:
            d_labitems = pd.read_csv(os.path.join(hosp_path, '_d_labitems.csv'))
            inr_items = d_labitems[d_labitems['label'].str.contains('INR|International Normalized Ratio|PT', 
                                                                  case=False, na=False)]
            
            if not inr_items.empty:
                inr_itemids = inr_items['itemid'].tolist()
                # Add the additional known PT/INR ID
                if 51675 not in inr_itemids:
                    inr_itemids.append(51675)
                print(f"Found {len(inr_itemids)} INR/PT itemids from d_labitems")
            else:
                # Fallback to known MIMIC-IV itemids, including both main IDs
                inr_itemids = [51237, 51675]
                print("No INR itemids found in d_labitems, using defaults: 51237, 51675")
        except Exception as e:
            # Fallback if d_labitems.csv is not available
            inr_itemids = [51237, 51675]  # Default INR itemid in MIMIC-IV
            print(f"Could not load d_labitems.csv: {str(e)}")
            print("Using default INR itemids: 51237, 51675")
        
        # Filter to only INR values
        subject_ids = set(cohort['subject_id'])
        inr_data = labs[labs['itemid'].isin(inr_itemids) & 
                        labs['subject_id'].isin(subject_ids)].copy()
        
        # Check if we have any INR data
        if inr_data.empty:
            print("No INR measurements found in the dataset.")
            # Create empty DataFrame with all subject IDs
            inr_df = pd.DataFrame({
                'subject_id': list(subject_ids),
                'inr_mean': np.nan,
                'inr_max': np.nan,
                'inr_measured': 0
            })
            return inr_df
        
        # Create dictionary for cohort info lookup
        cohort_dict = cohort.set_index('subject_id')[['intime', 'window_end']].to_dict('index')
        
        # Initialize results
        result_data = {
            'inr_mean': [],
            'inr_max': [], 
            'inr_measured': [],
            'subject_id': []
        }
        
        # Print some info about the distribution of the two different INR item IDs
        id_counts = inr_data['itemid'].value_counts()
        for item_id in inr_itemids:
            if item_id in id_counts:
                print(f"ItemID {item_id} has {id_counts[item_id]} measurements")
        
        for subject_id in tqdm(subject_ids, desc="Processing INR values"):
            if subject_id in cohort_dict:
                intime = cohort_dict[subject_id]['intime']
                window_end = cohort_dict[subject_id]['window_end']
                
                # Get INR measurements within time window
                pt_inr = inr_data[
                    (inr_data['subject_id'] == subject_id) & 
                    (inr_data['charttime'] >= intime) & 
                    (inr_data['charttime'] <= window_end)
                ]
                
                if not pt_inr.empty:
                    values = pt_inr['valuenum'].dropna()
                    
                    if not values.empty:
                        result_data['inr_mean'].append(values.mean())
                        result_data['inr_max'].append(values.max())
                        result_data['inr_measured'].append(1)
                    else:
                        result_data['inr_mean'].append(np.nan)
                        result_data['inr_max'].append(np.nan)
                        result_data['inr_measured'].append(0)
                else:
                    result_data['inr_mean'].append(np.nan)
                    result_data['inr_max'].append(np.nan)
                    result_data['inr_measured'].append(0)
                    
                result_data['subject_id'].append(subject_id)
        
        inr_df = pd.DataFrame(result_data)
        measured_count = inr_df['inr_measured'].sum()
        print(f"Extracted INR values for {measured_count} patients ({measured_count/len(subject_ids):.1%} of cohort)")
        
        # Ensure all cohort subject_ids are in the result
        all_subjects_df = pd.DataFrame({'subject_id': list(subject_ids)})
        inr_df = all_subjects_df.merge(inr_df, on='subject_id', how='left')
        inr_df['inr_measured'] = inr_df['inr_measured'].fillna(0).astype(int)
        
        return inr_df
        
    except Exception as e:
        print(f"Error extracting INR values: {str(e)}")
        print("Creating placeholder INR features...")
        
        # Create a dataframe with empty values for all patients
        inr_df = pd.DataFrame({
            'subject_id': cohort['subject_id'].unique(),
            'inr_mean': np.nan,
            'inr_max': np.nan,
            'inr_measured': 0
        })
        
        return inr_df


def extract_urine_output(cohort, icu_path):
    """Extract urine output measurements within the observation window using chartevents."""
    print("Extracting urine output measurements from chartevents...")
    
    try:
        # Load chart events data for urine
        charts = pd.read_csv(os.path.join(icu_path, '_chartevents.csv'),
                           usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
                           dtype={'itemid': int, 'stay_id': int, 'valuenum': float})
        charts['charttime'] = pd.to_datetime(charts['charttime'])
        
        # Urine output itemids in chartevents
        # Based on MIMIC-IV documentation and d_items
        urine_itemids = [
            226559,  # Foley
            226560,  # Void
            226561,  # Condom Cath
            226584,  # Ileoconduit
            226563,  # Suprapubic
            226564,  # R Nephrostomy
            226565,  # L Nephrostomy
            226567,  # Straight Cath
            226557,  # R Ureteral Stent
            226558,  # L Ureteral Stent
            227489,  # GU Irrigant/Urine Vol Out
            227488,  # GU Irrigant/Urine Volume Out
            226627,  # Urine Output
            226631   # Urine Output
        ]
        
        # Filter to only relevant itemids and cohort patients
        stay_ids = set(cohort['stay_id'])
        urine_data = charts[charts['itemid'].isin(urine_itemids) & 
                          charts['stay_id'].isin(stay_ids)].copy()
        
        # Dictionary for cohort info lookup
        cohort_dict = cohort.set_index('stay_id')[['intime', 'window_end']].to_dict('index')
        
        # Prepare results dataframe
        result = []
        
        for stay_id in tqdm(stay_ids, desc="Processing urine output"):
            if stay_id in cohort_dict:
                intime = cohort_dict[stay_id]['intime']
                window_end = cohort_dict[stay_id]['window_end']
                
                # Get urine measurements within time window
                pt_urine = urine_data[
                    (urine_data['stay_id'] == stay_id) & 
                    (urine_data['charttime'] >= intime) & 
                    (urine_data['charttime'] <= window_end)
                ]
                
                if not pt_urine.empty:
                    # Sum all urine output within window
                    total_urine = pt_urine['valuenum'].sum()
                    urine_measured = 1
                else:
                    total_urine = np.nan
                    urine_measured = 0
                
                result.append({
                    'stay_id': stay_id,
                    'urine_output_total': total_urine,
                    'urine_output_measured': urine_measured
                })
        
        urine_df = pd.DataFrame(result)
        print(f"Extracted urine output for {urine_df['urine_output_measured'].sum()} patients from chartevents")
        
        return urine_df
        
    except Exception as e:
        print(f"Error extracting urine output from chartevents: {str(e)}")
        print("Creating placeholder urine output features...")
        
        # Create a dataframe with empty values
        urine_df = pd.DataFrame({
            'stay_id': cohort['stay_id'],
            'urine_output_total': np.nan,
            'urine_output_measured': 0
        })
        
        return urine_df


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

