import os
import pandas as pd

def add_mortality_labels(features, label_path):
    """Add mortality outcome labels to feature dataset"""
    print("Adding mortality outcome labels...")
    
    # Initialize all patients with mortality = 0
    features_with_labels = features.copy()
    features_with_labels['mortality'] = 0
    
    try:
        # Load mortality labels (only contains patients who died)
        mortality_labels = pd.read_csv(os.path.join(label_path, '_label_death.csv'))
        print(f"Loaded mortality data for {len(mortality_labels)} patients who died")
        
        # Check required columns
        if 'subject_id' not in mortality_labels.columns:
            print("Warning: 'subject_id' not found in mortality labels file")
            return features_with_labels
            
        # Get list of subject_ids for patients who died
        died_subject_ids = set(mortality_labels['subject_id'])
        
        # Set mortality = 1 for patients who are in the death file
        features_with_labels.loc[features_with_labels['subject_id'].isin(died_subject_ids), 'mortality'] = 1
        
        # Report mortality rate
        mortality_count = features_with_labels['mortality'].sum()
        total_patients = len(features_with_labels)
        mortality_rate = mortality_count / total_patients
        
        print(f"Found {mortality_count} deaths out of {total_patients} patients")
        print(f"Overall mortality rate: {mortality_rate:.2%}")
        
    except FileNotFoundError:
        print(f"Warning: Mortality labels file not found at {os.path.join(label_path, '_label_death.csv')}")
        print("Proceeding with all patients marked as non-mortality (mortality=0)")
    except Exception as e:
        print(f"Error processing mortality labels: {str(e)}")
        print("Proceeding with all patients marked as non-mortality (mortality=0)")
        
    return features_with_labels
