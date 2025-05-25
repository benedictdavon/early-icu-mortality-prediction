import os
import pandas as pd

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

def save_features(features, output_dir):
    """Save features and generate descriptive statistics."""
    # Clean up and save
    final_features = features.copy(deep=True)

    # Save features to CSV
    features_path = os.path.join(output_dir, 'extracted_features.csv')
    final_features.to_csv(features_path, index=False)
    print(f"Saved {len(final_features)} patient records with {len(final_features.columns)} features to {features_path}")

    # Basic descriptive analysis for Table 1
    stats = final_features.describe().T
    stats['missing_pct'] = 100 * final_features.isnull().sum() / len(final_features)
    stats_path = os.path.join(output_dir, 'feature_statistics.csv')
    stats.to_csv(stats_path, index=True)
    print(f"Saved descriptive statistics to {stats_path}")
    
    return final_features, stats
