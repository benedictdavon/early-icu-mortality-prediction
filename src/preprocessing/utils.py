import pandas as pd

def load_data(path):
    """Load the extracted features dataset"""
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df


def save_processed_data(df, output_path):
    """Save the preprocessed dataframe"""
    print(f"Saving preprocessed data to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data with shape {df.shape}")

# preprocessing_report, visualizations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_preprocessing_report(original_df, processed_df, output_dir):
    """Generate a report on the preprocessing steps and their impact"""
    print("Generating preprocessing report...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    report = {
        'original_shape': original_df.shape,
        'processed_shape': processed_df.shape,
        'dropped_features': set(original_df.columns) - set(processed_df.columns),
        'new_features': set(processed_df.columns) - set(original_df.columns),
        'missing_before': original_df.isnull().sum().sum(),
        'missing_after': processed_df.isnull().sum().sum(),
    }
    
    # Generate report text
    report_text = [
        "=== Data Preprocessing Report ===",
        f"Original data shape: {report['original_shape'][0]} rows, {report['original_shape'][1]} columns",
        f"Processed data shape: {report['processed_shape'][0]} rows, {report['processed_shape'][1]} columns",
        f"Features dropped: {len(report['dropped_features'])}",
        f"New features created: {len(report['new_features'])}",
        f"Missing values before: {report['missing_before']}",
        f"Missing values after: {report['missing_after']}",
        "\nNew features created:",
        "\n".join([f"- {feat}" for feat in sorted(list(report['new_features']))]),
        "\nFeatures dropped:",
        "\n".join([f"- {feat}" for feat in sorted(list(report['dropped_features']))])
    ]
    
    # Write report to file
    with open(os.path.join(output_dir, 'preprocessing_report.txt'), 'w') as f:
        f.write('\n'.join(report_text))
    
    # Generate some visualizations
    # 1. Missing data comparison
    plt.figure(figsize=(12, 6))
    missing_before = (original_df.isnull().mean() * 100).sort_values(ascending=False)[:20]
    missing_after = (processed_df.isnull().mean() * 100).sort_values(ascending=False)[:20]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(missing_before)), missing_before.values)
    plt.title('Missing Data Before (Top 20)')
    plt.ylabel('Missing Percentage')
    plt.xticks([])
    
    plt.subplot(1, 2, 2)
    if len(missing_after) > 0 and missing_after.values[0] > 0:
        plt.bar(range(len(missing_after)), missing_after.values)
        plt.title('Missing Data After (Top 20)')
        plt.xticks([])
    else:
        plt.text(0.5, 0.5, 'No Missing Data', horizontalalignment='center', 
                verticalalignment='center', fontsize=16)
        plt.title('Missing Data After')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_data_comparison.png'))
    
    # 2. Feature count by type
    plt.figure(figsize=(10, 6))
    
    orig_numeric = len(original_df.select_dtypes(include=[np.number]).columns)
    orig_categorical = len(original_df.select_dtypes(exclude=[np.number]).columns)
    
    proc_numeric = len(processed_df.select_dtypes(include=[np.number]).columns)
    proc_categorical = len(processed_df.select_dtypes(exclude=[np.number]).columns)
    
    categories = ['Numeric', 'Categorical']
    before_counts = [orig_numeric, orig_categorical]
    after_counts = [proc_numeric, proc_categorical]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, before_counts, width, label='Before')
    plt.bar(x + width/2, after_counts, width, label='After')
    
    plt.ylabel('Feature Count')
    plt.title('Feature Count by Type')
    plt.xticks(x, categories)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_count_comparison.png'))
    
    return report

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