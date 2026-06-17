# Experiment Result Schema

The model suite writes aggregate result tables with a stable schema. The final
summary consumes these files directly.

## Files

| File | Description |
|---|---|
| `model_suite_status.json` | Run metadata, model status, dropped columns, nested reports |
| `model_suite_results.csv` | Validation and test rows for every model and threshold policy |
| `model_comparison_table.csv` | One comparison row per completed model |
| `<model>/threshold_policy_report.json` | Per-model aggregate report and calibration summary |
| `<model>/threshold_policy_results.csv` | Per-model aggregate threshold table |

## Result Columns

| Column | Meaning |
|---|---|
| `model_name` | Model registry name |
| `split` | `validation` or `test` |
| `auc_roc` | Threshold-independent discrimination metric |
| `average_precision` | Area under the precision-recall curve |
| `brier_score` | Probability calibration and sharpness metric |
| `threshold_policy` | Validation-selected policy name |
| `threshold` | Probability threshold selected on validation data |
| `accuracy` | Classification accuracy at threshold |
| `precision` | Positive predictive value at threshold |
| `recall` | Sensitivity at threshold |
| `specificity` | True negative rate at threshold |
| `f1` | Harmonic mean of precision and recall |
| `npv` | Negative predictive value at threshold |
| `n` | Number of rows in the evaluated split |
| `positive_rate` | Target prevalence in the evaluated split |

Bootstrap interval columns may be appended to `model_comparison_table.csv` when
bootstrap iterations are enabled.
