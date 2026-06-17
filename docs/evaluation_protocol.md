# Evaluation Protocol

The project uses retrospective train/validation/test evaluation for early ICU
mortality prediction.

## Split Policy

- Target: `mortality`
- Default split: 60% train, 20% validation, 20% test
- Stratification: enabled
- Patient grouping: enabled when `subject_id` is available
- Random seed: `42` unless explicitly overridden

The validation split is used for threshold selection. The held-out test split is
reserved for final aggregate reporting.

## Threshold Policies

Thresholds are selected on validation probabilities only, then applied unchanged
to validation and test records:

| Policy | Intent |
|---|---|
| `high_sensitivity` | Favor recall for screening-style use |
| `balanced_f1` | Balance precision and recall |
| `high_precision` | Reduce false positives |

The same threshold-policy names are used in every aggregate result table.

## Metrics

Primary aggregate metrics:

- AUC-ROC
- Average precision
- Brier score
- Accuracy
- Precision
- Recall
- Specificity
- F1
- NPV

Calibration summaries are stored in each model's
`threshold_policy_report.json`. Bootstrap confidence intervals can be requested
from the model-suite runner with `--bootstrap-iterations`.

## Reporting Rule

Only aggregate results are written. Row-level predictions are intentionally not
saved by the model-suite and final-summary workflows.
