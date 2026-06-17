# Experiment Result Schema

Aggregate validation and test results should use this schema:

```json
{
  "model_name": "xgboost",
  "split": "test",
  "auc_roc": 0.0,
  "average_precision": 0.0,
  "brier_score": 0.0,
  "threshold_policy": "balanced_f1",
  "threshold": 0.5,
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "specificity": 0.0,
  "f1": 0.0,
  "npv": 0.0,
  "n": 0,
  "positive_rate": 0.0
}
```

Rows are aggregate split-level records only. Do not save or commit row-level
probabilities, row-level predictions, patient identifiers, or restricted
patient-level feature values.

The canonical implementation lives in `src/evaluation/reporting.py`.
