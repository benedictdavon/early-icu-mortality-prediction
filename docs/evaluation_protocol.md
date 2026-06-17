# Evaluation Protocol

This project is retrospective academic and portfolio research. It is not a
clinical deployment system and must not be used for triage, treatment, alerting,
or care decisions.

## Splits

The leakage-safe baseline uses train, validation, and held-out test splits.
When patient identifiers are available, all rows for a patient must appear in
only one split.

- Training: model fitting, preprocessing fitting, imputation fitting, feature
  selection fitting, resampling, and class-weight estimation.
- Validation: threshold selection, calibration fitting, early stopping, and
  model-selection decisions.
- Test: final aggregate reporting only.

The test set must not be used for threshold selection, calibration fitting,
preprocessing fitting, imputation fitting, feature selection, resampling, or
early stopping.

## Metrics

Every model report should include:

- AUC-ROC
- average precision / PR-AUC
- Brier score
- calibration summaries
- accuracy
- precision
- recall / sensitivity
- specificity
- F1
- negative predictive value

Accuracy is retained for historical comparability but is not the main metric for
an imbalanced mortality outcome.

## Threshold Policies

Thresholds are selected on validation probabilities only, then applied unchanged
to the test probabilities:

- `high_sensitivity`: highest threshold satisfying validation recall >= 0.85 if
  possible.
- `balanced_f1`: threshold maximizing validation F1.
- `high_precision`: highest threshold satisfying validation precision >= 0.65 if
  possible.

Threshold choice is a policy decision as well as a model decision. Final test
tables should report each threshold policy separately.

## Historical Results Caveat

The threshold-specific numbers in the original report and README are historical
until the models are rerun under this corrected validation-threshold protocol.
Historical AUC-ROC values remain useful for orientation, but precision, recall,
F1, specificity, NPV, and threshold-specific accuracy should be interpreted as
legacy results until regenerated.
