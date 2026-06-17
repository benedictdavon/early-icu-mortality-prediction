# Model Card

## Intended Use

This project is a retrospective machine learning study for an academic AI in
EHR course. It explores whether early ICU measurements from the first 6 hours
of admission can predict in-hospital mortality.

## Not Intended For

This repository is not a clinical decision-support system. The models should
not be used for triage, treatment decisions, alarm generation, or deployment in
care settings without external validation, calibration, governance review, and
clinical safety evaluation.

## Data

The project uses a course-provided subset derived from MIMIC-IV. MIMIC-IV is
credentialed health data and is not redistributed in this repository.

## Model Families

- Logistic regression
- Random forest
- Random forest bagging
- XGBoost
- XGBoost probability-averaging ensemble

## Evaluation

The pipeline uses stratified train/validation/test splits. Thresholds should be
selected on validation data only; the test set is reserved for final reporting.

Primary metrics:

- AUC-ROC
- precision
- recall
- F1 score
- specificity for selected threshold analyses

## Limitations

- Single-source retrospective data limits generalizability.
- The repository cannot provide a fully runnable public demo without restricted
  MIMIC-IV data.
- Class imbalance makes threshold selection clinically sensitive.
- Feature engineering and imputation choices require independent review before
  any real-world use.
- The models do not replace clinical judgment.

## Ethical And Safety Notes

Mortality prediction can influence care allocation if misused. False negatives
may miss high-risk patients; false positives may cause unnecessary escalation or
alarm fatigue. Any future deployment would require prospective validation,
subgroup fairness analysis, calibration monitoring, and clinician-facing
explanations.
