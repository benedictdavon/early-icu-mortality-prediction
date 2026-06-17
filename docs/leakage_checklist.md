# Leakage Checklist

This checklist defines the minimum leakage controls for the public portfolio
project.

## Data Boundary

- Do not commit raw MIMIC files.
- Do not commit processed patient-level CSVs or parquet files.
- Do not commit row-level predictions.
- Do not commit model binaries, checkpoints, or artifacts containing restricted
  feature metadata.
- Do not commit subject, hospital admission, or ICU stay identifiers in public
  reports.

## First-6-Hour Window

- Feature values must be available no later than ICU admission + 6 hours.
- Features derived from discharge time, death time, total length of stay,
  post-window interventions, or post-window measurements are excluded.
- Cohort eligibility fields such as sufficient observation duration are not
  automatically model features.

## Split Discipline

- Patients must not overlap across train, validation, and test splits when
  patient identifiers are available.
- Preprocessing, imputation, scaling, encoding, feature selection, resampling,
  and class-weight calculations are fit on training data only.
- Validation is used for threshold selection, calibration fitting, early
  stopping, and model-selection decisions.
- Test is used only for final aggregate reporting.

## Feature Guards

Model features must not include:

- `mortality` or target aliases
- `subject_id`, `hadm_id`, `stay_id`, or related identifiers
- death, discharge, survival, outcome, expiration, or length-of-stay proxy
  columns
- row-level timestamps that imply information after the prediction window

The canonical processed-feature guard lives in `src/data/schema.py`.
Feature provenance metadata lives in `src/features/provenance.py`, with the
checked-in aggregate dictionary at `docs/feature_dictionary.csv`.

Known Phase 2 provenance note:

- `has_metastatic_cancer` is flagged as high leakage risk and not allowed by
  the provenance registry until the extractor is limited to prior or otherwise
  admission-time-known diagnoses.
- BMI extraction is now restricted to first-6-hour height/weight measurements.

## Automated Checks

Synthetic tests should run without restricted data and verify:

- validation-only threshold selection
- fixed threshold application to test probabilities
- no patient overlap across splits
- target and obvious leakage columns excluded from model features
- train-only preprocessing fit behavior where testable
