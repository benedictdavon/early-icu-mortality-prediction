# Leakage Checklist

Use this checklist before treating any model result as reportable.

## Cohort And Time Window

- Use the first ICU stay per patient.
- Use only data from ICU admission through the first 6 hours for Task 2.
- Exclude events after the observation window.
- Exclude patients without the required 6 hours before discharge.

## Feature Rules

- Do not use `mortality`, death time, discharge disposition, survival duration,
  or other outcome proxies as model features.
- Do not use `subject_id`, `hadm_id`, or `stay_id` as model features.
- Keep feature provenance documented in `docs/feature_dictionary.csv`.
- Treat missingness indicators as allowed only when they are derived from
  within-window measurement availability.

## Preprocessing Rules

- Fit imputation, scaling, encoding, and feature selection on training data only
  when evaluating models.
- Preserve the binary target as `0`/`1`; never scale or transform it.
- Validate categorical handling before sklearn probes or models that require
  numeric matrices.

## Evaluation Rules

- Select thresholds on validation probabilities only.
- Apply selected thresholds unchanged to the held-out test split.
- Save aggregate reports only; do not save row-level predictions.
- Treat retrospective performance as experimental, not deployment-ready.

## Helper Command

```bash
python tools/check_leakage.py --data-path data/processed/preprocessed_xgboost_features.csv
```
