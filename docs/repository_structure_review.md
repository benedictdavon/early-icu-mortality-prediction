# Repository Structure Review

## Current Strengths

- `src/models/` separates model families into packages rather than one flat
  directory.
- `src/preprocessing/` keeps imputation, scaling, feature selection, and
  validation helpers out of training code.
- `src/feature_extraction/` separates cohort, demographics, vitals, labs,
  labels, and reporting.
- `report/` and `figures/` preserve aggregate course-report evidence without
  committing patient-level data.

## Changes Applied

- Moved the ad hoc leakage check from `src/test.py` to
  `tools/check_leakage.py` so `src/` remains runtime code.
- Added `docs/data_contract.md` to document the restricted data boundary and
  expected file layout.
- Added `MODEL_CARD.md` to clarify intended use, limitations, and non-clinical
  status.
- Added `requirements.txt` as a lighter install surface than the full exported
  Conda environment.
- Updated `README.md` to match the current CLI and avoid clinical overclaiming.
- Split model code into package directories:
  - `src/models/base/model.py` and `src/models/base/persistence.py`
  - `src/models/xgboost/model.py`, `src/models/xgboost/tuning.py`, and
    `src/models/xgboost/ensemble.py`
  - `src/models/logistic_regression/model.py` and
    `src/models/logistic_regression/interpretation.py`
  - `src/models/random_forest/model.py` and
    `src/models/random_forest/bagging.py`
- Added `src/evaluation/` for shared metrics, threshold selection, and plots.
- Kept backward-compatible imports for `models.base_model` and
  `models.rf_bagging`.

## Recommended Future Cleanup

- Move root course PDFs into `report/` or remove them if `report/REPORT.md`
  covers the same content.
- Convert exploratory notebook output to a smaller, cleaned notebook or a
  script-backed report if the notebook is meant to stay public.
- Add a real `tests/` directory for path handling, threshold selection, and
  processed-feature validation.
- Continue shrinking large model classes after tests exist. The XGBoost class
  still has plotting and comprehensive-evaluation methods that can be moved
  safely in a follow-up.
- Consider packaging `src/` as an installable module if this repo grows beyond
  a course project.
