# Data Contract

This repository does not redistribute MIMIC-IV data or derived patient-level
datasets. To run the full pipeline, you must have authorized access to the
course-provided MIMIC-IV subset or the corresponding MIMIC-IV tables under the
PhysioNet credentialed health data license.

## Directory Layout

By default the code reads from `data/`. You can override this with:

```bash
set ICU_DATA_DIR=C:\path\to\icu-data
```

Expected layout:

```text
data/
  hosp/
    _patients.csv
    _labevents.csv
    _diagnoses_icd.csv
    _d_labitems.csv
  icu/
    _icustays.csv
    _chartevents.csv
  label/
    _label_death.csv
  processed/
    final_cohort.csv
    extracted_features.csv
    preprocessed_logistic_features.csv
    preprocessed_xgboost_features.csv
    preprocessed_random_forest_features.csv
```

## Required Processed Columns

Model training expects a processed CSV with:

- `mortality`: binary target column, where `1` indicates in-hospital mortality.
- optional identifiers: `subject_id`, `hadm_id`, `stay_id`; these are dropped
  before training.
- feature columns: numeric clinical, demographic, missingness, and derived
  features.

## Pipeline Order

```bash
python src/cohort_selection.py
python src/feature_extraction.py
python src/data_preprocessing.py
python src/main.py --model xgboost --no-tune --no-shap
```

The first three commands require local data access. They are documented for
reproducibility, but they cannot run from a fresh public clone without the
restricted data files.

## Privacy Boundary

Do not commit:

- raw MIMIC tables
- processed patient-level CSVs
- trained model artifacts containing feature metadata from restricted data
- generated result folders with row-level predictions
