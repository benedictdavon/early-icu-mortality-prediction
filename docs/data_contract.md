# Data Contract

This project expects authorized local access to the course-provided MIMIC-IV
subset. Raw data and patient-level processed data must not be committed.

## Raw Input Layout

By default, scripts read from `data/`. A local path can be supplied through
`ICU_DATA_DIR` or by editing an uncommitted `configs/paths.yaml`.

Expected raw tables depend on the stage being run:

| Stage | Expected sources |
|---|---|
| Cohort selection | ICU stays, admissions, patients |
| Feature extraction | ICU chartevents, outputevents, hospital labevents, diagnoses |
| Labels | Hospital admission outcome fields for in-hospital mortality |

The course data has recoded identifiers and edited dates. The code treats
`subject_id`, `hadm_id`, and `stay_id` as identifiers, not model features.

## Processed Feature Contract

Processed feature CSVs must contain:

| Column | Required | Notes |
|---|---:|---|
| `mortality` | yes | Binary target, values `0` or `1` |
| `subject_id` | preferred | Used only for patient-level split separation |
| Feature columns | yes | Numeric or categorical columns derived from allowed first-6h sources |

Outcome proxies, identifiers, and post-window columns are removed by
`data.schema.build_feature_matrix` before model fitting.

## Standard Artifacts

| Artifact | Producer |
|---|---|
| `data/processed/extracted_features.csv` | `src/feature_extraction.py` |
| `data/processed/preprocessed_<model>_features.csv` | `src/data_preprocessing.py` |
| `results/model_suite/` | `src/experiments/run_model_suite.py` |
| `report/FINAL_SUMMARY.md` | `src/experiments/generate_final_summary.py` |

Keep all patient-level CSVs, model artifacts, and row-level predictions local.
