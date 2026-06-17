# ICU Mortality Prediction

Retrospective machine learning study for early in-hospital mortality prediction
using ICU measurements from the first 6 hours of admission.

This project was built for the NYCU AI in EHR / AI Capstone 2025 course and is
based on a course-provided subset derived from MIMIC-IV.

## Important Disclaimer

This repository is for academic research and portfolio review only. It is not a
clinical decision-support system and should not be used for triage, treatment,
alarm generation, or care decisions. MIMIC-IV data is credentialed health data
and is not redistributed here.

## Project Summary

- Task: binary classification of in-hospital mortality.
- Observation window: first 6 hours after ICU admission.
- Cohort used in the original experiment: 16,922 ICU patients.
- Mortality rate in the original experiment: 20.87%.
- Best reported discrimination: XGBoost ensemble AUC-ROC around 0.846.
- Main focus: clinical feature engineering, missing-data handling, class
  imbalance, model comparison, and threshold tradeoffs.

## Repository Layout

```text
.
|-- src/
|   |-- cohort_selection.py        # Build final ICU cohort
|   |-- feature_extraction.py      # Extract demographics, vitals, labs, labels
|   |-- data_preprocessing.py      # Imputation, outliers, feature engineering
|   |-- main.py                    # Train/evaluate model families
|   |-- evaluation/                # Shared metrics, plots, threshold selection
|   |-- feature_extraction/        # Modular feature extraction helpers
|   |-- preprocessing/             # Modular preprocessing helpers
|   `-- models/
|       |-- base/                  # Shared base model and persistence helpers
|       |-- logistic_regression/   # Logistic regression model and interpretation
|       |-- random_forest/         # Random forest and bagging variants
|       `-- xgboost/               # XGBoost model, tuning, and ensemble helpers
|-- tools/
|   `-- check_leakage.py           # Lightweight processed-feature leakage checks
|-- docs/
|   `-- data_contract.md           # Expected restricted-data layout and columns
|-- report/
|   `-- REPORT.md                  # Course report and detailed methodology
|-- figures/                       # Selected non-patient aggregate figures
|-- MODEL_CARD.md                  # Intended use, limits, and safety notes
|-- requirements.txt
|-- environment.yml
`-- README.md
```

## Data Access

The code expects local MIMIC-IV-derived files. See
[`docs/data_contract.md`](docs/data_contract.md) for the expected directory
layout and processed CSV contract.

By default, the pipeline reads from `data/`. To use another location:

```powershell
$env:ICU_DATA_DIR = "C:\path\to\icu-data"
```

Do not commit raw MIMIC files, processed patient-level CSVs, model artifacts, or
row-level predictions.

## Setup

Minimal Python setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Conda setup from the original experiment environment:

```bash
conda env create -f environment.yml
conda activate capstone_final_gpu
```

The exported Conda environment is Windows/GPU-oriented. For a fresh machine,
prefer `requirements.txt` first, then install a CUDA-compatible XGBoost/PyTorch
stack only if needed.

## Pipeline

The full pipeline requires local access to the restricted data:

```bash
python src/cohort_selection.py
python src/feature_extraction.py
python src/data_preprocessing.py
```

Training examples:

```bash
# Fast smoke-style run if processed data already exists
python src/main.py --model xgboost --no-tune --no-shap

# XGBoost ensemble
python src/main.py --model xgboost_ensemble --ensemble-size 7 --no-tune

# Other model families
python src/main.py --model random_forest --no-tune --no-shap
python src/main.py --model logistic_regression --no-tune --no-shap
python src/main.py --model rf_bagging --no-tune --no-shap
```

Optional processed-feature leakage check:

```bash
python tools/check_leakage.py --data-path data/processed/preprocessed_xgboost_features.csv
```

## Evaluation Protocol

The training code uses stratified train/validation/test splits. Thresholds are
selected on validation data; the held-out test set is used for final reporting.

Primary metrics:

- AUC-ROC
- precision
- recall
- F1 score
- specificity / NPV for threshold analysis

## Reported Results

Results from the original course experiment:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7876 | 0.7439 | 0.4244 | 0.6402 | 0.5104 |
| Random forest | 0.8145 | 0.7900 | 0.4971 | 0.6091 | 0.5474 |
| XGBoost single | 0.8426 | 0.8095 | 0.5370 | 0.6275 | 0.5787 |
| XGBoost ensemble | 0.8465 | 0.8298 | 0.6419 | 0.4164 | 0.5052 |

Interpret these as retrospective experimental metrics, not deployment
performance. Re-running after code or threshold-protocol changes may produce
different F1/threshold values.

## Methods

Feature groups include:

- demographics
- early vital-sign statistics and trends
- lab summaries
- missingness indicators
- prior diagnosis summaries
- clinically derived features such as shock index and SIRS criteria count

Model families include:

- logistic regression
- random forest
- random forest bagging
- XGBoost
- XGBoost probability-averaging ensemble

## Course And Acknowledgment

Course: AI in EHR / AI Capstone 2025, National Yang Ming Chiao Tung University.

This repository reflects my personal implementation and exploration of a course
final project originally conducted as a group assignment.

## References

- MIMIC-IV: https://mimic.mit.edu/
- PhysioNet credentialed health data license: https://physionet.org/
- Full reference list: [`report/REPORT.md`](report/REPORT.md)

## License

Academic use only. Data access and use are governed by the MIMIC-IV / PhysioNet
credentialed health data license.
