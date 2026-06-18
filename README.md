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
|   |-- experiments/               # Model-suite and final-report entrypoints
|   |-- feature_extraction/        # Modular feature extraction helpers
|   |-- preprocessing/             # Modular preprocessing helpers
|   |-- training/                  # MAFNet training and neural callbacks
|   `-- models/
|       |-- base/                  # Shared base model and persistence helpers
|       |-- mafnet/                # Missingness-aware temporal fusion network
|       |-- logistic_regression/   # Logistic regression model and interpretation
|       |-- random_forest/         # Random forest and bagging variants
|       `-- xgboost/               # XGBoost model, tuning, and ensemble helpers
|-- tools/
|   `-- check_leakage.py           # Lightweight processed-feature leakage checks
|-- docs/
|   |-- data_contract.md           # Expected restricted-data layout and columns
|   |-- evaluation_protocol.md     # Split, threshold, and metric protocol
|   |-- experiment_result_schema.md # Aggregate result fields
|   |-- leakage_checklist.md       # Leakage controls for EHR modeling
|   `-- final_deliverables.md      # Final assignment deliverable mapping
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

Expanded tabular features are opt-in and write separate processed
artifacts:

```bash
python src/feature_extraction.py --enable-expanded-features
python src/data_preprocessing.py \
  --input-path data/processed/extracted_features_expanded.csv \
  --model-type xgboost \
  --output-path data/processed/preprocessed_xgboost_expanded_features.csv
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

Leakage-safe baseline vs expanded-feature XGBoost ablation:

```bash
python tools/run_xgboost_ablation.py \
  --baseline-data-path data/processed/preprocessed_xgboost_features.csv \
  --expanded-data-path data/processed/preprocessed_xgboost_expanded_features.csv
```

The ablation script uses validation data for early stopping and threshold-policy
selection, then applies selected thresholds unchanged to the held-out test set.
It saves aggregate reports only.

Tabular model suite:

```bash
python src/experiments/run_model_suite.py \
  --data-path data/processed/preprocessed_xgboost_expanded_features.csv \
  --output-dir results/model_suite
```

The suite runs Logistic Regression, Random Forest, ExtraTrees, XGBoost,
LightGBM, CatBoost, and EBM through a common interface when their dependencies
are installed. Missing optional backends are recorded as skipped. Outputs are
aggregate comparison tables and threshold-policy reports only; row-level
predictions are not written.

MAFNet first-6-hour temporal model:

```bash
python src/training/train_mafnet.py \
  --events-path data/processed/first6h_events.csv \
  --cohort-path data/processed/cohort.csv \
  --features-path data/processed/preprocessed_xgboost_expanded_features.csv \
  --output-dir results/mafnet \
  --evaluate-test
```

MAFNet ablations:

```bash
python src/experiments/run_mafnet_ablations.py \
  --events-path data/processed/first6h_events.csv \
  --cohort-path data/processed/cohort.csv \
  --features-path data/processed/preprocessed_xgboost_expanded_features.csv \
  --output-dir results/mafnet_ablations
```

The MAFNet runner writes aggregate metrics, calibration reports, training
curves, and local checkpoints. Checkpoints and patient-level predictions are
ignored by git and should not be committed.

Final deliverable summary:

```bash
python src/experiments/generate_final_summary.py \
  --model-suite-dir results/model_suite \
  --output-path report/FINAL_SUMMARY.md
```

The final summary consumes aggregate model-suite outputs and produces a
course-review summary covering TODO traceability, model-suite status, primary
test metrics, threshold policies, feature provenance, safety notes, and final
conclusions. See [`docs/final_deliverables.md`](docs/final_deliverables.md).

Final polish reports from a completed overnight run:

```bash
python src/experiments/generate_final_polish_reports.py \
  --run-dir results/overnight_20260618_111652
```

This writes aggregate-only `final_model_report.md`,
`calibration_report.md`, `bootstrap_ci_report.md`, and
`ensemble_report.md` under the run directory. Bootstrap paired comparisons and
multi-model calibrated ensembles require aligned local row-level prediction
files; the report generator states when those files are unavailable rather than
fabricating results.

Overnight GPU training run:

```bash
bash scripts/run_overnight_training.sh
```

The overnight runner executes cohort selection, baseline and expanded feature
extraction, preprocessing, the tabular model suite, XGBoost baseline-vs-expanded
ablation, XGBoost ensemble, MAFNet, and MAFNet ablations. It requires CUDA by
default and writes logs plus aggregate comparison CSVs under
`results/overnight_<timestamp>/`, including `all_training_results.csv` and
`result_csv_index.csv`.

Common overrides:

```bash
RUN_NAME=overnight_gpu_001 \
MAFNET_EVENTS_PATH=data/processed/first6h_events.csv \
MAFNET_COHORT_PATH=data/processed/final_cohort.csv \
bash scripts/run_overnight_training.sh
```

Set `SKIP_PREPROCESS=1` to reuse existing processed CSVs, `RUN_MAFNET=0` or
`RUN_MAFNET_ABLATIONS=0` to skip temporal neural runs, and `REQUIRE_GPU=0` only
for local dry runs. The script runs `pytest` before training by default; set
`RUN_PYTEST=0` only when you intentionally want to skip that preflight check.

Optional processed-feature leakage check:

```bash
python tools/check_leakage.py --data-path data/processed/preprocessed_xgboost_features.csv
```

## Evaluation Protocol

The training code uses stratified train/validation/test splits. Thresholds are
selected on validation data; the held-out test set is used for final reporting.
When patient identifiers are available, the split utilities keep patients from
appearing in more than one split.

Primary metrics:

- AUC-ROC
- average precision / PR-AUC
- Brier score
- calibration summaries
- precision
- recall
- F1 score
- specificity / NPV for threshold analysis

Threshold policies are selected on validation probabilities only and then
applied unchanged to test probabilities:

- high sensitivity
- balanced F1
- high precision

See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) and
[`docs/leakage_checklist.md`](docs/leakage_checklist.md). The aggregate result
format is documented in
[`docs/experiment_result_schema.md`](docs/experiment_result_schema.md).
Feature provenance is tracked in
[`docs/feature_dictionary.csv`](docs/feature_dictionary.csv).
The expanded first-6-hour feature plan is documented in
[`configs/features_expanded.yaml`](configs/features_expanded.yaml).

## Reported Results

Latest completed local run: `results/overnight_20260618_111652`
(`2026-06-18`). The run completed preprocessing, tabular model comparison,
XGBoost feature ablation, legacy XGBoost ensemble training, full MAFNet
training, MAFNet architecture ablations, final summary generation, and combined
aggregate CSV collection.

Primary tabular comparison uses the validation-selected `balanced_f1` threshold
applied unchanged to the held-out test split.

| Model | Test AUC-ROC | Test PR-AUC | Brier | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.8475 | 0.6289 | 0.1246 | 0.5755 | 0.6346 | 0.5264 |
| XGBoost | 0.8473 | 0.6271 | 0.1503 | 0.5808 | 0.7054 | 0.4936 |
| CatBoost | 0.8454 | 0.6223 | 0.1466 | 0.5742 | 0.7181 | 0.4783 |
| EBM | 0.8361 | 0.6104 | 0.1195 | 0.5740 | 0.6728 | 0.5005 |
| ExtraTrees | 0.8356 | 0.6014 | 0.1258 | 0.5713 | 0.6530 | 0.5077 |
| Random forest | 0.8286 | 0.5819 | 0.1267 | 0.5582 | 0.6898 | 0.4687 |
| Logistic regression | 0.8188 | 0.5629 | 0.1727 | 0.5431 | 0.6473 | 0.4678 |

LightGBM is the primary final model for reporting. XGBoost is statistically and
practically near-tied on discrimination in the available aggregate intervals and
had the highest F1 among the tabular suite at the selected threshold. EBM is a
useful calibrated/interpretable comparison because it had the best tabular Brier
score. Do not treat LightGBM as definitively superior unless a paired bootstrap
comparison on aligned predictions supports that claim.

Expanded features materially improved XGBoost:

| XGBoost feature set | Test AUC-ROC | Test PR-AUC | Brier | F1 | Recall |
|---|---:|---:|---:|---:|---:|
| Baseline features | 0.7853 | 0.5095 | 0.3483 | 0.4990 | 0.7238 |
| Expanded features | 0.8438 | 0.6228 | 0.1279 | 0.5743 | 0.7195 |

MAFNet was implemented and evaluated, but did not outperform the best boosted
tabular models in this run. The best neural temporal variant was MAFNet-T+S:

| MAFNet variant | Test AUC-ROC | Test PR-AUC | Brier |
|---|---:|---:|---:|
| Full MAFNet | 0.8061 | 0.5610 | 0.1282 |
| Best ablation, MAFNet-T+S | 0.8184 | 0.5896 | 0.1231 |

Interpret all numbers as retrospective academic metrics, not deployment
performance. See [`docs/results_summary.md`](docs/results_summary.md) for the
detailed run summary and artifact locations.

## Methods

Feature groups include:

- demographics
- early vital-sign statistics and trends
- hourly first-6-hour vital/lab bins
- first-vs-last trajectory summaries
- lab summaries
- measurement-process features such as early count and timing
- missingness indicators
- prior diagnosis summaries
- clinically derived features such as shock index, SIRS criteria count, organ
  dysfunction proxies, and focused clinical interactions

Phase 3 expanded feature builders are synthetic-tested for first-6-hour
filtering, but historical model metrics have not yet been rerun with those
features.

Model families include:

- logistic regression
- random forest
- random forest bagging
- XGBoost
- XGBoost probability-averaging ensemble
- ICU6H-MAFNet, a custom missingness-aware temporal fusion network

## Custom MAFNet Architecture

MAFNet consumes 15-minute first-6-hour tensors with values, observation masks,
time-since-last-measurement deltas, and measurement counts. The temporal branch
uses GRU-D-style value and hidden-state decay followed by optional transformer
refinement and temporal attention. Static and aggregate first-window feature
branches are encoded with MLPs, then fused with patient-specific gates before a
binary mortality classifier.

The training runner supports temporal self-supervised pretraining through
masked value reconstruction and next-bin measurement forecasting, followed by
supervised fine-tuning with weighted BCE and small auxiliary losses. Calibration
is fit on validation logits only, using Platt scaling by default with optional
isotonic calibration for ablation.

Architecture ablations include temporal-only, temporal+static, full
temporal+static+aggregate, no decay, no transformer, no auxiliary losses, no
gated fusion, and no pretraining. Ensemble helpers support averaging calibrated
XGBoost and MAFNet probabilities and an out-of-fold logistic stacker.

## Subgroup Analysis

Subgroup reporting utilities compute aggregate metrics across age group, sex,
major diagnosis group, lactate measured status, missingness level, measurement
intensity quartile, ICU unit type when available, and ventilation status when
available. Reports include subgroup size, mortality rate, AUC-ROC, average
precision, Brier score, calibration slope/intercept, and threshold metrics.
Small subgroups are flagged for cautious interpretation.

## Course And Acknowledgment

Course: Artificial Intelligence Capstone (人工智慧總整與實作) 2025, National Yang Ming Chiao Tung University.
Instructor: 王才沛

This repository reflects my personal implementation and exploration of a course
final project originally conducted as a group assignment.

## References

- MIMIC-IV: https://mimic.mit.edu/
- PhysioNet credentialed health data license: https://physionet.org/
- Full reference list: [`report/REPORT.md`](report/REPORT.md)

## License

Academic use only. Data access and use are governed by the MIMIC-IV / PhysioNet
credentialed health data license.
