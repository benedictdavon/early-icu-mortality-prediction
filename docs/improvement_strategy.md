# ICU 6-Hour Mortality Prediction: Full Improvement Strategy

**Project context**

This project predicts binary in-hospital mortality using patient information available during the first 6 hours after ICU admission. It is a retrospective academic and portfolio ML project, not a clinical deployment system.

Current historical baseline:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7876 | 0.7439 | 0.4244 | 0.6402 | 0.5104 |
| Random forest | 0.8145 | 0.7900 | 0.4971 | 0.6091 | 0.5474 |
| XGBoost single | 0.8426 | 0.8095 | 0.5370 | 0.6275 | 0.5787 |
| XGBoost ensemble | 0.8465 | 0.8298 | 0.6419 | 0.4164 | 0.5052 |

Important caveat: threshold-specific historical numbers should be treated as historical until the leakage-fixed pipeline is rerun. Threshold selection must use validation data only.

Public repository rule:

- Do not include raw MIMIC data.
- Do not include processed patient-level CSVs.
- Do not include model artifacts containing restricted metadata.
- Do not include row-level predictions.
- Do include code, synthetic fixtures, configuration files, generated aggregate reports, plots, model cards, and documentation.

---

## Strategic objective

The project should become a strong applied AI/ML portfolio project by demonstrating:

1. temporally constrained clinical feature engineering;
2. leakage-safe machine learning;
3. strong tabular baselines;
4. one custom missingness-aware temporal model;
5. calibrated risk prediction;
6. threshold-policy analysis;
7. subgroup and robustness evaluation;
8. honest limitations and non-deployment framing.

The target is not merely to increase AUC. A more credible target is:

```text
Improve discrimination where possible.
Improve PR-AUC and high-sensitivity operating points.
Improve probability calibration.
Add defensible leakage tests.
Add a custom model with clear clinical/time-series motivation.
Create portfolio-grade documentation and reports.
```

---

# 1. Fix the metric and threshold story

## Problem

The original report focuses heavily on AUC, accuracy, precision, recall, and F1. These are useful, but insufficient for an imbalanced clinical mortality prediction problem.

Mortality rate is approximately:

```text
3531 / 16922 = 20.87%
```

This means:

```text
No-skill PR-AUC baseline ≈ 0.2087
```

Accuracy can look acceptable even when the model misses too many mortality cases. AUC-ROC measures ranking across all thresholds, but does not fully communicate positive-class retrieval quality.

## Required changes

Add these metrics to every model report:

| Metric | Required? | Why |
|---|---:|---|
| AUC-ROC | yes | Main ranking metric, comparable to prior result. |
| Average precision / PR-AUC | yes | Positive-class retrieval metric; essential under imbalance. |
| Brier score | yes | Probability quality. |
| Calibration intercept | yes | Whether risks are systematically too high or low. |
| Calibration slope | yes | Whether risks are too extreme or underconfident. |
| Expected calibration error | yes | Portfolio-friendly calibration summary. |
| Accuracy | yes | Historical comparability only. |
| Precision | yes | Alert burden. |
| Recall / sensitivity | yes | Missed mortality cases. |
| Specificity | yes | False alert control. |
| NPV | yes | Useful for screening-style thresholds. |
| F1 | yes | Balanced threshold summary. |
| Bootstrap 95% confidence intervals | yes | Statistical credibility. |

## Required threshold policies

Do not report only one threshold. Report three validation-selected thresholds:

| Policy | Validation selection rule | Intended use |
|---|---|---|
| High-sensitivity screening | Highest threshold with recall >= 0.85 on validation. If impossible, choose max recall threshold with precision > base rate. | Reduce missed mortality cases. |
| Balanced F1 | Threshold maximizing F1 on validation. | General balanced comparison. |
| High-precision alert | Highest threshold with precision >= 0.65 on validation. If impossible, choose threshold with maximum precision subject to recall >= 0.20. | Fewer, more confident alerts. |

## Implementation tasks

- Create `src/evaluation/metrics.py`.
- Create `src/evaluation/thresholds.py`.
- Create `src/evaluation/bootstrap.py`.
- Ensure `thresholds.py` never reads test labels during threshold selection.
- Make validation thresholds serializable as JSON.
- Apply saved validation thresholds once to test predictions.

## Acceptance criteria

- Reports show both validation and final test metrics.
- Test thresholds are never optimized on test labels.
- A unit test fails if the test set is passed into threshold selection.
- README explicitly says threshold choice is a policy decision, not only a model decision.

---

# 2. Improve early-window feature engineering

## Problem

The current pipeline uses strong 6-hour aggregate features, but global aggregation can erase important trajectory information. In ICU data, the pattern of deterioration or recovery during the first 6 hours can matter as much as the absolute value.

## Required changes

Add temporal and care-process features while preserving the first-6-hour constraint.

## Feature family A: time-bin features

Use 15-minute or hourly bins.

Recommended default:

```text
15-minute bins for the custom model.
Hourly bins for tabular boosted models.
```

For tabular models, create features for each variable:

```text
0-1h mean/min/max/last
1-2h mean/min/max/last
2-3h mean/min/max/last
3-4h mean/min/max/last
4-5h mean/min/max/last
5-6h mean/min/max/last
```

For high-frequency vitals:

```text
heart_rate
respiratory_rate
sbp
dbp
map
temperature
spo2
```

For labs, use:

```text
last value per hour
observed flag per hour
measurement count per hour
```

## Feature family B: early-vs-late trajectory

Create:

```text
last_2h_mean - first_2h_mean
last_2h_min - first_2h_min
last_value - first_value
percent_change = (last - first) / (abs(first) + epsilon)
linear_slope over observed points
deterioration flag
recovery flag
```

Examples:

```text
spo2_last2h_mean_minus_first2h_mean
map_last2h_min_minus_first2h_min
lactate_last_minus_first
shock_index_last2h_mean_minus_first2h_mean
```

## Feature family C: instability

Create:

```text
range = max - min
std
coefficient_of_variation = std / (abs(mean) + epsilon)
number_of_abnormal_bins
longest_consecutive_abnormal_bins
worst_recent_value
```

Examples:

```text
map_range_0_6h
respiratory_rate_std_0_6h
spo2_longest_hypoxemia_run_0_6h
sbp_min_last2h
```

## Feature family D: measurement process features

These often carry clinical signal.

Create:

```text
variable_measured_flag
variable_measurement_count
variable_time_to_first_measurement
variable_time_since_last_measurement_at_6h
panel_missing_count
total_lab_count
total_vital_count
total_chart_event_count
```

Examples:

```text
lactate_measured_0_6h
lactate_count_0_6h
lactate_time_to_first_measurement
lactate_time_since_last_measurement
coag_panel_missing_count
total_lab_measurements_0_6h
```

## Feature family E: intervention features, if available

Add only if timestamps confirm availability inside first 6 hours.

Potential features:

```text
mechanical_ventilation_flag_0_6h
noninvasive_ventilation_flag_0_6h
oxygen_support_flag_0_6h
fio2_max_0_6h
vasopressor_flag_0_6h
norepinephrine_equivalent_max_0_6h
fluid_bolus_total_0_6h
urine_output_total_0_6h
oliguria_flag_0_6h
renal_replacement_therapy_flag_0_6h
antibiotic_started_0_6h
blood_culture_ordered_0_6h
```

If these are not available in the course subset, document them as future work rather than faking them.

## Feature family F: score-inspired organ dysfunction features

Do not claim exact SOFA/SAPS unless implemented exactly. Use names like:

```text
respiratory_dysfunction_proxy
cardiovascular_dysfunction_proxy
renal_dysfunction_proxy
hepatic_coag_dysfunction_proxy
inflammation_proxy
comorbidity_burden_proxy
```

Example logic:

```text
cardiovascular_dysfunction_proxy =
    1 if MAP_min < 65 or SBP_min < 90 or shock_index_max > 1.0 or lactate_max >= 2.0

respiratory_dysfunction_proxy =
    1 if SpO2_min < 90 or respiratory_rate_max > 30 or oxygen_support_flag == 1

renal_dysfunction_proxy =
    1 if creatinine_max high or BUN_max high or urine_output low

hepatic_coag_dysfunction_proxy =
    1 if bilirubin high or INR high or platelets low
```

## Implementation tasks

- Create `src/features/time_bins.py`.
- Create `src/features/trajectory.py`.
- Create `src/features/measurement_process.py`.
- Create `src/features/organ_dysfunction.py`.
- Add feature provenance rows for every new feature.
- Add small synthetic fixtures with known timestamps and expected outputs.

## Acceptance criteria

- Every engineered feature has a documented source table, window, aggregation, and leakage risk.
- No feature uses timestamps after ICU admission + 6 hours.
- Feature generation is reproducible from config.
- XGBoost and LightGBM can train on the expanded tabular feature set.

---

# 3. Add a stronger tabular model suite

## Problem

XGBoost is strong, but a portfolio project should compare several credible tabular learners under the same split, preprocessing rules, and threshold protocol.

## Required models

| Model | Role |
|---|---|
| Logistic regression | Interpretable baseline. |
| Random forest | Nonlinear bagging baseline. |
| ExtraTrees | More randomized tree ensemble baseline. |
| XGBoost | Current strong baseline. |
| LightGBM | Fast strong GBDT competitor. |
| CatBoost | Strong with categorical variables and missing values. |
| Explainable Boosting Machine | Interpretable nonlinear GAM-style baseline. |
| Stacked ensemble | Combines complementary models. |

## LightGBM experiments

Run:

```text
LightGBM-current:
  current processed features

LightGBM-expanded:
  expanded temporal/measurement-process features

LightGBM-nan-native:
  preserve NaNs where supported
  add missingness indicators
```

Suggested hyperparameter search:

```yaml
num_leaves: [15, 31, 63, 127]
max_depth: [-1, 3, 4, 5, 6]
learning_rate: [0.01, 0.02, 0.03, 0.05]
n_estimators: [1000, 3000, 5000]
min_child_samples: [20, 50, 100, 200]
subsample: [0.6, 0.8, 1.0]
colsample_bytree: [0.5, 0.7, 0.9, 1.0]
reg_alpha: [0, 0.1, 1, 5]
reg_lambda: [0.1, 1, 5, 10]
class_weight: [null, balanced]
```

## CatBoost experiments

Run:

```text
CatBoost-current:
  current features

CatBoost-categorical:
  preserve categorical features such as gender/admission/careunit if available

CatBoost-constrained:
  monotonic constraints for clinically obvious relationships only
```

Possible monotonic directions:

```text
age: increasing risk
lactate: increasing risk
shock_index: increasing risk
critical_value_count: increasing risk
diagnosis_count: increasing risk
SpO2_mean: decreasing risk as value increases
SBP_min: decreasing risk as value increases
MAP_min: decreasing risk as value increases
```

## XGBoost retuning

Search:

```yaml
max_depth: [2, 3, 4, 5, 6]
min_child_weight: [1, 3, 5, 10, 20]
learning_rate: [0.01, 0.02, 0.03, 0.05, 0.08]
subsample: [0.6, 0.75, 0.9, 1.0]
colsample_bytree: [0.5, 0.7, 0.9, 1.0]
reg_alpha: [0, 0.1, 1, 5, 10]
reg_lambda: [1, 3, 5, 10, 30]
gamma: [0, 0.5, 1, 5, 10]
scale_pos_weight: [1.0, 2.0, 3.8, 5.0]
```

Use early stopping on validation AUC or average precision.

## Stacked ensemble

Base learners:

```text
XGBoost
LightGBM
CatBoost
Logistic regression
ExtraTrees
MAFNet custom model
```

Meta-learner:

```text
L2-regularized logistic regression
```

Required rule:

```text
Train meta-learner only on out-of-fold predictions.
Never train the meta-learner on predictions from models fitted on the same rows.
```

## Implementation tasks

- Create `src/models/tabular/`.
- Create one wrapper per model with a common interface:
  - `fit(train, valid)`
  - `predict_proba(X)`
  - `save(path)`
  - `load(path)`
- Create `src/experiments/run_model_suite.py`.
- Create `src/models/stacking.py`.
- Track all configs and random seeds.

## Acceptance criteria

- Every model is evaluated on the same split.
- Every model outputs validation and test probabilities.
- Every threshold is selected from validation only.
- Stacked ensemble uses out-of-fold predictions.
- Final report includes an ablation table by model family.

---

# 4. Try modern tabular deep learning, but keep it bounded

## Problem

Deep learning on tabular data does not automatically beat gradient-boosted trees. But a strong portfolio project benefits from a fair comparison to modern tabular DL.

## Required experiments

Run these only after the baseline and stronger GBDT models are stable:

| Model | Why |
|---|---|
| TabPFN | Fast modern tabular baseline, useful for modest-sized datasets. |
| TabM | Recent efficient tabular deep learning model. |
| FT-Transformer or ResNet-like MLP | Standard neural tabular baseline. |

## Rules

- Use the same train/validation/test split.
- Do not use test labels for model selection.
- Do not spend more effort here than on the custom MAFNet model.
- Report if boosted trees remain stronger. That is an acceptable and credible result.

## Implementation tasks

- Add `src/models/tabular_dl/`.
- Add config files:
  - `configs/models/tabpfn.yaml`
  - `configs/models/tabm.yaml`
  - `configs/models/ft_transformer.yaml`
- Add a short report comparing DL tabular models against XGBoost/LightGBM/CatBoost.

## Acceptance criteria

- At least one modern tabular DL model is compared fairly.
- The README explains that tabular DL was tested but not assumed to dominate.
- If a DL model underperforms, document this honestly.

---

# 5. Build one custom model: ICU6H-MAFNet

## Problem

The current project uses engineered tabular summaries. That is strong, but it does not fully exploit irregular within-window trajectories, missingness patterns, and measurement intensity.

## Decision

Build:

```text
ICU6H-MAFNet
ICU 6-Hour Missingness-Aware Fusion Network
```

Core idea:

```text
Temporal branch:
  GRU-D-inspired missingness-aware sequence encoder
  small transformer refinement
  attention pooling

Static branch:
  demographics and prior diagnoses MLP

Aggregate branch:
  existing engineered feature MLP

Fusion:
  softmax modality gate
  final classifier

Auxiliary tasks:
  masked temporal value reconstruction
  next-bin measurement forecast
```

## Inputs

```text
x_temporal:     [B, 24, 43]
mask_temporal:  [B, 24, 43]
delta_temporal: [B, 24, 43]
count_temporal: [B, 24, 43]
x_static:       [B, S]
x_aggregate:    [B, A]
```

## Loss

```text
L_total = L_mortality + 0.05 * L_reconstruction + 0.01 * L_mask_forecast
```

Mortality loss:

```text
weighted BCEWithLogitsLoss
pos_weight = sqrt(n_negative_train / n_positive_train)
```

## Training

```text
Stage 1:
  self-supervised temporal pretraining
  20 epochs
  loss = reconstruction + 0.1 * mask forecast

Stage 2:
  supervised mortality fine-tuning
  max 150 epochs
  early stopping on validation average precision
```

## Implementation tasks

- Create `src/data/temporal_dataset.py`.
- Create `src/models/mafnet/model.py`.
- Create `src/training/train_mafnet.py`.
- Create `src/training/losses.py`.
- Create `configs/mafnet.yaml`.
- Add ablations:
  - temporal only
  - temporal + static
  - full fusion
  - no decay
  - no transformer
  - no auxiliary losses
  - no pretraining
  - no gated fusion
  - XGBoost + MAFNet probability ensemble

## Acceptance criteria

- MAFNet trains end-to-end.
- The model never sees post-6-hour data.
- The model exports validation and test logits.
- The model is calibrated using validation logits only.
- Ablation results are reported.
- MAFNet is compared against XGBoost and LightGBM.

See `docs/icu6h_mafnet_architecture.md` for full build details.

---

# 6. Handle class imbalance carefully

## Problem

Positive class proportion is approximately 20.87%. This is meaningful imbalance, but not extreme. Aggressive resampling can distort calibration and create unrealistic training distributions.

## Required comparison

Compare:

```text
No resampling + threshold tuning
Class weighting
scale_pos_weight for GBDT models
sqrt class weighting for neural model
Focal loss as ablation only
Random undersampling as ablation only
SMOTE/ADASYN only as optional ablation
```

## Recommended defaults

For XGBoost/LightGBM/CatBoost:

```text
Try scale_pos_weight/class_weight values:
1.0
2.0
3.8
5.0
```

For MAFNet:

```text
pos_weight = sqrt(n_negative_train / n_positive_train)
```

For your historical cohort, that is approximately:

```text
sqrt(13391 / 3531) ≈ 1.95
```

## Rules

- Validation and test splits must preserve the real event rate.
- Do not resample validation or test.
- If using resampling, apply it only to training.
- Report calibration after class weighting or resampling.

## Implementation tasks

- Add imbalance strategy to experiment configs.
- Compute class weights from training split only.
- Add calibration report per imbalance strategy.

## Acceptance criteria

- The final report states which imbalance strategy was selected and why.
- It includes PR-AUC and calibration, not only recall.

---

# 7. Improve feature provenance and leakage checks

## Problem

Clinical ML portfolios are often weakened by unclear feature provenance and leakage risk. This project should explicitly prove that features are available within the first 6 hours and that test data is not used during fitting, preprocessing, calibration, or threshold selection.

## Required feature dictionary

Create:

```text
docs/feature_dictionary.csv
```

Columns:

```text
feature_name
feature_group
source_table
source_variable
time_window_start
time_window_end
aggregation
clinical_rationale
missingness_handling
is_binary
is_missingness_indicator
leakage_risk
allowed_for_model
notes
```

Example:

```text
lactate_max_0_6h,
lab,
labevents,
lactate,
0h,
6h,
max,
marker of hypoperfusion/severity,
median_impute_plus_measured_flag,
false,
false,
low,
true,
only values charted before ICU intime + 6h
```

## Required leakage tests

Implement:

```text
test_no_post_6h_events_used()
test_threshold_selected_on_validation_only()
test_test_set_never_used_in_preprocessing_fit()
test_patient_ids_do_not_overlap_between_splits()
test_target_not_in_feature_names()
test_discharge_or_death_proxy_columns_removed()
test_imputer_fit_only_on_training_data()
test_scaler_fit_only_on_training_data()
test_feature_generation_small_synthetic_fixture()
test_no_row_level_predictions_exported_to_public_artifacts()
```

## Implementation tasks

- Create `tests/test_temporal_window.py`.
- Create `tests/test_splitting.py`.
- Create `tests/test_preprocessing_leakage.py`.
- Create `tests/test_thresholds.py`.
- Create `tests/fixtures/synthetic_patient_events.csv`.
- Create `docs/leakage_checklist.md`.

## Acceptance criteria

- CI runs tests without needing restricted raw data.
- Synthetic fixtures validate feature generation.
- Test split remains untouched until final evaluation.

---

# 8. Add subgroup and robustness analysis

## Problem

A single aggregate test AUC hides whether the model performs poorly in clinically meaningful subgroups.

## Required subgroup reports

Report by:

```text
age group: <50, 50-64, 65-79, 80+
sex
major diagnosis group
lactate measured vs not measured
high missingness vs low missingness
ICU unit type, if available
mechanical ventilation status, if available
```

For each subgroup:

```text
n
mortality rate
AUC-ROC
Average precision / PR-AUC
Brier score
calibration slope
calibration intercept
recall at screening threshold
precision at screening threshold
specificity at screening threshold
```

## Robustness checks

Add:

```text
seed stability across at least 5 seeds
bootstrap confidence intervals
performance by missingness quartile
performance by measurement intensity quartile
performance under temporal split if available
```

## Implementation tasks

- Create `src/evaluation/subgroups.py`.
- Create `src/evaluation/calibration.py`.
- Create `reports/subgroup_report.md`.
- Add plots:
  - subgroup AUC bar plot
  - subgroup calibration plot
  - missingness quartile performance plot

## Acceptance criteria

- Report includes subgroup sizes so small groups are not overinterpreted.
- Results are documented honestly, including weak subgroups.

---

# 9. Strengthen validation

## Problem

Random stratified splitting is useful but can overstate generalization. Clinical data has temporal, site, unit, and practice-pattern shifts.

## Required validation levels

Minimum:

```text
Stratified train/validation/test split with patient-level separation.
```

Recommended:

```text
5-fold cross-validation on training+validation for model selection.
Locked held-out test set for final reporting.
```

Strong portfolio version:

```text
Temporal split:
  train on earlier admissions
  validate on middle period
  test on later period

or

ICU unit split:
  train on some care units
  test on held-out care unit
```

External validation:

```text
eICU validation if access, feature alignment, and data-use rules allow it.
```

## Implementation tasks

- Create `src/data/splitting.py`.
- Add split strategies:
  - stratified random
  - temporal
  - group/patient-level
  - cross-validation folds
- Save split metadata without restricted row-level data in public repo.
- Include split summary:
  - n
  - positive rate
  - age distribution
  - sex distribution
  - major diagnosis distribution if allowed

## Acceptance criteria

- No patient appears in more than one split.
- All preprocessing fit steps use training only.
- Test set is used once for final model report.

---

# 10. Add clinical ML reporting artifacts

## Problem

To look like a serious research/portfolio project, the repo needs more than code and metrics.

## Required docs

Create:

```text
docs/model_card.md
docs/datasheet.md
docs/feature_dictionary.md or docs/feature_dictionary.csv
docs/leakage_checklist.md
docs/threshold_policy.md
docs/calibration_report.md
docs/tripod_ai_checklist.md
docs/probast_ai_notes.md
reports/final_model_report.md
```

## Model card sections

```text
Model overview
Intended use
Non-intended use
Data source
Cohort definition
Observation window
Target definition
Model families compared
Final selected model
Evaluation protocol
Metrics
Threshold policy
Calibration
Subgroup analysis
Limitations
Ethical and clinical caveats
Data/privacy restrictions
Reproducibility instructions
```

## Threshold policy report sections

```text
Why threshold choice matters
Validation-selected thresholds
Screening threshold
Balanced threshold
High-precision threshold
Test-set performance at fixed thresholds
Clinical/product interpretation
Limitations
```

## Acceptance criteria

- README links to all docs.
- Docs explicitly state that this is not a clinical deployment model.
- Docs do not include restricted data.

---

# 11. Concrete experiment roadmap

## Clean baseline

Goal:

```text
Reproduce the historical pipeline under the leakage-fixed threshold protocol.
```

Outputs:

```text
baseline_metrics.json
baseline_thresholds.json
baseline_report.md
ROC and PR plots
calibration plot
```

Success:

```text
XGBoost result is approximately reproducible.
Thresholds are selected on validation only.
Test report is generated once.
```

## Feature engineering sprint

Goal:

```text
Add temporal bins, trajectory features, measurement-process features, and organ dysfunction proxies.
```

Outputs:

```text
expanded_feature_dictionary.csv
feature_ablation_report.md
```

Success:

```text
Expanded features improve validation PR-AUC, AUC, or useful threshold behavior.
```

## Model suite

Goal:

```text
Run LightGBM, CatBoost, ExtraTrees, EBM, and retuned XGBoost.
```

Outputs:

```text
model_comparison_table.csv
model_comparison_report.md
```

Success:

```text
At least one non-XGBoost strong baseline is implemented and fairly evaluated.
```

## Custom MAFNet

Goal:

```text
Build the missingness-aware temporal fusion model.
```

Outputs:

```text
mafnet_validation_predictions_internal.parquet
mafnet_test_predictions_internal.parquet
mafnet_report.md
mafnet_ablation_report.md
```

Public repo note:

```text
Do not commit row-level prediction files.
Commit only aggregate reports and plots.
```

Success:

```text
MAFNet is competitive with boosted trees or contributes complementary signal to an ensemble.
```

## Ensembling and calibration

Goal:

```text
Combine complementary models and calibrate final probabilities.
```

Outputs:

```text
stacked_ensemble_report.md
calibration_report.md
threshold_policy_report.md
```

Success:

```text
Final model has the best validation performance and clean final test report.
```

## Robustness and documentation

Goal:

```text
Produce portfolio-grade clinical ML documentation.
```

Outputs:

```text
subgroup_report.md
model_card.md
datasheet.md
leakage_checklist.md
```

Success:

```text
The repository reads as a rigorous applied ML project, not a notebook-only experiment.
```

---

# Expected final positioning

Use this project title:

```text
Early ICU Mortality Prediction from First-6-Hour EHR Data:
Leakage-Controlled Feature Engineering, Calibrated Risk Modeling, and Missingness-Aware Temporal Fusion
```

Use this portfolio summary:

```text
This project builds a retrospective early ICU mortality prediction pipeline using only information available in the first 6 hours after ICU admission. It compares interpretable baselines, gradient-boosted trees, stacked ensembles, and a custom missingness-aware temporal fusion model. The project emphasizes leakage prevention, calibration, threshold-policy analysis, feature provenance, subgroup robustness, and honest non-deployment limitations.
```
