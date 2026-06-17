# ICU Mortality Project: Implementation Roadmap and Coding Checklist

This file converts the improvement strategy into an implementation plan for a coding agent.

The coding agent should treat this as the working roadmap. Do not implement roadmap sections out of order unless a prerequisite is already complete.

---

# Global implementation rules

## Data safety

- Do not commit raw MIMIC data.
- Do not commit processed patient-level CSVs.
- Do not commit row-level predictions.
- Do not commit model artifacts containing restricted metadata.
- Do not commit subject identifiers, stay identifiers, or admission identifiers in public artifacts.
- Synthetic fixtures are allowed.
- Aggregate reports and plots are allowed.

## Leakage rules

- All feature values must be available within the first 6 hours after ICU admission.
- Thresholds must be selected on validation data only.
- Calibration must be fit on validation data only.
- Test data must be used only for final reporting.
- Imputers, scalers, encoders, and feature selectors must be fit on training data only.
- Cross-validation meta-models must use out-of-fold predictions only.

## Coding style

Prefer modular files:

```text
src/
  data/
  features/
  models/
  training/
  evaluation/
  reports/
configs/
tests/
docs/
reports/
```

Every experiment should have:

```text
config file
random seed
input feature set name
split name
model name
validation metrics
test metrics
threshold policy
calibration status
```

---

# Repository structure and configuration

## Goal

Create the project structure needed for reproducible experiments.

## Files to create or update

```text
src/config.py
src/utils/io.py
src/utils/seed.py
src/utils/logging.py
src/data/schema.py
configs/base.yaml
configs/paths.example.yaml
docs/reproducibility.md
```

## Implementation checklist

- [x] Create a central config loader.
- [x] Add project root discovery.
- [x] Add controlled random seeding for:
  - [x] Python `random`
  - [x] NumPy
  - [x] PyTorch
  - [x] XGBoost/LightGBM/CatBoost where applicable
- [x] Add output directory naming:
  - [x] date
  - [x] experiment name
  - [x] model name
  - [x] split name
  - [x] seed
- [x] Add a `paths.example.yaml` file with placeholders only.
- [x] Ensure actual local data paths are ignored by git.
- [x] Add `.gitignore` entries for:
  - [x] raw data
  - [x] processed patient-level data
  - [x] model checkpoints
  - [x] prediction files
  - [x] local configs containing restricted paths

## Done criteria

- The repo can load config without requiring restricted data.
- Running a dry-run experiment creates a structured output directory.
- No restricted paths or files are committed.

---

# Leakage-safe split and baseline rerun

## Goal

Re-run the existing baseline with fixed threshold selection.

## Files to create or update

```text
src/data/splitting.py
src/evaluation/metrics.py
src/evaluation/thresholds.py
src/evaluation/bootstrap.py
src/evaluation/report_tables.py
src/experiments/run_baseline.py
tests/test_splitting.py
tests/test_thresholds.py
configs/baseline_xgboost.yaml
```

## Required implementation details

### Split function

Implement:

```python
make_train_valid_test_split(
    labels,
    patient_ids=None,
    test_size=0.20,
    valid_size=0.20,
    stratify=True,
    group_by_patient=True,
    random_state=42,
)
```

Behavior:

- If `patient_ids` is provided, no patient can appear in more than one split.
- Stratification should preserve mortality rate as closely as possible.
- Return row indices or internal keys; do not write identifiers to public reports.

### Threshold function

Implement:

```python
select_thresholds(y_valid, p_valid) -> dict
```

Return:

```python
{
    "screening": {
        "threshold": float,
        "rule": "highest threshold with recall >= 0.85"
    },
    "balanced_f1": {
        "threshold": float,
        "rule": "max validation F1"
    },
    "high_precision": {
        "threshold": float,
        "rule": "highest threshold with precision >= 0.65"
    }
}
```

### Metric function

Implement:

```python
compute_binary_metrics(y_true, p_pred, threshold) -> dict
```

Return:

```text
auc_roc
average_precision
brier_score
accuracy
precision
recall
f1
specificity
npv
tn
fp
fn
tp
threshold
```

### Bootstrap confidence intervals

Implement:

```python
bootstrap_metric_ci(y_true, p_pred, metric_fn, n_boot=1000, seed=42)
```

Rules:

- Sample rows with replacement.
- Skip bootstrap samples containing only one class for AUC.
- Report 2.5th and 97.5th percentiles.

## Implementation checklist

- [x] Build stratified split.
- [x] Verify no overlap between split IDs.
- [x] Train baseline logistic regression.
- [x] Train baseline random forest.
- [x] Train baseline XGBoost.
- [x] Select thresholds on validation.
- [x] Apply fixed thresholds to test.
- [x] Compute all metrics.
- [x] Generate ROC curve.
- [x] Generate PR curve.
- [x] Generate calibration plot.
- [x] Save aggregate JSON/CSV reports.
- [x] Do not save row-level predictions to public folder.

## Tests

- [x] `test_no_patient_overlap_between_splits`
- [x] `test_threshold_selection_accepts_only_validation_inputs`
- [x] `test_thresholds_are_not_recomputed_on_test`
- [x] `test_metrics_confusion_matrix_consistency`
- [x] `test_bootstrap_ci_shape`

## Done criteria

- Baseline report generated.
- Historical XGBoost performance is approximately reproduced.
- Threshold leakage is fixed.
- All leakage-safe split and baseline tests pass.

---

# Feature provenance and leakage fixtures

## Goal

Create transparent documentation for every feature and automated tests proving that feature generation respects the first-6-hour window.

## Files to create or update

```text
src/features/registry.py
src/features/provenance.py
docs/feature_dictionary.csv
docs/leakage_checklist.md
tests/fixtures/synthetic_events.csv
tests/fixtures/synthetic_cohort.csv
tests/test_temporal_window.py
tests/test_feature_provenance.py
```

## Feature dictionary schema

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

## Implementation checklist

- [x] Create a feature registry object or table.
- [x] Register existing demographic features.
- [x] Register existing vital features.
- [x] Register existing lab features.
- [x] Register prior diagnosis features.
- [x] Register derived clinical features.
- [x] Add leakage-risk labels:
  - [x] low
  - [x] medium
  - [x] high
  - [x] excluded
- [x] Write feature dictionary export function.
- [x] Create synthetic patient with events before 6h and after 6h.
- [x] Test that post-6h events are excluded.
- [x] Test that boundary timestamp at exactly 6h follows the chosen rule consistently.
- [x] Test that death/discharge proxy columns are excluded.

## Done criteria

- Feature dictionary exists and is machine-generated or automatically checked.
- Synthetic fixture tests pass.
- No feature lacks provenance metadata.

---

# Expanded feature engineering

## Goal

Add stronger first-6-hour feature signal for tabular models and MAFNet aggregate branch.

## Files to create or update

```text
src/features/time_bins.py
src/features/trajectory.py
src/features/measurement_process.py
src/features/organ_dysfunction.py
src/features/interactions.py
src/features/build_features.py
configs/features_expanded.yaml
tests/test_time_bins.py
tests/test_trajectory_features.py
tests/test_measurement_process.py
tests/test_organ_dysfunction.py
```

## Required feature additions

### Time-bin features

- [x] Hourly features for tabular models:
  - [x] mean
  - [x] min
  - [x] max
  - [x] last
  - [x] observed flag
  - [x] measurement count
- [x] 15-minute tensor features for MAFNet:
  - [x] last
  - [x] min
  - [x] max
  - [x] mask
  - [x] delta
  - [x] count

### Trajectory features

- [x] first observed value
- [x] last observed value
- [x] last minus first
- [x] percent change
- [x] first 2h mean
- [x] last 2h mean
- [x] last 2h minus first 2h
- [x] slope over observed values
- [x] deterioration flag
- [x] recovery flag

### Instability features

- [x] range
- [x] standard deviation
- [x] coefficient of variation
- [x] number of abnormal bins
- [x] longest consecutive abnormal run
- [x] worst recent value

### Measurement-process features

- [x] measured flag
- [x] measurement count
- [x] time to first measurement
- [x] time since last measurement at 6h
- [x] panel missing count
- [x] total lab count
- [x] total vital count
- [x] total chart event count if available

### Organ dysfunction proxy features

- [x] cardiovascular dysfunction proxy
- [x] respiratory dysfunction proxy
- [x] renal dysfunction proxy
- [x] hepatic/coagulation dysfunction proxy
- [x] inflammation/sepsis proxy
- [x] comorbidity burden proxy

### Interaction features

- [x] age × diagnosis burden
- [x] age × shock index
- [x] lactate × hypotension
- [x] RR × SpO2
- [x] creatinine × urine output if urine output exists
- [x] bilirubin × INR
- [x] platelets × INR
- [x] SIRS × lactate
- [x] critical value count × missing lab count

## Done criteria

- Expanded features can be generated from the same cohort.
- Feature dictionary includes all new features.
- XGBoost can train on expanded features.
- Feature ablation report shows which feature groups help.

---

# Stronger tabular model suite

## Goal

Establish a stronger non-neural and tabular-DL comparison set.

## Files to create or update

```text
src/models/base.py
src/models/tabular/logistic.py
src/models/tabular/random_forest.py
src/models/tabular/extra_trees.py
src/models/tabular/xgboost_model.py
src/models/tabular/lightgbm_model.py
src/models/tabular/catboost_model.py
src/models/tabular/ebm_model.py
src/experiments/run_model_suite.py
configs/models/logistic.yaml
configs/models/random_forest.yaml
configs/models/xgboost.yaml
configs/models/lightgbm.yaml
configs/models/catboost.yaml
configs/models/ebm.yaml
```

## Common model interface

Implement:

```python
class BaseRiskModel:
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        ...

    def predict_proba(self, X):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

## Implementation checklist

- [x] Wrap logistic regression.
- [x] Wrap random forest.
- [x] Wrap ExtraTrees.
- [x] Wrap XGBoost.
- [x] Wrap LightGBM.
- [x] Wrap CatBoost.
- [x] Add EBM if dependency is available.
- [x] Add hyperparameter search with Optuna or randomized search.
- [x] Use validation set for early stopping where applicable.
- [x] Compute identical metrics for every model.
- [x] Save aggregate comparison table.

## Done criteria

- Model suite runs from one command.
- All models use the same split.
- Output table includes AUC, AP, Brier, threshold metrics, and CIs.

---

# MAFNet temporal tensors

## Goal

Create the data tensors needed by ICU6H-MAFNet.

## Files to create or update

```text
src/data/temporal_dataset.py
src/features/temporal_tensor_builder.py
src/features/temporal_channels.py
configs/icu6h_mafnet.yaml
tests/test_temporal_tensor_builder.py
```

## Tensor specification

```text
x_temporal:     [B, 24, 43]
mask_temporal:  [B, 24, 43]
delta_temporal: [B, 24, 43]
count_temporal: [B, 24, 43]
x_static:       [B, S]
x_aggregate:    [B, A]
y:              [B]
```

## Temporal channel list

### Vitals: 21 channels

```text
heart_rate_last
heart_rate_min
heart_rate_max
respiratory_rate_last
respiratory_rate_min
respiratory_rate_max
sbp_last
sbp_min
sbp_max
dbp_last
dbp_min
dbp_max
map_last
map_min
map_max
temperature_last
temperature_min
temperature_max
spo2_last
spo2_min
spo2_max
```

### Labs: 17 channels

```text
wbc_last
hemoglobin_last
hematocrit_last
platelets_last
sodium_last
potassium_last
creatinine_last
bun_last
glucose_last
bilirubin_last
alkaline_phosphatase_last
alt_last
ast_last
inr_last
lactate_last
bicarbonate_last
anion_gap_last
```

### Derived: 5 channels

```text
shock_index_bin
hypotension_flag_bin
hypoxemia_flag_bin
sirs_count_bin
critical_value_count_bin
```

## Implementation checklist

- [ ] Build 15-minute bins.
- [ ] Compute per-bin values.
- [ ] Compute masks.
- [ ] Compute deltas per channel.
- [ ] Compute counts per channel.
- [ ] Fit temporal normalizer on training observed values only.
- [ ] Apply same normalizer to validation/test.
- [ ] Clip standardized values to [-5, 5].
- [ ] Store missing standardized values as 0.0.
- [ ] Log-transform deltas and counts.
- [ ] Add tests for a tiny synthetic patient with known expected tensor values.

## Done criteria

- Dataset returns PyTorch-ready tensors.
- Validation/test normalizers are not fit on validation/test.
- Unit tests verify mask, delta, and count logic.

---

# ICU6H-MAFNet implementation

## Goal

Build the custom missingness-aware temporal fusion model.

## Files to create or update

```text
src/models/icu6h_mafnet.py
src/training/losses.py
src/training/train_mafnet.py
src/training/callbacks.py
src/evaluation/mafnet_eval.py
configs/icu6h_mafnet.yaml
tests/test_mafnet_forward.py
tests/test_mafnet_loss.py
```

## Architecture modules

- [ ] `MissingnessAwareTemporalEncoder`
- [ ] `StaticEncoder`
- [ ] `AggregateEncoder`
- [ ] `GatedFusion`
- [ ] `MortalityClassifier`
- [ ] `TemporalReconstructionHead`
- [ ] `MeasurementForecastHead`

## Forward pass output

Return:

```python
{
    "mortality_logit": mortality_logit,
    "x_recon": x_recon,
    "mask_next_logit": mask_next_logit,
    "gate_weights": gate_weights,
    "temporal_attention": temporal_attention
}
```

## Training stages

### Stage 1: pretraining

- [ ] Train temporal encoder and auxiliary heads.
- [ ] Randomly hide 15% of observed values.
- [ ] Use reconstruction + mask forecast loss.
- [ ] Use training split only.
- [ ] Save checkpoint locally, not in public repo.

### Stage 2: supervised fine-tuning

- [ ] Load pretrained temporal encoder.
- [ ] Train full model.
- [ ] Use weighted BCE for mortality.
- [ ] Keep auxiliary losses active with small weights.
- [ ] Early stop on validation average precision.
- [ ] Save aggregate training curves.

## Done criteria

- One batch forward pass works.
- One epoch over synthetic data works.
- Full training runs without NaNs.
- Validation predictions are produced.
- Test predictions are produced only for final report.

---

# Calibration and threshold policy

## Goal

Turn model logits into usable calibrated risk scores and validation-selected threshold policies.

## Files to create or update

```text
src/evaluation/calibration.py
src/evaluation/threshold_policy.py
src/evaluation/plots.py
reports/calibration_report.md
reports/threshold_policy_report.md
tests/test_calibration.py
```

## Implementation checklist

- [ ] Fit Platt scaling on validation logits only.
- [ ] Optionally add isotonic calibration as ablation.
- [ ] Apply calibration to validation and test probabilities.
- [x] Select three thresholds using validation probabilities:
  - [x] screening
  - [x] balanced F1
  - [x] high precision
- [x] Apply fixed thresholds to test.
- [ ] Plot calibration curve.
- [x] Compute Brier score.
- [x] Compute ECE.
- [x] Save threshold report.

## Done criteria

- Calibration code does not access test labels during fitting.
- Threshold code does not access test labels during selection.
- Calibration and threshold tests pass.

---

# Ablations and ensembling

## Goal

Prove which parts of MAFNet matter and test whether MAFNet adds complementary signal to boosted trees.

## Files to create or update

```text
src/experiments/run_mafnet_ablations.py
src/models/stacking.py
src/experiments/run_stacking.py
reports/mafnet_ablation_report.md
reports/ensemble_report.md
tests/test_stacking_oof.py
```

## Required MAFNet ablations

- [ ] `MAFNet-T`: temporal branch only.
- [ ] `MAFNet-T+S`: temporal + static.
- [ ] `MAFNet-T+S+A`: full model.
- [ ] `NoDecay`: replace decay logic with zero-imputed GRU.
- [ ] `NoTransformer`: remove transformer encoder layer.
- [ ] `NoAux`: remove reconstruction and mask forecast losses.
- [ ] `NoGate`: replace gated fusion with concatenation.
- [ ] `NoPretrain`: train from scratch.
- [ ] `XGB+MAFNet`: average calibrated probabilities.
- [ ] `Stacked`: OOF logistic meta-learner over XGBoost, LightGBM, CatBoost, MAFNet.

## Done criteria

- Ablation table generated.
- Best single model and best ensemble are identified.
- Stacking uses out-of-fold predictions only.
- Test set remains locked until final evaluation.

---

# Subgroup and robustness reports

## Goal

Evaluate whether performance is stable across clinically meaningful subgroups and random seeds.

## Files to create or update

```text
src/evaluation/subgroups.py
src/evaluation/robustness.py
reports/subgroup_report.md
reports/seed_stability_report.md
tests/test_subgroups.py
```

## Subgroups

- [ ] Age: `<50`, `50-64`, `65-79`, `80+`
- [ ] Sex
- [ ] Major diagnosis group
- [ ] Lactate measured vs not measured
- [ ] High vs low missingness
- [ ] Measurement intensity quartile
- [ ] ICU unit type if available
- [ ] Ventilation status if available

## Metrics per subgroup

- [ ] n
- [ ] mortality rate
- [ ] AUC-ROC
- [ ] average precision
- [ ] Brier score
- [ ] calibration slope
- [ ] calibration intercept
- [ ] recall at screening threshold
- [ ] precision at screening threshold
- [ ] specificity at screening threshold

## Done criteria

- Small subgroups are flagged.
- Report includes cautious interpretation.
- No subgroup claims are overstated.

---

# Final documentation and portfolio polish

## Goal

Make the repository understandable to recruiters, engineers, and ML researchers without exposing restricted data.

## Files to create or update

```text
README.md
docs/model_card.md
docs/datasheet.md
docs/feature_dictionary.md
docs/leakage_checklist.md
docs/threshold_policy.md
docs/tripod_ai_checklist.md
docs/probast_ai_notes.md
reports/final_model_report.md
reports/final_plots/
```

## README structure

- [x] Project summary.
- [x] Problem framing.
- [x] Data privacy note.
- [x] Cohort definition.
- [x] First-6-hour observation window.
- [x] Models compared.
- [ ] Custom MAFNet architecture summary.
- [x] Evaluation protocol.
- [x] Main results.
- [x] Calibration and threshold policy.
- [ ] Subgroup analysis.
- [x] Leakage prevention.
- [x] Reproducibility instructions.
- [x] Limitations.
- [x] Non-clinical-deployment disclaimer.

## Final report plots

- [ ] ROC curve.
- [ ] PR curve.
- [ ] Calibration curve.
- [ ] Threshold tradeoff curve.
- [ ] Confusion matrix at three thresholds.
- [ ] SHAP feature importance for best tree model.
- [ ] MAFNet temporal attention plot using synthetic/deidentified example only.
- [ ] Subgroup performance plot.
- [ ] Ablation bar chart.

## Done criteria

- Final public repo contains only safe artifacts.
- Documentation explains the project clearly.
- Results are not overclaimed.
- The project reads as an applied clinical ML engineering project.

---

# Suggested implementation order summary

```text
0. Repo/config safety
1. Split + baseline metrics + threshold fix
2. Feature provenance + leakage tests
3. Expanded feature engineering
4. Stronger tabular model suite
5. MAFNet tensor builder
6. MAFNet model implementation
7. Calibration + threshold policy
8. Ablations + ensembling
9. Subgroup + robustness analysis
10. Final docs and reports
```
