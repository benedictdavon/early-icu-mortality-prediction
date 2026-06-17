# Architecture Decision Record: ICU6H-MAFNet

Decision date:

```text
2026-06-17
```

Status:

```text
Accepted for implementation
```

Decision:

```text
Build ICU6H-MAFNet: a GRU-D-inspired missingness-aware temporal encoder with lightweight transformer refinement, attention pooling, static/aggregate feature fusion, and auxiliary self-supervised temporal losses.
```

---

# 1. Context

The project predicts in-hospital mortality using data from the first 6 hours after ICU admission.

The dataset has:

```text
16,922 ICU patients
13,391 survivors
3,531 non-survivors
20.87% mortality rate
```

The current strongest historical model is an XGBoost ensemble with approximately:

```text
AUC-ROC ≈ 0.8465
```

The existing pipeline already includes strong engineered tabular features:

```text
vital summaries
lab summaries
trend features
shock index
SIRS criteria
organ dysfunction indicators
critical value counts
missingness indicators
prior diagnosis burden
```

The custom model should add a new research/portfolio contribution rather than merely reimplementing XGBoost in neural-network form.

---

# 2. Core modeling problem

The first 6 hours of ICU data are:

```text
short-window
sparse
irregularly measured
clinically heterogeneous
missing-not-at-random
mixed static + temporal + engineered tabular data
```

The model must use:

```text
1. values
2. whether values were measured
3. how recently values were measured
4. how often values were measured
5. static patient descriptors
6. existing aggregate clinical features
```

A plain tabular MLP cannot use temporal ordering well.

A standard GRU/LSTM can use ordering but does not explicitly model missingness and time gaps.

A full transformer over raw event tokens is more complex, more sensitive to implementation details, and less necessary for a 24-step sequence.

Therefore, the selected model combines:

```text
GRU-D-style missingness handling
small recurrent temporal encoder
small attention refinement layer
static/aggregate fusion
auxiliary self-supervised temporal learning
```

---

# 3. Decision summary

Build:

```text
ICU6H-MAFNet
```

With these modules:

```text
MissingnessAwareTemporalEncoder:
  GRU-D-inspired input decay
  hidden-state decay
  mask and delta inputs
  measurement count inputs
  GRUCell backbone
  1-layer transformer encoder
  attention pooling

StaticEncoder:
  small MLP over demographics and prior diagnosis features

AggregateEncoder:
  MLP over existing engineered 6-hour summary features

GatedFusion:
  modality weighting over temporal/static/aggregate representations

MortalityClassifier:
  MLP classifier outputting one logit

Auxiliary heads:
  masked value reconstruction
  next-bin measurement forecast
```

---

# 4. Why not just use XGBoost?

XGBoost is a strong baseline and may remain the best single model.

However, a custom model is still justified because XGBoost on aggregate features does not directly model:

```text
within-window temporal ordering
time gaps between observations
explicit decayed carry-forward dynamics
measurement intensity over time
temporal attention
self-supervised reconstruction of sparse clinical trajectories
```

Expected portfolio value:

```text
Even if XGBoost remains strongest, MAFNet can:
  show advanced model design,
  test whether temporal/missingness signals add value,
  contribute complementary probabilities to an ensemble,
  provide temporal attention visualizations,
  strengthen the research narrative.
```

---

# 5. Why GRU-D-inspired temporal encoding?

## Reason 1: ICU time series are irregular and sparse

In ICU data, missingness is not just noise. A lab may be missing because it was clinically unnecessary, unavailable, or not ordered. A lab may be measured repeatedly because clinicians are concerned.

Therefore the model should ingest:

```text
value
mask
time since last observation
measurement count
```

The GRU-D family of ideas is appropriate because it uses missingness masks and time intervals to handle clinical time series where observations are irregularly sampled.

## Reason 2: The sequence is short

The model only sees 6 hours. With 15-minute bins:

```text
T = 24
```

This is well suited to a recurrent encoder. A recurrent model over 24 steps is cheap, stable, and easy to debug.

## Reason 3: Learnable decay is clinically sensible

For missing values, the model should not assume that the last observed value remains fully valid forever. A blood pressure 15 minutes ago is more informative than one 5 hours ago. Learnable decay lets the model decide how quickly stale values lose relevance by channel.

Example:

```text
SpO2 and heart rate may decay quickly.
Age does not belong in the temporal branch.
Certain labs may decay slowly.
Lactate may remain informative for longer but worsening trend matters.
```

## Reason 4: Deltas are more expressive than imputation alone

Median imputation erases time-since-measured information. KNN or iterative imputation may introduce complexity and still hide the care-process signal. MAFNet keeps missingness explicit.

---

# 6. Why add measurement count?

The model receives:

```text
count[t, v]
```

because measurement frequency is itself informative in ICU data.

Examples:

```text
frequent blood gases may indicate respiratory instability
repeated lactate measurements may indicate shock/sepsis concern
frequent vitals may reflect higher acuity
no lactate measurement may indicate lower suspicion or missing data
```

Measurement count complements the binary mask:

```text
mask = whether measured
count = how intensely measured
```

---

# 7. Why use 15-minute bins?

The selected bin width is:

```text
15 minutes
```

This creates:

```text
24 time steps over 6 hours
```

This is a good compromise:

| Option | Problem |
|---|---|
| 5-minute bins | Too sparse; many empty bins; longer sequence. |
| 15-minute bins | Good balance of temporal resolution and sparsity. |
| 30-minute bins | More stable but may blur short hypotension/hypoxemia events. |
| 1-hour bins | Good for tabular features but too coarse for temporal model. |

15-minute bins preserve clinically meaningful short-term instability without making the sequence long.

For boosted tabular models, hourly aggregates are still useful. For the neural temporal model, 15-minute bins are preferable.

---

# 8. Why add a transformer layer after GRU-D?

The recurrent encoder processes information sequentially. A small transformer encoder layer then re-contextualizes all 24 hidden states.

This helps the model compare time steps directly.

Examples:

```text
early hypotension followed by recovery
late deterioration after stable first hours
repeated abnormal bins
changes in measurement intensity
```

The transformer is intentionally small:

```text
layers = 1
heads = 4
hidden_dim = 128
```

This avoids overbuilding a large attention model for a modest dataset.

---

# 9. Why attention pooling?

A final hidden state alone can overemphasize the last bin. Mortality risk may be signaled by:

```text
early severe instability
late deterioration
persistent abnormality
single extreme event
```

Attention pooling lets the model learn which time bins matter.

The model also includes:

```text
last hidden state
max pooled hidden state
attention pooled hidden state
```

This is more robust than attention pooling alone.

---

# 10. Why include static and aggregate branches?

## Static branch

Static patient factors are known important mortality predictors:

```text
age
diagnosis burden
metastatic cancer
prior respiratory/circulatory diagnoses
BMI
sex
```

These do not belong inside the temporal sequence.

## Aggregate branch

Your engineered features are already strong. Throwing them away would be a mistake.

Aggregate features include clinically meaningful summaries:

```text
shock index
SIRS count
critical value count
distance from normal
trend features
missingness indicators
organ dysfunction proxies
```

The model should learn from both:

```text
raw-ish temporal sequence
engineered clinical summaries
```

This makes MAFNet a fusion model rather than a pure sequence model.

---

# 11. Why gated fusion?

Simple concatenation forces the classifier to learn modality weighting implicitly.

Gated fusion explicitly learns patient-specific weights:

```text
temporal branch weight
static branch weight
aggregate branch weight
```

This is useful because different patients may have different information availability.

Examples:

```text
A patient with many temporal measurements may rely more on temporal branch.
A patient with sparse labs may rely more on static and aggregate branches.
A patient with strong prior diagnosis burden may rely more on static branch.
```

The design still concatenates the raw branch embeddings with the gated representation:

```text
z_fused = concat(z_gate, z_cat)
```

This avoids making the model overly dependent on the gate.

---

# 12. Why auxiliary reconstruction?

The mortality label is sparse and noisy. Auxiliary reconstruction forces the temporal encoder to learn relationships among variables and time.

Training procedure:

```text
randomly hide 15% of observed temporal values
predict the hidden standardized values
compute SmoothL1 loss only on hidden observed entries
```

Why this helps:

```text
encourages physiologic representation learning
uses all observed temporal data, not only the mortality label
regularizes the temporal encoder
improves stability on sparse data
```

SmoothL1 is selected because clinical variables can still contain heavy-tailed values after capping and standardization.

---

# 13. Why next-bin measurement forecast?

Measurement itself is part of the care process.

The model predicts:

```text
which variables will be measured in the next 15-minute bin
```

This auxiliary task teaches the model about monitoring intensity.

Examples:

```text
frequent lactate measurement may reflect shock concern
frequent blood gas measurement may reflect respiratory instability
frequent vitals may reflect acuity
```

The loss weight is intentionally small:

```text
lambda_mask = 0.01
```

This prevents the model from focusing too much on predicting clinician behavior instead of mortality.

---

# 14. Why weighted BCE instead of focal loss?

The main mortality loss is:

```text
BCEWithLogitsLoss(pos_weight=sqrt(n_negative_train / n_positive_train))
```

Reasons:

```text
stable
standard
easy to calibrate after training
fewer hyperparameters than focal loss
works well with threshold tuning
```

Full class ratio is around:

```text
n_negative / n_positive ≈ 3.79
```

But using the full ratio can overemphasize positives and hurt calibration. The square-root ratio is a moderate compromise:

```text
sqrt(3.79) ≈ 1.95
```

Focal loss is reserved for ablation only because it adds extra hyperparameters and may harm probability calibration.

---

# 15. Why Platt calibration?

Weighted losses can improve recall but distort probability calibration. Since this project reports risk probabilities, calibration is required.

Use:

```text
Platt scaling on validation logits only
```

Fit:

```text
p_calibrated = sigmoid(a * logit + b)
```

Then apply the fitted calibration model to test logits.

Do not fit calibration on test data.

---

# 16. Alternatives considered

## Alternative A: pure XGBoost only

Rejected as custom centerpiece.

Reason:

```text
Strong baseline, but not a custom neural architecture and does not fully exploit temporal/missingness dynamics.
```

Use it as:

```text
main benchmark
ensemble partner
SHAP interpretability model
```

## Alternative B: plain MLP over aggregate features

Rejected.

Reason:

```text
Ignores temporal sequence and missingness timing.
Likely weaker than GBDT on tabular features.
Weak portfolio novelty.
```

## Alternative C: standard LSTM/GRU over imputed bins

Rejected as final architecture.

Reason:

```text
Does not explicitly model missingness.
Depends too heavily on imputation quality.
Loses measurement-process signal.
```

But use it as an ablation:

```text
NoDecay
```

## Alternative D: full event-level transformer

Deferred.

Reason:

```text
More complex.
Requires careful event tokenization.
Potentially overkill for 24-step 6-hour window.
Harder to debug.
Higher risk of underperforming due to data size.
```

Potential future work:

```text
STraTS-style triplet transformer
```

## Alternative E: TCN over hourly bins

Rejected as main model.

Reason:

```text
Easier to implement but less directly suited to irregular missingness and time gaps.
```

Potential ablation:

```text
TCN temporal encoder
```

## Alternative F: TabNet/FT-Transformer only

Rejected as custom centerpiece.

Reason:

```text
Modern tabular DL is useful as a comparison, but the main scientific gap is irregular temporal/missingness modeling.
```

---

# 17. Risks and mitigations

## Risk 1: MAFNet underperforms XGBoost

Mitigation:

```text
Frame MAFNet as a temporal representation experiment.
Use MAFNet probabilities in ensemble with XGBoost/LightGBM.
Report ablations and complementary signal.
```

Success does not require MAFNet to be the best single model.

## Risk 2: Model overfits

Mitigation:

```text
small hidden size
dropout
weight decay
early stopping
auxiliary losses
bootstrap CIs
seed stability report
ablation study
```

## Risk 3: Auxiliary losses dominate mortality learning

Mitigation:

```text
lambda_recon = 0.05
lambda_mask = 0.01
monitor mortality validation AP
ablate NoAux
```

## Risk 4: Calibration is poor

Mitigation:

```text
validation-only Platt scaling
Brier score
calibration curve
ECE
calibration slope/intercept
```

## Risk 5: Temporal tensor generation contains leakage

Mitigation:

```text
strict timestamp filtering
synthetic fixture tests
feature provenance
test_no_post_6h_events_used
```

---

# 18. Final decision

Implement ICU6H-MAFNet exactly as specified in:

```text
03_icu6h_mafnet_model_architecture.md
```

Primary comparisons:

```text
XGBoost
LightGBM
CatBoost
MAFNet
XGBoost + MAFNet
Stacked ensemble
```

Primary evaluation metrics:

```text
AUC-ROC
average precision / PR-AUC
Brier score
calibration
threshold-policy metrics
subgroup robustness
```

Expected portfolio story:

```text
This project combines strong boosted-tree baselines with a custom missingness-aware temporal fusion model designed for sparse first-6-hour ICU data. The final system emphasizes leakage control, calibration, threshold policy, interpretability, and honest validation.
```
