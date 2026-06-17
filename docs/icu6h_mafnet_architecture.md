# ICU6H-MAFNet Model Architecture Specification

This file contains the detailed architecture for the custom model.

Model name:

```text
ICU6H-MAFNet
ICU 6-Hour Missingness-Aware Fusion Network
```

Purpose:

```text
Predict in-hospital mortality using only information available during the first 6 hours after ICU admission.
```

Primary design:

```text
Temporal branch:
  GRU-D-inspired missingness-aware recurrent encoder
  1-layer transformer refinement
  attention pooling

Static branch:
  MLP over demographics and prior diagnosis features

Aggregate branch:
  MLP over engineered 6-hour tabular features

Fusion:
  gated multimodal fusion

Auxiliary tasks:
  masked temporal value reconstruction
  next-bin measurement forecast
```

---

# 1. Input specification

## 1.1 Temporal input

Observation window:

```text
first 6 hours after ICU admission
```

Bin width:

```text
15 minutes
```

Number of time steps:

```text
T = 6 hours × 4 bins/hour = 24
```

Temporal tensors:

```text
x_temporal:     [B, 24, 43]
mask_temporal:  [B, 24, 43]
delta_temporal: [B, 24, 43]
count_temporal: [B, 24, 43]
```

Where:

```text
B = batch size
T = 24 time bins
V = 43 temporal channels
```

## 1.2 Static input

```text
x_static: [B, S]
```

Recommended static features:

```text
age
very_elderly_flag
gender_one_hot
bmi
prior_diagnosis_count
respiratory_diagnosis_count
circulatory_diagnosis_count
nervous_sensory_diagnosis_count
metastatic_cancer_flag
has_prior_diagnoses_flag
```

Do not include:

```text
subject_id
hadm_id
stay_id
death time
discharge time
length of stay
post-6-hour information
```

## 1.3 Aggregate input

```text
x_aggregate: [B, A]
```

Recommended aggregate features:

```text
6-hour vital summaries
6-hour lab summaries
trend features
delta features
percent change features
shock index summaries
SIRS features
organ dysfunction indicators
critical value count
distance-from-normal features
abnormal vital/lab flags
missingness indicators
measurement count features
measurement recency features
```

Expected dimension:

```text
A ≈ 100-160
```

---

# 2. Temporal channel specification

Total channels:

```text
V = 43
```

## 2.1 Vitals: 21 channels

For each vital, create last/min/max inside each 15-minute bin.

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

## 2.2 Labs: 17 channels

For each lab, use last observed value inside each 15-minute bin.

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

## 2.3 Dynamic derived features: 5 channels

```text
shock_index_bin
hypotension_flag_bin
hypoxemia_flag_bin
sirs_count_bin
critical_value_count_bin
```

Definitions:

```text
shock_index_bin = heart_rate_last / max(sbp_last, epsilon)

hypotension_flag_bin =
    1 if sbp_min < 90 or map_min < 65
    0 otherwise

hypoxemia_flag_bin =
    1 if spo2_min < 90
    0 otherwise

sirs_count_bin =
    count of:
      temperature > 38 or temperature < 36
      heart_rate > 90
      respiratory_rate > 20
      wbc > 12 or wbc < 4 if WBC available

critical_value_count_bin =
    count of clinically abnormal vital/lab flags available in the bin
```

Use clinically plausible thresholds consistently with the existing project.

---

# 3. Temporal preprocessing

## 3.1 Value tensor `x`

For each patient and each channel:

1. Use only measurements with timestamps:

```text
ICU intime <= measurement_time <= ICU intime + 6 hours
```

2. Assign each measurement to a 15-minute bin.

3. Compute bin statistic:

```text
vital_last: last value in bin
vital_min: minimum value in bin
vital_max: maximum value in bin
lab_last: last value in bin
derived: computed from available bin values
```

4. Standardize using observed training values only:

```text
x_z = (x_raw - train_mean[channel]) / train_std[channel]
```

5. Clip standardized values:

```text
x_z = clip(x_z, -5, 5)
```

6. Set missing entries to:

```text
0.0
```

Because features are standardized, `0.0` is the training mean. The mask tells the model whether the value was observed.

## 3.2 Mask tensor `mask`

```text
mask[t, v] = 1 if channel v was observed or computable in bin t
mask[t, v] = 0 otherwise
```

Derived channel mask rules:

```text
shock_index_bin:
  1 if heart rate and SBP are available in bin

hypotension_flag_bin:
  1 if SBP or MAP is available in bin

hypoxemia_flag_bin:
  1 if SpO2 is available in bin

sirs_count_bin:
  1 if at least one of HR, RR, temperature, or WBC is available

critical_value_count_bin:
  1 if at least one checked variable is available
```

## 3.3 Delta tensor `delta`

Definition:

```text
delta[t, v] = hours since channel v was last observed
```

Rules:

```text
if channel v observed in current bin:
    delta[t, v] = 0
elif channel v was previously observed:
    delta[t, v] = time since previous observation in hours
else:
    delta[t, v] = 6.0
```

Transform:

```text
delta_log = log1p(delta) / log1p(6.0)
```

## 3.4 Count tensor `count`

Definition:

```text
count[t, v] = number of raw observations for channel v in bin t
```

Transform:

```text
count_log = log1p(min(count, 10)) / log1p(10)
```

---

# 4. Model modules

## 4.1 High-level architecture

```text
Temporal inputs
  -> MissingnessAwareTemporalEncoder
  -> h_temp [B, 128]

Static inputs
  -> StaticEncoder
  -> h_static [B, 64]

Aggregate inputs
  -> AggregateEncoder
  -> h_agg [B, 96]

h_temp, h_static, h_agg
  -> GatedFusion
  -> z_fused [B, 512]

z_fused
  -> MortalityClassifier
  -> mortality_logit [B]

Temporal hidden sequence
  -> ReconstructionHead
  -> x_recon [B, 24, 43]

Temporal hidden sequence except final step
  -> MeasurementForecastHead
  -> mask_next_logit [B, 23, 43]
```

---

# 5. MissingnessAwareTemporalEncoder

## 5.1 Dimensions

```text
n_channels = 43
hidden_dim = 128
input_projection_dim = 128
n_time_steps = 24
transformer_layers = 1
transformer_heads = 4
transformer_ff_dim = 256
dropout = 0.15
```

## 5.2 GRU-D-inspired decay logic

At time step `t`, inputs are:

```text
x_t:     [B, V]
m_t:     [B, V]
delta_t: [B, V]
count_t: [B, V]
```

Maintain:

```text
x_last: [B, V]
h:      [B, 128]
```

Initialize:

```text
x_last = zeros
h = zeros
```

Input decay:

```text
gamma_x = exp(-relu(w_x * delta_t + b_x))
```

Shapes:

```text
w_x: [V]
b_x: [V]
gamma_x: [B, V]
```

Because standardized variable means are zero:

```text
x_decay = gamma_x * x_last
```

Fill missing values:

```text
x_hat = m_t * x_t + (1 - m_t) * x_decay
```

Update last observed value:

```text
x_last = m_t * x_t + (1 - m_t) * x_last
```

Hidden decay:

```text
gamma_h = exp(-relu(W_h(delta_t)))
h = gamma_h * h
```

Where:

```text
W_h: Linear(V -> 128)
```

Create recurrent input:

```text
r_t = concat(x_hat, m_t, delta_t, count_t)
```

Shape:

```text
r_t: [B, 4V] = [B, 172]
```

Input projection:

```text
u_t = Linear(172 -> 128)
u_t = LayerNorm(128)
u_t = GELU
u_t = Dropout(0.15)
```

GRU update:

```text
h_t = GRUCell(128, 128)(u_t, h)
```

Collect hidden states:

```text
H = [h_1, ..., h_24]
H: [B, 24, 128]
```

## 5.3 Transformer refinement

Add positional embedding:

```text
H = H + pos_emb
pos_emb: [1, 24, 128]
```

Transformer encoder:

```text
TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=256,
    dropout=0.10,
    activation="gelu",
    batch_first=True,
    norm_first=True
)
```

Use one layer:

```text
H_attn: [B, 24, 128]
```

## 5.4 Attention pooling

Compute time attention:

```text
scores = Linear(128 -> 1)(tanh(Linear(128 -> 128)(H_attn)))
alpha = softmax(scores, dim=time)
```

Then:

```text
h_att = sum_t alpha_t * H_attn_t
h_last = H_attn[:, -1, :]
h_max = max_pool(H_attn, dim=time)
```

Concatenate:

```text
h_temp_raw = concat(h_att, h_last, h_max)
h_temp_raw: [B, 384]
```

Project:

```text
h_temp = Linear(384 -> 128)
h_temp = LayerNorm(128)
h_temp = GELU
h_temp = Dropout(0.20)
```

Output:

```text
h_temp: [B, 128]
H_attn: [B, 24, 128]
alpha:  [B, 24]
```

---

# 6. StaticEncoder

Input:

```text
x_static: [B, S]
```

Architecture:

```text
Linear(S -> 64)
LayerNorm(64)
GELU
Dropout(0.10)

Linear(64 -> 64)
LayerNorm(64)
GELU
Dropout(0.10)
```

Output:

```text
h_static: [B, 64]
```

---

# 7. AggregateEncoder

Input:

```text
x_aggregate: [B, A]
```

Architecture:

```text
Linear(A -> 128)
LayerNorm(128)
GELU
Dropout(0.20)

Linear(128 -> 96)
LayerNorm(96)
GELU
Dropout(0.15)
```

Output:

```text
h_agg: [B, 96]
```

---

# 8. GatedFusion

Project all branches to 128 dimensions:

```text
z_temp = h_temp
z_static = Linear(64 -> 128)(h_static)
z_agg = Linear(96 -> 128)(h_agg)
```

Concatenate:

```text
z_cat = concat(z_temp, z_static, z_agg)
z_cat: [B, 384]
```

Gate network:

```text
gate_logits = Linear(384 -> 64)
gate_logits = GELU
gate_logits = Linear(64 -> 3)
gate_weights = softmax(gate_logits, dim=-1)
```

Weighted mixture:

```text
z_gate =
    gate_weights[:, 0] * z_temp
  + gate_weights[:, 1] * z_static
  + gate_weights[:, 2] * z_agg
```

Final fused representation:

```text
z_fused = concat(z_gate, z_cat)
z_fused: [B, 512]
```

---

# 9. MortalityClassifier

Input:

```text
z_fused: [B, 512]
```

Architecture:

```text
LayerNorm(512)

Linear(512 -> 128)
GELU
Dropout(0.25)

Linear(128 -> 64)
GELU
Dropout(0.15)

Linear(64 -> 1)
```

Output:

```text
mortality_logit: [B]
```

Inference probability:

```text
mortality_probability = sigmoid(mortality_logit)
```

---

# 10. Auxiliary heads

## 10.1 TemporalReconstructionHead

Input:

```text
H_attn: [B, 24, 128]
```

Architecture:

```text
Linear(128 -> 43)
```

Output:

```text
x_recon: [B, 24, 43]
```

Training target:

```text
randomly hidden observed values
```

Loss:

```text
SmoothL1Loss over artificially hidden observed values only
```

## 10.2 MeasurementForecastHead

Input:

```text
H_attn[:, :-1, :]: [B, 23, 128]
```

Architecture:

```text
Linear(128 -> 43)
```

Output:

```text
mask_next_logit: [B, 23, 43]
```

Target:

```text
mask[:, 1:, :]
```

Loss:

```text
BCEWithLogitsLoss
```

---

# 11. Loss functions

## 11.1 Mortality loss

Use logits.

```text
L_mortality = BCEWithLogitsLoss(pos_weight=pos_weight)
```

Compute positive weight from training split only:

```text
pos_weight = sqrt(n_negative_train / n_positive_train)
```

For the historical cohort:

```text
sqrt(13391 / 3531) ≈ 1.95
```

Do not hard-code this value. Compute it from the training split.

## 11.2 Reconstruction loss

Randomly hide 15% of observed temporal values.

```text
L_reconstruction = SmoothL1Loss(
    x_recon[recon_target_mask == 1],
    x_true[recon_target_mask == 1]
)
```

## 11.3 Measurement forecast loss

```text
L_mask = BCEWithLogitsLoss(
    mask_next_logit,
    mask[:, 1:, :]
)
```

## 11.4 Total loss

```text
L_total = L_mortality
        + 0.05 * L_reconstruction
        + 0.01 * L_mask
```

---

# 12. Training procedure

## 12.1 Stage 1: self-supervised pretraining

Train:

```text
MissingnessAwareTemporalEncoder
TemporalReconstructionHead
MeasurementForecastHead
```

Do not train:

```text
StaticEncoder
AggregateEncoder
GatedFusion
MortalityClassifier
```

Inputs:

```text
x_corrupt
mask_corrupt
delta
count
```

Loss:

```text
L_pretrain = L_reconstruction + 0.1 * L_mask
```

Settings:

```text
epochs = 20
batch_size = 256
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
gradient_clip_norm = 1.0
```

Use training split only for fitting.

## 12.2 Stage 2: supervised fine-tuning

Train full model.

Loss:

```text
L_total = L_mortality + 0.05 * L_reconstruction + 0.01 * L_mask
```

Settings:

```text
max_epochs = 150
batch_size = 256
optimizer = AdamW
learning_rate = 3e-4
weight_decay = 1e-4
gradient_clip_norm = 1.0
early_stopping_patience = 20
early_stopping_metric = validation average precision
```

Scheduler:

```text
ReduceLROnPlateau
monitor = validation average precision
factor = 0.5
patience = 5
min_lr = 1e-5
```

---

# 13. PyTorch skeleton

## 13.1 Full model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAFNet(nn.Module):
    def __init__(
        self,
        n_temporal_channels: int,
        n_static_features: int,
        n_aggregate_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.temporal_encoder = MissingnessAwareTemporalEncoder(
            n_channels=n_temporal_channels,
            hidden_dim=hidden_dim,
            n_time_steps=24,
            n_heads=4,
            transformer_ff_dim=256,
            dropout=0.15,
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(n_static_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.aggregate_encoder = nn.Sequential(
            nn.Linear(n_aggregate_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.static_proj = nn.Linear(64, 128)
        self.aggregate_proj = nn.Linear(96, 128)

        self.gate = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

        self.reconstruction_head = nn.Linear(hidden_dim, n_temporal_channels)
        self.mask_forecast_head = nn.Linear(hidden_dim, n_temporal_channels)

    def forward(self, x, mask, delta, count, x_static, x_aggregate):
        temporal_out = self.temporal_encoder(x, mask, delta, count)

        h_temp = temporal_out["patient_embedding"]
        h_seq = temporal_out["sequence_embedding"]

        h_static = self.static_encoder(x_static)
        h_agg = self.aggregate_encoder(x_aggregate)

        z_temp = h_temp
        z_static = self.static_proj(h_static)
        z_agg = self.aggregate_proj(h_agg)

        z_cat = torch.cat([z_temp, z_static, z_agg], dim=-1)

        gate_weights = torch.softmax(self.gate(z_cat), dim=-1)
        z_gate = (
            gate_weights[:, 0:1] * z_temp
            + gate_weights[:, 1:2] * z_static
            + gate_weights[:, 2:3] * z_agg
        )

        z_fused = torch.cat([z_gate, z_cat], dim=-1)
        mortality_logit = self.classifier(z_fused).squeeze(-1)

        x_recon = self.reconstruction_head(h_seq)
        mask_next_logit = self.mask_forecast_head(h_seq[:, :-1, :])

        return {
            "mortality_logit": mortality_logit,
            "x_recon": x_recon,
            "mask_next_logit": mask_next_logit,
            "gate_weights": gate_weights,
            "temporal_attention": temporal_out["attention_weights"],
        }
```

## 13.2 Temporal encoder

```python
class MissingnessAwareTemporalEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 128,
        n_time_steps: int = 24,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.n_time_steps = n_time_steps

        self.input_decay_w = nn.Parameter(torch.zeros(n_channels))
        self.input_decay_b = nn.Parameter(torch.zeros(n_channels))

        self.hidden_decay = nn.Linear(n_channels, hidden_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(4 * n_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, n_time_steps, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, x, mask, delta, count):
        B, T, V = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_dim, device=device)
        x_last = torch.zeros(B, V, device=device)

        hidden_states = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]
            c_t = count[:, t, :]

            gamma_x = torch.exp(
                -torch.relu(self.input_decay_w * d_t + self.input_decay_b)
            )

            x_decay = gamma_x * x_last
            x_hat = m_t * x_t + (1.0 - m_t) * x_decay

            x_last = m_t * x_t + (1.0 - m_t) * x_last

            gamma_h = torch.exp(-torch.relu(self.hidden_decay(d_t)))
            h = gamma_h * h

            r_t = torch.cat([x_hat, m_t, d_t, c_t], dim=-1)
            u_t = self.input_projection(r_t)

            h = self.gru_cell(u_t, h)
            hidden_states.append(h.unsqueeze(1))

        H = torch.cat(hidden_states, dim=1)

        H = H + self.pos_emb[:, :T, :]
        H = self.transformer(H)

        scores = self.attention(H).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)

        h_att = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        h_last = H[:, -1, :]
        h_max = torch.max(H, dim=1).values

        h_raw = torch.cat([h_att, h_last, h_max], dim=-1)
        patient_embedding = self.output_projection(h_raw)

        return {
            "patient_embedding": patient_embedding,
            "sequence_embedding": H,
            "attention_weights": alpha,
        }
```

## 13.3 Random observed masking

```python
def corrupt_observed_values(x, mask, mask_rate=0.15):
    observed = mask.bool()
    random_draw = torch.rand_like(x)
    hide = observed & (random_draw < mask_rate)

    x_corrupt = x.clone()
    mask_corrupt = mask.clone()

    x_corrupt[hide] = 0.0
    mask_corrupt[hide] = 0.0

    recon_target_mask = hide.float()

    return x_corrupt, mask_corrupt, recon_target_mask
```

## 13.4 Loss computation

```python
def compute_loss(
    outputs,
    y,
    x_true,
    mask_next_target,
    recon_target_mask,
    pos_weight,
    lambda_recon=0.05,
    lambda_mask=0.01,
):
    mortality_logit = outputs["mortality_logit"]
    x_recon = outputs["x_recon"]
    mask_next_logit = outputs["mask_next_logit"]

    mortality_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=y.device)
    )

    L_mort = mortality_loss_fn(mortality_logit, y.float())

    if recon_target_mask.sum() > 0:
        L_recon = F.smooth_l1_loss(
            x_recon[recon_target_mask.bool()],
            x_true[recon_target_mask.bool()],
        )
    else:
        L_recon = torch.tensor(0.0, device=y.device)

    L_mask = F.binary_cross_entropy_with_logits(
        mask_next_logit,
        mask_next_target.float(),
    )

    total_loss = L_mort + lambda_recon * L_recon + lambda_mask * L_mask

    return {
        "loss": total_loss,
        "mortality_loss": L_mort.detach(),
        "reconstruction_loss": L_recon.detach(),
        "mask_forecast_loss": L_mask.detach(),
    }
```

---

# 14. Recommended config

```yaml
data:
  time_window_hours: 6
  bin_minutes: 15
  num_time_steps: 24
  temporal_channels:
    vitals_use_last_min_max: true
    labs_use_last: true
    include_dynamic_derived_channels: true
  value_clip_z: 5.0
  delta_clip_hours: 6.0
  count_clip: 10

model:
  temporal_channels: 43
  temporal_hidden_dim: 128
  temporal_input_projection_dim: 128
  transformer_layers: 1
  transformer_heads: 4
  transformer_ff_dim: 256
  temporal_dropout: 0.15

  static_hidden_dim: 64
  aggregate_hidden_dim_1: 128
  aggregate_hidden_dim_2: 96
  fusion_dim: 128

  classifier_hidden_1: 128
  classifier_hidden_2: 64
  classifier_dropout_1: 0.25
  classifier_dropout_2: 0.15

training:
  batch_size: 256
  pretrain_epochs: 20
  max_epochs: 150
  early_stopping_patience: 20
  optimizer: adamw
  pretrain_lr: 0.001
  finetune_lr: 0.0003
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  pos_weight_strategy: sqrt_neg_pos_ratio
  scheduler: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 5
  min_lr: 0.00001

loss:
  mortality_loss: weighted_bce_with_logits
  lambda_recon: 0.05
  lambda_mask: 0.01
  random_observed_mask_rate: 0.15

evaluation:
  early_stopping_metric: average_precision
  calibration: platt_scaling
  threshold_selection_split: validation
  report_metrics:
    - auc_roc
    - average_precision
    - brier_score
    - accuracy
    - precision
    - recall
    - f1
    - specificity
    - npv
```

---

# 15. Required ablations

Run these after the full model trains.

| Experiment | Description |
|---|---|
| `MAFNet-T` | Temporal branch only. |
| `MAFNet-T+S` | Temporal + static branches. |
| `MAFNet-T+S+A` | Full model. |
| `NoDecay` | Replace missingness-aware decay with zero-imputed GRU. |
| `NoTransformer` | Remove transformer refinement. |
| `NoAux` | Remove reconstruction and mask-forecast losses. |
| `NoGate` | Replace gated fusion with plain concatenation. |
| `NoPretrain` | Skip self-supervised pretraining. |
| `XGB+MAFNet` | Average calibrated XGBoost and MAFNet probabilities. |
| `Stacked` | OOF logistic meta-learner over GBDT models and MAFNet. |

---

# 16. Evaluation

## Validation

For each epoch:

```text
AUC-ROC
average precision / PR-AUC
Brier score
mortality loss
```

Early stopping:

```text
maximize validation average precision
```

## Post-training

1. Fit Platt calibration on validation logits.
2. Convert validation and test logits to calibrated probabilities.
3. Select thresholds on validation probabilities only.
4. Apply fixed thresholds to test probabilities.
5. Report:
   - AUC-ROC
   - average precision
   - Brier score
   - calibration intercept/slope
   - ECE
   - threshold metrics
   - bootstrap confidence intervals

---

# 17. Expected result interpretation

Good result:

```text
MAFNet AUC: 0.835-0.850
MAFNet AP: competitive with XGBoost
Better recall at high-sensitivity threshold
Ablations show decay/missingness/temporal branch contributes signal
```

Strong result:

```text
MAFNet AUC: 0.850+
MAFNet improves AP or high-sensitivity threshold performance
MAFNet + XGBoost ensemble improves over both single models
```

Portfolio-winning result:

```text
XGBoost or LightGBM remains best single model,
but MAFNet captures complementary temporal signal,
and the calibrated ensemble is the best final model.
```

This is credible because boosted trees are very strong on engineered tabular ICU features. The custom model does not need to dominate to be valuable.
