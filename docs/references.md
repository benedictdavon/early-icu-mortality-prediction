# References for ICU Mortality Prediction Project

This file lists the main methodological and implementation references behind the improvement plan and ICU6H-MAFNet architecture.

The coding agent does not need to read every source before implementing. Use this file as the reference map for design decisions, documentation, and README citations.

---

# 1. Clinical time-series and missingness-aware models

## GRU-D

**Paper**

```text
Che, Z., Purushotham, S., Cho, K., Sontag, D., Liu, Y.
Recurrent Neural Networks for Multivariate Time Series with Missing Values.
Scientific Reports, 2018.
```

URL:

```text
https://www.nature.com/articles/s41598-018-24271-9
```

Why relevant:

```text
GRU-D is the main inspiration for MAFNet's learnable input decay, hidden-state decay, mask inputs, and delta-time inputs. It directly addresses multivariate clinical time series with informative missingness.
```

Use in project:

```text
Mention as inspiration, not as an exact reproduction unless implemented exactly.
```

---

## STraTS

**Paper**

```text
Tipirneni, S., Reddy, C. K.
Self-supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series.
KDD / arXiv, 2021.
```

URL:

```text
https://arxiv.org/abs/2107.14293
```

Why relevant:

```text
STraTS motivates modeling sparse irregular clinical measurements using time-aware representations and self-supervised objectives. It supports the idea that ICU time-series models should not rely only on dense imputation.
```

Use in project:

```text
Cite as motivation for future event-token transformer work and auxiliary temporal learning.
```

---

## BRITS

**Paper**

```text
Cao, W., Wang, D., Li, J., Zhou, H., Li, L., Li, Y.
BRITS: Bidirectional Recurrent Imputation for Time Series.
NeurIPS, 2018.
```

URL:

```text
https://arxiv.org/abs/1805.10572
```

Why relevant:

```text
BRITS supports the concept of learning imputation and temporal representation jointly rather than treating imputation as a fixed preprocessing step.
```

Use in project:

```text
Mention as related work, not as the selected architecture.
```

---

# 2. Tabular model references

## XGBoost

**Paper**

```text
Chen, T., Guestrin, C.
XGBoost: A Scalable Tree Boosting System.
KDD, 2016.
```

URL:

```text
https://arxiv.org/abs/1603.02754
```

Why relevant:

```text
XGBoost is the current strongest historical baseline and should remain a core comparison model.
```

---

## LightGBM documentation

URL:

```text
https://lightgbm.readthedocs.io/en/latest/
```

Advanced topics:

```text
https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
```

Why relevant:

```text
LightGBM is a strong gradient-boosted tree framework for tabular data. It supports missing value handling and categorical feature handling, making it a strong competitor to XGBoost.
```

---

## CatBoost

**Paper**

```text
Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A.
CatBoost: unbiased boosting with categorical features.
NeurIPS, 2018.
```

URL:

```text
https://arxiv.org/abs/1706.09516
```

Why relevant:

```text
CatBoost is useful if the project includes categorical variables such as sex, admission type, ICU unit, insurance, ethnicity, or diagnosis groups.
```

---

## AutoGluon-Tabular

**Paper**

```text
Erickson, N. et al.
AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.
arXiv, 2020.
```

URL:

```text
https://arxiv.org/abs/2003.06505
```

Why relevant:

```text
AutoGluon-Tabular motivates stacked ensembles and multi-model tabular comparison. The project can borrow the concept of diverse model families plus stacking without necessarily using AutoGluon directly.
```

---

# 3. Modern tabular deep learning

## Revisiting Deep Learning Models for Tabular Data

**Paper**

```text
Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A.
Revisiting Deep Learning Models for Tabular Data.
NeurIPS, 2021.
```

URL:

```text
https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html
```

Why relevant:

```text
This benchmark supports a cautious view: tabular deep learning can be useful, but gradient-boosted trees remain very strong and should not be dismissed.
```

---

## TabPFN / Prior Labs documentation

URL:

```text
https://docs.priorlabs.ai/models
```

Why relevant:

```text
TabPFN is a modern tabular model worth trying as a bounded experiment on a modest-sized tabular dataset.
```

Implementation note:

```text
Check current package version, license, input size limits, and intended usage before using it in final experiments.
```

---

## TabM

**Paper**

```text
Gorishniy, Y. et al.
TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.
arXiv, 2024.
```

URL:

```text
https://arxiv.org/abs/2410.24210
```

Why relevant:

```text
TabM is a recent practical tabular deep learning architecture. It is useful as a modern DL baseline, not necessarily as the main custom model.
```

---

# 4. Loss functions and PyTorch references

## BCEWithLogitsLoss

Documentation:

```text
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
```

Why relevant:

```text
MAFNet uses BCEWithLogitsLoss for mortality prediction. The `pos_weight` argument supports positive-class weighting for imbalanced binary classification.
```

---

## AdamW

Documentation:

```text
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
```

Why relevant:

```text
MAFNet uses AdamW with weight decay for stable neural-network training.
```

---

## Focal loss

**Paper**

```text
Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.
Focal Loss for Dense Object Detection.
ICCV, 2017.
```

URL:

```text
https://arxiv.org/abs/1708.02002
```

Why relevant:

```text
Focal loss is an optional ablation for imbalance handling. It is not the default because it adds hyperparameters and may complicate probability calibration.
```

---

# 5. Evaluation and calibration references

## scikit-learn average precision

Documentation:

```text
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
```

Why relevant:

```text
Average precision / PR-AUC is a required positive-class retrieval metric for the project.
```

---

## scikit-learn calibration

Documentation:

```text
https://scikit-learn.org/stable/modules/calibration.html
```

Why relevant:

```text
The project should report calibration curves, Brier score, and validation-only Platt scaling or isotonic calibration.
```

---

## Brier score

Documentation:

```text
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
```

Why relevant:

```text
The Brier score evaluates probability quality and should be included in final reports.
```

---

# 6. Clinical prediction reporting standards

## TRIPOD+AI

Website:

```text
https://www.tripod-statement.org/
```

Why relevant:

```text
TRIPOD+AI provides transparent reporting guidance for prediction model studies using regression or machine learning.
```

Use in project:

```text
Create docs/tripod_ai_checklist.md and explain which items the project satisfies.
```

---

## PROBAST+AI

Website:

```text
https://www.latitudes-network.org/tool/probastai/
```

Why relevant:

```text
PROBAST+AI supports risk-of-bias and applicability assessment for health prediction models using AI or regression methods.
```

Use in project:

```text
Create docs/probast_ai_notes.md with a candid risk-of-bias discussion.
```

---

# 7. Clinical data references

## MIMIC-IV

Website:

```text
https://physionet.org/content/mimiciv/
```

Why relevant:

```text
The project is based on a course-provided MIMIC-IV-derived subset. The public repository should respect data-use constraints and not include raw or processed patient-level data.
```

---

## eICU Collaborative Research Database

Website:

```text
https://physionet.org/content/eicu-crd/2.0/
```

Why relevant:

```text
Potential external validation source if access, feature alignment, and data-use rules allow it.
```

Use in project:

```text
Mention as future external validation, not as completed work unless implemented.
```

---

# 8. MIMIC-IV benchmark reference

## MIMIC-IV benchmark paper

URL:

```text
https://arxiv.org/pdf/2401.15290
```

Why relevant:

```text
This benchmark was used as contextual support that boosted trees can be strong for MIMIC-IV mortality prediction tasks. Do not treat its numbers as directly comparable unless cohort definition, prediction horizon, features, and split protocol match your project.
```

Use in project:

```text
Mention cautiously as external context, not as a direct performance target.
```

---

# 9. Recommended citation language for README

Use language like this:

```text
The custom temporal model is inspired by missingness-aware recurrent modeling ideas from GRU-D and by self-supervised sparse clinical time-series modeling concepts from STraTS. It is not claimed to be an exact reproduction of either method.
```

Use language like this for boosted trees:

```text
Because gradient-boosted decision trees are strong baselines for structured clinical data, the project compares the custom neural model against XGBoost, LightGBM, CatBoost, and stacked ensembles.
```

Use language like this for reporting:

```text
The evaluation protocol reports discrimination, precision-recall behavior, calibration, threshold tradeoffs, and subgroup robustness. The project is framed as retrospective academic research and is not intended for clinical deployment.
```
