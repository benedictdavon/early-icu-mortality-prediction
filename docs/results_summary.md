# Results Summary

Generated from local run `results/overnight_20260618_111652`, completed on
2026-06-18.

This file records aggregate, non-row-level results only. Raw MIMIC-IV files,
processed patient-level CSVs, row-level predictions, checkpoints, and model
artifacts must remain local and uncommitted.

## Run Status

- Pytest preflight: 72 passed.
- Cohort size: 16,922 ICU stays.
- Mortality rate: 20.87%.
- Train/validation/test rows for model suite and MAFNet: 10,152 / 3,385 / 3,385.
- Combined aggregate result rows: 71.
- Final polish reports are generated locally under the run directory:
  - `results/overnight_20260618_111652/final_model_report.md`
  - `results/overnight_20260618_111652/calibration_report.md`
  - `results/overnight_20260618_111652/bootstrap_ci_report.md`
  - `results/overnight_20260618_111652/ensemble_report.md`
- Main aggregate artifacts:
  - `results/overnight_20260618_111652/final_summary.md`
  - `results/overnight_20260618_111652/all_training_results.csv`
  - `results/overnight_20260618_111652/model_suite/model_comparison_table.csv`
  - `results/overnight_20260618_111652/xgboost_expanded_ablation/ablation_threshold_policy_results.csv`
  - `results/overnight_20260618_111652/legacy_xgboost_ensemble/xgboost/ensemble_threshold_results.csv`
  - `results/overnight_20260618_111652/mafnet_ablations/mafnet_ablation_results.csv`

## Primary Model Suite Results

The primary tabular comparison uses the validation-selected `balanced_f1`
threshold and applies it unchanged to the held-out test split.

| Model | Test AUC-ROC | Test PR-AUC | Brier | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.8475 | 0.6289 | 0.1246 | 0.8047 | 0.5264 | 0.6346 | 0.8496 | 0.5755 |
| XGBoost | 0.8473 | 0.6271 | 0.1503 | 0.7876 | 0.4936 | 0.7054 | 0.8093 | 0.5808 |
| CatBoost | 0.8454 | 0.6223 | 0.1466 | 0.7778 | 0.4783 | 0.7181 | 0.7936 | 0.5742 |
| EBM | 0.8361 | 0.6104 | 0.1195 | 0.7917 | 0.5005 | 0.6728 | 0.8231 | 0.5740 |
| ExtraTrees | 0.8356 | 0.6014 | 0.1258 | 0.7956 | 0.5077 | 0.6530 | 0.8331 | 0.5713 |
| Random forest | 0.8286 | 0.5819 | 0.1267 | 0.7722 | 0.4687 | 0.6898 | 0.7940 | 0.5582 |
| Logistic regression | 0.8188 | 0.5629 | 0.1727 | 0.7728 | 0.4678 | 0.6473 | 0.8059 | 0.5431 |

Summary:

- LightGBM was the best tabular model by test PR-AUC/AUC combination and is the
  primary final model for reporting.
- XGBoost had nearly identical AUC/PR-AUC and the highest tabular F1 at the
- validation-selected `balanced_f1` threshold. Treat it as a practical near-tie
  on discrimination unless paired bootstrap on aligned predictions supports a
  stronger claim.
- CatBoost produced the highest recall among the top three boosted models, with
  lower precision.
- EBM had the best tabular Brier score and is the main calibrated/interpretable
  comparison model.

## XGBoost Feature Expansion Ablation

| Feature set | Test AUC-ROC | Test PR-AUC | Brier | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline features | 0.7853 | 0.5095 | 0.3483 | 0.6969 | 0.3808 | 0.7238 | 0.6898 | 0.4990 |
| Expanded features | 0.8438 | 0.6228 | 0.1279 | 0.7775 | 0.4779 | 0.7195 | 0.7928 | 0.5743 |

Expanded first-6-hour features improved:

- AUC-ROC by 0.0586.
- PR-AUC by 0.1134.
- F1 by 0.0753.
- Brier score by 0.2203.

The expanded feature set is therefore the preferred tabular feature set for the
final report.

## Legacy XGBoost Ensemble

| Threshold policy | Threshold | AUC-ROC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.50 | 0.8482 | 0.6227 | 0.8287 | 0.6842 | 0.3314 | 0.9597 | 0.4466 |
| F1 optimized | 0.45 | 0.8482 | 0.6227 | 0.8310 | 0.6444 | 0.4235 | 0.9384 | 0.5111 |
| Balanced | 0.30 | 0.8482 | 0.6227 | 0.7947 | 0.5059 | 0.6671 | 0.8283 | 0.5754 |
| High sensitivity | 0.20 | 0.8482 | 0.6227 | 0.7149 | 0.4106 | 0.8428 | 0.6812 | 0.5522 |
| Clinical utility | 0.10 | 0.8482 | 0.6227 | 0.5282 | 0.3021 | 0.9632 | 0.4136 | 0.4599 |

The ensemble is competitive with the model-suite boosted models. The balanced
threshold gives a practical recall/precision tradeoff; lower thresholds may be
useful for sensitivity-focused analysis but increase false positives.

## MAFNet Results

Full MAFNet used first-6-hour long-form events converted into 15-minute tensors,
with Platt scaling fit on validation logits.

| MAFNet run | Validation AUC-ROC | Validation PR-AUC | Test AUC-ROC | Test PR-AUC | Test Brier |
|---|---:|---:|---:|---:|---:|
| Full MAFNet | 0.8005 | 0.5342 | 0.8061 | 0.5610 | 0.1282 |

MAFNet architecture ablations:

| Ablation | Test AUC-ROC | Test PR-AUC | Test Brier |
|---|---:|---:|---:|
| MAFNet-T+S | 0.8184 | 0.5896 | 0.1231 |
| NoGate | 0.8096 | 0.5657 | 0.1275 |
| NoPretrain | 0.8097 | 0.5620 | 0.1279 |
| NoAux | 0.8062 | 0.5615 | 0.1281 |
| Full MAFNet / MAFNet-T+S+A | 0.8061 | 0.5610 | 0.1282 |
| NoTransformer | 0.8080 | 0.5593 | 0.1287 |
| NoDecay | 0.8059 | 0.5525 | 0.1291 |
| MAFNet-T | 0.7747 | 0.5420 | 0.1321 |

Interpretation:

- MAFNet was successfully implemented and evaluated.
- It did not outperform the best boosted tabular models in this run.
- The best MAFNet ablation was `MAFNet-T+S`, suggesting temporal plus static
  features performed better than adding the aggregate branch for this dataset
  and configuration.
- Full MAFNet had acceptable calibration after Platt scaling, but lower
  discrimination than LightGBM/XGBoost.

## Final Takeaways

- Primary model for reporting: LightGBM, because it had the best tabular
  PR-AUC/AUC combination and strong Brier score.
- Alternative model: expanded-feature XGBoost or the XGBoost ensemble,
  especially when prioritizing threshold tradeoffs. This project is not a
  clinical deployment and should not be described as clinically useful without
  external validation and governance.
- Main feature-engineering result: expanded first-6-hour features materially
  improved XGBoost over baseline features.
- Custom model result: MAFNet is complete and evaluated, but should be reported
  as exploratory rather than the top performer. MAFNet-T+S was the best neural
  temporal variant; full MAFNet underperformed the boosted tabular models.

## Final Statistical Polish Notes

- Reusable bootstrap utilities now support AUC-ROC, average precision, Brier
  score, F1, recall, precision, specificity, NPV, and accuracy.
- Paired bootstrap comparison utilities are implemented for aligned local
  predictions, but were not run for this completed experiment because row-level
  prediction files were not saved.
- Platt and optional isotonic calibration utilities fit on validation
  predictions only, then apply the fitted mapping once to test predictions.
- MAFNet-T+S Platt scaling improved test Brier score from 0.1318 to 0.1231 and
  expected calibration error from 0.0643 to 0.0076.
- The calibrated multi-model ensemble and L2 logistic stacker are implemented
  for local aligned prediction files, but no ensemble result is claimed for this
  run because the required validation/test prediction files are missing.

## Commit Policy For Results

Safe to commit:

- `README.md` result summary.
- This `docs/results_summary.md` file.
- Code needed to reproduce the run.

Keep local and uncommitted:

- `data/`
- `results/`
- `*.csv` patient-level files
- `*.pt`, `*.joblib`, `*.pkl`, and checkpoints
- raw MIMIC-IV files
- row-level predictions
- large logs
