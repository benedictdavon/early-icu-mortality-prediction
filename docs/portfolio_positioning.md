# Portfolio Positioning

Project title:

```text
Early ICU Mortality Prediction from First-6-Hour EHR Data:
Leakage-Controlled Feature Engineering, Calibrated Risk Modeling, and
Missingness-Aware Temporal Fusion
```

This repository is a retrospective academic and portfolio ML project using a
course-provided MIMIC-IV-derived subset. It is not a clinical deployment model.

## Current Position

The historical strongest model is an XGBoost ensemble with AUC-ROC around
0.846. Those historical results are useful context, but threshold-specific
numbers are historical until rerun under the corrected validation-threshold
protocol.

The first portfolio upgrade is evaluation rigor:

- patient-separated train/validation/test structure when identifiers are
  available
- validation-only threshold selection
- final test-only aggregate reporting
- average precision and Brier score
- calibration-ready summaries
- leakage tests that run on synthetic data

## Near-Term Roadmap

Phase 2 now establishes feature provenance and first-6-hour leakage controls:

- feature dictionary with source table, time window, aggregation, missingness,
  and leakage-risk metadata
- synthetic timestamp fixtures proving post-6-hour events are excluded

Phase 3 expands tabular first-window feature engineering as tested building
blocks:

- temporal bin features
- trajectory features
- measurement-process features
- organ dysfunction proxy features
- focused clinical interaction features

Historical model metrics have not yet been rerun with Phase 3 features.

ICU6H-MAFNet should wait until the evaluation protocol, feature provenance, and
tabular baselines are stable.
