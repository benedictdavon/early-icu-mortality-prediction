# Final Deliverables

This document packages the project for course review. It does not introduce a new
model family; it turns the cohort, feature, preprocessing, and model-suite work
into reproducible final deliverables.

## Assignment Mapping

| Assignment item | Repository evidence |
|---|---|
| TODO 1: cohort selection and flow chart | `src/cohort_selection.py`, `report/REPORT.md` |
| TODO 2: feature extraction and descriptive analysis | `src/feature_extraction.py`, `docs/feature_dictionary.csv` |
| TODO 3: preprocessing strategies | `src/data_preprocessing.py`, `docs/leakage_checklist.md` |
| TODO 4: model development and evaluation | `src/main.py`, `src/experiments/run_model_suite.py` |
| Final sharing of code, results, descriptions, and conclusion | `src/experiments/generate_final_summary.py`, `report/FINAL_SUMMARY.md` |

## Command

After running the tabular model suite, generate the final summary:

```bash
python src/experiments/generate_final_summary.py \
  --model-suite-dir results/model_suite \
  --output-path report/FINAL_SUMMARY.md
```

The generated report includes assignment traceability, model-suite status,
primary test metrics, threshold-policy results, feature provenance, safety notes,
and the final conclusion.

## Privacy Boundary

The final summary reads aggregate CSV/JSON outputs only. It does not write patient-level
predictions or expose restricted MIMIC-IV data.
