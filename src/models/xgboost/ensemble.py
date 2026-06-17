"""XGBoost ensemble defaults."""

DEFAULT_ENSEMBLE_SEEDS = [42, 123, 2021, 777, 888]

ENSEMBLE_THRESHOLDS = {
    "standard": 0.50,
    "f1_optimized": 0.45,
    "balanced": 0.30,
    "high_sensitivity": 0.20,
    "clinical_utility": 0.10,
}
