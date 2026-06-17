from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.run_stacking import evaluate_probability_ensemble
from models.stacking import average_calibrated_probabilities, fit_oof_logistic_stacker


def test_average_calibrated_probabilities_for_xgb_mafnet_ensemble():
    xgb = np.asarray([0.10, 0.40, 0.80, 0.90])
    mafnet = np.asarray([0.20, 0.30, 0.70, 0.95])

    averaged = average_calibrated_probabilities(np.column_stack([xgb, mafnet]))

    assert np.allclose(averaged, [0.15, 0.35, 0.75, 0.925])
    assert np.all((averaged >= 0.0) & (averaged <= 1.0))


def test_probability_ensemble_report_is_aggregate_only():
    report = evaluate_probability_ensemble(
        y_true=[0, 0, 1, 1],
        probability_frame=pd.DataFrame(
            {
                "xgboost": [0.05, 0.30, 0.70, 0.90],
                "mafnet": [0.10, 0.20, 0.80, 0.85],
            }
        ),
        model_name="XGB+MAFNet",
        threshold=0.5,
    )

    assert report["model_name"] == "XGB+MAFNet"
    assert report["component_models"] == ["xgboost", "mafnet"]
    assert report["n"] == 4
    assert "probabilities" not in report


def test_oof_logistic_stacker_uses_out_of_fold_meta_features():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > np.median(X[:, 0] + 0.5 * X[:, 1])).astype(int)

    result = fit_oof_logistic_stacker(
        X,
        y,
        {
            "logistic_a": lambda: LogisticRegression(max_iter=500),
            "logistic_b": lambda: LogisticRegression(C=0.5, max_iter=500),
        },
        n_splits=4,
        random_state=11,
    )

    assert result.model_names == ["logistic_a", "logistic_b"]
    assert result.oof_meta_features.shape == (40, 2)
    assert np.all((result.oof_meta_features >= 0.0) & (result.oof_meta_features <= 1.0))
    assert set(result.base_models) == {"logistic_a", "logistic_b"}
    stacked = result.predict_proba_from_base_probabilities(result.oof_meta_features)
    assert stacked.shape == (40,)
    assert np.all((stacked >= 0.0) & (stacked <= 1.0))


def test_oof_stacker_rejects_too_small_single_class_data():
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_oof_logistic_stacker(
            np.zeros((4, 2)),
            np.zeros(4),
            {"logistic": lambda: LogisticRegression()},
        )
