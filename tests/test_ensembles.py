from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.ensembles import (
    average_probabilities,
    fit_validation_stacker,
    prediction_matrix,
    validate_prediction_frame,
)


def test_average_probabilities_returns_row_mean():
    matrix = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.70, 0.80, 0.90],
        ]
    )

    averaged = average_probabilities(matrix)

    assert np.allclose(averaged, [0.20, 0.80])


def test_stacker_trains_only_on_validation_predictions():
    validation_probs = np.asarray(
        [
            [0.10, 0.20],
            [0.15, 0.25],
            [0.70, 0.65],
            [0.85, 0.80],
        ]
    )
    validation_labels = np.asarray([0, 0, 1, 1])

    stacker = fit_validation_stacker(
        validation_probs,
        validation_labels,
        model_names=["lightgbm", "xgboost"],
        split="validation",
    )
    before = stacker.to_metadata()
    test_probs = stacker.predict_proba(np.asarray([[0.30, 0.35], [0.90, 0.88]]))
    after = stacker.to_metadata()

    assert before == after
    assert before["fit_split"] == "validation"
    assert before["fit_n"] == len(validation_labels)
    assert test_probs.shape == (2,)
    assert np.all((test_probs >= 0.0) & (test_probs <= 1.0))


def test_stacker_rejects_test_split_fit():
    with pytest.raises(ValueError, match="validation predictions only"):
        fit_validation_stacker(
            [[0.10, 0.20], [0.80, 0.70]],
            [0, 1],
            model_names=["lightgbm", "xgboost"],
            split="test",
        )


def test_prediction_frame_rejects_patient_identifiers():
    predictions = pd.DataFrame(
        {
            "split": ["validation"],
            "model_name": ["lightgbm"],
            "y_true": [0],
            "p_pred": [0.1],
            "subject_id": [123],
        }
    )

    with pytest.raises(ValueError, match="patient identifiers"):
        validate_prediction_frame(predictions)


def test_prediction_matrix_aligns_models_by_row_id():
    predictions = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "row_id": [1, 2, 1, 2],
            "model_name": ["lightgbm", "lightgbm", "xgboost", "xgboost"],
            "y_true": [0, 1, 0, 1],
            "p_pred": [0.10, 0.80, 0.20, 0.70],
        }
    )

    labels, matrix = prediction_matrix(
        predictions,
        split="validation",
        model_names=["lightgbm", "xgboost"],
    )

    assert labels.tolist() == [0, 1]
    assert np.allclose(matrix, [[0.10, 0.20], [0.80, 0.70]])
