from __future__ import annotations

import numpy as np
import pytest

from evaluation.calibration import (
    apply_isotonic_calibrator,
    apply_platt_scaler,
    apply_probability_calibrator,
    calibration_curve_data,
    fit_probability_calibrator,
    fit_isotonic_calibrator,
    fit_platt_scaler,
    logits_to_probabilities,
    platt_calibration_summary,
)
from evaluation.metrics import compute_binary_metrics


def test_platt_scaler_fits_validation_logits_and_returns_probabilities():
    validation_logits = np.asarray([-3.0, -1.4, -0.2, 0.8, 1.6, 3.1])
    validation_labels = np.asarray([0, 0, 0, 1, 1, 1])

    calibrator = fit_platt_scaler(validation_logits, validation_labels)
    probabilities = calibrator.predict_proba(validation_logits)
    metadata = calibrator.to_metadata()

    assert probabilities.shape == validation_logits.shape
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all(np.diff(probabilities[np.argsort(validation_logits)]) >= 0.0)
    assert metadata["method"] == "platt_scaling"
    assert metadata["fit_split"] == "validation"
    assert metadata["fit_n"] == len(validation_labels)
    assert np.isfinite(metadata["coefficient"])
    assert np.isfinite(metadata["intercept"])


def test_platt_scaler_applies_to_test_logits_without_refitting():
    calibrator = fit_platt_scaler(
        validation_logits=[-2.0, -0.8, 0.7, 2.0],
        validation_labels=[0, 0, 1, 1],
    )
    before = calibrator.to_metadata()

    test_probabilities = apply_platt_scaler(calibrator, logits=[-4.0, 0.0, 4.0])
    after = calibrator.to_metadata()

    assert test_probabilities.shape == (3,)
    assert np.all((test_probabilities >= 0.0) & (test_probabilities <= 1.0))
    assert after == before


def test_platt_scaler_requires_two_validation_classes():
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_platt_scaler(
            validation_logits=[-2.0, -1.0, 0.5, 1.0],
            validation_labels=[0, 0, 0, 0],
        )


def test_isotonic_calibrator_fits_validation_logits_and_applies_to_test():
    validation_logits = np.asarray([-3.0, -2.0, -1.0, 0.1, 1.0, 2.2, 3.0])
    validation_labels = np.asarray([0, 0, 0, 1, 0, 1, 1])
    calibrator = fit_isotonic_calibrator(validation_logits, validation_labels)
    metadata_before = calibrator.to_metadata()

    validation_probabilities = calibrator.predict_proba(validation_logits)
    test_probabilities = apply_isotonic_calibrator(
        calibrator,
        logits=[-5.0, -0.5, 0.5, 5.0],
    )

    assert validation_probabilities.shape == validation_logits.shape
    assert np.all((validation_probabilities >= 0.0) & (validation_probabilities <= 1.0))
    assert np.all(np.diff(validation_probabilities[np.argsort(validation_logits)]) >= 0.0)
    assert np.all((test_probabilities >= 0.0) & (test_probabilities <= 1.0))
    assert calibrator.to_metadata() == metadata_before
    assert metadata_before["method"] == "isotonic"
    assert metadata_before["fit_split"] == "validation"


def test_isotonic_calibrator_requires_two_validation_classes():
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_isotonic_calibrator(
            validation_logits=[-2.0, -1.0, 0.5, 1.0],
            validation_labels=[1, 1, 1, 1],
        )


def test_platt_calibration_summary_is_aggregate_only():
    validation_logits = np.asarray([-2.0, -1.0, 0.2, 1.0, 2.0, 3.0])
    validation_labels = np.asarray([0, 0, 0, 1, 1, 1])
    calibrator = fit_platt_scaler(validation_logits, validation_labels)

    summary = platt_calibration_summary(
        validation_labels,
        validation_logits,
        calibrator,
        n_bins=3,
    )

    assert "brier_score" in summary
    assert "expected_calibration_error" in summary
    assert "calibration_slope" in summary
    assert "calibration_intercept" in summary
    assert summary["fit_n"] == len(validation_labels)
    assert "probabilities" not in summary
    assert np.all((logits_to_probabilities(validation_logits) >= 0.0))


def test_probability_platt_calibrator_uses_validation_only():
    validation_probabilities = np.asarray([0.05, 0.12, 0.24, 0.65, 0.78, 0.91])
    validation_labels = np.asarray([0, 0, 0, 1, 1, 1])

    calibrator = fit_probability_calibrator(
        validation_probabilities,
        validation_labels,
        method="platt",
    )
    before = calibrator.to_metadata()
    test_probabilities = apply_probability_calibrator(
        calibrator,
        probabilities=[0.10, 0.50, 0.90],
    )
    after = calibrator.to_metadata()

    assert before == after
    assert before["fit_split"] == "validation"
    assert before["fit_n"] == len(validation_labels)
    assert test_probabilities.shape == (3,)
    assert np.all((test_probabilities >= 0.0) & (test_probabilities <= 1.0))


def test_calibration_curve_data_is_aggregate_only():
    rows = calibration_curve_data(
        y_true=[0, 0, 1, 1],
        p_pred=[0.10, 0.20, 0.70, 0.90],
        n_bins=2,
    )

    assert len(rows) == 2
    assert set(rows[0]) == {
        "bin",
        "lower",
        "upper",
        "n",
        "mean_predicted_probability",
        "observed_event_rate",
    }
    assert "subject_id" not in rows[0]


def test_selected_threshold_is_applied_unchanged_to_test():
    selected_threshold = 0.65
    test_scores = [0.10, 0.64, 0.65, 0.90]
    test_labels = [0, 1, 1, 1]

    metrics = compute_binary_metrics(test_labels, test_scores, threshold=selected_threshold)

    assert metrics["threshold"] == selected_threshold
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
