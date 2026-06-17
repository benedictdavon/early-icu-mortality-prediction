from __future__ import annotations

from evaluation.reporting import RESULT_SCHEMA_FIELDS, evaluate_validation_and_test
from evaluation.thresholds import select_thresholds


def test_threshold_selection_uses_validation_not_test():
    y_valid = [0, 0, 1, 1, 1]
    p_valid = [0.05, 0.20, 0.55, 0.70, 0.90]
    y_test = [0, 1, 0, 1, 0]
    p_test = [0.99, 0.98, 0.97, 0.10, 0.09]

    valid_thresholds = select_thresholds(y_valid, p_valid)
    test_thresholds = select_thresholds(y_test, p_test)
    report = evaluate_validation_and_test(
        "synthetic_model",
        y_valid,
        p_valid,
        y_test,
        p_test,
    )

    assert report["thresholds"] == valid_thresholds
    assert report["thresholds"] != test_thresholds
    for policy in report["thresholds"].values():
        assert policy["selected_on"] == "validation"


def test_selected_thresholds_are_applied_unchanged_to_test():
    report = evaluate_validation_and_test(
        "synthetic_model",
        y_valid=[0, 0, 1, 1],
        p_valid=[0.10, 0.40, 0.60, 0.90],
        y_test=[0, 1, 1, 0],
        p_test=[0.20, 0.50, 0.80, 0.95],
    )

    test_records = [row for row in report["results"] if row["split"] == "test"]
    assert test_records
    for row in test_records:
        expected_threshold = report["thresholds"][row["threshold_policy"]]["threshold"]
        assert row["threshold"] == expected_threshold


def test_result_schema_contains_required_fields():
    report = evaluate_validation_and_test(
        "synthetic_model",
        y_valid=[0, 0, 1, 1],
        p_valid=[0.10, 0.20, 0.80, 0.90],
        y_test=[0, 1, 1, 0],
        p_test=[0.20, 0.70, 0.60, 0.30],
    )

    first_record = report["results"][0]
    for field in RESULT_SCHEMA_FIELDS:
        assert field in first_record
