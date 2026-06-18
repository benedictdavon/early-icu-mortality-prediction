from __future__ import annotations

from sklearn.metrics import roc_auc_score

from evaluation.bootstrap import (
    bootstrap_binary_metrics_ci,
    bootstrap_metric_ci,
    paired_bootstrap_metric_difference,
)
from evaluation.metrics import compute_binary_metrics


def test_metrics_confusion_matrix_consistency():
    y_true = [0, 0, 1, 1]
    p_pred = [0.10, 0.60, 0.70, 0.20]

    metrics = compute_binary_metrics(y_true, p_pred, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["npv"] == 0.5
    assert metrics["n"] == 4
    assert metrics["positive_rate"] == 0.5


def test_bootstrap_ci_shape():
    y_true = [0, 0, 0, 1, 1, 1]
    p_pred = [0.05, 0.20, 0.30, 0.60, 0.80, 0.95]

    ci = bootstrap_metric_ci(
        y_true,
        p_pred,
        metric_fn=roc_auc_score,
        n_boot=50,
        seed=123,
    )

    assert set(ci) == {"mean", "ci_lower", "ci_upper", "n_boot", "n_success", "confidence"}
    assert ci["n_boot"] == 50
    assert ci["n_success"] > 0
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]


def test_bootstrap_binary_metrics_includes_threshold_metrics():
    y_true = [0, 0, 0, 1, 1, 1, 1, 0]
    p_pred = [0.05, 0.10, 0.30, 0.55, 0.62, 0.81, 0.94, 0.44]

    results = bootstrap_binary_metrics_ci(
        y_true,
        p_pred,
        threshold=0.5,
        n_boot=30,
        seed=123,
    )

    for metric in [
        "auc_roc",
        "average_precision",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "npv",
    ]:
        assert metric in results
        assert results[metric]["n_boot"] == 30
        assert results[metric]["n_success"] > 0


def test_bootstrap_skips_one_class_auc_samples_safely():
    y_true = [0, 0, 0, 0, 1]
    p_pred = [0.10, 0.20, 0.30, 0.40, 0.90]

    ci = bootstrap_metric_ci(
        y_true,
        p_pred,
        metric_fn=roc_auc_score,
        n_boot=100,
        seed=7,
    )

    assert ci["n_success"] < 100
    assert ci["n_success"] > 0


def test_paired_bootstrap_metric_difference_schema():
    y_true = [0, 0, 0, 1, 1, 1, 1, 0]
    model_a = [0.05, 0.10, 0.30, 0.55, 0.62, 0.81, 0.94, 0.44]
    model_b = [0.12, 0.20, 0.45, 0.50, 0.58, 0.75, 0.85, 0.35]

    result = paired_bootstrap_metric_difference(
        y_true,
        model_a,
        model_b,
        metric_name="average_precision",
        model_a_name="model_a",
        model_b_name="model_b",
        n_boot=25,
        seed=42,
    )

    assert set(result) == {
        "model_a",
        "model_b",
        "metric",
        "estimate_a",
        "estimate_b",
        "difference",
        "ci_lower",
        "ci_upper",
        "p_value_two_sided",
        "n_boot",
        "n_success",
        "confidence",
    }
    assert result["model_a"] == "model_a"
    assert result["model_b"] == "model_b"
    assert result["metric"] == "average_precision"
    assert result["n_boot"] == 25
