from __future__ import annotations

from sklearn.metrics import roc_auc_score

from evaluation.bootstrap import bootstrap_metric_ci
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
