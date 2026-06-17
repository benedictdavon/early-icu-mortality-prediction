"""Reusable classification metrics for model evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Calculate thresholded and ranking metrics for binary classifiers."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": auc(recall_curve, precision_curve),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }
