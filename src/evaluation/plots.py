"""Shared plotting helpers for model evaluation artifacts."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def save_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve(y_true, y_score, output_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_precision_recall_curve(y_true, y_score, output_path, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_calibration_curve(
    y_true,
    y_score,
    output_path,
    title="Calibration Curve",
    n_bins=10,
):
    """Save a reliability diagram for predicted probabilities."""
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="uniform",
    )
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Perfectly calibrated")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Rate")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluation_plot_paths(output_dir):
    return {
        "confusion_matrix": os.path.join(output_dir, "confusion_matrix.png"),
        "roc_curve": os.path.join(output_dir, "roc_curve.png"),
        "precision_recall_curve": os.path.join(output_dir, "precision_recall_curve.png"),
        "calibration_curve": os.path.join(output_dir, "calibration_curve.png"),
    }
