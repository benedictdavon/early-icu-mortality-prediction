"""Shared plotting helpers for model evaluation artifacts."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def save_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    _draw_confusion_matrix(cm, plt.gca())
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


def save_threshold_tradeoff_curve(
    threshold_table,
    output_path,
    title="Threshold Tradeoff Curve",
):
    """Plot precision, recall, specificity, and F1 across candidate thresholds."""
    table = pd.DataFrame(threshold_table).sort_values("threshold")
    required = {"threshold", "precision", "recall", "specificity", "f1"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"threshold_table is missing columns: {sorted(missing)}")
    plt.figure(figsize=(8, 6))
    for metric in ["precision", "recall", "specificity", "f1"]:
        plt.plot(table["threshold"], table[metric], marker="o", label=metric)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrices_at_thresholds(
    y_true,
    y_score,
    thresholds: dict,
    output_path,
    title="Confusion Matrices at Selected Thresholds",
):
    """Save side-by-side confusion matrices for fixed threshold policies."""
    policies = list(thresholds.items())
    if not policies:
        raise ValueError("thresholds must contain at least one policy")
    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4))
    if len(policies) == 1:
        axes = [axes]
    for ax, (name, policy) in zip(axes, policies):
        threshold = policy["threshold"] if isinstance(policy, dict) else float(policy)
        y_pred = (np.asarray(y_score) >= float(threshold)).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _draw_confusion_matrix(cm, ax)
        ax.set_title(f"{name}\nt={threshold:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _draw_confusion_matrix(cm, ax):
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")


def save_feature_importance_bar(
    feature_names,
    importances,
    output_path,
    title="Feature Importance",
    top_n: int = 20,
):
    """Save a top-N feature importance bar chart for tree/SHAP summaries."""
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.3 * len(frame))))
    plt.barh(frame["feature"][::-1], frame["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_temporal_attention_plot(
    attention_weights,
    output_path,
    title="MAFNet Temporal Attention",
    bin_minutes: int = 15,
):
    """Plot MAFNet attention over first-window bins for a synthetic/deidentified example."""
    weights = np.asarray(attention_weights, dtype=float).reshape(-1)
    hours = np.arange(len(weights)) * float(bin_minutes) / 60.0
    plt.figure(figsize=(8, 4))
    plt.plot(hours, weights, marker="o")
    plt.xlabel("Hours since ICU admission")
    plt.ylabel("Attention weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_subgroup_performance_plot(
    subgroup_report,
    output_path,
    metric: str = "average_precision",
    title="Subgroup Performance",
):
    """Save subgroup metric bars from an aggregate subgroup report."""
    frame = pd.DataFrame(subgroup_report)
    required = {"subgroup_variable", "subgroup", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"subgroup_report is missing columns: {sorted(missing)}")
    frame = frame.sort_values(metric, ascending=True)
    labels = frame["subgroup_variable"].astype(str) + ": " + frame["subgroup"].astype(str)
    plt.figure(figsize=(9, max(4, 0.28 * len(frame))))
    plt.barh(labels, frame[metric])
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_ablation_bar_chart(
    ablation_results,
    output_path,
    metric: str = "validation_average_precision",
    title="MAFNet Ablation Results",
):
    """Save an ablation comparison bar chart from aggregate ablation results."""
    frame = pd.DataFrame(ablation_results)
    required = {"ablation", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ablation_results is missing columns: {sorted(missing)}")
    frame = frame.sort_values(metric, ascending=True)
    plt.figure(figsize=(8, max(4, 0.35 * len(frame))))
    plt.barh(frame["ablation"], frame[metric])
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def final_report_plot_paths(output_dir):
    return {
        **evaluation_plot_paths(output_dir),
        "threshold_tradeoff_curve": os.path.join(output_dir, "threshold_tradeoff_curve.png"),
        "confusion_matrices_at_thresholds": os.path.join(
            output_dir,
            "confusion_matrices_at_thresholds.png",
        ),
        "shap_feature_importance": os.path.join(output_dir, "shap_feature_importance.png"),
        "mafnet_temporal_attention": os.path.join(output_dir, "mafnet_temporal_attention.png"),
        "subgroup_performance": os.path.join(output_dir, "subgroup_performance.png"),
        "ablation_bar_chart": os.path.join(output_dir, "ablation_bar_chart.png"),
    }
