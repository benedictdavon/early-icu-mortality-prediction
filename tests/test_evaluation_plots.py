from __future__ import annotations

from evaluation.plots import (
    evaluation_plot_paths,
    final_report_plot_paths,
    save_ablation_bar_chart,
    save_calibration_curve,
    save_confusion_matrices_at_thresholds,
    save_feature_importance_bar,
    save_subgroup_performance_plot,
    save_temporal_attention_plot,
    save_threshold_tradeoff_curve,
)


def test_calibration_curve_plot_is_written(tmp_path):
    output = tmp_path / "calibration.png"

    save_calibration_curve(
        [0, 0, 1, 1, 1, 0],
        [0.05, 0.20, 0.55, 0.70, 0.90, 0.30],
        output,
        n_bins=3,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_evaluation_plot_paths_include_calibration_curve(tmp_path):
    paths = evaluation_plot_paths(tmp_path)

    assert "calibration_curve" in paths
    assert paths["calibration_curve"].endswith("calibration_curve.png")


def test_final_report_plot_helpers_write_expected_artifacts(tmp_path):
    threshold_plot = tmp_path / "threshold_tradeoff.png"
    confusion_plot = tmp_path / "confusion_thresholds.png"
    importance_plot = tmp_path / "importance.png"
    attention_plot = tmp_path / "attention.png"
    subgroup_plot = tmp_path / "subgroup.png"
    ablation_plot = tmp_path / "ablation.png"

    save_threshold_tradeoff_curve(
        [
            {"threshold": 0.2, "precision": 0.4, "recall": 0.9, "specificity": 0.5, "f1": 0.55},
            {"threshold": 0.5, "precision": 0.7, "recall": 0.7, "specificity": 0.8, "f1": 0.70},
        ],
        threshold_plot,
    )
    save_confusion_matrices_at_thresholds(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.7, 0.9],
        {
            "screening": {"threshold": 0.2},
            "balanced_f1": {"threshold": 0.5},
            "high_precision": {"threshold": 0.8},
        },
        confusion_plot,
    )
    save_feature_importance_bar(["age", "lactate"], [0.2, 0.8], importance_plot)
    save_temporal_attention_plot([0.1, 0.2, 0.3, 0.4], attention_plot)
    save_subgroup_performance_plot(
        [
            {"subgroup_variable": "age_group", "subgroup": "<50", "average_precision": 0.6},
            {"subgroup_variable": "age_group", "subgroup": "80+", "average_precision": 0.4},
        ],
        subgroup_plot,
    )
    save_ablation_bar_chart(
        [
            {"ablation": "MAFNet-T", "validation_average_precision": 0.5},
            {"ablation": "MAFNet-T+S+A", "validation_average_precision": 0.7},
        ],
        ablation_plot,
    )

    for path in [
        threshold_plot,
        confusion_plot,
        importance_plot,
        attention_plot,
        subgroup_plot,
        ablation_plot,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_final_report_plot_paths_include_required_outputs(tmp_path):
    paths = final_report_plot_paths(tmp_path)

    assert "roc_curve" in paths
    assert "precision_recall_curve" in paths
    assert "calibration_curve" in paths
    assert "threshold_tradeoff_curve" in paths
    assert "confusion_matrices_at_thresholds" in paths
    assert "shap_feature_importance" in paths
    assert "mafnet_temporal_attention" in paths
    assert "subgroup_performance" in paths
    assert "ablation_bar_chart" in paths
