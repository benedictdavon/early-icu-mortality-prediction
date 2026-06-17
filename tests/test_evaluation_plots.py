from __future__ import annotations

from evaluation.plots import evaluation_plot_paths, save_calibration_curve


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
