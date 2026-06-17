from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.run_model_suite import run_model_suite
from models.tabular import get_model_class
from models.tabular.logistic import LogisticRiskModel


def _write_fast_configs(config_dir):
    config_dir.mkdir()
    (config_dir / "logistic.yaml").write_text(
        """
model:
  params:
    C: 1.0
    class_weight: balanced
    max_iter: 500
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "random_forest.yaml").write_text(
        """
model:
  params:
    n_estimators: 25
    max_depth: 4
    min_samples_leaf: 2
    class_weight: balanced_subsample
    n_jobs: 1
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "extra_trees.yaml").write_text(
        """
model:
  params:
    n_estimators: 25
    max_depth: 4
    min_samples_leaf: 2
    class_weight: balanced
    n_jobs: 1
""".strip(),
        encoding="utf-8",
    )


def _synthetic_processed_data(n_rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    age = rng.normal(67, 12, size=n_rows)
    lactate = rng.gamma(shape=2.0, scale=1.0, size=n_rows)
    map_min = rng.normal(72, 12, size=n_rows)
    shock_index = rng.normal(0.85, 0.22, size=n_rows)
    score = 0.04 * (age - 65) + 0.75 * lactate - 0.05 * (map_min - 70) + shock_index
    mortality = (score > np.quantile(score, 0.72)).astype(int)

    return pd.DataFrame(
        {
            "subject_id": np.arange(n_rows),
            "age": age,
            "lactate_max_0_6h": lactate,
            "map_min_0_6h": map_min,
            "shock_index_max_0_6h": shock_index,
            "sex": np.where(np.arange(n_rows) % 2 == 0, "F", "M"),
            "mortality": mortality,
        }
    )


def test_model_suite_writes_aggregate_reports_only(tmp_path):
    data_path = tmp_path / "synthetic_processed.csv"
    _synthetic_processed_data().to_csv(data_path, index=False)

    config_dir = tmp_path / "configs"
    _write_fast_configs(config_dir)

    out = tmp_path / "suite"
    result = run_model_suite(
        data_path=data_path,
        output_dir=out,
        models=["logistic", "random_forest", "extra_trees"],
        config_dir=config_dir,
        bootstrap_iterations=0,
        seed=7,
    )

    statuses = result["summary"]["statuses"]
    assert {row["status"] for row in statuses} == {"completed"}
    assert set(result["comparison"]["model_name"]) == {
        "logistic",
        "random_forest",
        "extra_trees",
    }

    all_results = result["results"]
    assert set(all_results["split"]) == {"validation", "test"}
    assert set(all_results["threshold_policy"]) == {
        "high_sensitivity",
        "balanced_f1",
        "high_precision",
    }
    assert (out / "model_suite_results.csv").exists()
    assert (out / "model_comparison_table.csv").exists()
    assert (out / "model_suite_status.json").exists()
    assert (out / "logistic" / "threshold_policy_report.json").exists()
    assert (out / "logistic" / "calibration_curve.png").exists()

    saved_prediction_files = [
        path for path in out.rglob("*") if "prediction" in path.name.lower()
    ]
    assert saved_prediction_files == []


def test_logistic_wrapper_save_load_round_trip(tmp_path):
    df = _synthetic_processed_data(64)
    X = df[["age", "lactate_max_0_6h", "map_min_0_6h", "shock_index_max_0_6h"]].to_numpy()
    y = df["mortality"].to_numpy()

    model = LogisticRiskModel(params={"max_iter": 500})
    model.fit(X, y)
    before = model.predict_proba(X[:8])

    path = tmp_path / "logistic.joblib"
    model.save(path)
    loaded = LogisticRiskModel.load(path)
    after = loaded.predict_proba(X[:8])

    assert np.allclose(before, after)


def test_model_suite_runner_supports_configured_randomized_search(tmp_path):
    data_path = tmp_path / "synthetic_processed.csv"
    _synthetic_processed_data().to_csv(data_path, index=False)

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "logistic.yaml").write_text(
        """
model:
  params:
    class_weight: balanced
    max_iter: 500
  search:
    enabled: true
    scoring: average_precision
    cv: 2
    n_iter: 2
    n_jobs: 1
    param_distributions:
      model__C: [0.1, 1.0]
""".strip(),
        encoding="utf-8",
    )

    result = run_model_suite(
        data_path=data_path,
        output_dir=tmp_path / "suite",
        models=["logistic"],
        config_dir=config_dir,
        bootstrap_iterations=0,
        seed=11,
    )

    status = result["summary"]["statuses"][0]
    assert status["status"] == "completed"
    assert status["search_enabled"] is True
    assert status["best_params"]["model__C"] in {0.1, 1.0}


def test_model_registry_errors_on_unknown_model():
    assert get_model_class("logistic") is LogisticRiskModel
    with pytest.raises(ValueError, match="Unknown tabular model"):
        get_model_class("unknown_model")
