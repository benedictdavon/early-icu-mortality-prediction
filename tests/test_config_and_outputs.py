from __future__ import annotations

from datetime import date

from config import load_project_config
from utils.io import build_experiment_output_dir, safe_path_token
from utils.seed import apply_backend_seed, backend_seed_params


def test_project_config_loads_base_without_private_paths():
    config = load_project_config()

    assert config["project"]["name"] == "early_icu_mortality_prediction"
    assert config["project"]["random_seed"] == 42
    assert "project_root" in config
    assert isinstance(config.get("paths", {}), dict)


def test_output_directory_name_is_traceable(tmp_path):
    output = build_experiment_output_dir(
        tmp_path,
        experiment_name="Expanded Features",
        model_name="XGBoost",
        split_name="patient_grouped",
        seed=7,
        run_date=date(2026, 6, 17),
    )

    assert output.exists()
    assert output.name == "2026-06-17__expanded-features__xgboost__patient_grouped__seed7"


def test_backend_seed_params_cover_supported_backends():
    seeds = backend_seed_params(123)

    assert seeds["xgboost"] == {"random_state": 123}
    assert seeds["lightgbm"] == {"random_state": 123}
    assert seeds["catboost"] == {"random_seed": 123}
    assert apply_backend_seed({"max_depth": 3}, "xgboost", 5) == {
        "max_depth": 3,
        "random_state": 5,
    }
    assert safe_path_token("XGBoost + LR") == "xgboost-lr"
