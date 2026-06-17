import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"

base_path = Path(os.getenv("ICU_DATA_DIR", PROJECT_ROOT / "data")).resolve()

hosp_path = str(base_path / "hosp")
icu_path = str(base_path / "icu")
label_path = str(base_path / "label")


def deep_merge(base: dict, override: dict | None) -> dict:
    """Recursively merge dictionaries without mutating inputs."""
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_file(path: str | Path | None) -> dict:
    """Load a YAML file, returning an empty dict when the path is missing."""
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_config(
    config_path: str | Path | None = None,
    paths_config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load base project config plus optional local paths and overrides."""
    base_config = load_yaml_file(config_path or DEFAULT_BASE_CONFIG)
    local_paths = load_yaml_file(paths_config_path or DEFAULT_PATHS_CONFIG)

    env_paths = {
        "data_dir": os.getenv("ICU_DATA_DIR"),
        "processed_dir": os.getenv("ICU_PROCESSED_DIR"),
        "results_dir": os.getenv("ICU_RESULTS_DIR"),
    }
    env_paths = {key: value for key, value in env_paths.items() if value}

    config = deep_merge(base_config, {"paths": local_paths})
    config = deep_merge(config, {"paths": env_paths})
    config = deep_merge(config, overrides or {})
    config["project_root"] = str(PROJECT_ROOT)
    return config
