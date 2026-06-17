from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import assert_no_leakage_columns, build_feature_matrix
from tools.check_leakage import prepare_model_matrix


def test_target_ids_and_obvious_leakage_columns_are_removed():
    df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "hadm_id": [10, 20, 30],
            "stay_id": [100, 200, 300],
            "mortality": [0, 1, 0],
            "death_time": ["", "2025-01-01", ""],
            "discharge_location": ["home", "expired", "home"],
            "duration_hours": [12.0, 8.0, 20.0],
            "age": [55, 70, 80],
            "lactate_max": [1.1, 3.4, 2.2],
        }
    )

    X, dropped = build_feature_matrix(df)

    assert list(X.columns) == ["age", "lactate_max"]
    assert {"subject_id", "hadm_id", "stay_id", "mortality"}.issubset(dropped)
    assert "death_time" in dropped
    assert "discharge_location" in dropped
    assert "duration_hours" in dropped
    assert_no_leakage_columns(X.columns)


def test_model_loader_fits_imputer_on_training_only(monkeypatch, tmp_path):
    import models.base.model as base_model_module
    from models.base.model import ICUMortalityBaseModel

    class TrackingImputer:
        fit_shapes = []
        transform_shapes = []

        def __init__(self, strategy="median"):
            self.strategy = strategy
            self.fill_values = None

        def fit_transform(self, X):
            TrackingImputer.fit_shapes.append(X.shape)
            self.fill_values = X.median(numeric_only=True)
            return X.fillna(self.fill_values).to_numpy()

        def transform(self, X):
            TrackingImputer.transform_shapes.append(X.shape)
            return X.fillna(self.fill_values).to_numpy()

    class DummyModel(ICUMortalityBaseModel):
        def tune_hyperparameters(self, cv=5, n_iter=50):
            return {}

        def train(self, params=None):
            return None

    monkeypatch.setattr(base_model_module, "SimpleImputer", TrackingImputer)

    n = 30
    df = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "mortality": [0, 1] * (n // 2),
            "age": np.linspace(40, 80, n),
            "lactate_max": [np.nan if i % 5 == 0 else float(i) for i in range(n)],
            "duration_hours": np.linspace(6, 30, n),
        }
    )
    data_path = tmp_path / "synthetic_processed.csv"
    df.to_csv(data_path, index=False)

    model = DummyModel(output_dir=str(tmp_path))
    model.load_data(data_path)

    assert len(TrackingImputer.fit_shapes) == 1
    assert len(TrackingImputer.transform_shapes) == 2
    assert TrackingImputer.fit_shapes[0][0] == len(model.split_indices["train"])
    assert TrackingImputer.transform_shapes[0][0] == len(model.split_indices["validation"])
    assert TrackingImputer.transform_shapes[1][0] == len(model.split_indices["test"])
    assert "duration_hours" not in model.feature_names
    assert "mortality" not in model.feature_names


def test_leakage_probe_matrix_encodes_categorical_features():
    df = pd.DataFrame(
        {
            "age_group": ["elderly", "adult", None],
            "age": [80.0, np.nan, 55.0],
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )

    X = prepare_model_matrix(df)

    assert X.isna().sum().sum() == 0
    assert "age_group_elderly" in X.columns
    assert "age_group_nan" in X.columns
