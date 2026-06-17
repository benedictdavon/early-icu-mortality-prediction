from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.temporal_channels import TEMPORAL_CHANNELS
from features.temporal_tensor_builder import (
    TemporalTensorNormalizer,
    build_15min_temporal_tensors,
    transform_temporal_splits,
)


def test_15min_temporal_tensor_builder_shapes_and_bins():
    cohort = pd.DataFrame(
        {
            "stay_id": [1],
            "intime": [pd.Timestamp("2026-01-01 00:00:00")],
        }
    )
    events = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "charttime": pd.to_datetime(
                [
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:20:00",
                    "2026-01-01 00:40:00",
                    "2026-01-01 06:00:00",
                    "2026-01-01 00:20:00",
                    "2026-01-01 00:20:00",
                    "2026-01-01 00:20:00",
                    "2026-01-01 00:20:00",
                ]
            ),
            "variable": [
                "heart_rate",
                "heart_rate",
                "lactate",
                "lactate",
                "sbp",
                "map",
                "spo2",
                "resp_rate",
            ],
            "valuenum": [80.0, 90.0, 2.5, 3.0, 100.0, 60.0, 88.0, 24.0],
        }
    )

    bundle = build_15min_temporal_tensors(events, cohort)

    assert len(TEMPORAL_CHANNELS) == 43
    assert bundle["x_temporal"].shape == (1, 24, 43)
    hr_last = bundle["channels"].index("heart_rate_last")
    hr_min = bundle["channels"].index("heart_rate_min")
    lactate = bundle["channels"].index("lactate_last")
    respiratory_rate = bundle["channels"].index("respiratory_rate_last")
    shock = bundle["channels"].index("shock_index_bin")
    hypotension = bundle["channels"].index("hypotension_flag_bin")
    hypoxemia = bundle["channels"].index("hypoxemia_flag_bin")
    critical = bundle["channels"].index("critical_value_count_bin")

    assert bundle["x_temporal"][0, 0, hr_last] == 80.0
    assert bundle["x_temporal"][0, 1, hr_last] == 90.0
    assert bundle["x_temporal"][0, 0, hr_min] == 80.0
    assert bundle["x_temporal"][0, 2, lactate] == 2.5
    assert bundle["x_temporal"][0, 23, lactate] == 3.0
    assert bundle["x_temporal"][0, 1, respiratory_rate] == 24.0
    assert bundle["x_temporal"][0, 1, shock] == pytest.approx(0.9)
    assert bundle["x_temporal"][0, 1, hypotension] == 1.0
    assert bundle["x_temporal"][0, 1, hypoxemia] == 1.0
    assert bundle["x_temporal"][0, 1, critical] == 2.0
    assert bundle["mask_temporal"][0, 3, hr_last] == 0.0
    assert bundle["delta_temporal"][0, 2, hr_last] == 0.25
    assert bundle["delta_temporal"][0, 0, lactate] == 6.0


def test_temporal_tensor_normalizer_uses_observed_values_only():
    x = np.array([[[10.0, 0.0], [20.0, 5.0]]], dtype=np.float32)
    mask = np.array([[[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)

    normalizer = TemporalTensorNormalizer().fit(x, mask)
    transformed = normalizer.transform(x, mask)

    assert transformed[0, 0, 0] == -1.0
    assert transformed[0, 1, 0] == 1.0
    assert transformed[0, 0, 1] == 0.0


def test_temporal_bundle_transform_normalizes_delta_and_count():
    bundle = {
        "x_temporal": np.array([[[10.0], [20.0]]], dtype=np.float32),
        "mask_temporal": np.array([[[1.0], [1.0]]], dtype=np.float32),
        "delta_temporal": np.array([[[0.0], [6.0]]], dtype=np.float32),
        "count_temporal": np.array([[[1.0], [12.0]]], dtype=np.float32),
    }

    normalizer = TemporalTensorNormalizer().fit(
        bundle["x_temporal"],
        bundle["mask_temporal"],
    )
    transformed = normalizer.transform_bundle(bundle)

    assert transformed["delta_temporal"][0, 0, 0] == 0.0
    assert transformed["delta_temporal"][0, 1, 0] == 1.0
    assert transformed["count_temporal"][0, 1, 0] == 1.0


def test_temporal_split_transform_fits_training_only():
    train = {
        "x_temporal": np.array([[[10.0]], [[20.0]]], dtype=np.float32),
        "mask_temporal": np.ones((2, 1, 1), dtype=np.float32),
        "delta_temporal": np.zeros((2, 1, 1), dtype=np.float32),
        "count_temporal": np.ones((2, 1, 1), dtype=np.float32),
    }
    validation = {
        "x_temporal": np.array([[[1000.0]]], dtype=np.float32),
        "mask_temporal": np.ones((1, 1, 1), dtype=np.float32),
        "delta_temporal": np.zeros((1, 1, 1), dtype=np.float32),
        "count_temporal": np.ones((1, 1, 1), dtype=np.float32),
    }

    transformed, normalizer = transform_temporal_splits(
        {"train": train, "validation": validation}
    )

    assert normalizer.mean_[0] == 15.0
    assert normalizer.std_[0] == 5.0
    assert transformed["validation"]["x_temporal"][0, 0, 0] == 5.0
