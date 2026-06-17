from __future__ import annotations

import numpy as np
import pandas as pd

from features.temporal_channels import TEMPORAL_CHANNELS
from features.temporal_tensor_builder import (
    TemporalTensorNormalizer,
    build_15min_temporal_tensors,
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
            "stay_id": [1, 1, 1, 1],
            "charttime": pd.to_datetime(
                [
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:20:00",
                    "2026-01-01 00:40:00",
                    "2026-01-01 06:00:00",
                ]
            ),
            "variable": ["heart_rate", "heart_rate", "lactate", "shock_index_bin"],
            "valuenum": [80.0, 90.0, 2.5, 1.2],
        }
    )

    bundle = build_15min_temporal_tensors(events, cohort)

    assert len(TEMPORAL_CHANNELS) == 43
    assert bundle["x_temporal"].shape == (1, 24, 43)
    hr_last = bundle["channels"].index("heart_rate_last")
    hr_min = bundle["channels"].index("heart_rate_min")
    lactate = bundle["channels"].index("lactate_last")
    shock = bundle["channels"].index("shock_index_bin")

    assert bundle["x_temporal"][0, 0, hr_last] == 80.0
    assert bundle["x_temporal"][0, 1, hr_last] == 90.0
    assert bundle["x_temporal"][0, 0, hr_min] == 80.0
    assert bundle["x_temporal"][0, 2, lactate] == 2.5
    assert bundle["x_temporal"][0, 23, shock] == 1.2
    assert bundle["mask_temporal"][0, 3, hr_last] == 0.0
    assert bundle["delta_temporal"][0, 2, hr_last] == 0.25


def test_temporal_tensor_normalizer_uses_observed_values_only():
    x = np.array([[[10.0, 0.0], [20.0, 5.0]]], dtype=np.float32)
    mask = np.array([[[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)

    normalizer = TemporalTensorNormalizer().fit(x, mask)
    transformed = normalizer.transform(x, mask)

    assert transformed[0, 0, 0] == -1.0
    assert transformed[0, 1, 0] == 1.0
    assert transformed[0, 0, 1] == 0.0
