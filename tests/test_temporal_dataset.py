from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.temporal_dataset import (
    TemporalFusionDataset,
    build_static_aggregate_matrices,
    build_temporal_fusion_dataset,
)


def _temporal_bundle():
    return {
        "stay_ids": np.array([101, 102]),
        "channels": ["heart_rate_last"],
        "x_temporal": np.ones((2, 24, 1), dtype=np.float32),
        "mask_temporal": np.ones((2, 24, 1), dtype=np.float32),
        "delta_temporal": np.zeros((2, 24, 1), dtype=np.float32),
        "count_temporal": np.ones((2, 24, 1), dtype=np.float32),
    }


def test_temporal_fusion_dataset_returns_torch_tensors():
    dataset = TemporalFusionDataset(
        _temporal_bundle(),
        y=np.array([0, 1], dtype=np.float32),
        x_static=np.ones((2, 2), dtype=np.float32),
        x_aggregate=np.ones((2, 3), dtype=np.float32),
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["x_temporal"].shape == (24, 1)
    assert sample["x_static"].shape == (2,)
    assert sample["x_aggregate"].shape == (3,)
    assert isinstance(sample["y"], torch.Tensor)


def test_build_static_aggregate_matrices_aligns_by_stay_id_and_drops_leakage():
    features = pd.DataFrame(
        {
            "stay_id": [102, 101],
            "subject_id": [2, 1],
            "mortality": [1, 0],
            "age": [80.0, 60.0],
            "gender_numeric": [1.0, 0.0],
            "lactate_max": [4.0, 1.5],
            "deathtime": ["x", "y"],
        }
    )

    matrices = build_static_aggregate_matrices(features, [101, 102])

    assert matrices.y.tolist() == [0.0, 1.0]
    assert matrices.static_columns == ["age", "gender_numeric"]
    assert matrices.x_static[:, 0].tolist() == [60.0, 80.0]
    assert matrices.aggregate_columns == ["lactate_max"]
    assert matrices.x_aggregate[:, 0].tolist() == [1.5, 4.0]


def test_build_temporal_fusion_dataset_uses_aligned_feature_matrices():
    features = pd.DataFrame(
        {
            "stay_id": [101, 102],
            "mortality": [0, 1],
            "age": [60.0, 80.0],
            "lactate_max": [1.5, 4.0],
        }
    )

    dataset, matrices = build_temporal_fusion_dataset(_temporal_bundle(), features)

    assert len(dataset) == 2
    assert matrices.static_columns == ["age"]
    assert matrices.aggregate_columns == ["lactate_max"]
    assert dataset[1]["y"].item() == 1.0
