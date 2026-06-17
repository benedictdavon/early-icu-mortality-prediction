from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.temporal_dataset import TemporalFusionDataset
from evaluation.mafnet_eval import predict_mafnet_probabilities
from models.mafnet import MAFNet
from training.losses import compute_pos_weight
from training.train_mafnet import train_mafnet_from_frames, train_mafnet_one_epoch


def _dataset(n_rows: int = 10):
    rng = np.random.default_rng(21)
    mask = (rng.random((n_rows, 24, 43)) > 0.45).astype("float32")
    x = rng.normal(size=(n_rows, 24, 43)).astype("float32") * mask
    bundle = {
        "stay_ids": np.arange(n_rows),
        "channels": [f"ch_{idx}" for idx in range(43)],
        "x_temporal": x,
        "mask_temporal": mask,
        "delta_temporal": rng.random((n_rows, 24, 43)).astype("float32"),
        "count_temporal": rng.random((n_rows, 24, 43)).astype("float32"),
    }
    y = np.asarray([0, 1] * (n_rows // 2), dtype="float32")
    return TemporalFusionDataset(
        bundle,
        y=y,
        x_static=rng.normal(size=(n_rows, 3)).astype("float32"),
        x_aggregate=rng.normal(size=(n_rows, 5)).astype("float32"),
    )


def test_one_synthetic_epoch_runs_without_nans_and_predicts_probabilities():
    torch.manual_seed(22)
    dataset = _dataset()
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    model = MAFNet(
        n_temporal_channels=43,
        n_static_features=3,
        n_aggregate_features=5,
        hidden_dim=32,
        n_heads=4,
        transformer_ff_dim=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_mafnet_one_epoch(
        model,
        loader,
        optimizer,
        pos_weight=compute_pos_weight(dataset.y),
    )
    y_true, probabilities = predict_mafnet_probabilities(model, loader)

    assert losses["loss"] > 0
    assert all(np.isfinite(value) for value in losses.values())
    assert y_true.shape == (10,)
    assert probabilities.shape == (10,)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_mafnet_training_runner_writes_local_aggregate_artifacts(tmp_path):
    n_rows = 24
    stay_ids = np.arange(1000, 1000 + n_rows)
    cohort = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "subject_id": stay_ids,
            "intime": pd.Timestamp("2026-01-01 00:00:00"),
        }
    )
    events = pd.DataFrame(
        [
            {
                "stay_id": stay_id,
                "charttime": pd.Timestamp("2026-01-01 00:05:00"),
                "variable": "heart_rate",
                "valuenum": 70.0 + (idx % 6) * 5,
            }
            for idx, stay_id in enumerate(stay_ids)
        ]
        + [
            {
                "stay_id": stay_id,
                "charttime": pd.Timestamp("2026-01-01 00:10:00"),
                "variable": "sbp",
                "valuenum": 110.0 - (idx % 4) * 5,
            }
            for idx, stay_id in enumerate(stay_ids)
        ]
        + [
            {
                "stay_id": stay_id,
                "charttime": pd.Timestamp("2026-01-01 01:00:00"),
                "variable": "lactate",
                "valuenum": 1.0 + (idx % 5) * 0.5,
            }
            for idx, stay_id in enumerate(stay_ids)
        ]
    )
    features = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "subject_id": stay_ids,
            "mortality": np.asarray([0, 1] * (n_rows // 2)),
            "age": np.linspace(45, 85, n_rows),
            "gender": np.where(stay_ids % 2 == 0, "F", "M"),
            "lactate_max": 1.0 + (np.arange(n_rows) % 5) * 0.5,
            "map_min": 65.0 + (np.arange(n_rows) % 4),
        }
    )

    result = train_mafnet_from_frames(
        events,
        cohort,
        features,
        output_dir=tmp_path / "mafnet_run",
        config={
            "data": {"valid_size": 0.20, "test_size": 0.20},
            "model": {
                "temporal_hidden_dim": 16,
                "transformer_heads": 4,
                "transformer_ff_dim": 32,
            },
            "training": {
                "seed": 123,
                "batch_size": 4,
                "pretrain_epochs": 1,
                "max_epochs": 2,
                "early_stopping_patience": 2,
            },
        },
        evaluate_test=True,
    )

    assert result.best_checkpoint_path.exists()
    assert result.pretrained_checkpoint_path is not None
    assert result.pretrained_checkpoint_path.exists()
    assert (result.output_dir / "training_history.json").exists()
    assert (result.output_dir / "validation_metrics.json").exists()
    assert (result.output_dir / "validation_raw_metrics.json").exists()
    assert result.platt_calibrator_path is not None
    assert result.platt_calibrator_path.exists()
    assert result.validation_calibration_path is not None
    assert result.validation_calibration_path.exists()
    assert (result.output_dir / "validation_calibration_curve.png").exists()
    assert (result.output_dir / "test_metrics.json").exists()
    assert (result.output_dir / "test_raw_metrics.json").exists()
    assert result.test_calibration_path is not None
    assert result.test_calibration_path.exists()
    assert (result.output_dir / "test_calibration_curve.png").exists()
    assert (result.output_dir / "training_curves.png").exists()
    assert result.validation_metrics["average_precision"] >= 0
    assert result.validation_metrics["probability_source"] == "platt_calibrated"
    assert result.test_metrics is not None
    assert result.test_metrics["probability_source"] == "platt_calibrated"
