from __future__ import annotations

import pytest
import torch

from models.mafnet import MAFNet
from training.losses import (
    compute_mafnet_loss,
    compute_pos_weight,
    compute_pretraining_loss,
    corrupt_observed_values,
)


def _batch(batch_size: int = 3):
    torch.manual_seed(11)
    mask = (torch.rand(batch_size, 24, 43) > 0.4).float()
    x = torch.randn(batch_size, 24, 43) * mask
    return {
        "x_temporal": x,
        "mask_temporal": mask,
        "delta_temporal": torch.rand(batch_size, 24, 43),
        "count_temporal": torch.rand(batch_size, 24, 43),
        "x_static": torch.randn(batch_size, 4),
        "x_aggregate": torch.randn(batch_size, 6),
        "y": torch.tensor([0.0, 1.0, 0.0]),
    }


def test_corrupt_observed_values_only_hides_observed_entries():
    batch = _batch()
    x_corrupt, mask_corrupt, target_mask = corrupt_observed_values(
        batch["x_temporal"],
        batch["mask_temporal"],
        mask_rate=1.0,
    )

    assert torch.equal(target_mask, batch["mask_temporal"])
    assert torch.count_nonzero(x_corrupt) == 0
    assert torch.count_nonzero(mask_corrupt) == 0


def test_compute_pos_weight_uses_sqrt_neg_pos_ratio():
    assert compute_pos_weight([0, 0, 0, 1]) == pytest.approx(3**0.5)
    assert compute_pos_weight([0, 0, 0]) == 1.0


def test_supervised_and_pretraining_losses_are_finite():
    batch = _batch()
    model = MAFNet(
        n_temporal_channels=43,
        n_static_features=4,
        n_aggregate_features=6,
        hidden_dim=32,
        n_heads=4,
        transformer_ff_dim=64,
    )
    x_corrupt, mask_corrupt, target_mask = corrupt_observed_values(
        batch["x_temporal"],
        batch["mask_temporal"],
        mask_rate=0.5,
    )
    output = model(
        x_corrupt,
        mask_corrupt,
        batch["delta_temporal"],
        batch["count_temporal"],
        batch["x_static"],
        batch["x_aggregate"],
    )

    supervised = compute_mafnet_loss(
        output,
        batch["y"],
        batch["x_temporal"],
        batch["mask_temporal"][:, 1:, :],
        target_mask,
        pos_weight=compute_pos_weight(batch["y"]),
    )
    pretrain = compute_pretraining_loss(
        output,
        batch["x_temporal"],
        batch["mask_temporal"][:, 1:, :],
        target_mask,
    )

    assert supervised["loss"].requires_grad
    assert torch.isfinite(supervised["loss"])
    assert torch.isfinite(pretrain["loss"])
