"""Loss and masking utilities for ICU6H-MAFNet."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MAFNetLossConfig:
    lambda_recon: float = 0.05
    lambda_mask: float = 0.01
    random_observed_mask_rate: float = 0.15


def compute_pos_weight(y_train) -> float:
    """Return sqrt(n_negative / n_positive) for weighted BCE."""
    y = torch.as_tensor(y_train, dtype=torch.float32)
    positives = float(torch.sum(y == 1).item())
    negatives = float(torch.sum(y == 0).item())
    if positives <= 0 or negatives <= 0:
        return 1.0
    return float(math.sqrt(negatives / positives))


def corrupt_observed_values(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_rate: float = 0.15,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly hide observed temporal values for reconstruction training."""
    observed = mask.bool()
    random_draw = torch.rand(
        x.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    hide = observed & (random_draw < mask_rate)
    x_corrupt = x.clone()
    mask_corrupt = mask.clone()
    x_corrupt[hide] = 0.0
    mask_corrupt[hide] = 0.0
    return x_corrupt, mask_corrupt, hide.float()


def compute_mafnet_loss(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
    x_true: torch.Tensor,
    mask_next_target: torch.Tensor,
    recon_target_mask: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor = 1.0,
    lambda_recon: float = 0.05,
    lambda_mask: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute supervised MAFNet mortality plus auxiliary losses."""
    mortality_logit = outputs["mortality_logit"]
    x_recon = outputs["x_recon"]
    mask_next_logit = outputs["mask_next_logit"]
    y = y.float()
    pos_weight_tensor = torch.as_tensor(pos_weight, dtype=y.dtype, device=y.device)
    mortality_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    mortality_loss = mortality_loss_fn(mortality_logit, y)

    if recon_target_mask.sum() > 0:
        reconstruction_loss = F.smooth_l1_loss(
            x_recon[recon_target_mask.bool()],
            x_true[recon_target_mask.bool()],
        )
    else:
        reconstruction_loss = x_true.new_tensor(0.0)

    mask_forecast_loss = F.binary_cross_entropy_with_logits(
        mask_next_logit,
        mask_next_target.float(),
    )
    total = (
        mortality_loss
        + lambda_recon * reconstruction_loss
        + lambda_mask * mask_forecast_loss
    )
    return {
        "loss": total,
        "mortality_loss": mortality_loss,
        "reconstruction_loss": reconstruction_loss,
        "mask_forecast_loss": mask_forecast_loss,
    }


def compute_pretraining_loss(
    outputs: dict[str, torch.Tensor],
    x_true: torch.Tensor,
    mask_next_target: torch.Tensor,
    recon_target_mask: torch.Tensor,
    *,
    lambda_mask: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Compute self-supervised reconstruction and next-measurement losses."""
    x_recon = outputs["x_recon"]
    mask_next_logit = outputs["mask_next_logit"]
    if recon_target_mask.sum() > 0:
        reconstruction_loss = F.smooth_l1_loss(
            x_recon[recon_target_mask.bool()],
            x_true[recon_target_mask.bool()],
        )
    else:
        reconstruction_loss = x_true.new_tensor(0.0)
    mask_forecast_loss = F.binary_cross_entropy_with_logits(
        mask_next_logit,
        mask_next_target.float(),
    )
    total = reconstruction_loss + lambda_mask * mask_forecast_loss
    return {
        "loss": total,
        "reconstruction_loss": reconstruction_loss,
        "mask_forecast_loss": mask_forecast_loss,
    }


__all__ = [
    "MAFNetLossConfig",
    "compute_mafnet_loss",
    "compute_pos_weight",
    "compute_pretraining_loss",
    "corrupt_observed_values",
]
