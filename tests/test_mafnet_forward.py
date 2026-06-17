from __future__ import annotations

import torch

from models.mafnet import MAFNet, MissingnessAwareTemporalEncoder


def _batch(batch_size: int = 4, n_static: int = 5, n_aggregate: int = 7):
    torch.manual_seed(7)
    x = torch.randn(batch_size, 24, 43)
    mask = (torch.rand(batch_size, 24, 43) > 0.35).float()
    x = x * mask
    return {
        "x_temporal": x,
        "mask_temporal": mask,
        "delta_temporal": torch.rand(batch_size, 24, 43),
        "count_temporal": torch.rand(batch_size, 24, 43),
        "x_static": torch.randn(batch_size, n_static),
        "x_aggregate": torch.randn(batch_size, n_aggregate),
        "y": torch.randint(0, 2, (batch_size,)).float(),
    }


def test_missingness_aware_temporal_encoder_shapes():
    batch = _batch()
    encoder = MissingnessAwareTemporalEncoder(
        n_channels=43,
        hidden_dim=32,
        n_heads=4,
        transformer_ff_dim=64,
    )

    output = encoder(
        batch["x_temporal"],
        batch["mask_temporal"],
        batch["delta_temporal"],
        batch["count_temporal"],
    )

    assert output["patient_embedding"].shape == (4, 32)
    assert output["sequence_embedding"].shape == (4, 24, 32)
    assert output["attention_weights"].shape == (4, 24)
    assert torch.allclose(
        output["attention_weights"].sum(dim=1),
        torch.ones(4),
        atol=1e-5,
    )


def test_mafnet_forward_output_contract():
    batch = _batch()
    model = MAFNet(
        n_temporal_channels=43,
        n_static_features=5,
        n_aggregate_features=7,
        hidden_dim=32,
        n_heads=4,
        transformer_ff_dim=64,
    )

    output = model(
        batch["x_temporal"],
        batch["mask_temporal"],
        batch["delta_temporal"],
        batch["count_temporal"],
        batch["x_static"],
        batch["x_aggregate"],
    )

    assert set(output) == {
        "mortality_logit",
        "x_recon",
        "mask_next_logit",
        "gate_weights",
        "temporal_attention",
    }
    assert output["mortality_logit"].shape == (4,)
    assert output["x_recon"].shape == (4, 24, 43)
    assert output["mask_next_logit"].shape == (4, 23, 43)
    assert output["gate_weights"].shape == (4, 3)
    assert output["temporal_attention"].shape == (4, 24)
    assert torch.allclose(output["gate_weights"].sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(output["mortality_logit"]).all()


def test_mafnet_ablation_switches_keep_forward_contract():
    batch = _batch(batch_size=3, n_static=5, n_aggregate=7)
    common_kwargs = {
        "n_temporal_channels": 43,
        "n_static_features": 5,
        "n_aggregate_features": 7,
        "hidden_dim": 32,
        "n_heads": 4,
        "transformer_ff_dim": 64,
    }
    variants = [
        MAFNet(
            **common_kwargs,
            use_static_branch=False,
            use_aggregate_branch=False,
        ),
        MAFNet(**common_kwargs, use_transformer=False, transformer_layers=0),
        MAFNet(**common_kwargs, use_decay=False),
        MAFNet(**common_kwargs, use_gated_fusion=False),
    ]

    for model in variants:
        output = model(
            batch["x_temporal"],
            batch["mask_temporal"],
            batch["delta_temporal"],
            batch["count_temporal"],
            batch["x_static"],
            batch["x_aggregate"],
        )
        assert output["mortality_logit"].shape == (3,)
        assert output["x_recon"].shape == (3, 24, 43)
        assert output["mask_next_logit"].shape == (3, 23, 43)
        assert output["gate_weights"].shape == (3, 3)
