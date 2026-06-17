from __future__ import annotations

import pytest

from experiments.run_mafnet_ablations import MAFNET_ABLATIONS, mafnet_ablation_config


def test_required_mafnet_ablation_configs_are_defined():
    assert set(MAFNET_ABLATIONS) == {
        "MAFNet-T",
        "MAFNet-T+S",
        "MAFNet-T+S+A",
        "NoDecay",
        "NoTransformer",
        "NoAux",
        "NoGate",
        "NoPretrain",
    }


def test_mafnet_ablation_config_overrides_only_targeted_options():
    base = {
        "model": {
            "use_static_branch": True,
            "use_aggregate_branch": True,
            "use_decay": True,
            "use_transformer": True,
            "use_gated_fusion": True,
        },
        "training": {"pretrain_epochs": 2},
        "loss": {"lambda_recon": 0.05, "lambda_mask": 0.01},
    }

    temporal_only = mafnet_ablation_config(base, "MAFNet-T")
    no_decay = mafnet_ablation_config(base, "NoDecay")
    no_aux = mafnet_ablation_config(base, "NoAux")
    no_pretrain = mafnet_ablation_config(base, "NoPretrain")

    assert temporal_only["model"]["use_static_branch"] is False
    assert temporal_only["model"]["use_aggregate_branch"] is False
    assert no_decay["model"]["use_decay"] is False
    assert no_aux["loss"]["lambda_recon"] == 0.0
    assert no_aux["loss"]["lambda_mask"] == 0.0
    assert no_pretrain["training"]["pretrain_epochs"] == 0
    assert base["model"]["use_decay"] is True


def test_unknown_mafnet_ablation_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown MAFNet ablation"):
        mafnet_ablation_config({}, "unknown")
