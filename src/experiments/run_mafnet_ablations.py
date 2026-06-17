"""Run MAFNet architecture ablations with aggregate-only outputs."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_mafnet import DEFAULT_CONFIG, load_mafnet_config, train_mafnet_from_frames


MAFNET_ABLATIONS: dict[str, dict] = {
    "MAFNet-T": {
        "model": {"use_static_branch": False, "use_aggregate_branch": False},
    },
    "MAFNet-T+S": {
        "model": {"use_static_branch": True, "use_aggregate_branch": False},
    },
    "MAFNet-T+S+A": {
        "model": {"use_static_branch": True, "use_aggregate_branch": True},
    },
    "NoDecay": {
        "model": {"use_decay": False},
    },
    "NoTransformer": {
        "model": {"use_transformer": False, "transformer_layers": 0},
    },
    "NoAux": {
        "loss": {"lambda_recon": 0.0, "lambda_mask": 0.0},
    },
    "NoGate": {
        "model": {"use_gated_fusion": False},
    },
    "NoPretrain": {
        "training": {"pretrain_epochs": 0},
    },
}


def _deep_merge(base: dict, override: dict | None) -> dict:
    merged = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def mafnet_ablation_config(base_config: dict, ablation_name: str) -> dict:
    """Return a config for one named MAFNet ablation."""
    if ablation_name not in MAFNET_ABLATIONS:
        available = ", ".join(MAFNET_ABLATIONS)
        raise ValueError(f"Unknown MAFNet ablation `{ablation_name}`. Available: {available}")
    cfg = _deep_merge(base_config, MAFNET_ABLATIONS[ablation_name])
    cfg.setdefault("experiment", {})["ablation_name"] = ablation_name
    return cfg


def _metric_record(name: str, result) -> dict:
    record = {
        "ablation": name,
        "output_dir": str(result.output_dir),
        "best_checkpoint_path": str(result.best_checkpoint_path),
        "validation_average_precision": result.validation_metrics.get("average_precision"),
        "validation_auc_roc": result.validation_metrics.get("auc_roc"),
        "validation_brier_score": result.validation_metrics.get("brier_score"),
        "validation_probability_source": result.validation_metrics.get("probability_source"),
    }
    if result.test_metrics:
        record.update(
            {
                "test_average_precision": result.test_metrics.get("average_precision"),
                "test_auc_roc": result.test_metrics.get("auc_roc"),
                "test_brier_score": result.test_metrics.get("brier_score"),
                "test_probability_source": result.test_metrics.get("probability_source"),
            }
        )
    return record


def run_mafnet_ablations_from_frames(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    output_dir,
    config: dict | None = None,
    config_path=DEFAULT_CONFIG,
    ablations: list[str] | tuple[str, ...] | None = None,
    device: str = "cpu",
    evaluate_test: bool = False,
) -> dict:
    """Run selected MAFNet ablations and save aggregate comparison artifacts."""
    base_config = load_mafnet_config(config_path, config)
    selected = list(ablations or MAFNET_ABLATIONS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    statuses = []
    for name in selected:
        run_out = out / _safe_name(name)
        status = {"ablation": name, "status": "started"}
        try:
            result = train_mafnet_from_frames(
                events,
                cohort,
                feature_frame,
                output_dir=run_out,
                config=mafnet_ablation_config(base_config, name),
                config_path=config_path,
                device=device,
                evaluate_test=evaluate_test,
            )
            records.append(_metric_record(name, result))
            status.update({"status": "completed", "output_dir": str(run_out)})
        except Exception as exc:
            status.update({"status": "failed", "reason": str(exc)})
            raise
        finally:
            statuses.append(status)

    table = pd.DataFrame(records)
    if not table.empty:
        sort_cols = [
            col
            for col in ["test_average_precision", "validation_average_precision"]
            if col in table.columns
        ]
        table = table.sort_values(sort_cols, ascending=False)
    table.to_csv(out / "mafnet_ablation_results.csv", index=False)
    summary = {
        "output_dir": str(out),
        "ablations": selected,
        "statuses": statuses,
        "results_path": str(out / "mafnet_ablation_results.csv"),
    }
    with (out / "mafnet_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return {"summary": summary, "results": table}


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("+", "_plus_")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAFNet architecture ablations")
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--cohort-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "mafnet_ablations"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablations", default=",".join(MAFNET_ABLATIONS))
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.ablations.split(",") if item.strip()]
    result = run_mafnet_ablations_from_frames(
        pd.read_csv(args.events_path),
        pd.read_csv(args.cohort_path),
        pd.read_csv(args.features_path),
        output_dir=args.output_dir,
        config_path=args.config,
        ablations=selected,
        device=args.device,
        evaluate_test=args.evaluate_test,
    )
    print(f"MAFNet ablations complete. Results: {result['summary']['results_path']}")


__all__ = [
    "MAFNET_ABLATIONS",
    "mafnet_ablation_config",
    "run_mafnet_ablations_from_frames",
]


if __name__ == "__main__":
    main()
