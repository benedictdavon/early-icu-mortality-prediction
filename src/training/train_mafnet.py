"""Training runner and train-step helpers for MAFNet."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from data.schema import build_feature_matrix
from data.splitting import make_train_valid_test_split
from data.temporal_dataset import (
    DEFAULT_STATIC_FEATURE_CANDIDATES,
    TemporalFusionDataset,
)
from features.temporal_tensor_builder import (
    build_15min_temporal_tensors,
    transform_temporal_splits,
)
from evaluation.calibration import (
    calibration_summary,
    fit_isotonic_calibrator,
    fit_platt_scaler,
    logits_to_probabilities,
)
from evaluation.plots import save_calibration_curve
from models.mafnet import MAFNet
from training.callbacks import EarlyStopping
from training.losses import (
    MAFNetLossConfig,
    compute_mafnet_loss,
    compute_pos_weight,
    compute_pretraining_loss,
    corrupt_observed_values,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "mafnet.yaml"


@dataclass(frozen=True)
class MAFNetDataBundle:
    datasets: dict[str, TemporalFusionDataset]
    splits: dict[str, np.ndarray]
    temporal_normalizer: object
    branch_preprocessor: "BranchFeaturePreprocessor"


@dataclass(frozen=True)
class MAFNetRunResult:
    output_dir: Path
    history: list[dict]
    validation_metrics: dict
    test_metrics: dict | None
    best_checkpoint_path: Path
    pretrained_checkpoint_path: Path | None
    summary_path: Path
    platt_calibrator_path: Path | None
    isotonic_calibrator_path: Path | None
    calibration_model_path: Path | None
    validation_calibration_path: Path | None
    test_calibration_path: Path | None


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["x_temporal"],
        batch["mask_temporal"],
        batch["delta_temporal"],
        batch["count_temporal"],
        batch["x_static"],
        batch["x_aggregate"],
    )


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _simple_imputer(strategy: str) -> SimpleImputer:
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


def _build_column_preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer | None:
    if not columns:
        return None
    numeric_cols = frame[columns].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", _simple_imputer("median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", _simple_imputer("most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_names(preprocessor: ColumnTransformer | None) -> list[str]:
    if preprocessor is None:
        return []
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


class BranchFeaturePreprocessor:
    """Train-only preprocessing for MAFNet static and aggregate branches."""

    def __init__(
        self,
        *,
        static_candidates: tuple[str, ...] | list[str] = DEFAULT_STATIC_FEATURE_CANDIDATES,
        target_col: str = "mortality",
    ) -> None:
        self.static_candidates = tuple(static_candidates)
        self.target_col = target_col
        self.static_columns_: list[str] = []
        self.aggregate_columns_: list[str] = []
        self.static_feature_names_: list[str] = []
        self.aggregate_feature_names_: list[str] = []
        self.static_preprocessor_: ColumnTransformer | None = None
        self.aggregate_preprocessor_: ColumnTransformer | None = None
        self.safe_feature_frame_: pd.DataFrame | None = None

    def fit(self, aligned_frame: pd.DataFrame, train_indices: np.ndarray) -> "BranchFeaturePreprocessor":
        safe_features, _ = build_feature_matrix(aligned_frame, target_col=self.target_col)
        self.safe_feature_frame_ = safe_features
        static_columns = [
            col for col in self.static_candidates if col in safe_features.columns
        ]
        if "gender" in safe_features.columns and "gender" not in static_columns:
            static_columns.append("gender")
        self.static_columns_ = static_columns
        self.aggregate_columns_ = [
            col for col in safe_features.columns if col not in self.static_columns_
        ]

        train_frame = safe_features.iloc[np.asarray(train_indices, dtype=int)]
        self.static_preprocessor_ = _build_column_preprocessor(
            safe_features,
            self.static_columns_,
        )
        self.aggregate_preprocessor_ = _build_column_preprocessor(
            safe_features,
            self.aggregate_columns_,
        )
        if self.static_preprocessor_ is not None:
            self.static_preprocessor_.fit(train_frame[self.static_columns_])
        if self.aggregate_preprocessor_ is not None:
            self.aggregate_preprocessor_.fit(train_frame[self.aggregate_columns_])
        self.static_feature_names_ = _feature_names(self.static_preprocessor_)
        self.aggregate_feature_names_ = _feature_names(self.aggregate_preprocessor_)
        return self

    def transform(self, aligned_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.safe_feature_frame_ is None:
            raise ValueError("BranchFeaturePreprocessor must be fit before transform")
        safe_features, _ = build_feature_matrix(aligned_frame, target_col=self.target_col)
        n_rows = len(safe_features)
        if self.static_preprocessor_ is None:
            x_static = np.zeros((n_rows, 0), dtype=np.float32)
        else:
            x_static = self.static_preprocessor_.transform(
                safe_features[self.static_columns_]
            ).astype(np.float32)
        if self.aggregate_preprocessor_ is None:
            x_aggregate = np.zeros((n_rows, 0), dtype=np.float32)
        else:
            x_aggregate = self.aggregate_preprocessor_.transform(
                safe_features[self.aggregate_columns_]
            ).astype(np.float32)
        return {"x_static": x_static, "x_aggregate": x_aggregate}


def _align_features_to_stays(
    feature_frame: pd.DataFrame,
    stay_ids,
    *,
    id_col: str = "stay_id",
    target_col: str = "mortality",
) -> pd.DataFrame:
    if id_col not in feature_frame.columns:
        raise ValueError(f"feature_frame must contain `{id_col}`")
    if target_col not in feature_frame.columns:
        raise ValueError(f"feature_frame must contain `{target_col}`")
    indexed = feature_frame.drop_duplicates(subset=[id_col]).set_index(id_col)
    missing = [stay_id for stay_id in stay_ids if stay_id not in indexed.index]
    if missing:
        raise ValueError(f"feature_frame is missing {len(missing)} temporal stay ids")
    return indexed.loc[list(stay_ids)].reset_index()


def subset_temporal_bundle(bundle: dict, indices) -> dict:
    """Subset row-aligned temporal arrays while preserving bundle metadata."""
    idx = np.asarray(indices, dtype=int)
    subset = dict(bundle)
    for key in ["stay_ids", "x_temporal", "mask_temporal", "delta_temporal", "count_temporal"]:
        subset[key] = np.asarray(bundle[key])[idx]
    return subset


def prepare_mafnet_datasets(
    raw_temporal_bundle: dict,
    feature_frame: pd.DataFrame,
    *,
    target_col: str = "mortality",
    patient_id_col: str = "subject_id",
    valid_size: float = 0.20,
    test_size: float = 0.20,
    seed: int = 42,
    evaluate_test: bool = False,
) -> MAFNetDataBundle:
    """Create train/validation(/test) datasets with train-only fitting."""
    aligned = _align_features_to_stays(
        feature_frame,
        raw_temporal_bundle["stay_ids"],
        target_col=target_col,
    )
    y = aligned[target_col].astype(int).to_numpy()
    patient_ids = aligned[patient_id_col] if patient_id_col in aligned.columns else None
    splits = make_train_valid_test_split(
        y,
        patient_ids=patient_ids,
        valid_size=valid_size,
        test_size=test_size,
        stratify=True,
        group_by_patient=patient_ids is not None,
        random_state=seed,
    )

    requested_splits = ["train", "validation"] + (["test"] if evaluate_test else [])
    raw_split_bundles = {
        split_name: subset_temporal_bundle(raw_temporal_bundle, splits[split_name])
        for split_name in requested_splits
    }
    transformed_bundles, temporal_normalizer = transform_temporal_splits(
        raw_split_bundles,
        train_split="train",
    )
    branch_preprocessor = BranchFeaturePreprocessor(target_col=target_col).fit(
        aligned,
        splits["train"],
    )
    branch_arrays = branch_preprocessor.transform(aligned)

    datasets = {}
    for split_name in requested_splits:
        idx = splits[split_name]
        datasets[split_name] = TemporalFusionDataset(
            transformed_bundles[split_name],
            y=y[idx],
            x_static=branch_arrays["x_static"][idx],
            x_aggregate=branch_arrays["x_aggregate"][idx],
        )

    return MAFNetDataBundle(
        datasets=datasets,
        splits=splits,
        temporal_normalizer=temporal_normalizer,
        branch_preprocessor=branch_preprocessor,
    )


def mafnet_supervised_train_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    pos_weight: float = 1.0,
    loss_config: MAFNetLossConfig | None = None,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Run one supervised full-model optimization step."""
    config = loss_config or MAFNetLossConfig()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_corrupt, mask_corrupt, recon_target_mask = corrupt_observed_values(
        batch["x_temporal"],
        batch["mask_temporal"],
        mask_rate=config.random_observed_mask_rate,
    )
    outputs = model(
        x_corrupt,
        mask_corrupt,
        batch["delta_temporal"],
        batch["count_temporal"],
        batch["x_static"],
        batch["x_aggregate"],
    )
    losses = compute_mafnet_loss(
        outputs,
        batch["y"],
        batch["x_temporal"],
        batch["mask_temporal"][:, 1:, :],
        recon_target_mask,
        pos_weight=pos_weight,
        lambda_recon=config.lambda_recon,
        lambda_mask=config.lambda_mask,
    )
    losses["loss"].backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def mafnet_pretrain_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    mask_rate: float = 0.15,
    lambda_mask: float = 0.10,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Run one self-supervised temporal optimization step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_corrupt, mask_corrupt, recon_target_mask = corrupt_observed_values(
        batch["x_temporal"],
        batch["mask_temporal"],
        mask_rate=mask_rate,
    )
    outputs = model(
        x_corrupt,
        mask_corrupt,
        batch["delta_temporal"],
        batch["count_temporal"],
        batch["x_static"],
        batch["x_aggregate"],
    )
    losses = compute_pretraining_loss(
        outputs,
        batch["x_temporal"],
        batch["mask_temporal"][:, 1:, :],
        recon_target_mask,
        lambda_mask=lambda_mask,
    )
    losses["loss"].backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def train_mafnet_one_epoch(
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    pos_weight: float = 1.0,
    loss_config: MAFNetLossConfig | None = None,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Train a full MAFNet model for one epoch and return mean losses."""
    totals: dict[str, float] = {}
    n_batches = 0
    model.to(device)
    for batch in loader:
        batch = batch_to_device(batch, device)
        losses = mafnet_supervised_train_step(
            model,
            batch,
            optimizer,
            pos_weight=pos_weight,
            loss_config=loss_config,
            gradient_clip_norm=gradient_clip_norm,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    if n_batches == 0:
        raise ValueError("loader produced no batches")
    return {key: value / n_batches for key, value in totals.items()}


def pretrain_mafnet_one_epoch(
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    mask_rate: float = 0.15,
    lambda_mask: float = 0.10,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Pretrain temporal and auxiliary modules for one epoch."""
    totals: dict[str, float] = {}
    n_batches = 0
    model.to(device)
    for batch in loader:
        batch = batch_to_device(batch, device)
        losses = mafnet_pretrain_step(
            model,
            batch,
            optimizer,
            mask_rate=mask_rate,
            lambda_mask=lambda_mask,
            gradient_clip_norm=gradient_clip_norm,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    if n_batches == 0:
        raise ValueError("loader produced no batches")
    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def predict_logits(
    model,
    loader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and mortality logits for aggregate evaluation only."""
    model.eval()
    model.to(device)
    labels = []
    logits = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = forward_batch(model, batch)
        labels.append(batch["y"].detach().cpu().numpy().reshape(-1))
        logits.append(outputs["mortality_logit"].detach().cpu().numpy().reshape(-1))
    if not labels:
        raise ValueError("loader produced no batches")
    return np.concatenate(labels), np.concatenate(logits)


@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and uncalibrated sigmoid probabilities."""
    labels, logits = predict_logits(model, loader, device=device)
    return labels, logits_to_probabilities(logits)


def _probability_metrics(y_true, probabilities) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    metrics = {
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    if len(np.unique(y)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y, p))
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


def _probability_metrics_with_source(y_true, probabilities, source: str) -> dict:
    metrics = _probability_metrics(y_true, probabilities)
    metrics["probability_source"] = source
    return metrics


def _calibrated_probability_source(method: str) -> str:
    if method == "platt_scaling":
        return "platt_calibrated"
    return f"{method}_calibrated"


def _calibration_report(
    y_true,
    raw_probabilities,
    calibrated_probabilities,
    calibrator,
    *,
    n_bins: int,
) -> dict:
    method = calibrator.to_metadata()["method"]
    probability_source = _calibrated_probability_source(method)
    return {
        "probability_source": probability_source,
        "calibrator": calibrator.to_metadata(),
        "raw_sigmoid": calibration_summary(y_true, raw_probabilities, n_bins=n_bins),
        probability_source: calibration_summary(
            y_true,
            calibrated_probabilities,
            n_bins=n_bins,
        ),
    }


def _read_yaml(path: str | Path) -> dict:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict, override: dict | None) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_mafnet_config(
    config_path: str | Path = DEFAULT_CONFIG,
    overrides: dict | None = None,
) -> dict:
    """Load MAFNet YAML config with optional nested overrides."""
    return _deep_merge(_read_yaml(config_path), overrides or {})


def _loader(dataset, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _temporal_aux_parameters(model: MAFNet):
    return list(model.temporal_encoder.parameters()) + list(
        model.reconstruction_head.parameters()
    ) + list(model.measurement_forecast_head.parameters())


def save_temporal_pretraining_checkpoint(model: MAFNet, path: str | Path) -> Path:
    """Save only temporal encoder and auxiliary heads."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "temporal_encoder": model.temporal_encoder.state_dict(),
            "reconstruction_head": model.reconstruction_head.state_dict(),
            "measurement_forecast_head": model.measurement_forecast_head.state_dict(),
        },
        output,
    )
    return output


def load_pretrained_temporal_modules(model: MAFNet, path: str | Path) -> None:
    """Load temporal encoder and auxiliary heads into a MAFNet model."""
    checkpoint = torch.load(path, map_location="cpu")
    model.temporal_encoder.load_state_dict(checkpoint["temporal_encoder"])
    model.reconstruction_head.load_state_dict(checkpoint["reconstruction_head"])
    model.measurement_forecast_head.load_state_dict(checkpoint["measurement_forecast_head"])


def _save_json(path: str | Path, payload: dict | list) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    return output


def _save_training_curve(history: list[dict], path: str | Path) -> Path | None:
    if not history:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history if row.get("stage") == "finetune"]
    train_loss = [
        row.get("train_loss")
        for row in history
        if row.get("stage") == "finetune"
    ]
    validation_ap = [
        row.get("validation_average_precision")
        for row in history
        if row.get("stage") == "finetune"
    ]
    if not epochs:
        return None
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, validation_ap, marker="o", label="Validation AP")
    plt.xlabel("Epoch")
    plt.title("MAFNet Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def _checkpoint_payload(
    model: MAFNet,
    *,
    config: dict,
    epoch: int,
    validation_metrics: dict,
    data_bundle: MAFNetDataBundle,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
        "static_feature_names": data_bundle.branch_preprocessor.static_feature_names_,
        "aggregate_feature_names": data_bundle.branch_preprocessor.aggregate_feature_names_,
    }


def train_mafnet_from_bundle(
    raw_temporal_bundle: dict,
    feature_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: dict | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    device: torch.device | str = "cpu",
    evaluate_test: bool = False,
    pretrained_checkpoint_path: str | Path | None = None,
) -> MAFNetRunResult:
    """Run pretraining and supervised MAFNet fine-tuning from in-memory data."""
    cfg = load_mafnet_config(config_path, config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    evaluation_cfg = cfg.get("evaluation", {})

    seed = int(train_cfg.get("seed", cfg.get("project", {}).get("random_seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_bundle = prepare_mafnet_datasets(
        raw_temporal_bundle,
        feature_frame,
        target_col=data_cfg.get("target_col", "mortality"),
        patient_id_col=data_cfg.get("patient_id_col", "subject_id"),
        valid_size=float(data_cfg.get("valid_size", 0.20)),
        test_size=float(data_cfg.get("test_size", 0.20)),
        seed=seed,
        evaluate_test=evaluate_test,
    )

    batch_size = int(train_cfg.get("batch_size", 256))
    train_loader = _loader(
        data_bundle.datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        data_bundle.datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = (
        _loader(data_bundle.datasets["test"], batch_size=batch_size, shuffle=False, seed=seed)
        if evaluate_test
        else None
    )

    sample = data_bundle.datasets["train"][0]
    model = MAFNet(
        n_temporal_channels=int(sample["x_temporal"].shape[-1]),
        n_static_features=int(sample["x_static"].shape[-1]),
        n_aggregate_features=int(sample["x_aggregate"].shape[-1]),
        hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
        n_time_steps=int(data_cfg.get("num_time_steps", 24)),
        n_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_ff_dim=int(model_cfg.get("transformer_ff_dim", 256)),
        transformer_layers=int(model_cfg.get("transformer_layers", 1)),
        temporal_dropout=float(model_cfg.get("temporal_dropout", 0.15)),
        use_static_branch=bool(model_cfg.get("use_static_branch", True)),
        use_aggregate_branch=bool(model_cfg.get("use_aggregate_branch", True)),
        use_decay=bool(model_cfg.get("use_decay", True)),
        use_transformer=bool(model_cfg.get("use_transformer", True)),
        use_gated_fusion=bool(model_cfg.get("use_gated_fusion", True)),
    ).to(device)

    if pretrained_checkpoint_path is not None:
        load_pretrained_temporal_modules(model, pretrained_checkpoint_path)

    history: list[dict] = []
    pretrain_checkpoint_path = None
    pretrain_epochs = int(train_cfg.get("pretrain_epochs", 20))
    gradient_clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    if pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.AdamW(
            _temporal_aux_parameters(model),
            lr=float(train_cfg.get("pretrain_lr", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        )
        for epoch in range(1, pretrain_epochs + 1):
            losses = pretrain_mafnet_one_epoch(
                model,
                train_loader,
                pretrain_optimizer,
                device=device,
                mask_rate=float(loss_cfg.get("random_observed_mask_rate", 0.15)),
                lambda_mask=0.10,
                gradient_clip_norm=gradient_clip_norm,
            )
            history.append({"stage": "pretrain", "epoch": epoch, **losses})
        pretrain_checkpoint_path = save_temporal_pretraining_checkpoint(
            model,
            out / "pretrained_temporal.pt",
        )

    loss_config = MAFNetLossConfig(
        lambda_recon=float(loss_cfg.get("lambda_recon", 0.05)),
        lambda_mask=float(loss_cfg.get("lambda_mask", 0.01)),
        random_observed_mask_rate=float(loss_cfg.get("random_observed_mask_rate", 0.15)),
    )
    pos_weight = compute_pos_weight(data_bundle.datasets["train"].y)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("finetune_lr", 0.0003)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 5)),
        min_lr=float(train_cfg.get("min_lr", 0.00001)),
    )
    stopper = EarlyStopping(
        mode="max",
        patience=int(train_cfg.get("early_stopping_patience", 20)),
    )

    best_score = float("-inf")
    best_metrics: dict = {}
    best_checkpoint_path = out / "best_model.pt"
    max_epochs = int(train_cfg.get("max_epochs", 150))
    for epoch in range(1, max_epochs + 1):
        losses = train_mafnet_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            pos_weight=pos_weight,
            loss_config=loss_config,
            gradient_clip_norm=gradient_clip_norm,
        )
        y_valid, p_valid = predict_probabilities(model, validation_loader, device=device)
        validation_metrics = _probability_metrics(y_valid, p_valid)
        validation_ap = validation_metrics["average_precision"]
        scheduler.step(validation_ap)

        history.append(
            {
                "stage": "finetune",
                "epoch": epoch,
                "train_loss": losses["loss"],
                "train_mortality_loss": losses["mortality_loss"],
                "train_reconstruction_loss": losses["reconstruction_loss"],
                "train_mask_forecast_loss": losses["mask_forecast_loss"],
                "validation_average_precision": validation_ap,
                "validation_auc_roc": validation_metrics["auc_roc"],
                "validation_brier_score": validation_metrics["brier_score"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if validation_ap > best_score:
            best_score = float(validation_ap)
            best_metrics = validation_metrics
            torch.save(
                _checkpoint_payload(
                    model,
                    config=cfg,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    data_bundle=data_bundle,
                ),
                best_checkpoint_path,
            )

        if stopper.step(validation_ap):
            break

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    calibration_method_raw = evaluation_cfg.get("calibration", "platt_scaling")
    calibration_method = (
        "none" if calibration_method_raw is None else str(calibration_method_raw).lower()
    )
    calibration_bins = int(evaluation_cfg.get("calibration_bins", 10))

    y_valid, validation_logits = predict_logits(model, validation_loader, device=device)
    validation_raw_probabilities = logits_to_probabilities(validation_logits)
    validation_raw_metrics = _probability_metrics_with_source(
        y_valid,
        validation_raw_probabilities,
        "raw_sigmoid",
    )
    _save_json(out / "validation_raw_metrics.json", validation_raw_metrics)

    platt_calibrator_path = None
    isotonic_calibrator_path = None
    calibration_model_path = None
    validation_calibration_path = None
    test_calibration_path = None

    if calibration_method in {"platt", "platt_scaling"}:
        calibrator = fit_platt_scaler(validation_logits, y_valid)
        platt_calibrator_path = out / "platt_calibrator.joblib"
        calibration_model_path = platt_calibrator_path
    elif calibration_method == "isotonic":
        calibrator = fit_isotonic_calibrator(validation_logits, y_valid)
        isotonic_calibrator_path = out / "isotonic_calibrator.joblib"
        calibration_model_path = isotonic_calibrator_path
    elif calibration_method in {"none", "raw", "uncalibrated"}:
        calibrator = None
        validation_probabilities = validation_raw_probabilities
        validation_metrics = validation_raw_metrics
    else:
        raise ValueError(f"Unsupported MAFNet calibration method: {calibration_method_raw}")

    if calibrator is not None:
        joblib.dump(calibrator, calibration_model_path)
        calibration_metadata = calibrator.to_metadata()
        probability_source = _calibrated_probability_source(calibration_metadata["method"])
        validation_probabilities = calibrator.predict_proba(validation_logits)
        validation_metrics = _probability_metrics_with_source(
            y_valid,
            validation_probabilities,
            probability_source,
        )
        validation_metrics["calibration_method"] = calibration_metadata["method"]
        validation_calibration_path = _save_json(
            out / "validation_calibration.json",
            _calibration_report(
                y_valid,
                validation_raw_probabilities,
                validation_probabilities,
                calibrator,
                n_bins=calibration_bins,
            ),
        )
        save_calibration_curve(
            y_valid,
            validation_probabilities,
            out / "validation_calibration_curve.png",
            title="MAFNet Validation Calibration Curve",
            n_bins=calibration_bins,
        )

    test_metrics = None
    if test_loader is not None:
        y_test, test_logits = predict_logits(model, test_loader, device=device)
        test_raw_probabilities = logits_to_probabilities(test_logits)
        test_raw_metrics = _probability_metrics_with_source(
            y_test,
            test_raw_probabilities,
            "raw_sigmoid",
        )
        _save_json(out / "test_raw_metrics.json", test_raw_metrics)
        if calibrator is not None:
            calibration_metadata = calibrator.to_metadata()
            probability_source = _calibrated_probability_source(calibration_metadata["method"])
            test_probabilities = calibrator.predict_proba(test_logits)
            test_metrics = _probability_metrics_with_source(
                y_test,
                test_probabilities,
                probability_source,
            )
            test_metrics["calibration_method"] = calibration_metadata["method"]
            test_calibration_path = _save_json(
                out / "test_calibration.json",
                _calibration_report(
                    y_test,
                    test_raw_probabilities,
                    test_probabilities,
                    calibrator,
                    n_bins=calibration_bins,
                ),
            )
            save_calibration_curve(
                y_test,
                test_probabilities,
                out / "test_calibration_curve.png",
                title="MAFNet Test Calibration Curve",
                n_bins=calibration_bins,
            )
        else:
            test_metrics = test_raw_metrics
        _save_json(out / "test_metrics.json", test_metrics)

    _save_json(out / "training_history.json", history)
    _save_json(out / "validation_metrics.json", validation_metrics)
    _save_training_curve(history, out / "training_curves.png")
    joblib.dump(data_bundle.temporal_normalizer, out / "temporal_normalizer.joblib")
    joblib.dump(data_bundle.branch_preprocessor, out / "branch_preprocessor.joblib")

    summary = {
        "output_dir": str(out),
        "best_checkpoint_path": str(best_checkpoint_path),
        "pretrained_checkpoint_path": str(pretrain_checkpoint_path) if pretrain_checkpoint_path else None,
        "history_path": str(out / "training_history.json"),
        "validation_metrics_path": str(out / "validation_metrics.json"),
        "validation_raw_metrics_path": str(out / "validation_raw_metrics.json"),
        "validation_calibration_path": str(validation_calibration_path)
        if validation_calibration_path
        else None,
        "test_metrics_path": str(out / "test_metrics.json") if test_metrics else None,
        "test_raw_metrics_path": str(out / "test_raw_metrics.json") if test_metrics else None,
        "test_calibration_path": str(test_calibration_path) if test_calibration_path else None,
        "platt_calibrator_path": str(platt_calibrator_path) if platt_calibrator_path else None,
        "isotonic_calibrator_path": str(isotonic_calibrator_path)
        if isotonic_calibrator_path
        else None,
        "calibration_model_path": str(calibration_model_path) if calibration_model_path else None,
        "calibration_method": calibration_method,
        "n_train": int(len(data_bundle.datasets["train"])),
        "n_validation": int(len(data_bundle.datasets["validation"])),
        "n_test": int(len(data_bundle.datasets["test"])) if evaluate_test else 0,
        "static_feature_count": len(data_bundle.branch_preprocessor.static_feature_names_),
        "aggregate_feature_count": len(data_bundle.branch_preprocessor.aggregate_feature_names_),
    }
    summary_path = _save_json(out / "run_summary.json", summary)

    return MAFNetRunResult(
        output_dir=out,
        history=history,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_checkpoint_path=best_checkpoint_path,
        pretrained_checkpoint_path=pretrain_checkpoint_path,
        summary_path=summary_path,
        platt_calibrator_path=platt_calibrator_path,
        isotonic_calibrator_path=isotonic_calibrator_path,
        calibration_model_path=calibration_model_path,
        validation_calibration_path=validation_calibration_path,
        test_calibration_path=test_calibration_path,
    )


def train_mafnet_from_frames(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: dict | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    device: torch.device | str = "cpu",
    evaluate_test: bool = False,
) -> MAFNetRunResult:
    """Build temporal tensors from long-form events and run MAFNet training."""
    cfg = load_mafnet_config(config_path, config)
    data_cfg = cfg.get("data", {})
    raw_bundle = build_15min_temporal_tensors(
        events,
        cohort,
        window_hours=float(data_cfg.get("time_window_hours", 6.0)),
        bin_minutes=int(data_cfg.get("bin_minutes", 15)),
    )
    return train_mafnet_from_bundle(
        raw_bundle,
        feature_frame,
        output_dir=output_dir,
        config=cfg,
        config_path=config_path,
        device=device,
        evaluate_test=evaluate_test,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAFNet from local first-window data")
    parser.add_argument("--events-path", required=True, help="Long-form temporal event CSV")
    parser.add_argument("--cohort-path", required=True, help="Cohort CSV with stay_id and intime")
    parser.add_argument("--features-path", required=True, help="Processed feature CSV with target")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "mafnet"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"training": {}}
    if args.pretrain_epochs is not None:
        overrides["training"]["pretrain_epochs"] = args.pretrain_epochs
    if args.max_epochs is not None:
        overrides["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if not overrides["training"]:
        overrides = None

    events = pd.read_csv(args.events_path)
    cohort = pd.read_csv(args.cohort_path)
    features = pd.read_csv(args.features_path)
    result = train_mafnet_from_frames(
        events,
        cohort,
        features,
        output_dir=args.output_dir,
        config=overrides,
        config_path=args.config,
        device=args.device,
        evaluate_test=args.evaluate_test,
    )
    print(f"MAFNet training complete. Summary: {result.summary_path}")


__all__ = [
    "batch_to_device",
    "BranchFeaturePreprocessor",
    "forward_batch",
    "load_mafnet_config",
    "load_pretrained_temporal_modules",
    "mafnet_pretrain_step",
    "mafnet_supervised_train_step",
    "prepare_mafnet_datasets",
    "predict_logits",
    "predict_probabilities",
    "pretrain_mafnet_one_epoch",
    "save_temporal_pretraining_checkpoint",
    "subset_temporal_bundle",
    "train_mafnet_from_bundle",
    "train_mafnet_from_frames",
    "train_mafnet_one_epoch",
]


if __name__ == "__main__":
    main()
