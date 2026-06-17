"""PyTorch-ready dataset wrappers for ICU6H-MAFNet tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.schema import build_feature_matrix


DEFAULT_STATIC_FEATURE_CANDIDATES = (
    "age",
    "age_very_elderly",
    "very_elderly_flag",
    "gender_numeric",
    "gender_f",
    "gender_m",
    "bmi",
    "prev_dx_count_total",
    "prior_diagnosis_count",
    "prev_dx_respiratory_count",
    "respiratory_diagnosis_count",
    "prev_dx_circulatory_count",
    "circulatory_diagnosis_count",
    "prev_dx_nervous_sensory_count",
    "nervous_sensory_diagnosis_count",
    "has_metastatic_cancer",
    "metastatic_cancer_flag",
    "has_prior_diagnoses",
    "has_prior_diagnoses_flag",
)


@dataclass(frozen=True)
class MAFNetFeatureMatrices:
    x_static: np.ndarray
    x_aggregate: np.ndarray
    y: np.ndarray
    static_columns: list[str]
    aggregate_columns: list[str]


def _as_float_array(values, *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name} row count {len(array)} does not match {rows}")
    return array


def build_static_aggregate_matrices(
    feature_frame: pd.DataFrame,
    stay_ids,
    *,
    static_columns: list[str] | tuple[str, ...] | None = None,
    id_col: str = "stay_id",
    target_col: str = "mortality",
) -> MAFNetFeatureMatrices:
    """Align processed features to temporal stay order and split branch inputs."""
    if id_col not in feature_frame.columns:
        raise ValueError(f"feature_frame must contain `{id_col}` for temporal alignment")
    if target_col not in feature_frame.columns:
        raise ValueError(f"feature_frame must contain `{target_col}`")

    indexed = feature_frame.drop_duplicates(subset=[id_col]).set_index(id_col)
    missing_ids = [stay_id for stay_id in stay_ids if stay_id not in indexed.index]
    if missing_ids:
        raise ValueError(f"feature_frame is missing {len(missing_ids)} stay ids")

    aligned = indexed.loc[list(stay_ids)].reset_index()
    safe_features, _ = build_feature_matrix(aligned, target_col=target_col)
    safe_features = pd.get_dummies(safe_features, drop_first=False)

    requested_static = list(static_columns or DEFAULT_STATIC_FEATURE_CANDIDATES)
    available_static = [col for col in requested_static if col in safe_features.columns]
    aggregate_columns = [col for col in safe_features.columns if col not in available_static]

    return MAFNetFeatureMatrices(
        x_static=safe_features[available_static].to_numpy(dtype=np.float32),
        x_aggregate=safe_features[aggregate_columns].to_numpy(dtype=np.float32),
        y=aligned[target_col].to_numpy(dtype=np.float32),
        static_columns=available_static,
        aggregate_columns=aggregate_columns,
    )


class TemporalFusionDataset(Dataset):
    """Dataset returning tensors for temporal, static, aggregate, and target data."""

    def __init__(
        self,
        temporal_bundle: dict,
        *,
        y,
        x_static=None,
        x_aggregate=None,
    ) -> None:
        row_count = int(temporal_bundle["x_temporal"].shape[0])
        self.x_temporal = torch.as_tensor(
            _as_float_array(temporal_bundle["x_temporal"], name="x_temporal"),
            dtype=torch.float32,
        )
        self.mask_temporal = torch.as_tensor(
            _as_float_array(temporal_bundle["mask_temporal"], name="mask_temporal", rows=row_count),
            dtype=torch.float32,
        )
        self.delta_temporal = torch.as_tensor(
            _as_float_array(temporal_bundle["delta_temporal"], name="delta_temporal", rows=row_count),
            dtype=torch.float32,
        )
        self.count_temporal = torch.as_tensor(
            _as_float_array(temporal_bundle["count_temporal"], name="count_temporal", rows=row_count),
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(_as_float_array(y, name="y", rows=row_count), dtype=torch.float32)

        if x_static is None:
            x_static = np.zeros((row_count, 0), dtype=np.float32)
        if x_aggregate is None:
            x_aggregate = np.zeros((row_count, 0), dtype=np.float32)
        self.x_static = torch.as_tensor(
            _as_float_array(x_static, name="x_static", rows=row_count),
            dtype=torch.float32,
        )
        self.x_aggregate = torch.as_tensor(
            _as_float_array(x_aggregate, name="x_aggregate", rows=row_count),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x_temporal": self.x_temporal[index],
            "mask_temporal": self.mask_temporal[index],
            "delta_temporal": self.delta_temporal[index],
            "count_temporal": self.count_temporal[index],
            "x_static": self.x_static[index],
            "x_aggregate": self.x_aggregate[index],
            "y": self.y[index],
        }


def build_temporal_fusion_dataset(
    temporal_bundle: dict,
    feature_frame: pd.DataFrame,
    *,
    static_columns: list[str] | tuple[str, ...] | None = None,
    id_col: str = "stay_id",
    target_col: str = "mortality",
) -> tuple[TemporalFusionDataset, MAFNetFeatureMatrices]:
    """Create a PyTorch dataset aligned by stay_id without exposing IDs as features."""
    matrices = build_static_aggregate_matrices(
        feature_frame,
        temporal_bundle["stay_ids"],
        static_columns=static_columns,
        id_col=id_col,
        target_col=target_col,
    )
    dataset = TemporalFusionDataset(
        temporal_bundle,
        y=matrices.y,
        x_static=matrices.x_static,
        x_aggregate=matrices.x_aggregate,
    )
    return dataset, matrices
