"""Feature provenance and leakage-checking helpers."""

from features.provenance import (
    FEATURE_DICTIONARY_COLUMNS,
    FeatureProvenance,
    export_feature_dictionary,
    feature_dictionary_dataframe,
    match_feature_name,
    validate_feature_provenance,
)
from features.temporal_window import filter_events_to_observation_window
from features.time_bins import build_hourly_time_bin_features
from features.trajectory import compute_trajectory_features
from features.measurement_process import compute_measurement_process_features
from features.organ_dysfunction import compute_organ_dysfunction_features
from features.interactions import compute_clinical_interaction_features
from features.instability import compute_instability_features
from features.temporal_channels import TEMPORAL_CHANNELS
from features.temporal_tensor_builder import (
    TemporalTensorNormalizer,
    build_15min_temporal_tensors,
    fit_transform_temporal_bundle,
)

__all__ = [
    "FEATURE_DICTIONARY_COLUMNS",
    "FeatureProvenance",
    "build_hourly_time_bin_features",
    "compute_clinical_interaction_features",
    "compute_instability_features",
    "compute_measurement_process_features",
    "compute_organ_dysfunction_features",
    "compute_trajectory_features",
    "export_feature_dictionary",
    "feature_dictionary_dataframe",
    "match_feature_name",
    "validate_feature_provenance",
    "filter_events_to_observation_window",
    "TEMPORAL_CHANNELS",
    "TemporalTensorNormalizer",
    "build_15min_temporal_tensors",
    "fit_transform_temporal_bundle",
]
