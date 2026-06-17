"""Opt-in expanded first-6-hour feature extraction."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from features import (
    build_hourly_time_bin_features,
    compute_clinical_interaction_features,
    compute_instability_features,
    compute_measurement_process_features,
    compute_organ_dysfunction_features,
    compute_trajectory_features,
    validate_feature_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPANDED_FEATURE_CONFIG = PROJECT_ROOT / "configs" / "features_expanded.yaml"

VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "resp_rate": [220210],
    "map": [220052],
    "temp": [223761, 223762],
    "sbp": [220179],
    "dbp": [220180],
    "spo2": [220277],
}

LAB_ITEMIDS = {
    "bun": [51006],
    "alkaline_phosphatase": [50863],
    "bilirubin": [50885],
    "creatinine": [50912],
    "glucose": [50931],
    "platelets": [51265],
    "hemoglobin": [51222],
    "wbc": [51301],
    "sodium": [50983],
    "potassium": [50971],
    "lactate": [50813],
    "hematocrit": [51221],
    "chloride": [50902],
    "bicarbonate": [50882],
    "anion_gap": [50868],
    "inr": [51237, 51675],
}


def load_expanded_feature_config(config_path=None) -> dict:
    """Load the expanded feature config if present."""
    path = Path(config_path) if config_path else DEFAULT_EXPANDED_FEATURE_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded


def expanded_features_enabled(config: dict | None, override: bool | None = None) -> bool:
    """Resolve the expanded-feature on/off flag from CLI override and config."""
    if override is not None:
        return bool(override)
    feature_engineering = (config or {}).get("feature_engineering", {})
    return bool(
        feature_engineering.get(
            "enabled",
            False,
        )
    )


def _group_enabled(config: dict | None, group_name: str, default: bool = True) -> bool:
    group_config = ((config or {}).get("feature_groups", {}) or {}).get(group_name, {})
    return bool(group_config.get("enabled", default))


def _flatten_item_map(item_map: dict[str, list[int]]) -> dict[int, str]:
    return {
        int(itemid): variable
        for variable, itemids in item_map.items()
        for itemid in itemids
    }


def _vital_events_from_chart_data(
    chart_data: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    item_to_variable = _flatten_item_map(VITAL_ITEMIDS)
    required_cols = {"stay_id", "charttime", "itemid", "valuenum"}
    missing_cols = required_cols - set(chart_data.columns)
    if missing_cols:
        raise ValueError(f"chart_data missing columns: {sorted(missing_cols)}")

    stay_ids = set(cohort["stay_id"])
    events = chart_data[
        chart_data["stay_id"].isin(stay_ids)
        & chart_data["itemid"].isin(item_to_variable)
    ][["stay_id", "charttime", "itemid", "valuenum"]].copy()

    events = events.dropna(subset=["stay_id", "charttime", "itemid", "valuenum"])
    events["itemid"] = events["itemid"].astype(int)
    events["variable"] = events["itemid"].map(item_to_variable)
    events["source"] = "vital"
    return events[["stay_id", "charttime", "variable", "source", "valuenum"]]


def _lab_events_from_hospital_path(
    cohort: pd.DataFrame,
    hospital_path: str,
) -> pd.DataFrame:
    item_to_variable = _flatten_item_map(LAB_ITEMIDS)
    lab_path = os.path.join(hospital_path, "_labevents.csv")
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Lab events file not found: {lab_path}")

    labs = pd.read_csv(
        lab_path,
        usecols=["subject_id", "charttime", "itemid", "valuenum"],
    )
    labs = labs.dropna(subset=["subject_id", "charttime", "itemid", "valuenum"])
    labs["itemid"] = labs["itemid"].astype(int)

    subject_ids = set(cohort["subject_id"])
    labs = labs[
        labs["subject_id"].isin(subject_ids)
        & labs["itemid"].isin(item_to_variable)
    ].copy()
    if labs.empty:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "variable", "source", "valuenum"]
        )

    stay_lookup = cohort[["subject_id", "stay_id"]].drop_duplicates()
    events = labs.merge(stay_lookup, on="subject_id", how="inner")
    events["variable"] = events["itemid"].map(item_to_variable)
    events["source"] = "lab"
    return events[["stay_id", "charttime", "variable", "source", "valuenum"]]


def build_expanded_long_events(
    cohort: pd.DataFrame,
    chart_data: pd.DataFrame,
    hospital_path: str,
) -> pd.DataFrame:
    """Build long-form vital/lab events for expanded first-window features."""
    vital_events = _vital_events_from_chart_data(chart_data, cohort)
    lab_events = _lab_events_from_hospital_path(cohort, hospital_path)
    events = pd.concat([vital_events, lab_events], ignore_index=True)
    if events.empty:
        return events
    events["charttime"] = pd.to_datetime(events["charttime"])
    return events


def _validate_new_features(feature_names, context: str) -> None:
    validation = validate_feature_provenance(feature_names)
    if validation["unmatched"] or validation["disallowed"]:
        raise ValueError(
            f"{context} produced features without allowed provenance: "
            f"unmatched={validation['unmatched']}, "
            f"disallowed={validation['disallowed']}"
        )


def extract_expanded_event_features(
    cohort: pd.DataFrame,
    chart_data: pd.DataFrame,
    hospital_path: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """Extract opt-in event-derived first-6-hour features."""
    result = cohort[["stay_id"]].drop_duplicates().copy()
    events = build_expanded_long_events(cohort, chart_data, hospital_path)
    if events.empty:
        return result

    if _group_enabled(config, "hourly_time_bins"):
        bin_config = ((config or {}).get("feature_groups", {}) or {}).get(
            "hourly_time_bins", {}
        )
        aggregations = tuple(
            bin_config.get("aggregations", ["mean", "min", "max", "last"])
        )
        binned = build_hourly_time_bin_features(
            events,
            cohort,
            aggregations=aggregations,
            include_observed=bool(bin_config.get("include_observed_flags", True)),
            include_count=bool(bin_config.get("include_counts", True)),
        )
        result = result.merge(binned, on="stay_id", how="left")

    if _group_enabled(config, "trajectory"):
        trajectory = compute_trajectory_features(events, cohort)
        result = result.merge(trajectory, on="stay_id", how="left")

    if _group_enabled(config, "instability"):
        instability = compute_instability_features(events, cohort)
        result = result.merge(instability, on="stay_id", how="left")

    if _group_enabled(config, "measurement_process"):
        expected_variables = sorted(set(VITAL_ITEMIDS) | set(LAB_ITEMIDS))
        measurement = compute_measurement_process_features(
            events,
            cohort,
            source_col="source",
            expected_variables=expected_variables,
        )
        result = result.merge(measurement, on="stay_id", how="left")

    new_features = [col for col in result.columns if col != "stay_id"]
    _validate_new_features(new_features, "expanded event extraction")
    return result


def add_expanded_derived_features(
    features: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Add opt-in derived proxy and interaction features."""
    result = features.copy()
    before_cols = set(result.columns)

    if _group_enabled(config, "organ_dysfunction"):
        result = compute_organ_dysfunction_features(result)

    if _group_enabled(config, "interactions"):
        result = compute_clinical_interaction_features(result)

    new_features = sorted(set(result.columns) - before_cols)
    _validate_new_features(new_features, "expanded derived feature extraction")
    return result
