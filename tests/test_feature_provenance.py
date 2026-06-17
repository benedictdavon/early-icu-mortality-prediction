from __future__ import annotations

import pandas as pd

from features.provenance import (
    FEATURE_DICTIONARY_COLUMNS,
    feature_dictionary_dataframe,
    match_feature_name,
    validate_feature_provenance,
)


def test_feature_dictionary_has_required_schema_and_groups():
    dictionary = feature_dictionary_dataframe()

    assert list(dictionary.columns) == FEATURE_DICTIONARY_COLUMNS
    assert not dictionary.empty
    assert {
        "demographic",
        "vital",
        "lab",
        "diagnosis",
        "missingness",
        "derived_clinical",
        "trajectory",
    }.issubset(set(dictionary["feature_group"]))
    assert set(dictionary["leakage_risk"]).issubset(
        {"low", "medium", "high", "excluded"}
    )


def test_existing_feature_families_have_provenance_matches():
    feature_names = [
        "age",
        "gender_numeric",
        "bmi",
        "bmi_measured",
        "heart_rate_mean",
        "heart_rate_measured",
        "lactate_delta",
        "lactate_measured",
        "heart_rate_hour_6",
        "lactate_change_0to6",
        "has_prior_diagnoses",
        "prev_dx_respiratory_count",
        "shock_index",
        "has_tachypnea",
        "age_comorbidity",
        "lactate_mean_missing",
        "spo2_pct_change",
        "spo2_mean_dist_from_normal",
    ]

    result = validate_feature_provenance(feature_names)

    assert result["unmatched"] == []
    assert result["disallowed"] == []


def test_known_uncertain_legacy_feature_is_flagged_as_disallowed():
    result = validate_feature_provenance(["has_metastatic_cancer"])

    assert result["unmatched"] == []
    assert result["disallowed"] == ["has_metastatic_cancer"]
    assert match_feature_name("has_metastatic_cancer").leakage_risk == "high"


def test_checked_in_feature_dictionary_csv_matches_schema():
    dictionary = pd.read_csv("docs/feature_dictionary.csv")

    for column in FEATURE_DICTIONARY_COLUMNS:
        assert column in dictionary.columns
    assert not dictionary.empty
