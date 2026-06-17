"""Machine-checkable feature provenance metadata.

The public dictionary is feature-family based: each row names a concrete feature
or a regex pattern for a generated family. That keeps the metadata stable while
the preprocessing pipeline creates model-specific feature subsets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import pandas as pd


FEATURE_DICTIONARY_COLUMNS = [
    "feature_name",
    "feature_pattern",
    "feature_group",
    "source_table",
    "source_variable",
    "time_window_start",
    "time_window_end",
    "aggregation",
    "clinical_rationale",
    "missingness_handling",
    "is_binary",
    "is_missingness_indicator",
    "leakage_risk",
    "allowed_for_model",
    "notes",
]

ALLOWED_LEAKAGE_RISKS = {"low", "medium", "high", "excluded"}


@dataclass(frozen=True)
class FeatureProvenance:
    feature_name: str
    feature_pattern: str
    feature_group: str
    source_table: str
    source_variable: str
    time_window_start: str
    time_window_end: str
    aggregation: str
    clinical_rationale: str
    missingness_handling: str
    is_binary: bool
    is_missingness_indicator: bool
    leakage_risk: str
    allowed_for_model: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if self.leakage_risk not in ALLOWED_LEAKAGE_RISKS:
            raise ValueError(f"Invalid leakage risk: {self.leakage_risk}")

    def matches(self, feature_name: str) -> bool:
        return re.fullmatch(self.feature_pattern, feature_name) is not None

    def to_record(self) -> dict:
        return asdict(self)


VITALS = "heart_rate|resp_rate|map|temp|sbp|dbp|spo2"
LABS = (
    "bun|alkaline_phosphatase|bilirubin|creatinine|glucose|platelets|"
    "hemoglobin|wbc|sodium|potassium|lactate|hematocrit|chloride|"
    "bicarbonate|anion_gap|inr"
)


FEATURE_PROVENANCE_REGISTRY = (
    FeatureProvenance(
        feature_name="age",
        feature_pattern=r"age|age_(young_adult|middle_aged|elderly|very_elderly)",
        feature_group="demographic",
        source_table="hosp/patients + hosp/admissions",
        source_variable="anchor_age, anchor_year, admittime",
        time_window_start="pre-ICU",
        time_window_end="ICU admission",
        aggregation="age at admission, capped at 90",
        clinical_rationale="Age is a strong baseline mortality risk factor.",
        missingness_handling="no imputation expected; downstream median if missing",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="gender_numeric",
        feature_pattern=r"gender_numeric|gender_[A-Za-z0-9_]+",
        feature_group="demographic",
        source_table="hosp/patients",
        source_variable="gender",
        time_window_start="pre-ICU",
        time_window_end="ICU admission",
        aggregation="binary or one-hot encoding",
        clinical_rationale="Sex may capture baseline demographic risk differences.",
        missingness_handling="mode imputation if missing",
        is_binary=True,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="bmi",
        feature_pattern=r"bmi",
        feature_group="demographic",
        source_table="icu/chartevents",
        source_variable="height, weight",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="first valid height and weight in first 6 hours",
        clinical_rationale="Body habitus can affect acuity and physiologic reserve.",
        missingness_handling="downstream median imputation plus bmi_measured flag",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
        notes="Legacy extraction previously used a wider BMI window; current extraction restricts this to 6h.",
    ),
    FeatureProvenance(
        feature_name="bmi_measured",
        feature_pattern=r"bmi_measured|bmi_missing",
        feature_group="missingness",
        source_table="icu/chartevents",
        source_variable="height, weight",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="observed flag",
        clinical_rationale="Anthropometric measurement availability can reflect care process.",
        missingness_handling="binary missingness indicator",
        is_binary=True,
        is_missingness_indicator=True,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="vital_summary",
        feature_pattern=rf"({VITALS})_(mean|min|max)",
        feature_group="vital",
        source_table="icu/chartevents",
        source_variable="vital itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="time-weighted mean, min, or max",
        clinical_rationale="Early vital instability reflects acute physiologic risk.",
        missingness_handling="downstream imputation plus measured flags",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="vital_measured",
        feature_pattern=rf"({VITALS})(_(mean|min|max))?_(measured|missing)",
        feature_group="missingness",
        source_table="icu/chartevents",
        source_variable="vital itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="observed flag",
        clinical_rationale="Measurement availability is informative in ICU care.",
        missingness_handling="binary missingness indicator",
        is_binary=True,
        is_missingness_indicator=True,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="lab_summary",
        feature_pattern=rf"({LABS})_(mean|min|max|delta)",
        feature_group="lab",
        source_table="hosp/labevents",
        source_variable="lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="mean, min, max, or last minus first",
        clinical_rationale="Early laboratory derangements reflect organ dysfunction and severity.",
        missingness_handling="downstream imputation plus measured flags",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="lab_measured",
        feature_pattern=rf"({LABS})(_(mean|min|max|delta))?_(measured|missing)",
        feature_group="missingness",
        source_table="hosp/labevents",
        source_variable="lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="observed flag",
        clinical_rationale="Lab ordering and missingness can reflect clinical concern.",
        missingness_handling="binary missingness indicator",
        is_binary=True,
        is_missingness_indicator=True,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="urine_output_total",
        feature_pattern=r"urine_output_(total|measured)",
        feature_group="fluid_balance",
        source_table="icu/chartevents",
        source_variable="urine output itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="sum or observed flag",
        clinical_rationale="Early oliguria can indicate renal or circulatory dysfunction.",
        missingness_handling="downstream imputation plus measured flag",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="time_window_value",
        feature_pattern=rf"({VITALS}|{LABS})_hour_(0|1|2|3|4|6)",
        feature_group="time_window",
        source_table="icu/chartevents or hosp/labevents",
        source_variable="vital/lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="closest observed value inside configured first-6h window",
        clinical_rationale="Within-window trajectory can capture early deterioration.",
        missingness_handling="downstream imputation",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="time_window_change",
        feature_pattern=rf"({VITALS}|{LABS})_(change_0to6|hourly_change|delta_[0-9]+to[0-9]+)",
        feature_group="trajectory",
        source_table="derived from first-6h vital/lab features",
        source_variable="time-window vital/lab values",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="difference or rate of change",
        clinical_rationale="Direction of change can indicate improvement or deterioration.",
        missingness_handling="downstream imputation",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="hourly_bin_summary",
        feature_pattern=rf"({VITALS}|{LABS})_bin_[0-5]_[1-6]h_(mean|min|max|last|observed|count)",
        feature_group="time_window",
        source_table="icu/chartevents or hosp/labevents",
        source_variable="vital/lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="hourly bin summary, observed flag, or count",
        clinical_rationale="Preserves early within-window timing without using post-window events.",
        missingness_handling="value features use downstream imputation; observed/count features use zero when absent",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="trajectory_summary",
        feature_pattern=rf"({VITALS}|{LABS})_(first|last|last_minus_first|percent_change|first_2h_mean|last_2h_mean|last2h_minus_first2h|slope_0_6h|deterioration_flag|recovery_flag)",
        feature_group="trajectory",
        source_table="icu/chartevents or hosp/labevents",
        source_variable="vital/lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="first, last, first-vs-last change, early/late mean, or linear slope",
        clinical_rationale="Captures early improvement or deterioration within the prediction window.",
        missingness_handling="downstream imputation",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="instability_summary",
        feature_pattern=rf"({VITALS}|{LABS})_(range_0_6h|std_0_6h|cv_0_6h|abnormal_count_0_6h|longest_abnormal_run_0_6h|worst_recent_value_0_6h)",
        feature_group="instability",
        source_table="icu/chartevents or hosp/labevents",
        source_variable="vital/lab itemids",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="range, variability, abnormal event count, abnormal run length, or recent worst value",
        clinical_rationale="Captures short-window physiologic instability and repeated abnormal measurements.",
        missingness_handling="downstream imputation; counts use zero when absent",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="measurement_process",
        feature_pattern=rf"(({VITALS}|{LABS})_(measured_0_6h|measurement_count_0_6h|time_to_first_measurement|time_since_last_measurement_at_6h)|total_(lab|vital|chart)_measurements_0_6h|total_measurements_0_6h|total_chart_event_count_0_6h|panel_missing_count_0_6h)",
        feature_group="measurement_process",
        source_table="icu/chartevents or hosp/labevents",
        source_variable="vital/lab event timestamps",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="measurement availability, count, and timing",
        clinical_rationale="Early measurement intensity can reflect acuity and care process.",
        missingness_handling="count/flag features use zero when absent; timing features use downstream imputation",
        is_binary=False,
        is_missingness_indicator=True,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="prior_diagnosis",
        feature_pattern=r"has_prior_diagnoses|prev_dx_[A-Za-z0-9_]+_count|prev_dx_count_total",
        feature_group="diagnosis",
        source_table="hosp/admissions + hosp/diagnoses_icd",
        source_variable="diagnoses before ICU admission",
        time_window_start="pre-ICU",
        time_window_end="ICU admission",
        aggregation="ICD chapter counts from prior admissions",
        clinical_rationale="Comorbidity burden affects early mortality risk.",
        missingness_handling="zero when no prior diagnosis found",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="has_metastatic_cancer",
        feature_pattern=r"has_metastatic_cancer",
        feature_group="diagnosis",
        source_table="hosp/diagnoses_icd",
        source_variable="ICD metastatic cancer codes",
        time_window_start="uncertain",
        time_window_end="uncertain",
        aggregation="any matching diagnosis for subject",
        clinical_rationale="Metastatic cancer is a major mortality risk factor.",
        missingness_handling="zero when no matching diagnosis found",
        is_binary=True,
        is_missingness_indicator=False,
        leakage_risk="high",
        allowed_for_model=False,
        notes="Current legacy extractor is not clearly limited to prior or admission-time-known diagnoses.",
    ),
    FeatureProvenance(
        feature_name="derived_clinical",
        feature_pattern=(
            r"has_[A-Za-z0-9_]+|shock_index|shock_lactate_interaction|"
            r"bun_creatinine_ratio|resp_distress(_score)?|sirs_criteria_count|"
            r"[A-Za-z0-9_]+_dysfunction|organ_dysfunction_count|"
            r"age_comorbidity|[A-Za-z0-9_]+_x_[A-Za-z0-9_]+|hr_sbp_ratio|"
            r"map_variance_x_lactate|sirs_x_lactate"
        ),
        feature_group="derived_clinical",
        source_table="derived from first-6h demographics, vitals, labs, and diagnoses",
        source_variable="base feature columns",
        time_window_start="0h or pre-ICU",
        time_window_end="6h",
        aggregation="clinical rule, ratio, count, or interaction",
        clinical_rationale="Combines physiologic signals into clinically interpretable risk proxies.",
        missingness_handling="inherits from source features",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="organ_dysfunction_proxy",
        feature_pattern=r"(cardiovascular|respiratory|renal|hepatic|coagulation|metabolic)_dysfunction|organ_dysfunction_count",
        feature_group="organ_dysfunction",
        source_table="derived from first-6h vitals, labs, and urine output",
        source_variable="summary clinical feature columns",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="thresholded dysfunction proxy flags and count",
        clinical_rationale="Approximates organ-system derangement with transparent first-window rules.",
        missingness_handling="missing source features do not trigger a proxy flag",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
        notes="These are not exact SOFA scores.",
    ),
    FeatureProvenance(
        feature_name="clinical_interaction",
        feature_pattern=(
            r"age_x_prev_dx_count|age_x_shock_index|lactate_x_hypotension|"
            r"resp_rate_x_spo2_deficit|platelets_x_inr|sirs_x_lactate|"
            r"creatinine_x_urine_output|bilirubin_x_inr|critical_count_x_missing_lab_count"
        ),
        feature_group="interaction",
        source_table="derived from first-6h clinical summaries and prior diagnoses",
        source_variable="base feature columns",
        time_window_start="0h or pre-ICU",
        time_window_end="6h",
        aggregation="pairwise clinical interaction",
        clinical_rationale="Captures clinically plausible effect modification without post-window information.",
        missingness_handling="inherits from source features",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="temporal_trend",
        feature_pattern=rf"({VITALS}|{LABS})_(pct_change|trend)",
        feature_group="trajectory",
        source_table="derived from first-6h summaries",
        source_variable="min/max or time-window values",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="percent change or signed trend",
        clinical_rationale="Trend features summarize early movement in physiologic state.",
        missingness_handling="downstream imputation",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
    FeatureProvenance(
        feature_name="distance_or_polynomial",
        feature_pattern=rf"({VITALS}|{LABS})(_(mean|min|max))?_(squared|dist_from_normal|log)",
        feature_group="derived_clinical",
        source_table="derived from first-6h summaries",
        source_variable="base vital/lab feature",
        time_window_start="0h",
        time_window_end="6h",
        aggregation="nonlinear transform",
        clinical_rationale="Captures nonlinear risk from abnormal values.",
        missingness_handling="inherits from source features",
        is_binary=False,
        is_missingness_indicator=False,
        leakage_risk="low",
        allowed_for_model=True,
    ),
)


def feature_dictionary_dataframe() -> pd.DataFrame:
    """Return the project feature dictionary as a DataFrame."""
    records = [entry.to_record() for entry in FEATURE_PROVENANCE_REGISTRY]
    return pd.DataFrame(records, columns=FEATURE_DICTIONARY_COLUMNS)


def export_feature_dictionary(path) -> Path:
    """Write the aggregate feature dictionary to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary_dataframe().to_csv(output_path, index=False)
    return output_path


def match_feature_name(feature_name: str) -> FeatureProvenance | None:
    """Return the first provenance entry matching a feature name."""
    for entry in FEATURE_PROVENANCE_REGISTRY:
        if entry.matches(feature_name):
            return entry
    return None


def validate_feature_provenance(
    feature_names,
    *,
    require_allowed: bool = True,
) -> dict:
    """Validate feature names against provenance metadata.

    Returns lists instead of raising so callers can decide whether to warn or
    fail. Tests use this as a hard gate.
    """
    unmatched = []
    disallowed = []
    matched = {}
    for feature_name in feature_names:
        entry = match_feature_name(feature_name)
        if entry is None:
            unmatched.append(feature_name)
            continue
        matched[feature_name] = entry.feature_name
        if require_allowed and not entry.allowed_for_model:
            disallowed.append(feature_name)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "disallowed": disallowed,
    }
