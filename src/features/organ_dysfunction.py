"""Clinically interpretable first-window organ dysfunction proxies."""

from __future__ import annotations

import pandas as pd


def _combine_rules(df: pd.DataFrame, rules: list[tuple[str, str, float]]):
    used = False
    condition = pd.Series(False, index=df.index)
    for column, operator, threshold in rules:
        if column not in df.columns:
            continue
        used = True
        if operator == "<":
            condition = condition | (df[column] < threshold)
        elif operator == "<=":
            condition = condition | (df[column] <= threshold)
        elif operator == ">":
            condition = condition | (df[column] > threshold)
        elif operator == ">=":
            condition = condition | (df[column] >= threshold)
        else:
            raise ValueError(f"unsupported operator: {operator}")
    if not used:
        return None
    return condition.astype(int)


def compute_organ_dysfunction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight organ dysfunction proxy flags.

    These are transparent portfolio features, not exact SOFA scores.
    """
    df = features.copy()

    proxy_rules = {
        "cardiovascular_dysfunction": [
            ("map_min", "<", 65),
            ("sbp_min", "<", 90),
            ("shock_index", ">", 0.9),
        ],
        "respiratory_dysfunction": [
            ("spo2_min", "<", 92),
            ("resp_rate_max", ">", 30),
            ("resp_distress_score", ">", 30),
        ],
        "renal_dysfunction": [
            ("creatinine_max", ">=", 1.5),
            ("bun_max", ">=", 40),
            ("urine_output_total", "<", 300),
        ],
        "hepatic_dysfunction": [
            ("bilirubin_max", ">=", 2.0),
            ("inr_max", ">=", 1.5),
        ],
        "coagulation_dysfunction": [
            ("platelets_min", "<", 100),
            ("hemoglobin_min", "<", 8),
        ],
        "metabolic_dysfunction": [
            ("lactate_max", ">=", 2.0),
            ("bicarbonate_min", "<", 18),
            ("anion_gap_max", ">", 16),
        ],
    }

    created_cols = []
    for feature_name, rules in proxy_rules.items():
        values = _combine_rules(df, rules)
        if values is None:
            continue
        df[feature_name] = values
        created_cols.append(feature_name)

    if created_cols:
        df["organ_dysfunction_count"] = df[created_cols].sum(axis=1).astype(int)

    return df
