import pandas as pd
import numpy as np

def identify_and_handle_outliers(df):
    """Identify and handle outliers using clinical knowledge and statistical methods"""
    print("Handling outliers...")

    processed_df = df.copy()

    # 1. Clinical range constraints - adjust physiologically impossible values
    clinical_ranges = {
        "heart_rate_mean": (30, 200),
        "resp_rate_mean": (5, 60),
        "map_mean": (40, 180),
        "sbp_mean": (60, 220),
        "dbp_mean": (30, 120),
        "temp_mean": (33, 42),
        "spo2_mean": (60, 100),
        "glucose_mean": (30, 600),
        "creatinine_mean": (0.1, 15),
        "lactate_mean": (0.1, 30),
    }

    # Apply clinical range constraints
    for feature, (lower, upper) in clinical_ranges.items():
        if feature in processed_df.columns:
            # Calculate how many outliers were found
            outliers_count = (
                (processed_df[feature] < lower) | (processed_df[feature] > upper)
            ).sum()

            # Apply capping
            processed_df[feature] = processed_df[feature].clip(lower=lower, upper=upper)

            if outliers_count > 0:
                print(
                    f"  - Capped {outliers_count} outliers in {feature} to range [{lower}, {upper}]"
                )

    # 2. Statistical outlier handling for other numeric features
    # Use Winsorizing (capping at percentiles) for features with high skew
    numeric_features = processed_df.select_dtypes(include=[np.number]).columns

    for feature in numeric_features:
        # Skip features already handled and ID columns
        if (
            feature in clinical_ranges.keys()
            or "id" in feature.lower()
            or "_missing" in feature
        ):
            continue

        # Calculate skewness
        skew = processed_df[feature].skew()

        # For highly skewed data (abs(skew) > 2), apply winsorizing at 1% and 99%
        if abs(skew) > 2:
            lower_bound = processed_df[feature].quantile(0.01)
            upper_bound = processed_df[feature].quantile(0.99)

            # Count outliers
            outliers_count = (
                (processed_df[feature] < lower_bound)
                | (processed_df[feature] > upper_bound)
            ).sum()

            # Apply winsorizing
            processed_df[feature] = processed_df[feature].clip(
                lower=lower_bound, upper=upper_bound
            )

            if outliers_count > 0:
                print(
                    f"  - Winsorized {outliers_count} outliers in {feature} (skew={skew:.2f})"
                )

    return processed_df
