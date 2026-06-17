from __future__ import annotations

import pandas as pd

from preprocessing.scaling import standardize_features, verify_feature_scaling


def test_scaling_helpers_do_not_transform_mortality_target():
    df = pd.DataFrame(
        {
            "mortality": [0, 1, 0, 1],
            "large_feature": [10.0, 1000.0, 2000.0, 3000.0],
            "subject_id": [1, 2, 3, 4],
        }
    )

    scaled = verify_feature_scaling(df.copy(), excluded_cols=["mortality"])
    standardized = standardize_features(scaled, excluded_cols=["mortality"])

    assert standardized["mortality"].tolist() == [0, 1, 0, 1]
    assert standardized["subject_id"].tolist() == [1, 2, 3, 4]
