"""Lightweight checks for obvious target leakage in processed ICU features.

This script is intentionally separate from the runtime package. It expects a
processed feature CSV with a `mortality` target column and reports suspicious
feature names, high correlations with the target, and duplicate rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def prepare_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric matrix suitable for lightweight sklearn probes."""
    numeric_df = pd.get_dummies(df, dummy_na=True)
    return numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed features for leakage signals")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/preprocessed_xgboost_features.csv"),
        help="Path to a processed feature CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)

    if "mortality" not in df.columns:
        raise ValueError("Expected a `mortality` target column")

    corr_with_target = df.corr(numeric_only=True)["mortality"].abs().sort_values(ascending=False)
    print("Top numeric correlations with mortality:")
    print(corr_with_target.head(10))

    suspicious_terms = ["death", "expire", "mortality", "outcome", "survival", "died"]
    suspicious_features = [
        col
        for col in df.columns
        if col != "mortality" and any(term in col.lower() for term in suspicious_terms)
    ]
    print("Suspicious feature names:", suspicious_features)

    duplicate_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {duplicate_count}")

    X = prepare_model_matrix(df.drop("mortality", axis=1))
    y = df["mortality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Random forest accuracy with alternate split: {rf.score(X_test, y_test):.4f}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    print(f"Logistic regression accuracy with alternate split: {lr.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
