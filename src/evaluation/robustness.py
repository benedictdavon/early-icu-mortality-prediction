"""Robustness summaries across random seeds or repeated runs."""

from __future__ import annotations

import pandas as pd


def summarize_seed_stability(records, metric_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Summarize metric variation across seeds without row-level outputs."""
    frame = pd.DataFrame(records)
    if "seed" not in frame.columns:
        raise ValueError("records must include a seed column")
    missing = [column for column in metric_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"records are missing metric columns: {missing}")
    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "n_seeds": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["summarize_seed_stability"]
