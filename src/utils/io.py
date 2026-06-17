"""File and output-directory helpers."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def safe_path_token(value: str | int | None, default: str = "run") -> str:
    """Convert a label into a compact filesystem-safe token."""
    text = str(value if value not in (None, "") else default).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or default


def build_experiment_output_dir(
    base_dir: str | Path,
    *,
    experiment_name: str,
    model_name: str,
    split_name: str,
    seed: int,
    run_date: date | str | None = None,
    create: bool = True,
) -> Path:
    """Build a traceable output directory name for an experiment run."""
    if run_date is None:
        date_token = date.today().isoformat()
    elif isinstance(run_date, date):
        date_token = run_date.isoformat()
    else:
        date_token = safe_path_token(run_date)

    parts = [
        date_token,
        safe_path_token(experiment_name),
        safe_path_token(model_name),
        safe_path_token(split_name),
        f"seed{int(seed)}",
    ]
    output_dir = Path(base_dir) / "__".join(parts)
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
