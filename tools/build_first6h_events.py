"""Build long-form first-6-hour event CSV for MAFNet training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction.expanded_features import LAB_ITEMIDS, VITAL_ITEMIDS
from feature_extraction.expanded_features import _flatten_item_map
from features._events import prepare_windowed_events


def _read_filtered_chart_events(
    chart_path: Path,
    cohort: pd.DataFrame,
    *,
    chunksize: int,
) -> pd.DataFrame:
    item_to_variable = _flatten_item_map(VITAL_ITEMIDS)
    stay_ids = set(cohort["stay_id"])
    chunks = []

    print(f"Reading chart events from {chart_path}")
    for i, chunk in enumerate(
        pd.read_csv(
            chart_path,
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            chunksize=chunksize,
        ),
        start=1,
    ):
        chunk = chunk.dropna(subset=["stay_id", "charttime", "itemid", "valuenum"]).copy()
        chunk["itemid"] = chunk["itemid"].astype(int)
        filtered = chunk[
            chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(item_to_variable)
        ].copy()
        if not filtered.empty:
            filtered["variable"] = filtered["itemid"].map(item_to_variable)
            filtered["source"] = "vital"
            chunks.append(
                filtered[["stay_id", "charttime", "variable", "source", "valuenum"]]
            )
        if i % 25 == 0:
            kept = sum(len(part) for part in chunks)
            print(f"  chart chunks processed={i}, kept_rows={kept}")

    if not chunks:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "variable", "source", "valuenum"]
        )
    return pd.concat(chunks, ignore_index=True)


def _read_filtered_lab_events(
    lab_path: Path,
    cohort: pd.DataFrame,
    *,
    chunksize: int,
) -> pd.DataFrame:
    item_to_variable = _flatten_item_map(LAB_ITEMIDS)
    subject_ids = set(cohort["subject_id"])
    stay_lookup = cohort[["subject_id", "stay_id"]].drop_duplicates()
    chunks = []

    print(f"Reading lab events from {lab_path}")
    for i, chunk in enumerate(
        pd.read_csv(
            lab_path,
            usecols=["subject_id", "charttime", "itemid", "valuenum"],
            chunksize=chunksize,
        ),
        start=1,
    ):
        chunk = chunk.dropna(
            subset=["subject_id", "charttime", "itemid", "valuenum"]
        ).copy()
        chunk["itemid"] = chunk["itemid"].astype(int)
        filtered = chunk[
            chunk["subject_id"].isin(subject_ids) & chunk["itemid"].isin(item_to_variable)
        ].copy()
        if not filtered.empty:
            filtered = filtered.merge(stay_lookup, on="subject_id", how="inner")
            filtered["variable"] = filtered["itemid"].map(item_to_variable)
            filtered["source"] = "lab"
            chunks.append(
                filtered[["stay_id", "charttime", "variable", "source", "valuenum"]]
            )
        if i % 25 == 0:
            kept = sum(len(part) for part in chunks)
            print(f"  lab chunks processed={i}, kept_rows={kept}")

    if not chunks:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "variable", "source", "valuenum"]
        )
    return pd.concat(chunks, ignore_index=True)


def build_first6h_events(
    *,
    cohort_path: Path,
    icu_dir: Path,
    hosp_dir: Path,
    output_path: Path,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path)
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort = cohort[["subject_id", "stay_id", "intime"]].drop_duplicates()

    chart_events = _read_filtered_chart_events(
        icu_dir / "_chartevents.csv",
        cohort,
        chunksize=chunksize,
    )
    lab_events = _read_filtered_lab_events(
        hosp_dir / "_labevents.csv",
        cohort,
        chunksize=chunksize,
    )
    events = pd.concat([chart_events, lab_events], ignore_index=True)
    if events.empty:
        raise ValueError("No eligible MAFNet events found")

    events["charttime"] = pd.to_datetime(events["charttime"])
    windowed = prepare_windowed_events(
        events,
        cohort,
        id_col="stay_id",
        event_time_col="charttime",
        intime_col="intime",
        window_hours=6.0,
    )
    output_cols = ["stay_id", "charttime", "variable", "source", "valuenum"]
    windowed = windowed[output_cols].sort_values(["stay_id", "charttime", "variable"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    windowed.to_csv(output_path, index=False)
    print(f"Wrote {len(windowed)} first-6-hour events to {output_path}")
    return windowed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAFNet first-6-hour event CSV")
    parser.add_argument("--cohort-path", default=str(ROOT / "data" / "processed" / "final_cohort.csv"))
    parser.add_argument("--icu-dir", default=str(ROOT / "data" / "icu"))
    parser.add_argument("--hosp-dir", default=str(ROOT / "data" / "hosp"))
    parser.add_argument("--output-path", default=str(ROOT / "data" / "processed" / "first6h_events.csv"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_first6h_events(
        cohort_path=Path(args.cohort_path),
        icu_dir=Path(args.icu_dir),
        hosp_dir=Path(args.hosp_dir),
        output_path=Path(args.output_path),
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
