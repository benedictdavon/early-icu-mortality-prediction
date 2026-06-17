"""Leakage-safe train/validation/test splitting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _stratify_or_none(labels, stratify: bool):
    if not stratify:
        return None
    y = np.asarray(labels)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return None
    return y


def _group_frame(labels, patient_ids) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(labels)),
            "patient_id": np.asarray(patient_ids),
            "label": np.asarray(labels).astype(int),
        }
    )
    group_labels = rows.groupby("patient_id", sort=False)["label"].max().reset_index()
    return group_labels


def _indices_for_groups(patient_ids, selected_groups) -> np.ndarray:
    mask = pd.Series(patient_ids).isin(set(selected_groups)).to_numpy()
    return np.flatnonzero(mask)


def make_train_valid_test_split(
    labels,
    patient_ids=None,
    test_size: float = 0.20,
    valid_size: float = 0.20,
    stratify: bool = True,
    group_by_patient: bool = True,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create train/validation/test row indices.

    If patient_ids are provided, all rows for a patient are assigned to exactly
    one split. `valid_size` and `test_size` are fractions of the full dataset.
    """
    y = np.asarray(labels).astype(int)
    if len(y) == 0:
        raise ValueError("labels must contain at least one row")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1")
    if test_size + valid_size >= 1:
        raise ValueError("test_size + valid_size must be less than 1")

    if patient_ids is not None and group_by_patient:
        if len(patient_ids) != len(y):
            raise ValueError("patient_ids must have the same length as labels")

        groups = _group_frame(y, patient_ids)
        train_valid_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=random_state,
            stratify=_stratify_or_none(groups["label"], stratify),
        )
        valid_relative_size = valid_size / (1.0 - test_size)
        train_groups, valid_groups = train_test_split(
            train_valid_groups,
            test_size=valid_relative_size,
            random_state=random_state,
            stratify=_stratify_or_none(train_valid_groups["label"], stratify),
        )
        splits = {
            "train": _indices_for_groups(patient_ids, train_groups["patient_id"]),
            "validation": _indices_for_groups(patient_ids, valid_groups["patient_id"]),
            "test": _indices_for_groups(patient_ids, test_groups["patient_id"]),
        }
        assert_no_group_overlap(splits, patient_ids)
        return splits

    indices = np.arange(len(y))
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_or_none(y, stratify),
    )
    valid_relative_size = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_relative_size,
        random_state=random_state,
        stratify=_stratify_or_none(y[train_valid_idx], stratify),
    )
    return {
        "train": np.asarray(train_idx),
        "validation": np.asarray(valid_idx),
        "test": np.asarray(test_idx),
    }


def assert_no_group_overlap(splits: dict[str, np.ndarray], patient_ids) -> None:
    """Assert no patient identifier appears in more than one split."""
    ids = np.asarray(patient_ids)
    split_sets = {
        name: set(ids[np.asarray(indexes, dtype=int)])
        for name, indexes in splits.items()
    }
    names = list(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                sample = sorted(str(value) for value in list(overlap)[:5])
                raise ValueError(
                    f"Patient IDs overlap between {left} and {right}: {sample}"
                )
