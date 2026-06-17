from __future__ import annotations

import numpy as np

from data.splitting import assert_no_group_overlap, make_train_valid_test_split


def test_no_patient_overlap_between_splits():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    patient_ids = np.array([1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8])

    splits = make_train_valid_test_split(
        labels,
        patient_ids=patient_ids,
        test_size=0.25,
        valid_size=0.25,
        stratify=True,
        group_by_patient=True,
        random_state=7,
    )

    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(indexes) for indexes in splits.values()) == len(labels)
    assert_no_group_overlap(splits, patient_ids)


def test_split_indices_cover_each_row_once_without_patient_ids():
    labels = np.array([0, 1] * 20)
    splits = make_train_valid_test_split(labels, random_state=3)
    combined = np.concatenate(list(splits.values()))

    assert sorted(combined.tolist()) == list(range(len(labels)))
