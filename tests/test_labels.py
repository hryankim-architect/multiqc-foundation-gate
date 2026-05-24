"""Unit tests for `src/multiqc_gate/labels.py`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multiqc_gate import labels


@pytest.fixture
def labels_csv() -> Path:
    """The committed Hour 3 label sheet (10 base + 10 dropout = 20 rows)."""
    p = Path(__file__).parent.parent / "data" / "labels.csv"
    if not p.exists():
        pytest.skip(f"labels.csv missing: {p}")
    return p


def test_load_labels_parses_committed_sheet(labels_csv):
    df = labels.load_labels(labels_csv)
    # Phase C complete (Hour 4.B): 10 base + 10 dropout + 30 FASTQ-augmented = 50
    assert len(df) == 50
    assert set(df.columns) == {
        "report_id",
        "label",
        "augmentation",
        "label_source",
        "notes",
    }
    assert set(df["label"].unique()) == {"include", "exclude", "manual-review"}


def test_load_labels_rejects_unknown_label(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "report_id,label,augmentation,label_source,notes\n"
        "SRR1,unknown_label,original,base,bad row\n"
    )
    with pytest.raises(ValueError):
        labels.load_labels(bad_csv)


def test_stratified_split_preserves_class_in_holdout(labels_csv):
    df = labels.load_labels(labels_csv)
    train, holdout = labels.stratified_split(df, holdout_per_class=2, seed=42)

    # Phase C complete: 3 classes present (include 10 / exclude 20 /
    # manual-review 20). holdout = 2 per class = 6 total, train = 44.
    expected_holdout = 2 * len(df["label"].unique())
    assert len(holdout) == expected_holdout
    assert len(train) == len(df) - expected_holdout

    # Each class present in the sheet must appear in the holdout.
    for label in df["label"].unique():
        assert (holdout["label"] == label).sum() == 2


def test_stratified_split_is_deterministic(labels_csv):
    df = labels.load_labels(labels_csv)
    train_a, holdout_a = labels.stratified_split(df, holdout_per_class=2, seed=42)
    train_b, holdout_b = labels.stratified_split(df, holdout_per_class=2, seed=42)
    pd.testing.assert_frame_equal(
        holdout_a.sort_values("report_id").reset_index(drop=True),
        holdout_b.sort_values("report_id").reset_index(drop=True),
    )


def test_label_index_roundtrip():
    for label in labels.LABELS:
        idx = labels.label_to_idx(label)
        assert labels.idx_to_label(idx) == label
