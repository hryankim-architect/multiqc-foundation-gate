"""Label sheet loader + stratified train/holdout split.

The label CSV at `data/labels.csv` is the canonical source of truth for
which MultiQC report belongs to which class. Schema:

    report_id, label, augmentation, label_source, notes

Where:
    - `report_id`: directory name under `data/multiqc_reports/`
    - `label`: one of {"include", "exclude", "manual-review"}
    - `augmentation`: one of {"original", "module_dropout",
        "adapter_injection", "quality_degradation", "mixed_issue"}
    - `label_source`: who set the label ("base" for originals, "augment_v0.1"
        for synthetic augmentations, "manual" for hand-labeled overrides)
    - `notes`: free-text comment

Train / holdout split is stratified by `label` and uses a fixed seed (42)
so the demo is byte-reproducible across re-runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Class labels in the order the classifier head will use.
LABELS = ("include", "exclude", "manual-review")
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


class LabelRow(NamedTuple):
    """One row of the label sheet."""

    report_id: str
    label: str
    augmentation: str
    label_source: str
    notes: str


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Load and validate the label sheet."""
    df = pd.read_csv(csv_path)
    expected = {"report_id", "label", "augmentation", "label_source", "notes"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {sorted(missing)}")
    bad = set(df["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"labels CSV has unknown labels: {sorted(bad)}")
    df = df.fillna("")
    df["report_id"] = df["report_id"].astype(str)
    return df


def stratified_split(
    df: pd.DataFrame,
    holdout_per_class: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train + holdout stratified by label.

    With `holdout_per_class=2` and the default 50-row label sheet
    (10 include, 20 exclude, 20 manual-review after Phase C completes),
    the holdout is 6 samples (2 per class) and the train set is 44.

    With the Hour 3 partial sheet (10 include + 10 module-dropout,
    n=20 total), `holdout_per_class=2` gives a 4-sample holdout.

    Args:
        df: DataFrame returned by `load_labels`.
        holdout_per_class: How many samples per class go into the holdout.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, holdout_df) — both with the same column schema as `df`.
    """
    rng_kwargs = {"random_state": seed, "replace": False}
    holdout_rows: list[pd.DataFrame] = []

    for label in LABELS:
        class_df = df[df["label"] == label]
        if len(class_df) < holdout_per_class:
            # Smaller than expected — use all samples for holdout, no train.
            # The classifier will warn; this is intentional so the user sees
            # the imbalance.
            holdout_rows.append(class_df)
        else:
            holdout_rows.append(class_df.sample(n=holdout_per_class, **rng_kwargs))

    holdout_df = pd.concat(holdout_rows, ignore_index=True)
    train_df = df.drop(holdout_df.index).reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    return train_df, holdout_df


def label_to_idx(label: str) -> int:
    """Map a class label to its integer index for cross-entropy loss."""
    return LABEL_TO_IDX[label]


def idx_to_label(idx: int) -> str:
    """Inverse of `label_to_idx`."""
    return IDX_TO_LABEL[idx]
