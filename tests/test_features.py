"""Unit tests for `src/multiqc_gate/features.py`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multiqc_gate import augment, features


@pytest.fixture
def reports_dir() -> Path:
    p = Path(__file__).parent.parent / "data" / "multiqc_reports"
    if not p.exists():
        pytest.skip(f"reports dir missing: {p}")
    return p


@pytest.fixture
def base_sample_json(reports_dir) -> Path:
    p = reports_dir / "SRR1039513" / "multiqc_data" / "multiqc_data.json"
    if not p.exists():
        pytest.skip(f"fixture missing: {p}")
    return p


def test_feature_vector_has_correct_shape(base_sample_json):
    vec = features.extract_features(base_sample_json)
    assert vec.shape == (features.N_FEATURES,)
    assert vec.dtype == np.float32
    assert features.N_FEATURES == 28


def test_feature_vector_has_named_dimensions():
    assert len(features.FEATURE_NAMES) == features.N_FEATURES
    # Spot-check ordering
    assert features.FEATURE_NAMES[0].startswith("num__")
    assert any("status__adapter_content" == n for n in features.FEATURE_NAMES)
    assert features.FEATURE_NAMES[-1].startswith("present__fastqc")


def test_base_sample_features_are_clean(base_sample_json):
    """A real Himes baseline sample should have all modules present and pass."""
    vec = features.extract_features(base_sample_json)
    # Total Sequences > 0 (numerical position 0)
    assert vec[0] > 0
    # %GC in plausible range (numerical position 1)
    assert 30.0 <= vec[1] <= 70.0
    # All module-presence flags == 1.0 (last 8 positions)
    presence = vec[-len(features.FASTQC_PLOT_MODULES):]
    assert np.all(presence == 1.0)
    # All module-status values should be 0 (pass) since this is a clean fixture
    status_slice = slice(
        len(features.NUMERICAL_FIELDS),
        len(features.NUMERICAL_FIELDS) + len(features.FASTQC_STATUS_MODULES),
    )
    statuses = vec[status_slice]
    assert (statuses == 0.0).all(), f"unexpected non-pass statuses: {statuses}"


def test_dropout_sample_has_missing_modules(reports_dir, tmp_path):
    """An augmented dropout sample should have presence==0 for dropped modules."""
    source = reports_dir / "SRR1039513"
    target = tmp_path / "SRR1039513_dropout_2mod"
    dropped = augment.augment_module_dropout(source, target, n_drop=2, seed=42)
    assert len(dropped) == 2

    json_path = target / "multiqc_data" / "multiqc_data.json"
    vec = features.extract_features(json_path)

    # Presence flags for dropped modules should now be 0.
    presence_slice = slice(
        features.N_FEATURES - len(features.FASTQC_PLOT_MODULES),
        features.N_FEATURES,
    )
    presence = vec[presence_slice]
    # 2 of 8 should be 0 (dropped), 6 should be 1
    assert int(presence.sum()) == len(features.FASTQC_PLOT_MODULES) - 2


def test_extract_features_batch_returns_matrix(reports_dir):
    sample_ids = [
        "SRR1039513", "SRR1039514", "SRR1039515",
        "SRR1039516", "SRR1039517", "SRR1039518",
        "SRR1039519", "SRR1039520", "SRR1039521", "SRR1039522",
    ]
    X, used = features.extract_features_batch(reports_dir, sample_ids)
    assert X.shape == (10, features.N_FEATURES)
    assert used == sample_ids
    # All baseline samples should have positive Total Sequences
    assert (X[:, 0] > 0).all()


def test_extract_features_batch_skips_missing(reports_dir):
    sample_ids = ["SRR1039513", "SRR999999_not_real", "SRR1039514"]
    X, used = features.extract_features_batch(reports_dir, sample_ids)
    assert X.shape == (2, features.N_FEATURES)
    assert used == ["SRR1039513", "SRR1039514"]
