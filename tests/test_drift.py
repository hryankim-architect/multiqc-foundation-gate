"""Unit tests for `src/multiqc_gate/drift.py`."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from multiqc_gate import drift


def test_no_drift_when_samples_identical():
    """Same distribution -> no feature should flag drift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(50, 5))
    test = rng.normal(size=(50, 5))
    results = drift.detect_drift(ref, test, alpha=0.05)
    n_drifted = sum(1 for r in results if r.drifted)
    # With alpha=0.05 and matched distributions, expect ~5% false positive
    # rate. For n_features=5, this is at most 1 with high probability.
    assert n_drifted <= 2


def test_drift_detected_when_distributions_differ():
    """Large mean shift -> all features should flag drift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=(50, 3))
    test = rng.normal(loc=5.0, scale=1.0, size=(50, 3))  # 5-sigma shift
    results = drift.detect_drift(ref, test, alpha=0.05)
    assert all(r.drifted for r in results)


def test_rejects_dim_mismatch():
    ref = np.zeros((10, 5))
    test = np.zeros((10, 7))
    with pytest.raises(ValueError, match="feature-dim mismatch"):
        drift.detect_drift(ref, test)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        drift.detect_drift(np.zeros((10,)), np.zeros((10, 5)))


def test_constant_column_edge_case():
    """Two constant columns at same value -> no drift; different values -> drift."""
    ref = np.full((20, 2), 5.0)
    test_same = np.full((20, 2), 5.0)
    res = drift.detect_drift(ref, test_same)
    assert not any(r.drifted for r in res)

    test_diff = np.full((20, 2), 7.0)
    res = drift.detect_drift(ref, test_diff)
    assert all(r.drifted for r in res)


def test_summarize_drift_returns_expected_keys():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(30, 4))
    test = rng.normal(loc=3.0, size=(30, 4))
    results = drift.detect_drift(ref, test)
    summary = drift.summarize_drift(results)
    assert set(summary.keys()) >= {
        "n_features", "n_drifted", "fraction_drifted", "alpha", "top_5_drifted"
    }
    assert summary["n_features"] == 4
