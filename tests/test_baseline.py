"""Smoke tests for `src/multiqc_gate/baseline.py`."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from multiqc_gate import baseline


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(42)
    X = np.concatenate([
        rng.normal(loc=0.0, scale=0.5, size=(10, 28)),
        rng.normal(loc=2.0, scale=0.5, size=(10, 28)),
        rng.normal(loc=-2.0, scale=0.5, size=(10, 28)),
    ]).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    return X, y


def test_build_classifier_rejects_unknown():
    with pytest.raises(ValueError):
        baseline._build_classifier("xgboost_not_supported", seed=42)


def test_random_forest_baseline_runs(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    X, y = tiny_dataset
    results = baseline.run_baseline(X, y, "random_forest", job_id="t", n_splits=3, seed=42)
    assert len(results) == 3
    for r in results:
        assert r.method == "random_forest"
        assert 0 <= r.accuracy <= 1


def test_logistic_regression_baseline_runs(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    X, y = tiny_dataset
    results = baseline.run_baseline(X, y, "logistic_regression", job_id="t", n_splits=3, seed=42)
    assert len(results) == 3
    for r in results:
        assert r.method == "logistic_regression"
        assert 0 <= r.accuracy <= 1


def test_run_all_baselines_returns_both(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    X, y = tiny_dataset
    out = baseline.run_all_baselines(X, y, job_id="t", n_splits=3, seed=42)
    assert set(out.keys()) == {"random_forest", "logistic_regression"}
