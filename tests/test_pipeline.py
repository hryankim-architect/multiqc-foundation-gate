"""End-to-end smoke tests for the P2 MultiQC-gate pipeline.

These tests exercise the substrate bracket (audit_start -> work ->
audit_end + chain validity + tamper detection) using a small synthetic
dataset. A full real-data smoke run is in `scripts/run_lab.sh` /
`make run`; the unit tests stay fast by using n=30 synthetic features.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # pipeline needs torch
sklearn = pytest.importorskip("sklearn")

from multiqc_gate import audit, pipeline  # noqa: E402
from multiqc_gate.features import N_FEATURES  # noqa: E402
from multiqc_gate.labels import LABELS  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path):
    """Lay down a synthetic features.npy + labels.csv at tmp_path/data/."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rng = np.random.default_rng(42)
    n_per_class = 10
    X = np.concatenate([
        rng.normal(loc=0.0, scale=0.5, size=(n_per_class, N_FEATURES)),
        rng.normal(loc=2.0, scale=0.5, size=(n_per_class, N_FEATURES)),
        rng.normal(loc=-2.0, scale=0.5, size=(n_per_class, N_FEATURES)),
    ]).astype(np.float32)

    sample_ids = [f"sample_{i:03d}" for i in range(X.shape[0])]
    labels = [LABELS[0]] * n_per_class + [LABELS[1]] * n_per_class + [LABELS[2]] * n_per_class

    np.save(data_dir / "features.npy", X)
    (data_dir / "feature_sample_ids.txt").write_text("\n".join(sample_ids))

    with (data_dir / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["report_id", "label", "augmentation", "label_source", "notes"])
        for sid, lbl in zip(sample_ids, labels, strict=True):
            w.writerow([sid, lbl, "synthetic", "test", ""])

    return data_dir


def test_pipeline_runs_end_to_end_with_synthetic_data(fake_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    out_dir = tmp_path / "artifacts"
    result = pipeline.run_pipeline("smoke", out_dir)

    assert result["status"] == "success"
    assert "mlp_accuracy_mean" in result["metrics"]
    assert "random_forest_accuracy_mean" in result["metrics"]
    assert "logistic_regression_accuracy_mean" in result["metrics"]
    assert "drift_n_drifted" in result["metrics"]

    # Comparison + drift JSON artifacts should exist.
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "drift.json").exists()
    assert (out_dir / "eval_mlp.json").exists()


def test_audit_chain_is_valid_after_pipeline(fake_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts")

    ok, n_entries, first_bad = audit.verify()
    assert ok, f"audit chain invalid at {first_bad}"
    # pipeline_start + dataset_loaded + (5 folds x 2 + ~epochs) +
    # comparison_table + drift_summary + pipeline_end >= 10
    assert n_entries >= 10


def test_audit_chain_detects_tamper(fake_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts")
    ledger = audit.DEFAULT_LEDGER

    lines = ledger.read_text().splitlines()
    assert len(lines) >= 2
    tampered = json.loads(lines[0])
    tampered["fields"]["out_dir"] = "/etc/evil"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")

    ok, _, first_bad = audit.verify()
    assert not ok
    assert first_bad is not None


def test_pipeline_records_failure_in_audit_when_features_missing(tmp_path, monkeypatch):
    """If features.npy is missing, the pipeline raises but still emits pipeline_end."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline("smoke", tmp_path / "artifacts")

    # pipeline_start + pipeline_error + pipeline_end should all be in the ledger.
    ok, n, _ = audit.verify()
    assert ok
    assert n >= 2  # start + end (error in between also recorded)
