"""Smoke tests for `src/multiqc_gate/train.py`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multiqc_gate import train


@pytest.fixture
def tiny_dataset():
    """Synthetic 30-sample 3-class dataset (10 per class)."""
    rng = np.random.default_rng(42)
    X = np.concatenate([
        rng.normal(loc=0.0, scale=0.5, size=(10, 28)),
        rng.normal(loc=2.0, scale=0.5, size=(10, 28)),
        rng.normal(loc=-2.0, scale=0.5, size=(10, 28)),
    ]).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    return X, y


def test_pick_device_returns_string():
    assert train.pick_device("cpu") == "cpu"
    auto = train.pick_device("auto")
    assert auto in {"cpu", "mps"}


def test_class_weights_inverse_frequency():
    y = np.array([0, 0, 0, 1, 1, 2])
    w = train._class_weights(y, n_classes=3)
    # Class 0 has 3 samples, class 2 has 1 -> class 2 weight > class 0 weight
    assert w[2] > w[0]
    assert (w > 0).all()


def test_train_one_fold_runs_and_returns_predictions(tiny_dataset):
    X, y = tiny_dataset
    config = train.TrainConfig(
        max_epochs=3,
        patience=10,
        device="cpu",
        seed=42,
    )
    # 20 train / 10 val stratified split
    train_idx = list(range(0, 7)) + list(range(10, 17)) + list(range(20, 27))
    val_idx = [7, 8, 9, 17, 18, 19, 27, 28, 29]
    fr = train.train_one_fold(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        config=config,
        fold=0,
        job_id="test-train",
    )
    assert fr.fold == 0
    assert 0 <= fr.best_epoch < config.max_epochs
    assert len(fr.holdout_y_pred) == len(val_idx)
    assert len(fr.epoch_history) >= 1


def test_train_cv_returns_n_splits_results(tiny_dataset, tmp_path, monkeypatch):
    """Full CV smoke — substrate emits to tmp dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    X, y = tiny_dataset
    config = train.TrainConfig(max_epochs=3, patience=10, device="cpu", seed=42)
    results = train.train_cv(X, y, config, job_id="smoke", n_splits=3)
    assert len(results) == 3
    for r in results:
        assert r.holdout_y_pred  # non-empty


def test_labels_to_indices_roundtrip():
    from multiqc_gate.labels import LABELS

    indices = train.labels_to_indices(list(LABELS))
    assert indices.tolist() == list(range(len(LABELS)))
