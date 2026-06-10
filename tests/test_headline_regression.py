"""Regression guard: pin the sklearn-vs-MLP headline on the committed n=50 data.

The portfolio headline is "sklearn beats a collapsing MLP": on the real n=50 dataset,
LogisticRegression (StandardScaler-normalized, so lbfgs converges and the score is
version-stable) and RandomForest both clear the MLP's 0.40. This locks LR = 0.80 and
RF = 0.84 to the committed features + split so the number cannot silently drift again —
the unscaled LR scored 0.86 on sklearn 1.7 but 0.56 on a newer release, which is exactly
what this test exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from multiqc_gate import baseline  # noqa: E402
from multiqc_gate import eval as evalmod  # noqa: E402
from multiqc_gate.labels import LABELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def _load_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load the committed n=50 features + labels, aligned by sample-id order."""
    X = np.load(DATA / "features.npy")
    sample_ids = (DATA / "feature_sample_ids.txt").read_text().splitlines()
    df = pd.read_csv(DATA / "labels.csv").fillna("")
    label_map = dict(zip(df["report_id"].astype(str), df["label"], strict=False))
    y = np.array([LABEL_TO_IDX[label_map[s]] for s in sample_ids], dtype=np.int64)
    return X, y


def _accuracy(method: str, X: np.ndarray, y: np.ndarray) -> float:
    folds = baseline.run_baseline(X, y, method, job_id="regress", n_splits=5, seed=42)
    summary = evalmod.aggregate_folds(
        method,
        fold_y_true=[f.holdout_y_true for f in folds],
        fold_y_pred=[f.holdout_y_pred for f in folds],
    )
    return round(summary.accuracy_mean, 2)


def test_sklearn_headline_is_pinned(tmp_path, monkeypatch):
    # Isolate the audit-ledger write to a temp dir (audit emits relative to cwd).
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)

    X, y = _load_xy()
    assert X.shape == (50, 28), f"committed feature matrix changed shape: {X.shape}"

    lr = _accuracy("logistic_regression", X, y)
    rf = _accuracy("random_forest", X, y)

    # The headline numbers, pinned. LR is StandardScaler-wrapped (baseline.py) so this
    # is the converged, version-stable 0.80 — not the fragile unscaled 0.86.
    assert lr == 0.80, f"LogReg drifted to {lr} (expected 0.80; is StandardScaler still applied?)"
    assert rf == 0.84, f"RandomForest drifted to {rf} (expected 0.84)"

    # The qualitative claim: both sklearn baselines clear the MLP's 0.40 collapse.
    assert lr > 0.40 and rf > 0.40
