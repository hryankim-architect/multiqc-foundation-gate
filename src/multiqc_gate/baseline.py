"""Sklearn baselines for honest comparison against the MLP.

Two baselines:
    - RandomForestClassifier: typically the strong baseline on tabular
      feature vectors with n < 100 samples.
    - LogisticRegression: a simple linear baseline that establishes the
      lower bound on signal in the feature vector.

Both run the same stratified K-fold split as the MLP so the comparison
is apples-to-apples. The README honest-scope statement is built around
the observation that on n=50 the sklearn baselines may match or beat
the MLP, which is the *correct* signal — the substrate value is in the
audit / MLflow / drift framework, not in the MLP-beats-everything claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from multiqc_gate import audit


@dataclass
class BaselineFoldResult:
    """Per-fold output of a single sklearn baseline."""

    method: str
    fold: int
    accuracy: float
    holdout_y_true: list[int]
    holdout_y_pred: list[int]
    holdout_y_proba: list[list[float]] = field(default_factory=list)


def _build_classifier(method: str, seed: int):
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=seed,
            class_weight="balanced",
        )
    if method == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            solver="lbfgs",
        )
    raise ValueError(f"unknown baseline method: {method}")


def run_baseline(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    job_id: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[BaselineFoldResult]:
    """Stratified K-fold for one sklearn baseline."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: list[BaselineFoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = _build_classifier(method, seed=seed + fold)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        try:
            y_proba = clf.predict_proba(X[val_idx]).tolist()
        except AttributeError:
            y_proba = []
        acc = float((y_pred == y[val_idx]).mean())
        audit.emit(
            action="baseline_fold_end",
            job_id=job_id,
            fields={
                "method": method,
                "fold": fold,
                "accuracy": acc,
            },
        )
        results.append(BaselineFoldResult(
            method=method,
            fold=fold,
            accuracy=acc,
            holdout_y_true=y[val_idx].tolist(),
            holdout_y_pred=y_pred.tolist(),
            holdout_y_proba=y_proba,
        ))
    return results


def run_all_baselines(
    X: np.ndarray,
    y: np.ndarray,
    job_id: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, list[BaselineFoldResult]]:
    """Convenience runner: returns {method_name: [fold_results, ...]}."""
    return {
        "random_forest": run_baseline(X, y, "random_forest", job_id, n_splits, seed),
        "logistic_regression": run_baseline(X, y, "logistic_regression", job_id, n_splits, seed),
    }
