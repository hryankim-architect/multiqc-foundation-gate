"""Evaluation aggregation across folds + classification reporting.

Takes a list of FoldResult (or BaselineFoldResult) and produces:
    - mean +/- std accuracy across folds
    - macro-averaged F1 across folds
    - per-class precision / recall / F1 (sklearn classification_report)
    - aggregated confusion matrix (sum across folds)
    - JSON summary saved to artifacts/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from multiqc_gate.labels import LABELS


@dataclass
class EvalSummary:
    """Aggregated metrics for one classifier across all folds."""

    method: str  # "mlp", "random_forest", "logistic_regression"
    n_folds: int
    n_samples_total: int
    accuracy_mean: float
    accuracy_std: float
    f1_macro_mean: float
    f1_macro_std: float
    confusion_matrix: list[list[int]]  # shape (n_classes, n_classes)
    classification_report: dict[str, Any]  # sklearn dict format
    per_fold_accuracy: list[float]


def aggregate_folds(
    method: str,
    fold_y_true: list[list[int]],
    fold_y_pred: list[list[int]],
) -> EvalSummary:
    """Aggregate per-fold predictions into one EvalSummary."""
    accuracies: list[float] = []
    f1s: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for y_t, y_p in zip(fold_y_true, fold_y_pred):
        y_t_arr = np.array(y_t)
        y_p_arr = np.array(y_p)
        if len(y_t_arr) == 0:
            continue
        acc = float((y_t_arr == y_p_arr).mean())
        accuracies.append(acc)
        f1 = float(f1_score(y_t_arr, y_p_arr, average="macro", zero_division=0.0))
        f1s.append(f1)
        all_y_true.extend(y_t)
        all_y_pred.extend(y_p)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(len(LABELS))))
    report = classification_report(
        all_y_true,
        all_y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0.0,
    )

    return EvalSummary(
        method=method,
        n_folds=len(accuracies),
        n_samples_total=len(all_y_true),
        accuracy_mean=float(np.mean(accuracies)) if accuracies else 0.0,
        accuracy_std=float(np.std(accuracies)) if accuracies else 0.0,
        f1_macro_mean=float(np.mean(f1s)) if f1s else 0.0,
        f1_macro_std=float(np.std(f1s)) if f1s else 0.0,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        per_fold_accuracy=accuracies,
    )


def write_summary_json(summary: EvalSummary, out_path: Path) -> None:
    """Persist EvalSummary to JSON for downstream substrate consumers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": summary.method,
        "n_folds": summary.n_folds,
        "n_samples_total": summary.n_samples_total,
        "accuracy_mean": summary.accuracy_mean,
        "accuracy_std": summary.accuracy_std,
        "f1_macro_mean": summary.f1_macro_mean,
        "f1_macro_std": summary.f1_macro_std,
        "confusion_matrix": summary.confusion_matrix,
        "classification_report": summary.classification_report,
        "per_fold_accuracy": summary.per_fold_accuracy,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def compare_methods(summaries: list[EvalSummary]) -> dict[str, Any]:
    """Produce a side-by-side comparison table for the README climax.

    Returns a dict with method-keyed entries containing the headline
    metrics. The README climax section then prints this as a markdown
    table.
    """
    comparison = {}
    for s in summaries:
        comparison[s.method] = {
            "accuracy_mean": round(s.accuracy_mean, 3),
            "accuracy_std": round(s.accuracy_std, 3),
            "f1_macro_mean": round(s.f1_macro_mean, 3),
            "f1_macro_std": round(s.f1_macro_std, 3),
            "n_folds": s.n_folds,
            "n_samples": s.n_samples_total,
        }
    return comparison
