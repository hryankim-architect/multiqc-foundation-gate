"""Unit tests for `src/multiqc_gate/eval.py`."""

from __future__ import annotations

import json

import pytest

from multiqc_gate import eval as eval_mod
from multiqc_gate.labels import LABELS


def test_aggregate_folds_handles_perfect_predictions():
    fold_y_true = [[0, 1, 2], [0, 1, 2]]
    fold_y_pred = [[0, 1, 2], [0, 1, 2]]
    summary = eval_mod.aggregate_folds("test", fold_y_true, fold_y_pred)
    assert summary.accuracy_mean == 1.0
    assert summary.f1_macro_mean == 1.0
    assert summary.n_folds == 2
    assert summary.n_samples_total == 6


def test_aggregate_folds_handles_misclassification():
    fold_y_true = [[0, 1, 2]]
    fold_y_pred = [[1, 1, 2]]  # 1/3 wrong on class 0
    summary = eval_mod.aggregate_folds("test", fold_y_true, fold_y_pred)
    assert summary.accuracy_mean == pytest.approx(2 / 3)
    assert 0 < summary.f1_macro_mean < 1


def test_aggregate_folds_confusion_matrix_shape():
    fold_y_true = [[0, 1, 2, 0, 1, 2]]
    fold_y_pred = [[0, 1, 2, 1, 1, 2]]
    summary = eval_mod.aggregate_folds("test", fold_y_true, fold_y_pred)
    cm = summary.confusion_matrix
    assert len(cm) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in cm)


def test_write_summary_json_roundtrip(tmp_path):
    fold_y_true = [[0, 1, 2]]
    fold_y_pred = [[0, 1, 2]]
    summary = eval_mod.aggregate_folds("test_method", fold_y_true, fold_y_pred)
    out = tmp_path / "summary.json"
    eval_mod.write_summary_json(summary, out)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["method"] == "test_method"
    assert payload["accuracy_mean"] == 1.0


def test_compare_methods_keyed_by_method():
    s1 = eval_mod.aggregate_folds("mlp", [[0, 1]], [[0, 1]])
    s2 = eval_mod.aggregate_folds("random_forest", [[0, 1]], [[1, 1]])
    comp = eval_mod.compare_methods([s1, s2])
    assert set(comp.keys()) == {"mlp", "random_forest"}
    assert "accuracy_mean" in comp["mlp"]
