#!/usr/bin/env python3
"""Calibration diagnostic for the MultiQC include/exclude gate.

A gate that decides whether a sample is *included* in downstream analysis is
exactly the kind of component whose **confidence should be calibrated** — an
over-confident wrong "include" is a silent data-quality failure. This script
measures that, mirroring the honest-evaluation posture of dmoi-brca-poc v0.13.

The classifier is 3-class (include / exclude / manual-review), so we use the
general **confidence-ECE** (Guo et al. 2017: max-probability confidence vs
accuracy) plus a multiclass Brier score, computed on the pooled held-out
predictions of the existing stratified 5-fold baselines (LogReg, RandomForest).

Honest scope: with n=50 samples across 3 classes, post-hoc *re-calibration*
(temperature / isotonic / Platt) cannot be reliably estimated — each calibration
bin would hold a handful of points. So this is reported as a **diagnostic**
(is the gate over- or under-confident?), not a calibration *fix*. Reporting that
limit honestly is the point, same as the v0.13 null result.

Reproduce:  python scripts/calibrate_gate.py
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from multiqc_gate import audit, tracking  # noqa: E402
from multiqc_gate.baseline import run_baseline  # noqa: E402
from multiqc_gate.labels import LABELS, load_labels  # noqa: E402
from multiqc_gate.train import labels_to_indices  # noqa: E402

DATA = REPO / "data"
AUDIT = REPO / "audit"
JOB_ID = "multiqc-gate-calibration"
N_BINS = 10
METHODS = ("logistic_regression", "random_forest")


def _load_xy():
    features_path = DATA / "features.npy"
    sample_ids_path = DATA / "feature_sample_ids.txt"
    labels_path = DATA / "labels.csv"
    for p in (features_path, sample_ids_path, labels_path):
        if not p.exists():
            sys.stderr.write(
                f"ERROR: missing {p}. Run scripts/run_phase_c.py + feature "
                "extraction first (see README).\n"
            )
            raise SystemExit(1)
    x = np.load(features_path)
    sample_ids = sample_ids_path.read_text().splitlines()
    labels_df = load_labels(labels_path)
    label_map = dict(zip(labels_df["report_id"], labels_df["label"], strict=True))
    y = labels_to_indices([label_map[sid] for sid in sample_ids])
    return x, np.asarray(y)


def _pool_holdout(results):
    """Concatenate per-fold held-out (y_true, proba) across folds."""
    y_true, proba = [], []
    for r in results:
        if not r.holdout_y_proba:
            continue
        y_true.extend(r.holdout_y_true)
        proba.extend(r.holdout_y_proba)
    return np.asarray(y_true), np.asarray(proba, dtype=float)


def _confidence_ece(y_true, proba, n_bins=N_BINS):
    """Guo et al. confidence-ECE: |confidence - accuracy| per bin, weighted.

    Returns (ece, mean_confidence, accuracy, bins) where bins is a list of
    (lo, hi, count, mean_conf, acc).
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt:
            mc, ac = float(conf[mask].mean()), float(correct[mask].mean())
            ece += (cnt / n) * abs(mc - ac)
            bins.append((lo, hi, cnt, mc, ac))
    return ece, float(conf.mean()), float(correct.mean()), bins


def _multiclass_brier(y_true, proba):
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def main() -> int:
    x, y = _load_xy()
    print(f"=== MultiQC gate calibration diagnostic (n={len(y)}, "
          f"{len(LABELS)} classes) ===")
    dist = {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))}
    print(f"  class distribution: {dist}")

    AUDIT.mkdir(exist_ok=True)
    rel_path = AUDIT / "gate_calibration_reliability.tsv"
    rows, rel_lines = [], ["method\tbin_lo\tbin_hi\tcount\tmean_conf\taccuracy"]
    for method in METHODS:
        results = run_baseline(x, y, method=method, job_id=JOB_ID, n_splits=5, seed=42)
        y_true, proba = _pool_holdout(results)
        if len(y_true) == 0:
            print(f"  {method}: no probabilities available, skipping")
            continue
        ece, mean_conf, acc, bins = _confidence_ece(y_true, proba)
        brier = _multiclass_brier(y_true, proba)
        gap = mean_conf - acc
        tendency = "over-confident" if gap > 0.02 else ("under-confident" if gap < -0.02 else "well-matched")
        rows.append((method, ece, brier, mean_conf, acc, tendency))
        for lo, hi, cnt, mc, ac in bins:
            rel_lines.append(f"{method}\t{lo:.2f}\t{hi:.2f}\t{cnt}\t{mc:.4f}\t{ac:.4f}")
        print(f"  {method:20s} ECE={ece:.4f}  Brier={brier:.4f}  "
              f"conf={mean_conf:.3f} vs acc={acc:.3f} ({tendency})")
    rel_path.write_text("\n".join(rel_lines) + "\n")

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    table = "\n".join(
        f"| {m} | {ece:.4f} | {brier:.4f} | {mc:.3f} | {ac:.3f} | {t} |"
        for (m, ece, brier, mc, ac, t) in rows
    )
    (AUDIT / "gate_calibration.md").write_text(
        "# MultiQC gate — calibration diagnostic\n\n"
        f"Generated: {ts}\n\n"
        "## Why\n\n"
        "An include/exclude gate's *confidence* matters: an over-confident wrong "
        "'include' is a silent data-quality failure. This measures whether the "
        "gate's probabilities are calibrated (confidence-ECE, Guo et al.), on the "
        "pooled held-out predictions of the stratified 5-fold baselines.\n\n"
        f"- Samples: n={len(y)} across {len(LABELS)} classes {dist}.\n"
        "- Metric: confidence-ECE (max-probability vs accuracy) + multiclass Brier.\n\n"
        "## Results (pooled 5-fold held-out)\n\n"
        "| Method | ECE | Brier | mean conf | accuracy | tendency |\n"
        "|---|---|---|---|---|---|\n" + table + "\n\n"
        "Reliability bins: `gate_calibration_reliability.tsv`.\n\n"
        "## Honest scope\n\n"
        "This is a calibration **diagnostic, not a fix**. With n=50 across 3 "
        "classes, post-hoc re-calibration (temperature / isotonic / Platt) cannot "
        "be reliably estimated — each reliability bin holds only a handful of "
        "points, so a fitted calibrator would overfit. The deliverable is the "
        "honest measurement of whether the gate is over- or under-confident, in "
        "the same spirit as dmoi-brca-poc v0.13's cross-cohort calibration result. "
        "A production gate trained on a larger label set is where a fitted "
        "calibrator becomes appropriate.\n\n"
        "## Reproduce\n\n```bash\npython scripts/calibrate_gate.py\n```\n"
    )
    print(f"\nWrote {AUDIT / 'gate_calibration.md'}")

    if rows:
        audit.emit(
            action="gate_calibration", job_id=JOB_ID,
            fields={m: {"ece": ece, "brier": brier} for (m, ece, brier, _mc, _ac, _t) in rows},
        )
        try:
            if tracking.is_enabled():
                tracking.log_metrics({f"{m}_ece": ece for (m, ece, *_ ) in rows})
        except Exception as exc:  # noqa: BLE001 — tracking must never be fatal
            print(f"  (MLflow logging skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
