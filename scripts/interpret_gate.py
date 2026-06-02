#!/usr/bin/env python3
"""Interpretability diagnostic for the MultiQC include/exclude gate.

Answers the oversight question a reviewer actually cares about: *which MultiQC
QC signals drive the gate's include / exclude / manual-review decision?* A gate
whose decisions are legible (and trace to sensible QC metrics) is far easier to
trust than an opaque accuracy number — the interpretability analogue of the
calibration diagnostic in `scripts/calibrate_gate.py`.

Method: model-agnostic **permutation importance** (sklearn) computed on the
held-out fold of the same stratified 5-fold split used everywhere else, averaged
across folds, scored by macro-F1 (the cohort is class-imbalanced:
include=10 / exclude=20 / manual-review=20). Importances are mapped to the 28
named features in `multiqc_gate.features.FEATURE_NAMES`.

Honest scope: at n=50 permutation importance is noisy — magnitudes are
indicative, not precise, and near-zero features are not "proven irrelevant". We
report the ranking with that caveat and average over repeats + folds to damp the
variance, in the same honest-evaluation spirit as the rest of the repo.

Reproduce:  python scripts/interpret_gate.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from multiqc_gate import audit, tracking  # noqa: E402
from multiqc_gate.baseline import _build_classifier  # noqa: E402
from multiqc_gate.features import FEATURE_NAMES  # noqa: E402
from multiqc_gate.labels import LABELS, load_labels  # noqa: E402
from multiqc_gate.train import labels_to_indices  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA = REPO / "data"
AUDIT = REPO / "audit"
JOB_ID = "multiqc-gate-interpretability"
METHODS = ("random_forest", "logistic_regression")
N_REPEATS = 20
TOP_K = 10


def _load_xy():
    fp, sp, lp = DATA / "features.npy", DATA / "feature_sample_ids.txt", DATA / "labels.csv"
    for p in (fp, sp, lp):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing {p}. See README for data setup.\n")
            raise SystemExit(1)
    x = np.load(fp)
    sample_ids = sp.read_text().splitlines()
    labels_df = load_labels(lp)
    label_map = dict(zip(labels_df["report_id"], labels_df["label"], strict=True))
    y = labels_to_indices([label_map[sid] for sid in sample_ids])
    return x, np.asarray(y)


def _fold_permutation_importance(x, y, method):
    """Mean permutation importance over held-out folds (macro-F1 scoring)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = np.zeros(x.shape[1], dtype=float)
    n_folds = 0
    for fold, (tr, va) in enumerate(skf.split(x, y)):
        clf = _build_classifier(method, seed=42 + fold)
        clf.fit(x[tr], y[tr])
        # Need >=2 classes present in the val fold for f1_macro to be meaningful.
        if len(np.unique(y[va])) < 2:
            continue
        r = permutation_importance(
            clf, x[va], y[va], scoring="f1_macro",
            n_repeats=N_REPEATS, random_state=42 + fold,
        )
        acc += r.importances_mean
        n_folds += 1
    return acc / max(n_folds, 1), n_folds


def main() -> int:
    x, y = _load_xy()
    dist = {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))}
    print(f"=== MultiQC gate interpretability (n={len(y)}, {dist}) ===")
    if x.shape[1] != len(FEATURE_NAMES):
        print(f"  NOTE: X has {x.shape[1]} cols but {len(FEATURE_NAMES)} named "
              "features; ranking will fall back to indices where unnamed.")

    AUDIT.mkdir(exist_ok=True)
    tsv_lines = ["method\trank\tfeature\timportance"]
    md_blocks = []
    emit_fields = {}
    for method in METHODS:
        imp, n_folds = _fold_permutation_importance(x, y, method)
        order = np.argsort(imp)[::-1]
        print(f"\n  {method} (mean over {n_folds} folds, top {TOP_K}):")
        rows = []
        for rank, j in enumerate(order[:TOP_K], 1):
            name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feat_{j}"
            print(f"    {rank:2d}. {name:28s} {imp[j]:+.4f}")
            tsv_lines.append(f"{method}\t{rank}\t{name}\t{imp[j]:.5f}")
            rows.append(f"| {rank} | `{name}` | {imp[j]:+.4f} |")
        md_blocks.append(f"### {method} (mean over {n_folds} folds)\n\n"
                         "| Rank | Feature | Δ macro-F1 when permuted |\n|---|---|---|\n"
                         + "\n".join(rows))
        top_name = FEATURE_NAMES[order[0]] if order[0] < len(FEATURE_NAMES) else f"feat_{order[0]}"
        emit_fields[method] = {"top_feature": top_name, "top_importance": float(imp[order[0]])}

    (AUDIT / "gate_feature_importance_reliability.tsv").write_text("\n".join(tsv_lines) + "\n")
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    (AUDIT / "gate_feature_importance.md").write_text(
        "# MultiQC gate — feature-importance (interpretability) diagnostic\n\n"
        f"Generated: {ts}\n\n"
        "## Why\n\n"
        "Which QC signals drive the gate's include / exclude / manual-review "
        "decision? A legible gate that keys on sensible MultiQC metrics is far "
        "easier to trust (oversight) than an opaque accuracy number. This is the "
        "interpretability companion to `audit/gate_calibration.md`.\n\n"
        f"- Samples: n={len(y)} {dist}.\n"
        "- Method: model-agnostic permutation importance (sklearn), scored by "
        f"macro-F1, n_repeats={N_REPEATS}, averaged over the held-out folds of the "
        "stratified 5-fold split. Features named per `features.FEATURE_NAMES` (28).\n\n"
        "## Top features by permuted macro-F1 drop\n\n"
        + "\n\n".join(md_blocks) + "\n\n"
        "Full ranking: `gate_feature_importance_reliability.tsv`.\n\n"
        "## Honest scope\n\n"
        "At n=50, permutation-importance magnitudes are **indicative, not "
        "precise**; near-zero features are not proven irrelevant, and fold-to-fold "
        "ranking will jitter. We average over repeats and folds to damp variance "
        "and report the ranking as a legibility diagnostic, not a definitive "
        "feature-selection claim — the same honest posture as the calibration "
        "diagnostic and the repo's sklearn-beats-MLP baseline.\n\n"
        "## Reproduce\n\n```bash\npython scripts/interpret_gate.py\n```\n"
    )
    print(f"\nWrote {AUDIT / 'gate_feature_importance.md'}")

    audit.emit(action="gate_interpretability", job_id=JOB_ID, fields=emit_fields)
    try:
        if tracking.is_enabled():
            tracking.log_params({m: f["top_feature"] for m, f in emit_fields.items()})
    except Exception as exc:  # noqa: BLE001 — tracking must never be fatal
        print(f"  (MLflow logging skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
