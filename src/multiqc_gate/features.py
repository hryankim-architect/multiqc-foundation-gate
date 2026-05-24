"""Feature extraction: MultiQC JSON -> numerical vector for the classifier.

Reads `multiqc_data/multiqc_data.json` from a per-sample report directory
and emits a fixed-shape 28-dimensional `np.float32` feature vector that
the MLP (and the sklearn baseline) can consume.

Feature schema (28 dimensions total):

    (1) Numerical aggregates (mean across R1+R2 samples in the report): 7
        - Total Sequences            (e.g. 100,000)
        - %GC                        (e.g. 48.0)
        - avg_sequence_length        (e.g. 63.0)
        - total_deduplicated_percentage  (e.g. 95.4)
        - Sequences flagged as poor quality
        - median_sequence_length
        - Sequence length

    (2) Per-module status (worst-of pass/warn/fail across R1+R2): 10
        Each module encoded as: pass=0, warn=1, fail=2, missing=-1
        Modules:
        - basic_statistics, per_base_sequence_quality,
          per_sequence_quality_scores, per_base_sequence_content,
          per_sequence_gc_content, per_base_n_content,
          sequence_length_distribution, sequence_duplication_levels,
          overrepresented_sequences, adapter_content

    (3) Aggregate counts across all samples x all modules: 3
        - n_modules_pass, n_modules_warn, n_modules_fail

    (4) Plot-module presence flags (1 if present in report_plot_data): 8
        Mirrors `augment.FASTQC_MODULES_DROPPABLE` so the classifier can
        learn to detect upstream-tool-failure (module dropout) patterns
        as the manual-review signal.

The 28-dim vector is small enough for an MLP with ~2-3k parameters on
n=50 samples (the canonical capability-portrait n) without immediately
overfitting. The sklearn RandomForest baseline reads the same features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

NUMERICAL_FIELDS = (
    "Total Sequences",
    "%GC",
    "avg_sequence_length",
    "total_deduplicated_percentage",
    "Sequences flagged as poor quality",
    "median_sequence_length",
    "Sequence length",
)

FASTQC_STATUS_MODULES = (
    "basic_statistics",
    "per_base_sequence_quality",
    "per_sequence_quality_scores",
    "per_base_sequence_content",
    "per_sequence_gc_content",
    "per_base_n_content",
    "sequence_length_distribution",
    "sequence_duplication_levels",
    "overrepresented_sequences",
    "adapter_content",
)

STATUS_TO_INT = {"pass": 0, "warn": 1, "fail": 2}
STATUS_MISSING = -1  # Encoded value when the module was dropped/missing.

FASTQC_PLOT_MODULES = (
    "fastqc_sequence_counts_plot",
    "fastqc_per_base_sequence_quality_plot",
    "fastqc_per_sequence_quality_scores_plot",
    "fastqc_per_sequence_gc_content_plot",
    "fastqc_per_base_n_content_plot",
    "fastqc_sequence_duplication_levels_plot",
    "fastqc_adapter_content_plot",
    "fastqc-status-check-heatmap",
)

FEATURE_NAMES = (
    *[f"num__{f.replace(' ', '_').replace('%', 'pct')}" for f in NUMERICAL_FIELDS],
    *[f"status__{m}" for m in FASTQC_STATUS_MODULES],
    "agg__n_modules_pass",
    "agg__n_modules_warn",
    "agg__n_modules_fail",
    *[f"present__{m}" for m in FASTQC_PLOT_MODULES],
)
N_FEATURES = len(FEATURE_NAMES)  # 28


# ---------------------------------------------------------------------------
# Single-report extraction
# ---------------------------------------------------------------------------


def extract_features(json_path: Path) -> np.ndarray:
    """Read one ``multiqc_data.json`` and return the 28-dim feature vector."""
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    fastqc = data.get("report_saved_raw_data", {}).get("multiqc_fastqc", {})
    plot_data = data.get("report_plot_data", {})

    samples = list(fastqc.values())  # 1 (single-end) or 2 (paired-end)
    if not samples:
        return _empty_features()

    features: list[float] = []

    # (1) Numerical aggregates — mean across R1+R2 samples.
    for field in NUMERICAL_FIELDS:
        nums: list[float] = []
        for s in samples:
            raw_val = s.get(field, 0)
            try:
                nums.append(float(raw_val))
            except (TypeError, ValueError):
                # Strings like "6.3 Mbp" coerce to 0 — caller doesn't lose
                # the numeric signal because we have Total Sequences too.
                nums.append(0.0)
        features.append(float(np.mean(nums)) if nums else 0.0)

    # (2) Per-module status — take the WORST across R1+R2 (fail > warn > pass).
    #     Worst-of is more conservative than mean; one bad sample in a paired
    #     pair should pull the report toward "manual-review" or "exclude".
    for module in FASTQC_STATUS_MODULES:
        worst = "pass"
        seen_any = False
        for s in samples:
            st = s.get(module)
            if st is None:
                continue
            seen_any = True
            if st == "fail":
                worst = "fail"
                break
            if st == "warn" and worst != "fail":
                worst = "warn"
        if not seen_any:
            features.append(float(STATUS_MISSING))
        else:
            features.append(float(STATUS_TO_INT[worst]))

    # (3) Aggregate counts across all (sample, module) pairs.
    pass_n = warn_n = fail_n = 0
    for s in samples:
        for module in FASTQC_STATUS_MODULES:
            st = s.get(module)
            if st == "pass":
                pass_n += 1
            elif st == "warn":
                warn_n += 1
            elif st == "fail":
                fail_n += 1
    features.extend([float(pass_n), float(warn_n), float(fail_n)])

    # (4) Plot-module presence — 1 if MultiQC produced the plot, 0 if dropped.
    for module in FASTQC_PLOT_MODULES:
        features.append(1.0 if module in plot_data else 0.0)

    arr = np.array(features, dtype=np.float32)
    if arr.shape[0] != N_FEATURES:
        raise RuntimeError(
            f"feature shape mismatch: got {arr.shape[0]}, expected {N_FEATURES}"
        )
    return arr


def _empty_features() -> np.ndarray:
    """Zero vector with all-missing module statuses (for entirely empty reports)."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    # Status block (positions 7..16) should be STATUS_MISSING, not 0.
    status_start = len(NUMERICAL_FIELDS)
    status_end = status_start + len(FASTQC_STATUS_MODULES)
    vec[status_start:status_end] = float(STATUS_MISSING)
    return vec


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_features_batch(
    reports_dir: Path,
    sample_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract features for many reports at once.

    Args:
        reports_dir: Parent directory containing one subdirectory per sample
            (e.g. ``data/multiqc_reports/``).
        sample_ids: List of subdirectory names (e.g.
            ``["SRR1039513", "SRR1039513_dropout_1mod", ...]``).

    Returns:
        (X, used_ids): X has shape ``(n_valid, 28)`` and ``used_ids`` is the
        subset of ``sample_ids`` that were successfully extracted (missing
        report directories are silently skipped — the caller can compare
        lengths to detect drops).
    """
    features: list[np.ndarray] = []
    used: list[str] = []
    for sid in sample_ids:
        json_path = reports_dir / sid / "multiqc_data" / "multiqc_data.json"
        if not json_path.exists():
            continue
        features.append(extract_features(json_path))
        used.append(sid)
    if not features:
        return np.zeros((0, N_FEATURES), dtype=np.float32), []
    return np.stack(features), used
