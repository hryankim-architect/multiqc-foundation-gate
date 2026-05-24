"""Per-feature distribution drift detection via Kolmogorov-Smirnov tests.

The drift module compares two feature matrices (reference vs. test) and
flags any feature whose distribution differs at the alpha=0.05 level.
In the v0.1 demo, the "reference" is the augmented training set's
include-class samples (n=10 clean baselines) and the "test" is the
exclude / manual-review augmentation samples (n=40). A working drift
detector should flag many of the 28 features as drifted because
augmentation is designed to shift their distributions.

Drift output feeds directly into the substrate's audit ledger so a
production consumer (downstream of P2) can sound an alarm when
incoming MultiQC reports drift away from the training distribution
without re-running the full eval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

from multiqc_gate.features import FEATURE_NAMES


@dataclass
class DriftResult:
    """Per-feature drift flag + KS statistic + p-value."""

    feature_name: str
    feature_index: int
    ks_statistic: float
    p_value: float
    drifted: bool  # True if p_value < alpha


def detect_drift(
    reference: np.ndarray,
    test: np.ndarray,
    alpha: float = 0.05,
) -> list[DriftResult]:
    """Run a Kolmogorov-Smirnov 2-sample test per feature.

    Args:
        reference: shape (n_ref, n_features). The "expected" distribution.
        test: shape (n_test, n_features). The candidate batch.
        alpha: significance threshold (default 0.05).

    Returns:
        One DriftResult per feature; `drifted=True` means p < alpha.
    """
    if reference.ndim != 2 or test.ndim != 2:
        raise ValueError(
            f"both reference and test must be 2D arrays; got "
            f"{reference.ndim}D / {test.ndim}D"
        )
    if reference.shape[1] != test.shape[1]:
        raise ValueError(
            f"feature-dim mismatch: reference={reference.shape[1]}, "
            f"test={test.shape[1]}"
        )

    results: list[DriftResult] = []
    n_features = reference.shape[1]
    for j in range(n_features):
        ref_col = reference[:, j]
        test_col = test[:, j]
        # Constant-column edge case: KS undefined when both columns are
        # the same single value. Treat as "no drift" (p=1.0).
        if np.allclose(ref_col, ref_col[0]) and np.allclose(test_col, test_col[0]):
            if ref_col[0] == test_col[0]:
                stat, p = 0.0, 1.0
            else:
                stat, p = 1.0, 0.0
        else:
            res = ks_2samp(ref_col, test_col)
            stat = float(res.statistic)
            p = float(res.pvalue)

        results.append(DriftResult(
            feature_name=FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feat_{j}",
            feature_index=j,
            ks_statistic=stat,
            p_value=p,
            drifted=p < alpha,
        ))
    return results


def summarize_drift(results: list[DriftResult], alpha: float = 0.05) -> dict:
    """Aggregate drift results into a small dict for logging."""
    n_drifted = sum(1 for r in results if r.drifted)
    top_drifted = sorted(results, key=lambda r: r.ks_statistic, reverse=True)[:5]
    return {
        "n_features": len(results),
        "n_drifted": n_drifted,
        "fraction_drifted": n_drifted / max(len(results), 1),
        "alpha": alpha,
        "top_5_drifted": [
            {
                "feature": r.feature_name,
                "ks_statistic": round(r.ks_statistic, 3),
                "p_value": round(r.p_value, 4),
            }
            for r in top_drifted
        ],
    }
