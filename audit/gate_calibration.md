# MultiQC gate — calibration diagnostic

Generated: 2026-06-10T18:40:14Z (regenerated after the LogReg StandardScaler fix — see README "Engineering lessons")

## Why

An include/exclude gate's *confidence* matters: an over-confident wrong 'include' is a silent data-quality failure. This measures whether the gate's probabilities are calibrated (confidence-ECE, Guo et al.), on the pooled held-out predictions of the stratified 5-fold baselines.

- Samples: n=50 across 3 classes {'include': 10, 'exclude': 20, 'manual-review': 20}.
- Metric: confidence-ECE (max-probability vs accuracy) + multiclass Brier.

## Results (pooled 5-fold held-out)

| Method | ECE | Brier | mean conf | accuracy | tendency |
|---|---|---|---|---|---|
| logistic_regression | 0.1475 | 0.2974 | 0.830 | 0.800 | over-confident |
| random_forest | 0.0521 | 0.2198 | 0.843 | 0.840 | well-matched |

Reliability bins: `gate_calibration_reliability.tsv`.

## Limitations

This is a calibration **diagnostic, not a fix**. With n=50 across 3 classes, post-hoc re-calibration (temperature / isotonic / Platt) cannot be reliably estimated — each reliability bin holds only a handful of points, so a fitted calibrator would overfit. The output is a measurement of whether the gate is over- or under-confident. A production gate trained on a larger label set is where a fitted calibrator becomes appropriate.

## Reproduce

```bash
python scripts/calibrate_gate.py
```
