# MultiQC gate — calibration diagnostic

Generated: 2026-06-02T20:47:53Z

## Why

An include/exclude gate's *confidence* matters: an over-confident wrong 'include' is a silent data-quality failure. This measures whether the gate's probabilities are calibrated (confidence-ECE, Guo et al.), on the pooled held-out predictions of the stratified 5-fold baselines.

- Samples: n=50 across 3 classes {'include': 10, 'exclude': 20, 'manual-review': 20}.
- Metric: confidence-ECE (max-probability vs accuracy) + multiclass Brier.

## Results (pooled 5-fold held-out)

| Method | ECE | Brier | mean conf | accuracy | tendency |
|---|---|---|---|---|---|
| logistic_regression | 0.1139 | 0.2204 | 0.880 | 0.860 | well-matched |
| random_forest | 0.0525 | 0.2206 | 0.842 | 0.840 | well-matched |

Reliability bins: `gate_calibration_reliability.tsv`.

## Honest scope

This is a calibration **diagnostic, not a fix**. With n=50 across 3 classes, post-hoc re-calibration (temperature / isotonic / Platt) cannot be reliably estimated — each reliability bin holds only a handful of points, so a fitted calibrator would overfit. The deliverable is the honest measurement of whether the gate is over- or under-confident, in the same spirit as dmoi-brca-poc v0.13's cross-cohort calibration result. A production gate trained on a larger label set is where a fitted calibrator becomes appropriate.

## Reproduce

```bash
python scripts/calibrate_gate.py
```
