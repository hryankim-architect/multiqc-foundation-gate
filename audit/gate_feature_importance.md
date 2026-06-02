# MultiQC gate — feature-importance (interpretability) diagnostic

Generated: 2026-06-02T20:53:45Z

## Why

Which QC signals drive the gate's include / exclude / manual-review decision? A legible gate that keys on sensible MultiQC metrics is far easier to trust (oversight) than an opaque accuracy number. This is the interpretability companion to `audit/gate_calibration.md`.

- Samples: n=50 {'include': 10, 'exclude': 20, 'manual-review': 20}.
- Method: model-agnostic permutation importance (sklearn), scored by macro-F1, n_repeats=20, averaged over the held-out folds of the stratified 5-fold split. Features named per `features.FEATURE_NAMES` (28).

## Top features by permuted macro-F1 drop

### random_forest (mean over 5 folds)

| Rank | Feature | Δ macro-F1 when permuted |
|---|---|---|
| 1 | `present__fastqc_per_sequence_quality_scores_plot` | +0.0721 |
| 2 | `num__median_sequence_length` | +0.0447 |
| 3 | `num__avg_sequence_length` | +0.0156 |
| 4 | `num__Sequence_length` | +0.0156 |
| 5 | `agg__n_modules_warn` | +0.0054 |
| 6 | `present__fastqc-status-check-heatmap` | +0.0000 |
| 7 | `status__per_base_n_content` | +0.0000 |
| 8 | `num__Sequences_flagged_as_poor_quality` | +0.0000 |
| 9 | `status__basic_statistics` | +0.0000 |
| 10 | `status__per_base_sequence_quality` | +0.0000 |

### logistic_regression (mean over 5 folds)

| Rank | Feature | Δ macro-F1 when permuted |
|---|---|---|
| 1 | `agg__n_modules_pass` | +0.3663 |
| 2 | `present__fastqc_per_sequence_quality_scores_plot` | +0.0870 |
| 3 | `num__avg_sequence_length` | +0.0810 |
| 4 | `num__median_sequence_length` | +0.0810 |
| 5 | `num__Sequence_length` | +0.0810 |
| 6 | `status__overrepresented_sequences` | +0.0126 |
| 7 | `present__fastqc_adapter_content_plot` | +0.0026 |
| 8 | `present__fastqc-status-check-heatmap` | +0.0000 |
| 9 | `status__per_base_n_content` | +0.0000 |
| 10 | `num__Sequences_flagged_as_poor_quality` | +0.0000 |

Full ranking: `gate_feature_importance_reliability.tsv`.

## Honest scope

At n=50, permutation-importance magnitudes are **indicative, not precise**; near-zero features are not proven irrelevant, and fold-to-fold ranking will jitter. We average over repeats and folds to damp variance and report the ranking as a legibility diagnostic, not a definitive feature-selection claim — the same honest posture as the calibration diagnostic and the repo's sklearn-beats-MLP baseline.

## Reproduce

```bash
python scripts/interpret_gate.py
```
