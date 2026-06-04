# Architecture, `multiqc-foundation-gate`

This repo layers two concerns on top of the shared scaffold substrate:

1. **Data layer**: ENA SRR download -> FastQC -> MultiQC -> JSON ->
   28-dim feature vector (`features.py`).
2. **Classifier layer**: stratified 5-fold CV across one PyTorch MLP plus
   two sklearn baselines (RandomForest + LogisticRegression), folded into
   a comparison table, drift summary, and per-fold confusion matrices.

`pipeline.py` wraps both layers in the substrate's
`audit_start -> tracking -> body -> audit_end` bracket, so a single
hash-chained NDJSON ledger covers the entire classifier comparison and
drift detection run.

---

## Left-to-right control flow

```
 Operator
    |
    v
 scripts/run_lab.sh
    |
    v
 make run
    |
    v
 pipeline.run_pipeline(run_name)          [src/multiqc_gate/pipeline.py]
    |
    +-- audit.emit('pipeline_start') -----+
    |                                     |
    +-- tracking.start_run() ------------>|
    |                                     |
    v                                     |
 Load features.npy + labels.csv           |
    |                                     |
    +-- audit.emit('dataset_loaded',      |
    |     n_samples, class_distribution)  |
    |                                     |
    v                                     |
 MLP 5-fold CV  [train.py / PyTorch]      |
    |  per fold:                          |
    |    audit.emit('fold_start')         |
    |    per epoch:                       |
    |      audit.emit('epoch_end',        |
    |        train_loss, val_loss,        |
    |        val_acc)                     |
    |      tracking.log_metric(...)       |
    |    audit.emit('fold_end',           |
    |      best_epoch, best_val_acc)      |
    |                                     |
    v                                     |
 Sklearn baselines  [baseline.py]         |
    |  RandomForest + LogisticRegression  |
    |  same 5 folds as MLP               |
    |  per (method, fold):               |
    |    audit.emit('baseline_fold_end') |
    |                                     |
    v                                     |
 Aggregation  [eval.py]                   |
    |  EvalSummary -> eval_<method>.json  |
    |  comparison.json (side-by-side)     |
    +-- audit.emit('comparison_table')    |
    |                                     |
    v                                     |
 Drift detection  [drift.py]              |
    |  include-class baseline vs rest     |
    |  KS test per feature (alpha=0.05)   |
    |  drift.json + summarize_drift       |
    +-- audit.emit('drift_summary')  ---->|
    |                                     |
    +-- tracking.log_metrics(...)         |
    +-- tracking.log_artifact(...)        |
    +-- audit.emit('pipeline_end',        |
    |     status, metrics) ---------------+
    |
    v
 audit/local-demo.ndjson
   (~215 entries on n=50 demo, hash-chained)
```

---

## Audit chain composition (n=50 demo, ~215 entries)

| Position | Event | Source | Count |
|---|---|---|---|
| 1 | `pipeline_start` | `pipeline.py` outer bracket | 1 |
| 2 | `dataset_loaded` | `pipeline.py` after load | 1 |
| ~3-12 | `fold_start`, `fold_end` (MLP, 5 folds each) | `train.train_cv` | 10 |
| ~13-152 | `epoch_end` (MLP, 5 folds x ~28 epochs early-stopped avg) | `train.train_one_fold` | ~140 |
| ~153-162 | `baseline_fold_end` (RF + LR, 5 folds each) | `baseline.run_baseline` | 10 |
| ~163 | `comparison_table` | `pipeline.py` after aggregation | 1 |
| ~164 | `drift_summary` | `pipeline.py` after `detect_drift` | 1 |
| ~165 | `pipeline_end` | `pipeline.py` outer bracket | 1 |
| (rest) | canary + miscellaneous substrate entries | various | ~50 |

Entries are wall-clock monotonic. Each entry's `prev_hash` field holds the
SHA-256 of the canonical JSON encoding of the preceding entry. `audit.verify()`
walks the chain and confirms every link is intact.

---

## Feature vector schema (28 dims)

See `src/multiqc_gate/features.py` for the schema constants. Summary:

| Block | Dims | Source |
|---|---|---|
| Numerical aggregates (mean across R1+R2 samples) | 7 | `report_saved_raw_data.multiqc_fastqc.*` (Total Sequences, %GC, avg_sequence_length, total_deduplicated_percentage, Sequences flagged as poor quality, median_sequence_length, Sequence length) |
| Per-module status (worst-of pass/warn/fail across R1+R2) | 10 | FastQC module status fields (basic_statistics, per_base_sequence_quality, per_sequence_quality_scores, per_base_sequence_content, per_sequence_gc_content, per_base_n_content, sequence_length_distribution, sequence_duplication_levels, overrepresented_sequences, adapter_content). Encoded pass=0, warn=1, fail=2, missing=-1. |
| Aggregate status counts (across all sample x module pairs) | 3 | `n_modules_pass`, `n_modules_warn`, `n_modules_fail` |
| Plot-module presence flags | 8 | 1 if module in `report_plot_data`, 0 if dropped (mirror of `augment.FASTQC_MODULES_DROPPABLE`) |

This 28-dim vector is the only input the classifier sees. All five
augmentation strategies (originals, module dropout, adapter injection,
quality degradation, mixed) project into the same 28-dim space, so the
classifier learns to distinguish their patterns through vector geometry
rather than raw read data.

---

## Classifier comparison (why 3 methods, same folds)

The deliverable is the **comparison artifact**, not any single model:

| Method | When it wins | When it loses |
|---|---|---|
| LogisticRegression | small n, roughly linear decision boundary, interpretable coefficients | non-linear feature interactions |
| RandomForest | small-to-medium n, non-linear, captures feature interactions | very small n (tree depth limited), no probability calibration by default |
| MLP (PyTorch) | n > ~hundreds-to-thousands, expressive non-linear function, MLflow + audit integration straightforward | n=50 with 3 classes -> over-parameterized regardless of LayerNorm and dropout |

The v0.1 demo intentionally exposes the MLP weakness at n=50. A production
run at n=3,000+ flips this ranking; `comparison.json` is regenerated each
run, so the substrate reflects the new winner automatically.

---

## Substrate integration points

| Channel | Module | Env var | Endpoint |
|---|---|---|---|
| Audit (immutable record) | `multiqc_gate.audit` | `AUDIT_HOST` | `http://${AUDIT_HOST}/events` |
| MLflow (experiment tracking) | `multiqc_gate.tracking` | `MLFLOW_TRACKING_URI` | configurable |
| Canary (daily probe) | `multiqc_gate.canary` | `BIOSCAFFOLD_CANARY_FIXTURE` | invoked by `lab_semantic_check.py` |
| Drift (per-feature KS) | `multiqc_gate.drift` | (none, called by `pipeline.py`) | results land in audit + `drift.json` |

The canary is the entry point that `lab_semantic_check.py` probes on its
daily schedule. All four channels degrade to no-ops when the substrate is
absent. The local NDJSON ledger is written regardless of whether the remote
POST succeeds, so audit history is never lost to a network failure.

---

## Why a tiny MLP and not a transformer

The original spec called for a 4-layer transformer
encoder (~1M parameters). On 50 samples with 3 classes (n_per_class:
10 / 20 / 20), 1M parameters is roughly 100x over-parameterized. Even with
strong regularization (LayerNorm + dropout 0.3 + class-weighted CE),
the model would memorize training samples and produce noise on the holdout.

The v0.1 picks the simpler path:

- **Input**: 28-dim feature vector (LayerNorm-normalised to handle
  heterogeneous scales: Total Sequences ~1e5, status codes ~[-1, 2],
  presence flags ~[0, 1]).
- **Hidden**: 28 -> 32 -> 16 with ReLU + dropout 0.3.
- **Output**: 3 logits.
- **Total trainable parameters**: ~1.5k.

This sits at the right capacity level for n=50 in the regime where
LogReg still wins, but the per-epoch audit and MLflow emission is identical
to what a production n=3,000 MLP would produce. The architecture decision
is therefore substrate-preserving as data scale grows.

---

## What this architecture intentionally avoids

- **No DAG engine.** No Nextflow / Airflow / Prefect / Dagster. The
  pipeline is a single Python process; `pipeline.py` calls each stage
  in sequence. The orchestration project (`healthomics-lab-orchestrator`)
  covers DAG-engine concerns; this repo runs inside one Nextflow process
  when deployed at production scale.
- **No GPU dependency.** PyTorch MPS is used opportunistically on
  Apple Silicon; CPU is fully sufficient for n=50.
- **No data validation framework beyond Pydantic-on-demand.** The
  manifest is plain YAML, the label sheet is plain CSV. Pydantic
  appears only where substrate POSTs need structured payloads.
- **No model registry beyond MLflow artifacts.** Trained model state
  is logged via `tracking.log_artifact(...)`. A versioned registry
  belongs to a production deployment, not v0.1.

The contract is small and the implementation is small. Expansion happens
through additive features (multi-cohort data, larger MLP, transformer
when n permits) without re-architecting the pipeline.
