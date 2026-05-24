# Architecture — `multiqc-foundation-gate`

This repo composes two layers on top of the shared scaffold substrate:

1. **Data layer**: ENA SRR download -> FastQC -> MultiQC -> JSON ->
   28-dim feature vector (`features.py`).
2. **Classifier layer**: stratified 5-fold CV across one PyTorch MLP +
   two sklearn baselines (RandomForest + LogisticRegression), aggregated
   into a comparison table + drift summary + per-fold confusion matrices.

The outer pipeline (`pipeline.py`) brackets both layers with the standard
substrate `audit_start -> tracking -> body -> audit_end` pattern, giving
the substrate a single hash-chained NDJSON ledger that spans the entire
classifier comparison + drift detection run.

---

## Two-layer control flow

```
                         Operator
                            |
                            v
            scripts/run_lab.sh   (substrate endpoints)
                            |
                            v
                       make run
                            |
                            v
        +---------------------------------------+
        |  pipeline.run_pipeline(run_name)      |
        |  (src/multiqc_gate/pipeline.py)       |
        +---------------------------------------+
                            |
                            +--->  audit.emit('pipeline_start')
                            |
                            +--->  tracking.run().start_run()
                            |
                            v
        +---------------------------------------+
        |  Load features.npy + labels.csv       |
        |    -> audit.emit('dataset_loaded',     |
        |       n_samples, class_distribution)   |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  MLP 5-fold CV (PyTorch)              |
        |    per fold:                          |
        |      audit.emit('fold_start')         |
        |      per epoch:                       |
        |        audit.emit('epoch_end',         |
        |          train_loss, val_loss, val_acc)|
        |        tracking.log_metric             |
        |      audit.emit('fold_end',            |
        |        best_epoch, best_val_acc)       |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Sklearn baselines (same folds)       |
        |    RandomForest + LogisticRegression  |
        |    per (method, fold):                |
        |      audit.emit('baseline_fold_end')  |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Aggregation (eval.py)                |
        |    per method:                        |
        |      EvalSummary -> eval_<method>.json|
        |    comparison.json (side-by-side)     |
        |    audit.emit('comparison_table')     |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Drift detection (drift.py)           |
        |    include-baseline vs everything else|
        |    KS test per feature (alpha=0.05)   |
        |    drift.json + summarize_drift       |
        |    audit.emit('drift_summary')        |
        +---------------------------------------+
                            |
                            +--->  tracking.log_metrics(...) + log_artifact
                            |
                            +--->  audit.emit('pipeline_end', status, metrics)
                            |
                            v
                  audit/local-demo.ndjson
                  (~215 entries on n=50 demo, hash-chained)
```

---

## Audit chain composition (n=50 demo, 215 entries)

| Position | Action | Source | Count |
|---|---|---|---|
| 1 | `pipeline_start` | `pipeline.py` outer bracket | 1 |
| 2 | `dataset_loaded` | `pipeline.py` (post-load) | 1 |
| ~3-12 | `fold_start`, `fold_end` (MLP, 5 folds each) | `train.train_cv` | 10 |
| ~13-152 | `epoch_end` (MLP, 5 folds x ~28 epochs early-stopped avg) | `train.train_one_fold` | ~140 |
| ~153-162 | `baseline_fold_end` (RF + LR, 5 folds each) | `baseline.run_baseline` | 10 |
| ~163 | `comparison_table` | `pipeline.py` after aggregation | 1 |
| ~164 | `drift_summary` | `pipeline.py` after `detect_drift` | 1 |
| ~165 | `pipeline_end` | `pipeline.py` outer bracket | 1 |
| (rest) | canary + miscellaneous substrate entries | various | ~50 |

The ordering is wall-clock monotonic. Hash chain links every entry's
`prev_hash` field to the SHA-256 of the canonical encoding of the
preceding entry — `audit.verify()` walks the chain and confirms every
link.

---

## Feature vector schema (28 dims)

See `src/multiqc_gate/features.py` for the schema constants. Summary:

| Block | Dims | Source |
|---|---|---|
| Numerical aggregates (mean across R1+R2 samples) | 7 | `report_saved_raw_data.multiqc_fastqc.*` (Total Sequences, %GC, avg_sequence_length, total_deduplicated_percentage, Sequences flagged as poor quality, median_sequence_length, Sequence length) |
| Per-module status (worst-of pass/warn/fail across R1+R2) | 10 | FastQC module status fields (basic_statistics, per_base_sequence_quality, per_sequence_quality_scores, per_base_sequence_content, per_sequence_gc_content, per_base_n_content, sequence_length_distribution, sequence_duplication_levels, overrepresented_sequences, adapter_content). Encoded pass=0, warn=1, fail=2, missing=-1. |
| Aggregate status counts (across all sample x module pairs) | 3 | `n_modules_pass`, `n_modules_warn`, `n_modules_fail` |
| Plot-module presence flags | 8 | 1 if module in `report_plot_data`, 0 if dropped (mirror of `augment.FASTQC_MODULES_DROPPABLE`) |

This 28-dim vector is the *only* thing the classifier sees. All five
augmentation strategies (originals, module dropout, adapter injection,
quality degradation, mixed) project into this same 28-dim space, so the
classifier learns to distinguish their patterns via the vector geometry.

---

## Classifier comparison (why 3 methods, same folds)

The substrate value is the **comparison artifact**, not any single model:

| Method | When it wins | When it loses |
|---|---|---|
| LogisticRegression | small n, ~linear decision boundary, interpretable coefficients | non-linear interactions between features |
| RandomForest | small-to-medium n, non-linear, captures feature interactions | very small n (tree depth limited), no probability calibration by default |
| MLP (PyTorch) | n > ~hundreds-to-thousands, expressive non-linear function, MLflow + audit integration straightforward | n=50 with 3 classes -> over-parameterized regardless of LayerNorm and dropout |

On n=50 the v0.1 demo intentionally exposes the MLP weakness, because
a portrait that pretended otherwise would be a lie. A production version
at n=3,000+ flips this ranking; the substrate flips with it automatically
because `comparison.json` is regenerated each run.

---

## Substrate integration points

Same four-channel substrate as every other capability-portrait repo:

| Channel | Module | Env var | Substrate endpoint |
|---|---|---|---|
| Audit (immutable record) | `multiqc_gate.audit` | `AUDIT_HOST` | `http://${AUDIT_HOST}/events` |
| MLflow (experiment tracking) | `multiqc_gate.tracking` | `MLFLOW_TRACKING_URI` | configurable |
| Canary (daily probe) | `multiqc_gate.canary` | `HEALTHOMICS_LAB_CANARY_FIXTURE` | invoked by `lab_semantic_check.py` |
| Drift (per-feature KS) | `multiqc_gate.drift` | (none — invoked by `pipeline.py`) | results land in audit + `drift.json` |

All channels degrade to no-ops when the substrate is absent. The
deterministic local NDJSON ledger remains the source of truth for audit
even when remote POST fails.

---

## Why a tiny MLP and not a transformer

The original Capability-Showcase spec called for a 4-layer transformer
encoder (~1M parameters). On 50 samples with 3 classes (n_per_class:
10 / 20 / 20), 1M parameters is 100x over-parameterized. Even with
strong regularization (LayerNorm + dropout 0.3 + class-weighted CE),
the model would memorize the training set and the holdout would be
noise.

The v0.1 picks the simpler-but-honest path:

- **Input**: 28-dim feature vector (LayerNorm-normalised to handle
  heterogeneous scales: Total Sequences ~1e5, status codes ~[-1, 2],
  presence flags ~[0, 1]).
- **Hidden**: 28 -> 32 -> 16 with ReLU + dropout 0.3.
- **Output**: 3 logits.
- **Total trainable parameters**: ~1.5k.

This sits at the right capacity level for n=50 in the regime where
LogReg still wins, but where the substrate framing (per-epoch audit +
MLflow + comparison artifact) is *identical* to what a production
n=3,000 MLP would emit. The architecture decision is therefore
substrate-preserving even as the data scale changes.

---

## What this architecture intentionally avoids

- **No DAG engine.** No Nextflow / Airflow / Prefect / Dagster. The
  pipeline is a single Python process — `pipeline.py` calls each stage
  in sequence. P1 (`healthomics-lab-orchestrator`) is the DAG-engine
  capability portrait; P2 is the analytical-method portrait that runs
  inside one Nextflow process when deployed at production scale.
- **No GPU dependency.** PyTorch MPS is used opportunistically on
  Apple Silicon; CPU is fully sufficient for n=50.
- **No data validation framework beyond Pydantic-on-demand.** The
  manifest is plain YAML, the label sheet is plain CSV. Pydantic
  appears only where the substrate POSTs need structured payloads.
- **No model registry beyond MLflow artifacts.** Trained model state
  is logged via `tracking.log_artifact(...)`. Production deployment
  would add a versioned model registry; v0.1 does not.

The contract is small and the implementation is small; expansion happens
through additive features (multi-cohort data, larger MLP, transformer
when n permits) without re-architecting.
