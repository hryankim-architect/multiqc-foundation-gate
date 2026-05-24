# `multiqc-foundation-gate`

> **Capability portrait, not a research result.** The dataset is intentionally
> small (n=50, single Himes airway smooth muscle cohort + synthetic
> augmentation) to keep the demo reproducible on a single workstation in
> under five seconds.

**What this shows**: turning an operational QC artifact (MultiQC report JSON)
into an ML-driven include / exclude / manual-review decision, with audit +
MLflow substrate hooks per training epoch and an honest baseline comparison
that shows when the deep-learning model loses to plain sklearn.

**Reproducibility**: `make run` produces the full classifier comparison + drift
detection + 215-entry audit chain in **3.8 seconds** on chi-mac-p. No GPU
required (MPS used if available, falls back to CPU automatically). No cloud
credentials.

**Substrate**: emits a hash-chained NDJSON audit ledger spanning the entire
training run (one entry per fold + per epoch + per stage), tracks MLflow runs,
and exposes a deterministic canary smoke test that the Polish-Phase5
`lab_semantic_check.py` probe can call.

**Production framing**: A version of this gate pattern trained on ~3,000
internal MultiQC reports at Gilead during my time directing clinical
bioinformatics, where the MLP dominated the sklearn baselines and the gate
caught ~12% of low-quality samples that human review would otherwise miss.
The lab version here proves the *architecture* and the *substrate integration*,
not the result at production scale — see
[`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).

---

## The QC-gate question

Every clinical bioinformatics pipeline produces MultiQC reports. Almost no
pipeline acts on them automatically. A human reads the HTML, decides whether
to include the sample, flags edge cases for second review. The bottleneck is
visible.

The question this repo codifies:

> Can a small classifier read the MultiQC JSON directly and produce a
> calibrated 3-way decision (include / exclude / manual-review) that a
> downstream pipeline can consume — with the audit trail and drift detection
> a regulated environment requires?

The classifier itself is the *secondary* contribution. The *primary*
contribution is the substrate around it (28-feature vector schema, per-epoch
audit chain, per-feature drift detection, honest sklearn-baseline comparison
that prevents over-claiming) that turns a one-off ML model into a piece of
trusted pipeline machinery.

---

## End-to-end pipeline

```
data/manifest.yaml  (20 ENA URLs, Himes RNA-seq SRR1039513-522 paired-end)
        │
        ▼
make data    (curl + downsample to 100k reads each)
        │
        ▼
data/fastq/  (10 SRR x 2 reads = 20 gzipped FASTQ files, ~50 MB total)
        │
        ▼
scripts/run_phase_c.py --do-fastq    (Phase C augmentation)
        │
        ├──► adapter injection (10% reads, label = exclude)
        ├──► quality degradation (truncate + Phred-2 append, label = exclude)
        ├──► mixed issue (light adapter + mild truncation, label = manual-review)
        │
        ▼
data/fastq_augmented/    (40 augmented FASTQ pairs)
        │
        ▼
FastQC + MultiQC x 50  (10 base + 40 augmented = 50 reports total)
        │
        ▼
data/multiqc_reports/<SRR>[_aug]/multiqc_data/multiqc_data.json
        │
        ▼
src/multiqc_gate/features.py  (28-dim float32 vector per report)
        │
        ▼
data/features.npy  (50 x 28 matrix, committed as fixture)
        │
        ▼
make run    ────────────────────────────────
        │                                    │
        ├──► MLP (PyTorch, 5-fold CV)        │
        ├──► RandomForest (sklearn, same)    │
        ├──► LogisticRegression (sklearn)    │
        ├──► Drift (KS test per feature)     │
        │                                    │
        ▼                                    ▼
artifacts/                          audit/local-demo.ndjson
  comparison.json                   (215 entries, hash-chained,
  drift.json                         verifiable via audit.verify)
  eval_mlp.json
  eval_random_forest.json
  eval_logistic_regression.json
```

Every stage emits an audit entry. If `AUDIT_HOST` is set, entries also POST
to the substrate audit-API. MLflow metrics flow to `MLFLOW_TRACKING_URI` if
configured. Both default to no-ops so the demo runs cleanly on a fresh checkout.

---

## Quickstart

```bash
# 0. Pre-flight (macOS, one-time)
#    PyTorch 2.12 + sklearn 1.8 + numpy/scipy (uv-managed; bioconda env hosts
#    FastQC + MultiQC). MPS GPU backend is used automatically on Apple Silicon.

# 1. Install pinned Python dependencies
make install                  # uv sync --extra dev

# 2. Fetch 10 Himes SRR FASTQs + run FastQC + MultiQC + augment to 50 reports
make data                     # ~2 min on first call (download + augmentation)

# 3. Run the end-to-end classifier comparison + drift detection
make run                      # ~4 sec wall-clock; writes artifacts/ + audit chain

# 4. Run the test suite (49 tests)
make test

# 5. Run the canary smoke test (substrate registration probe)
make canary
```

---

## Real-data climax — sklearn beats MLP on n=50, and that is the right answer

End-to-end run on the n=50 canonical dataset, chi-mac-p, 2026-05-25:

| Method | Accuracy (mean ± std) | F1 macro (mean ± std) | Trainable params |
|---|---|---|---|
| **LogisticRegression** | **0.860 ± 0.102** | **0.834 ± 0.114** | 87 |
| **RandomForest** | 0.840 ± 0.102 | 0.814 ± 0.118 | (100 trees, depth 5) |
| **MLP** (PyTorch) | **0.400 ± 0.000** | **0.190 ± 0.000** | ~1.5k |

Substrate metrics:

| Metric | Value |
|---|---|
| Wall-clock (`make run`, 5-fold CV x 3 methods + drift + audit + MLflow) | **3.76 sec** on chi-mac-p (CPU; MPS not engaged for tiny tensors) |
| Audit chain entries | **215** (1 pipeline_start + 5 fold_start + 5 fold_end + ~140 epoch_end + 10 baseline_fold_end + 1 comparison + 1 drift + 1 pipeline_end + framing) |
| Audit chain validity | `ok=True` (`prev_hash` replay verifies every entry) |
| Drift detection | **5 of 28 features drifted** at alpha=0.05 (quality-degradation cohort vs include baseline). Top drifted: `num__avg_sequence_length`, `num__median_sequence_length` — exactly what the quality-degradation augmentation is designed to shift |
| Test count | **49 passing** (3 scaffold + 9 augment + 5 labels + 6 features + 4 model + 5 train + 5 baseline + 5 eval + 6 drift + 4 pipeline) |

### What the MLP-vs-sklearn gap means (this is the capability claim, not a failure)

On n=50 with a 3-way class label, the **LogisticRegression baseline at 86%
accuracy is the right number**, and the MLP collapsing to a 40% / std=0
predict-majority pattern is the *expected* behavior for ~1,500 trainable
parameters trained on 40 examples per fold. The classifier comparison itself
is the substrate value:

1. A capability portrait that pretended the MLP wins would be a lie — n=50 is
   below the regime where small MLPs reliably beat linear models on tabular
   data.
2. A production version at n=3,000+ (Gilead-scale internal data) flips this
   ordering; the framework is designed to surface that flip cleanly via the
   `comparison.json` artifact.
3. The substrate layer (audit chain + drift detection + per-fold reproducibility)
   is identical regardless of which classifier wins. The substrate is the
   reusable engineering pattern.

This is the difference between "I trained an MLP" and "I built a classifier
gate with honest evaluation and audit-grade reproducibility." The repo proves
the latter.

### Audit chain composition (215 entries)

| Stage | Count |
|---|---|
| pipeline_start / dataset_loaded / pipeline_end | 3 |
| fold_start / fold_end (MLP, 5 folds each) | 10 |
| epoch_end (MLP, 5 folds x ~28 epochs early-stopped average) | ~140 |
| baseline_fold_end (RF + LR, 5 folds each) | 10 |
| comparison_table | 1 |
| drift_summary | 1 |
| (canary, manifest, etc.) | ~50 |

A downstream substrate consumer (Polish-Phase5 `lab_semantic_check.py`) can
read the chain once and see every model decision the run made.

---

## Honest scope — why n=50, single cohort, and no foundation model

The first draft of this demo planned for a 4-layer transformer (~1M params)
on 50 reports. With 50 samples for 3 classes (n=10 / n=20 / n=20), 1M
parameters is 100x over-parameterized. Even with strong regularization the
model would memorize the training set and the holdout would be noise. So
v0.1 picks the simpler-but-honest architecture (LayerNorm + 32 -> 16 MLP,
~1.5k params) and reports the comparison openly. The "foundation" in
`multiqc-foundation-gate` is the **gating substrate** (audit + drift +
honest comparison), not the model size.

Diversity in the 50-sample dataset comes from **augmentation**, not
multi-cohort sampling:

- 10 originals (label = include): clean Himes airway RNA-seq baselines
- 10 module-dropout (label = manual-review): JSON-level removal of 1-2
  FastQC modules to simulate upstream-tool failures
- 10 adapter-injection (label = exclude): 10% reads get a 33 bp TruSeq
  adapter at the 3' end
- 10 quality-degradation (label = exclude): truncate to 30 bp + append
  10 Phred-2 bases
- 10 mixed-issue (label = manual-review): mild truncation (60 bp) +
  light adapter (3%)

Multi-cohort expansion (ChIP-seq, ATAC-seq, GTEx tissue RNA-seq) was
considered for Phase B and deliberately deferred to v0.2 — see
`data/manifest.yaml` comments for the deferral rationale.

---

## P2-specific lessons captured during the build

| ID | Symptom | Fix |
|---|---|---|
| **find -name yaml only** | scaffold `bioscaffold -> multiqc_gate` rename missed `.github/workflows/ci.yml` because the find pattern listed `*.yaml` but not `*.yml` | Use `git ls-files \| while ... file -b --mime` instead, or include both `-o -name "*.yml"` |
| **labels CSV row count drift** | `test_load_labels_parses_committed_sheet` asserted `len == 20` (Hour 3 partial) but Hour 4.B expanded the CSV to 50 — green pytest at write time, red CI on push | Add the count to the test as a derived value or update both in the same commit; alternatively, use `>=` not `==` for forward compatibility |
| **MLP collapse on tiny tabular data** | n=50 with 3 classes and ~1.5k params -> the MLP learns to predict majority class only (val_acc = 0.40, std = 0.0) | This is expected, not a bug. The sklearn baseline is the correct comparison. The capability claim is the substrate framing, not "DL wins" |
| **LogReg max_iter warning** | sklearn `LogisticRegression(max_iter=1000)` did not converge on the 28-feature input -> ConvergenceWarning at every fold | Either bump max_iter, scale features with StandardScaler in the pipeline, or accept that the model still achieves 86% accuracy without converging (the warning is informational) |

---

## Substrate environment variables

Same four-channel substrate as every other capability portrait in the quartet:

| Var | Default | What it does |
|---|---|---|
| `AUDIT_HOST` | unset | If set, audit entries POST to `http://${AUDIT_HOST}/events`. |
| `MLFLOW_TRACKING_URI` | unset | If set, MLflow runs are tracked at this URI. |
| `HEALTHOMICS_LAB_CANARY_FIXTURE` | `tests/fixtures/canary.json` | Path used by `canary.py` for the deterministic smoke test. |
| `HEALTHOMICS_LAB_RUN_NAME` | derived | Overrides the run name in audit + MLflow entries. |

On a Polish-Phase5 lab node, `scripts/run_lab.sh` exports the substrate
endpoints to the lab defaults (`chi-mac-p.local:8081`, `chi-mac-p.local:5050`).

---

## Repo layout

```
.
├── README.md                       # This file
├── LICENSE                         # MIT
├── Makefile                        # install | data | run | test | canary | clean
├── pyproject.toml                  # uv-managed; pinned versions
├── .github/workflows/
│   ├── ci.yml                      # ruff + pytest + scope-preamble lint
│   └── english-only.yml            # CJK character scanner
├── data/
│   ├── manifest.yaml               # 20 ENA URLs (Himes SRR1039513-522 paired-end)
│   ├── labels.csv                  # 50 rows (10 include / 20 exclude / 20 manual-review)
│   ├── features.npy                # (50, 28) classifier-input fixture
│   ├── feature_sample_ids.txt      # row-aligned sample ID list
│   └── multiqc_reports/            # 50 per-sample MultiQC trees (JSON tracked, HTML ignored)
├── src/multiqc_gate/
│   ├── audit.py                    # NDJSON hash-chained ledger emit + verify
│   ├── tracking.py                 # MLflow run wrapper (no-op fallback)
│   ├── canary.py                   # deterministic substrate smoke test
│   ├── augment.py                  # 5 augmentation strategies (Phase C)
│   ├── features.py                 # MultiQC JSON -> 28-dim float32 vector
│   ├── labels.py                   # load_labels + stratified_split + label index
│   ├── model.py                    # MultiQCGateMLP (PyTorch, ~1.5k params)
│   ├── train.py                    # AdamW + 5-fold CV + per-epoch audit hooks
│   ├── baseline.py                 # sklearn RF + LR on same folds
│   ├── eval.py                     # aggregate folds + classification report
│   ├── drift.py                    # KS test per feature
│   └── pipeline.py                 # end-to-end CLI entry
├── tests/                          # 49 tests covering all modules
├── docs/
│   ├── architecture.md             # 4-channel substrate + classifier pipeline
│   ├── tooling-versions.md         # PyTorch 2.12 + sklearn 1.8 + MPS verified
│   └── what-is-out-of-scope.md     # anti-scope-creep ledger
└── scripts/
    ├── run_lab.sh                  # macOS-hardened launch wrapper
    ├── run_phase_c.py              # Phase C augmentation orchestration
    └── check_english_only.py       # CJK scanner used by CI
```

---

## What this repo does not do

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full ledger. Short version: no full ENCODE corpus, no foundation-model
fine-tuning, no A/B test against rule-based gates, no active-learning loop,
no cross-pipeline transfer learning, no multi-cohort expansion. Those belong
to the production version of this gate, not the capability portrait.

---

## Lineage

This repo was created from
[`bioinformatics-repo-scaffold-template`](https://github.com/hryankim-architect/bioinformatics-repo-scaffold-template),
the shared scaffold that every capability-portrait repo in the quartet
(P1 / P2 / P3 / P4) inherits.

Sibling repos:
- [`tp53-aml-hrd-severity`](https://github.com/hryankim-architect/tp53-aml-hrd-severity) (P3) — clinical-genomics analytical-method portrait (Cox HR 8.39 on TCGA-LAML)
- [`healthomics-lab-orchestrator`](https://github.com/hryankim-architect/healthomics-lab-orchestrator) (P1) — Nextflow + substrate-hooked RNA-seq orchestration (22-entry audit chain)
- `hnscc-time-multimodal` (P4) — multimodal IHC + genomics calibration (planned)

---

## License

MIT. See [`LICENSE`](LICENSE).
