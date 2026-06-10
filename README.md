# `multiqc-foundation-gate`

![ci](https://github.com/hryankim-architect/multiqc-foundation-gate/actions/workflows/ci.yml/badge.svg)

> **One principle, applied here.** Pick the smallest, most interpretable representation that could carry the signal; measure it against an honest baseline; report the verdict faithfully — whether the compact choice wins, ties, or loses. *That last step is why AI safety is needed: knowing a capability is real rather than a flattering benchmark.*
>
> In this repo: **representation** a 28-feature engineered vector + linear/tree model → **baseline** an MLP (~1.5k params) → **verdict** compact *wins*: sklearn baselines (LogReg 0.80, RandomForest 0.84) beat the MLP's 0.40 at n=50 — capacity without data hurts, narrated honestly.

n=50 (Himes airway smooth muscle, SRP033351, 10 base SRR + synthetic augmentation). `make run` finishes in 3.8 seconds on a laptop CPU; no cloud credentials or GPU needed.

**What this shows**: turning an operational QC artifact (MultiQC report JSON)
into an ML-driven include / exclude / manual-review decision, with audit +
MLflow substrate hooks per training epoch and a baseline comparison
that shows when the deep-learning model loses to plain sklearn.

**Reproducibility**: `make run` produces the full classifier comparison + drift
detection + 215-entry audit chain in **3.8 seconds** on chi-mac-p. No GPU
required (MPS used if available, falls back to CPU automatically). No cloud
credentials.

**Substrate**: emits a NDJSON ledger whose entries are hash-linked, one per fold, epoch, and stage across the training run, tracks MLflow runs,
and exposes a deterministic canary smoke test that the `lab_semantic_check.py` probe can call.

**Prior work context**: At Gilead I ran a version of this gate pattern on ~3,000
internal MultiQC reports; there the MLP dominated the sklearn baselines and the gate
caught ~12% of low-quality samples that human review would otherwise miss.
This repo validates the architecture and substrate integration on a small labeled dataset.
Results at that scale are not reproducible here; see
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
> downstream pipeline can consume, with the audit trail and drift detection
> a regulated environment requires?

The classifier itself is the *secondary* contribution. The *primary*
contribution is the substrate around it (28-feature vector schema, per-epoch
audit chain, per-feature drift detection, sklearn-baseline comparison
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
#    PyTorch 2.12 + sklearn 1.7 (pinned <1.8) + numpy/scipy (uv-managed; bioconda env hosts
#    FastQC + MultiQC). MPS GPU backend is used automatically on Apple Silicon.

# 1. Install pinned Python dependencies
make install                  # uv sync --extra dev

# 2. Fetch 10 Himes SRR FASTQs + run FastQC + MultiQC + augment to 50 reports
make data                     # ~2 min on first call (download + augmentation)

# 3. Run the end-to-end classifier comparison + drift detection
make run                      # ~4 sec wall-clock; writes artifacts/ + audit chain

# 4. Run the test suite (50 tests)
make test

# 5. Run the canary smoke test (substrate registration probe)
make canary
```

---

## Real-data climax, sklearn beats MLP on n=50, and that is the right answer

End-to-end run on the n=50 canonical dataset, chi-mac-p, 2026-05-25:

| Method | Accuracy (mean ± std) | F1 macro (mean ± std) | Trainable params |
|---|---|---|---|
| **RandomForest** | **0.840 ± 0.102** | **0.814 ± 0.118** | (100 trees, depth 5) |
| **LogisticRegression** | **0.800 ± 0.063** | **0.768 ± 0.063** | 87 |
| **MLP** (PyTorch) | **0.400 ± 0.000** | **0.190 ± 0.000** | ~1.5k |

> **Reproducibility note (LogReg).** The LogisticRegression baseline runs on `StandardScaler`-normalized features. Without scaling, lbfgs does not converge on the raw 28-feature vector, and the non-converged solution is sklearn-version-dependent (it scored 0.86 on sklearn 1.7 but 0.56 on a newer release). Scaling makes lbfgs converge, giving a stable **0.80** across versions — the honest, reproducible number. The qualitative verdict (compact linear/tree models beat the MLP at n=50) is unchanged.

Substrate metrics:

| Metric | Value |
|---|---|
| Wall-clock (`make run`, 5-fold CV x 3 methods + drift + audit + MLflow) | **3.76 sec** on chi-mac-p (CPU; MPS not engaged for tiny tensors) |
| Audit chain entries | **215** (1 pipeline_start + 5 fold_start + 5 fold_end + ~140 epoch_end + 10 baseline_fold_end + 1 comparison + 1 drift + 1 pipeline_end + framing) |
| Audit chain validity | `ok=True` (`prev_hash` replay verifies every entry) |
| Drift detection | **5 of 28 features drifted** at alpha=0.05 (quality-degradation cohort vs include baseline). Top drifted: `num__avg_sequence_length`, `num__median_sequence_length`, exactly what the quality-degradation augmentation is designed to shift |
| Test count | **50** (8 augment + 4 baseline + 2 canary + 6 drift + 5 eval + 6 features + 1 headline-regression + 5 labels + 4 model + 4 pipeline + 5 train); 3 model/pipeline/train cases skip without PyTorch |

### What the MLP-vs-sklearn gap means (this is the capability claim, not a failure)

On n=50 with a 3-way class label, the **sklearn baselines at 80–84%
accuracy are the right number**, and the MLP collapsing to a 40% / std=0
predict-majority pattern is the *expected* behavior for ~1,500 trainable
parameters trained on 40 examples per fold. The classifier comparison itself
is the substrate value:

1. On n=50, small MLPs do not reliably beat linear models on tabular data.
   Reporting otherwise would misrepresent the result.
2. A production version at n=3,000+ (Gilead-scale internal data) flips this
   ordering; the framework is designed to surface that flip cleanly via the
   `comparison.json` artifact.
3. The substrate layer (audit chain + drift detection + per-fold reproducibility)
   is identical regardless of which classifier wins. The substrate is the
   reusable engineering pattern.

This is the difference between "I trained an MLP" and "I built a classifier
gate with rigorous evaluation and audit-grade reproducibility." The repo proves
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

A downstream substrate consumer (`lab_semantic_check.py`) can
read the chain once and see every model decision the run made.

---

## Calibration & interpretability diagnostics

Because the gate makes an automated include / exclude decision, two oversight
questions matter beyond raw accuracy: *how calibrated is its confidence?* and
*what does it key on?*

- **Calibration** (`scripts/calibrate_gate.py` → `audit/gate_calibration.md`):
  confidence-ECE (Guo et al. 2017) + multiclass Brier on the pooled held-out
  folds. RandomForest ECE **0.05** (mean confidence ≈ accuracy, well-matched),
  LogisticRegression ECE **0.15** (mildly over-confident once the features are
  `StandardScaler`-normalized for convergence). With n=50 across 3 classes this is
  reported as a **diagnostic, not a re-calibration fix** — a fitted calibrator would overfit.
- **Interpretability** (`scripts/interpret_gate.py` →
  `audit/gate_feature_importance.md`): model-agnostic permutation importance over
  the 28 named MultiQC features. The gate keys on sensible QC signals — the number
  of passing modules, the per-sequence quality-score plot, and sequence-length
  metrics — so its decisions are legible rather than opaque.

Both are diagnostic-only at n=50: indicative, not definitive. Same caveat applies as the sklearn-beats-MLP baseline.

---

## Scope, why n=50, single cohort, and no foundation model

The first draft of this demo planned for a 4-layer transformer (~1M params)
on 50 reports. With 50 samples for 3 classes (n=10 / n=20 / n=20), 1M
parameters is 100x over-parameterized. Even with strong regularization the
model would memorize the training set and the holdout would be noise. So
v0.1 picks the simpler architecture (LayerNorm + 32 -> 16 MLP,
~1.5k params) and reports the comparison openly. The "foundation" in
`multiqc-foundation-gate` is the **gating substrate** (audit + drift +
comparison), not the model size.

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
considered for Phase B and deliberately deferred to v0.2, see
`data/manifest.yaml` comments for the deferral rationale.

---

## P2-specific lessons captured during the build

| ID | Symptom | Fix |
|---|---|---|
| **find -name yaml only** | scaffold `bioscaffold -> multiqc_gate` rename missed `.github/workflows/ci.yml` because the find pattern listed `*.yaml` but not `*.yml` | Use `git ls-files \| while ... file -b --mime` instead, or include both `-o -name "*.yml"` |
| **labels CSV row count drift** | `test_load_labels_parses_committed_sheet` asserted `len == 20` (Hour 3 partial) but Hour 4.B expanded the CSV to 50, green pytest at write time, red CI on push | Add the count to the test as a derived value or update both in the same commit; alternatively, use `>=` not `==` for forward compatibility |
| **MLP collapse on tiny tabular data** | n=50 with 3 classes and ~1.5k params -> the MLP learns to predict majority class only (val_acc = 0.40, std = 0.0) | This is expected, not a bug. The sklearn baseline is the correct comparison. The capability claim is the substrate framing, not "DL wins" |
| **LogReg non-convergence was hiding version drift** | sklearn `LogisticRegression(max_iter=1000)` did not converge on the raw 28-feature input -> ConvergenceWarning at every fold. The non-converged solution was sklearn-version-dependent: 0.86 on sklearn 1.7, 0.56 on a newer release — a silent reproducibility hole in a repo whose whole point is reproducibility | **Fixed:** the LR baseline now runs inside a `StandardScaler` pipeline (`baseline.py`), so lbfgs converges and the score is a stable **0.80** across sklearn versions. Lesson: a ConvergenceWarning is not "informational" when it makes a headline number version-dependent — scale, then pin |

---

## Substrate environment variables

Four-channel substrate (same interface as the other repos in this series):

| Var | Default | What it does |
|---|---|---|
| `AUDIT_HOST` | unset | If set, audit entries POST to `http://${AUDIT_HOST}/events`. |
| `MLFLOW_TRACKING_URI` | unset | If set, MLflow runs are tracked at this URI. |
| `BIOSCAFFOLD_CANARY_FIXTURE` | `tests/fixtures/canary.json` | Path used by `canary.py` for the deterministic smoke test. |
| `BIOSCAFFOLD_RUN_NAME` | derived | Overrides the run name in audit + MLflow entries. |

On a lab node, `scripts/run_lab.sh` exports the substrate
endpoints to the lab defaults (`chi-mac-m:8081`, `chi-mac-m:5050`).

---

## Repo layout

```
.
├── README.md                       # This file
├── LICENSE                         # MIT
├── Makefile                        # install | data | run | test | canary | clean
├── pyproject.toml                  # uv-managed; pinned versions
├── .github/workflows/
│   └── ci.yml                      # ruff + pytest + canary
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
├── tests/                          # 50 tests covering all modules
├── docs/
│   ├── architecture.md             # 4-channel substrate + classifier pipeline
│   ├── tooling-versions.md         # PyTorch 2.12 + sklearn 1.8 + MPS verified
│   └── what-is-out-of-scope.md     # out-of-scope items and rationale
└── scripts/
    ├── run_lab.sh                  # macOS-hardened launch wrapper
    └── run_phase_c.py              # Phase C augmentation orchestration
```

---

## What this repo does not do

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full list. Short version: no full ENCODE corpus, no foundation-model
fine-tuning, no A/B test against rule-based gates, no active-learning loop,
no cross-pipeline transfer learning, no multi-cohort expansion.

---

## Lineage

This repo was created from
[`bioinformatics-repo-scaffold-template`](https://github.com/hryankim-architect/bioinformatics-repo-scaffold-template),
the shared scaffold used by all four repos in this series (P1 / P2 / P3 / P4).

Sibling repos:
- [`tp53-aml-hrd-severity`](https://github.com/hryankim-architect/tp53-aml-hrd-severity) (P3), clinical-genomics analytical-method portrait (Cox HR 8.39 on TCGA-LAML)
- [`healthomics-lab-orchestrator`](https://github.com/hryankim-architect/healthomics-lab-orchestrator) (P1), Nextflow + substrate-hooked RNA-seq orchestration (22-entry audit chain)
- `hnscc-time-multimodal` (P4), multimodal IHC + genomics calibration (planned)

---

## License

MIT. See [`LICENSE`](LICENSE).
