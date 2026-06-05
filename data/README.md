# `data/`

This directory holds the canonical classifier-input fixtures committed to the
repo, plus downloaded FASTQ data that is git-ignored.

**Committed fixtures** (required for `make run`):
- `manifest.yaml` — 20 ENA FTP URLs with SHA-256 placeholders.
- `labels.csv` — 50-row label sheet (10 include / 20 exclude / 20 manual-review).
- `features.npy` — (50, 28) float32 feature matrix; direct classifier input.
- `feature_sample_ids.txt` — row-aligned sample IDs for `features.npy`.
- `multiqc_reports/` — 50 per-sample MultiQC data trees. HTML reports and raw
  FastQC dirs are git-ignored (see `multiqc_reports/.gitignore`); only the
  parsed JSON and small text summaries are tracked.

**Git-ignored** (downloaded or generated on demand):
- `fastq/` — raw FASTQ files (~50 MB). Fetched via `make data`.
- `fastq_augmented/` — augmented FASTQ pairs from `scripts/run_phase_c.py`.
