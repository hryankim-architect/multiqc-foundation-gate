# Tooling Versions

## Installed and verified (Hour 1, 2026-05-25)

| Tool | Version | Source / Env | Notes |
|---|---|---|---|
| Python | 3.12.13 | `.venv/bin/python` (uv-managed) | hosts the `multiqc_gate` package + PyTorch + sklearn |
| **PyTorch** | **2.12.0** | `.venv` via `uv add torch` | MPS backend for chi-mac-p Apple Silicon, verified Hour 1 |
| torchvision | 0.27.0 | `.venv` | included for transforms (currently unused; reserved for future image-feature work) |
| scikit-learn | **1.8.0** | `.venv` | Option C baseline (RandomForest + LogisticRegression) |
| numpy / pandas | pulled in by torch / mlflow | `.venv` | |
| **MultiQC** | 1.35 | dedicated `multiqc` conda env (`conda run -n multiqc multiqc ...`) | inherited from P1 lesson L-phi (libtiff dylib mismatch in base env) |
| **FastQC** | 0.12.1 | bioconda (base) | re-used from P1 toolchain |
| seqkit | 2.13.0 | bioconda (base) | re-used for read sampling / augmentation |

## MPS availability

PyTorch MPS (Metal Performance Shaders) is the Apple Silicon GPU backend. On
chi-mac-p, verified via:

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Result on chi-mac-p (Hour 1, 2026-05-25): **True** (both `is_available()` and
`is_built()`). Smoke test: `torch.randn(100, 100, device='mps') @ x.T`
produced a `torch.float32` tensor on `mps:0` without error.

If MPS is unavailable, the demo falls back to CPU. With n=50 samples and a
~2-3k parameter MLP, training time is sub-30 seconds on CPU either way, so
MPS is a quality-of-life win, not a hard requirement.

## P1 lessons inherited

| ID | Pattern | P2 applicability |
|---|---|---|
| L-phi | conda runtime dylib mismatch in base env | MultiQC stays in dedicated env; FastQC stays in base |
| L-psi | macOS conda Java 25 vs Nextflow 23.04 | **N/A** — P2 does not use Nextflow |
| L-omega | `conda deactivate` removes venv python | **N/A** — P2 does not shell out to subprocess from Nextflow |
| L-alpha2 | `conda deactivate` removes bioconda tools | partially relevant — `conda run -n multiqc/base` form is robust |
| L-beta2 | Nextflow `-resume` cache key ignores env vars | **N/A** — no Nextflow |
| L-chi | `!` in commit-msg body triggers zsh BANG_HIST | `git commit -F file` with quoted heredoc |
| hostname-mismatch | SSH alias != macOS LocalHostName | `scripts/run_lab.sh` already defaults to `chi-mac-p.local` |

## P2-specific lessons (captured as they appear)

This section gets entries as Hour 1-15 surfaces new issues. Watch for:

- PyTorch MPS quirks on small batch sizes (known: MPS sometimes slower than
  CPU for tiny tensors due to kernel-launch overhead)
- sklearn vs torch random-seed determinism across runs
- MultiQC JSON schema variation across versions (FastQC modules can change
  between MultiQC releases)
- ENCODE FASTQ download rate limits (mirror via ENA if hit)
