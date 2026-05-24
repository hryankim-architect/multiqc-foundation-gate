# `bioinformatics-repo-scaffold-template`

> **Template repository for capability-portrait bioinformatics projects.**
> Click *Use this template* to start a new repo with the same shared substrate:
> audit-chained NDJSON ledger, MLflow tracking, English-only CI, anti-scope-creep
> guardrails, and a single `make run` entry point that reproduces the demo
> end-to-end on a single workstation in under N minutes.

A house style for reproducible bioinformatics R&D.

---

## What this template gives you

A new repo created from this template ships with:

- **One-command demo**: `make run` reproduces the pipeline on a tiny public-data subset.
- **Substrate hooks**: every run emits a hash-chained NDJSON audit entry, tracks
  parameters and metrics to MLflow, and exposes a canary smoke test that the
  Polish-Phase5 `lab_semantic_check.py` can probe.
- **Anti-scope-creep guardrails**: required `docs/what-is-out-of-scope.md`,
  CI runtime budget, and `data/manifest.yaml` cap that forces explicit friction
  when adding samples.
- **English-only CI**: CJK-character scanner fails CI if non-English content
  enters a public artifact (lessons-learned, code comments, docs).
- **Reproducibility baseline**: pinned dependencies via `pyproject.toml`,
  containerless `uv` workflow, no external services required for the demo.
- **Honest-scope README template**: the six-line preamble below is enforced
  in CI; production framing and lab-scope framing are kept distinct.

---

## The honest-scope preamble (template — paste into the new repo's README)

```markdown
# <new-repo-name>

> **Capability portrait, not a research result.** Public data is intentionally
> subsetted to keep the demo small and reproducible on a single workstation.

**What this shows**: <one-line capability claim>

**Reproducibility**: `make run` produces the demo output in < N minutes on a
single Mac/Linux box.

**Substrate**: emits audit (NDJSON), tracks MLflow runs, observable via
AgentOps SSE.

**Production framing**: A version of this method ran at full cohort scale on
proprietary data during my time in industry. The lab version here proves the
*method* and the *engineering*, not the result. See
[`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).
```

The lint job in `.github/workflows/ci.yml` checks that the new repo's README
contains the phrase "Capability portrait, not a research result." If a
contributor strips the preamble, CI fails.

---

## Layout

```
.
├── README.md                # This file (replaced per new repo)
├── LICENSE                  # MIT
├── Makefile                 # data | run | test | report | clean
├── pyproject.toml           # uv-managed; pinned versions
├── .github/
│   └── workflows/
│       ├── ci.yml           # ruff + pytest + scope-preamble lint
│       └── english-only.yml # CJK character scanner
├── data/
│   ├── .gitignore           # raw data never committed
│   └── manifest.yaml        # public URLs + checksums for the tiny subset
├── src/bioscaffold/
│   ├── __init__.py
│   ├── pipeline.py          # CLI entry; demonstrates audit + tracking pattern
│   ├── audit.py             # NDJSON hash-chained ledger emit
│   ├── tracking.py          # MLflow run wrapper
│   └── canary.py            # smoke test interface for lab_semantic_check.py
├── tests/
│   ├── test_pipeline.py
│   └── test_canary.py
├── notebooks/
│   └── demo.ipynb           # rendered output committed alongside .ipynb
├── docs/
│   ├── architecture.md      # substrate integration diagram
│   └── what-is-out-of-scope.md  # required anti-scope-creep page
└── scripts/
    └── run_lab.sh           # one-liner to execute on a lab node
```

Rename `src/bioscaffold/` to your project package name when creating the new
repo. The substrate modules (`audit.py`, `tracking.py`, `canary.py`) are
designed to be copy-and-edit, not pip-installed, so each capability repo can
diverge as needed without coordinating releases.

---

## Quickstart (in a new repo created from this template)

```bash
# 1. Install deps
uv sync

# 2. Run the demo end-to-end
make run

# 3. Run tests
make test

# 4. Produce the HTML report
make report
```

The demo prints an audit entry to `audit/local-demo.ndjson` and (if the
`AUDIT_HOST` env var is set) posts it to the substrate audit-API. MLflow runs
appear at `MLFLOW_TRACKING_URI` if configured.

---

## Substrate environment variables

The substrate hooks read these at runtime; the defaults are no-ops, so the
demo works without the substrate present:

| Var | Default | What it does |
|---|---|---|
| `AUDIT_HOST` | unset | If set, audit entries are POSTed to `http://${AUDIT_HOST}/events`. |
| `MLFLOW_TRACKING_URI` | unset | If set, MLflow runs are tracked at this URI. |
| `BIOSCAFFOLD_CANARY_FIXTURE` | `tests/fixtures/canary.json` | Path used by `canary.py` for the deterministic smoke test. |
| `BIOSCAFFOLD_RUN_NAME` | derived | Overrides the run name in audit + MLflow entries. |

On a Polish-Phase5 lab node, `scripts/run_lab.sh` sets these to the lab
defaults before invoking `make run`.

---

## What this template intentionally does not do

- It does not install a package globally; each repo owns its own deps.
- It does not enforce a directory structure beyond the substrate hooks.
- It does not gate the demo on cloud credentials.
- It does not commit raw data — only manifests, checksums, and licenses.
- It does not impose a specific deconvolution / segmentation / annotation
  tool — those are project-level choices.

---

## License

MIT. See [`LICENSE`](LICENSE).
