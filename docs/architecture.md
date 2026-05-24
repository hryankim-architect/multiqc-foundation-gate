# Architecture

This template imposes a deliberately small architecture. The point is that
every capability-portrait repo inheriting the scaffold has the same shape, so
a reviewer can navigate any of them in 30 seconds.

## The control flow

```
                  ┌──────────────────────────────────────┐
                  │  make run                            │
                  │  (or scripts/run_lab.sh on a node)   │
                  └─────────────────┬────────────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  bioscaffold.pipeline.run_pipeline   │
                  └─────────────────┬────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
┌───────────────┐         ┌───────────────────┐         ┌─────────────────┐
│ audit.emit    │         │ tracking.run      │         │  body  (per-    │
│ (NDJSON +     │         │ (MLflow active    │         │  project, e.g.  │
│  optional     │         │  run context)     │         │  VCF → HRD,     │
│  POST to      │         │                   │         │  Cellpose seg., │
│  AUDIT_HOST)  │         │                   │         │  classifier.)   │
└───────┬───────┘         └─────────┬─────────┘         └────────┬────────┘
        │                           │                            │
        └───────────────────────────┴────────────────────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  artifact JSON + metrics             │
                  └──────────────────────────────────────┘
```

## Substrate integration points

The scaffold integrates with the Polish-Phase5 substrate through three
loosely-coupled channels:

| Channel | Module | Env var | Substrate endpoint |
|---|---|---|---|
| Audit (immutable record) | `bioscaffold.audit` | `AUDIT_HOST` | `http://${AUDIT_HOST}/events` |
| MLflow (experiment tracking) | `bioscaffold.tracking` | `MLFLOW_TRACKING_URI` | configurable |
| Canary (daily probe) | `bioscaffold.canary` | `BIOSCAFFOLD_CANARY_FIXTURE` | invoked by `lab_semantic_check.py` |

All three channels degrade to no-ops when the substrate is absent. The
deterministic local NDJSON ledger remains the source of truth for audit
even when the remote post fails.

## Why a hash-chained NDJSON ledger

Every entry's `prev_hash` is the SHA-256 of the canonical (sorted, separator-
controlled) JSON of the preceding entry. Tampering anywhere in the chain
invalidates the hash of every following entry. The `audit.verify()` function
walks the chain and returns `(ok, n_entries, first_bad_ts)`.

In the Polish-Phase5 substrate this runs at ~6.19 µs/entry up to 10k entries,
with a measured tamper-detect of ~6 ms (full chain re-verify). Capability-
portrait repos do not need that scale; they inherit the format for
*consistency* across the quartet, so the substrate's `gatk_audit.py` verifier
works against any of them.

## Why MLflow

Three reasons, in order:

1. **Experiment tracking** — parameters and metrics are version-controlled
   alongside the run, so the demo's output is reproducible.
2. **Substrate consistency** — every repo in the quartet posts to the same
   MLflow server, so a reviewer can compare runs across projects.
3. **No-op when absent** — the wrapper means the demo works without an MLflow
   server, so a recruiter cloning the repo on a laptop still sees `make run`
   succeed.

## Why a deterministic canary

The canary is the entry point that the Polish-Phase5 `lab_semantic_check.py`
probes daily. Its requirements are:

- Deterministic input (fixture-driven).
- Under 30 seconds to complete.
- Exits 0 on success, non-zero on any deviation.
- No external services required.

A daily-green canary across the quartet means substrate-level monitoring
catches regressions in any of the four capability projects without the
projects themselves needing custom alerting.

## What this architecture intentionally avoids

- No microservices.
- No async runtime.
- No process supervisor.
- No container per pipeline (the scaffold runs in a single Python process).
- No data validation framework beyond Pydantic-on-demand.
- No DAG engine (Nextflow, Airflow, etc.) — those belong inside the body
  when a project needs them (P1 specifically), not in the scaffold.

The point is that the scaffold is the *contract*, not the implementation.
