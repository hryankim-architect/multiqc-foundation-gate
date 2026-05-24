# What is out of scope

This file is **required** in every repo created from the scaffold template.
The CI lint job verifies that this file exists; the PR template references
it as part of the review checklist.

## Why this file exists

A capability-portrait repo's value comes from being *small and complete*. The
single largest risk to that value is the steady accumulation of "while we're
here, let's also..." additions. This file is the anti-scope-creep ledger.
If a PR proposes something on this list, the PR template asks the contributor
to answer one question:

> Why is this still out of scope?

If the answer is good, edit this file in the same PR. If the answer is not
good, the PR doesn't land.

## Default out-of-scope items

(Copy and edit these into the derived repo's `what-is-out-of-scope.md`.)

- **Statistical-power claims**. The demo uses a tiny public subset; effect
  sizes and p-values are illustrative, not conclusive.
- **Full-cohort reproduction**. Adding samples beyond the manifest cap
  requires editing both `data/manifest.yaml` and the README's
  "minimum subset" claim.
- **Multi-cohort meta-analysis**. Out of scope unless this repo's capability
  *is* meta-analysis.
- **Production hardening** (HA, RBAC, multi-tenant). The substrate provides
  the foundation; the capability portrait does not re-implement it.
- **Cost optimization for cloud deployment**. The demo runs on a single
  workstation; cloud cost is by definition out of scope.

## Per-project out-of-scope items

The derived repo replaces this section with its own list, written at v0.1
and amended as PRs land. Examples:

### P3 (`tp53-aml-hrd-severity`)
- BeatAML extension
- Therapy-response prediction
- Multi-cohort survival meta-analysis

### P1 (`healthomics-lab-orchestrator`)
- Production-scale parallelism
- Full reference genome (uses chr22 subset)
- Differential expression analysis

### P2 (`multiqc-foundation-gate`)
- Production-scale training corpus
- Cross-pipeline transfer learning
- Active learning loop

### P4 (`hnscc-time-multimodal`)
- DeepLIIF virtual stain translation (deferred to v0.2)
- Foundation-model embeddings (Approach C)
- Patient-level paired integration (impossible across PMC and TCGA cohorts)
- Outcome prediction (immunotherapy response)

## How to add an item

Open a PR that:

1. Adds the item to the appropriate section above.
2. Adds a one-sentence reason in italics.
3. Links to the PR or issue where the item was originally proposed.

That's it. The friction is intentional.
