# What is out of scope (P2 — `multiqc-foundation-gate`)

This file is the anti-scope-creep ledger for the P2 capability portrait.
The repo's value comes from being *small and complete* — every item below
is something a reviewer might reasonably ask for that the v0.1 demo
deliberately does not attempt.

If a future PR proposes any of these, the contributor must answer one
question: **why is this still out of scope?** If the answer is good, edit
this file in the same PR. If not, the PR doesn't land.

---

## Full ENCODE-scale training corpus

The capability claim is anchored at n=50 (10 base SRR x 5 augmentation
strategies). A production version would train on ~3,000 internal MultiQC
reports across multiple cohorts (RNA-seq, ChIP-seq, ATAC-seq, scRNA-seq).

**Why out of scope**: a full corpus run would need controlled-access
internal data + an order-of-magnitude longer training time, breaking the
"reproducible in 4 seconds on a single workstation" contract. The
v0.1 substrate framing (audit + drift + comparison) is the production
contribution regardless of dataset size.

---

## Foundation-model fine-tuning (Geneformer, scBERT, scGPT)

Single-cell foundation models could plausibly be fine-tuned on per-sample
gene-expression vectors derived from MultiQC report inputs.

**Why out of scope**: foundation-model fine-tuning requires GPU-scale
compute and ~5 GB of pre-trained weights, breaking the "no GPU required"
contract. The 28-feature engineered vector is the right input modality
for a 50-sample dataset — foundation models on this input would over-
parameterize even more aggressively than the current MLP.

---

## A/B test against rule-based gates

A rule-based gate (e.g., `if avg_seq_length < 50 OR adapter_content > 20%
then exclude`) is the current state-of-the-art at most clinical
bioinformatics labs. A real comparison would A/B test against such a
gate on hold-out samples.

**Why out of scope**: rule-based gates vary by lab, and "the right rule"
itself is an internal decision. The v0.1 sklearn baselines (LogReg + RF)
are the closest proxy — they learn a (linear or tree-based) decision
boundary from the same features the rule-based gate would use, and the
LogReg coefficient signs serve as an interpretable rule extraction.
A formal A/B test belongs to a production deployment, not the capability
portrait.

---

## Active-learning loop

A production gate would query human reviewers on its lowest-confidence
predictions and retrain. v0.1 trains once and reports cross-validation
accuracy.

**Why out of scope**: active learning requires a label-acquisition
oracle (a human or a wet-lab assay), which is out-of-scope for a
clone-and-run capability portrait. The drift detection module is the
forward-looking substrate hook for triggering active-learning loops
in a production deployment.

---

## Cross-pipeline transfer learning

A gate trained on Himes RNA-seq MultiQC reports may or may not transfer
to ChIP-seq, ATAC-seq, or single-cell RNA-seq reports. v0.1 does not test
this — all 50 samples come from a single Himes airway smooth muscle cohort.

**Why out of scope**: cross-pipeline transfer learning needs multi-cohort
data (deferred to v0.2 — see Phase B comments in `data/manifest.yaml`).
The 28-feature schema is *designed* to be pipeline-agnostic (FastQC module
status + summary stats are common to all FastQC-targeting workflows), so
v0.2 expansion is a feature-vector reuse, not a redesign.

---

## Multi-cohort expansion (Phase B)

Cohort 2 in the original manifest comments envisioned 15 additional SRR
across ChIP-seq + ATAC-seq + GTEx-style RNA-seq. v0.1 implements Cohort 1
only (Himes 10 SRR) + Phase C augmentation.

**Why out of scope**: ENA URL discovery for arbitrary ENCODE / GTEx
accessions has substantial manual overhead, and the v0.1 capability
claim (substrate + classifier comparison + drift) is fully demonstrated
on Cohort 1 + augmentation. Cohort 2 is a Phase 2 paper-grade extension,
not a v0.1 substrate test.

---

## Production hardening (HA, RBAC, multi-tenant)

The pipeline runs in a single Python process. There is no HA, no RBAC,
no per-tenant isolation, no input streaming, no retry/backoff, no
distributed orchestration.

**Why out of scope**: the substrate (`audit.py`, `tracking.py`,
`canary.py`) provides the building blocks; the capability portrait does
not re-implement Polish-Phase5 production infrastructure. Production
hardening belongs to the orchestration project (P1
`healthomics-lab-orchestrator`), not the analytical-gate portrait.

---

## Drift remediation policy

The drift module flags features whose distributions differ from the
include-class baseline. v0.1 *reports* drift; it does not *act* on it
(no automatic retraining, no alerting, no model rollback).

**Why out of scope**: drift remediation policy is operational, not
analytical — it depends on the deployment SLA, the cost of false-positive
exclusion, and the lab's review capacity. The drift signal itself is the
substrate contribution; the policy on top is a deployment-tier decision.

---

## Explainability beyond LogReg coefficients + RF feature importance

A full clinical-grade gate would surface SHAP values, per-sample
explanations, and counterfactual analysis ("what would the report need to
look like for the gate to switch its decision?").

**Why out of scope**: counterfactual analysis on a 28-feature vector
requires either a generative model of MultiQC reports (out-of-scope ML)
or a constrained optimization formulation (out-of-scope methodology
deferred to a Phase 2 paper, possibly fed into the DMOI dialectical
framework for the comparison-of-hypotheses framing).

---

## Adding an item

Open a PR that:

1. Adds the item to the appropriate section above (or creates a new
   section if none fits).
2. Adds a one-sentence reason in italics for why it remains out of scope.
3. Links to the upstream feature request or issue if there is one.

That's it. The friction is intentional.
