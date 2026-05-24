#!/usr/bin/env python3
"""Phase C augmentation orchestration: 10 base SRR -> 40 augmented FASTQ pairs.

Reads `data/labels.csv`, takes the 10 base samples (label=include,
augmentation=original), applies the 4 augmentation strategies, writes
augmented FASTQs to `data/fastq_augmented/`, and prints the bash
one-liner that the operator runs to re-run FastQC + MultiQC on each
augmented FASTQ pair.

Strategies (10 outputs each, 40 total):
    - module_dropout     (JSON-level, generated here directly)
    - adapter_injection  (FASTQ-level, then FastQC + MultiQC)
    - quality_degradation (FASTQ-level, then FastQC + MultiQC)
    - mixed_issue        (FASTQ-level, then FastQC + MultiQC)

Together with the 10 originals, this gives the canonical 50-sample
dataset that the classifier trains on.

Run from the repo root:

    .venv/bin/python scripts/run_phase_c.py --plan
        Print what would be generated (no writes).

    .venv/bin/python scripts/run_phase_c.py --do-json
        Generate the JSON-level (module_dropout) augmentations only.
        Fast (~10 ms / sample) so this runs in-process.

    .venv/bin/python scripts/run_phase_c.py --do-fastq
        Generate the FASTQ-level augmentations (adapter / quality / mixed).
        Slower (~1 s / sample for the FASTQ rewrite; FastQC + MultiQC
        are a separate step printed at the end).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from multiqc_gate import augment  # noqa: E402

REPO_ROOT = ROOT
FASTQ_DIR = REPO_ROOT / "data" / "fastq"
FASTQ_AUG_DIR = REPO_ROOT / "data" / "fastq_augmented"
REPORTS_DIR = REPO_ROOT / "data" / "multiqc_reports"

# The 10 base samples from manifest.yaml cohorts 1+2.
BASE_SAMPLES = [f"SRR1039{n}" for n in range(513, 523)]

# Augmentation suffixes — must match labels.csv report_id naming.
DROPOUT_SUFFIXES_PER_SAMPLE = [
    ("dropout_1mod", {"n_drop": 1}),
    ("dropout_2mod", {"n_drop": 2}),
]


def _augment_jsonlevel_one(sample_id: str, suffix: str, seed_offset: int, **kwargs) -> str:
    """Generate one JSON-level (module dropout) augmented report."""
    source_dir = REPORTS_DIR / sample_id
    target_id = f"{sample_id}_{suffix}"
    target_dir = REPORTS_DIR / target_id
    dropped = augment.augment_module_dropout(
        source_dir, target_dir, seed=hash(target_id) % (2**31), **kwargs
    )
    return f"  [json] {target_id}  dropped={dropped}"


def _augment_fastq_pair_one(sample_id: str, suffix: str, strategy: str) -> tuple[str, list[Path]]:
    """Generate one FASTQ-level augmented PAIRED-END FASTQ pair.

    Returns (log_line, [out_r1_path, out_r2_path]) so the caller can
    print the bash commands to re-run FastQC + MultiQC.
    """
    out_dir = FASTQ_AUG_DIR / f"{sample_id}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    seed = hash(f"{sample_id}_{suffix}") % (2**31)

    for r in (1, 2):
        src = FASTQ_DIR / f"{sample_id}_{r}.fastq.gz"
        tgt = out_dir / f"{sample_id}_{suffix}_{r}.fastq.gz"
        if not src.exists():
            return (f"  [skip] {sample_id} (missing source {src})", [])
        if strategy == "adapter":
            n = augment.augment_adapter_injection(src, tgt, fraction=0.10, seed=seed + r)
        elif strategy == "quality":
            n = augment.augment_quality_degradation(
                src, tgt, truncate_to=30, append_low_qual_bases=10, seed=seed + r
            )
        elif strategy == "mixed":
            n = augment.augment_mixed_issue(
                src, tgt, adapter_fraction=0.03, truncate_to=60, seed=seed + r
            )
        else:
            raise ValueError(f"unknown FASTQ-level strategy: {strategy}")
        out_paths.append(tgt)

    return (f"  [{strategy}] {sample_id}_{suffix}  wrote {n} reads x 2 mates", out_paths)


def plan() -> None:
    print("Phase C augmentation plan:")
    print(f"  Source: {len(BASE_SAMPLES)} base samples ({BASE_SAMPLES[0]}-{BASE_SAMPLES[-1]})")
    print(f"  JSON-level (module_dropout): {len(BASE_SAMPLES)} outputs (10 reports)")
    print(f"  FASTQ-level (adapter/quality/mixed): {len(BASE_SAMPLES) * 3} outputs (30 reports)")
    print(f"  Total: 10 originals + 10 dropout + 10 adapter + 10 quality + 10 mixed = 50 reports")
    print(f"  Output:")
    print(f"    JSON: {REPORTS_DIR}/<SRR>_<suffix>/multiqc_data/multiqc_data.json")
    print(f"    FASTQ: {FASTQ_AUG_DIR}/<SRR>_<suffix>/*_<R>.fastq.gz")


def do_json() -> None:
    print(f"Generating {len(BASE_SAMPLES)} JSON-level (module_dropout) augmentations...")
    # 10 dropout samples: 6x 1-module-drop + 4x 2-module-drop (matches labels.csv)
    drop_plan = ["1mod"] * 6 + ["2mod"] * 4
    for sid, suffix_kind in zip(BASE_SAMPLES, drop_plan):
        n_drop = 1 if suffix_kind == "1mod" else 2
        suffix = f"dropout_{suffix_kind}"
        log_line = _augment_jsonlevel_one(sid, suffix, seed_offset=0, n_drop=n_drop)
        print(log_line)
    print(f"\nDone. Verify with: ls -d data/multiqc_reports/SRR*_dropout_*/")


def do_fastq() -> None:
    print(f"Generating {len(BASE_SAMPLES) * 3} FASTQ-level augmentations...")
    multiqc_jobs: list[tuple[str, str]] = []  # (input dir, output dir)

    for strategy in ("adapter", "quality", "mixed"):
        print(f"\n--- strategy: {strategy} ---")
        for sid in BASE_SAMPLES:
            suffix = strategy
            log_line, out_paths = _augment_fastq_pair_one(sid, suffix, strategy)
            print(log_line)
            if out_paths:
                fastq_dir = out_paths[0].parent
                report_dir = REPORTS_DIR / f"{sid}_{suffix}"
                multiqc_jobs.append((str(fastq_dir), str(report_dir)))

    print("\n=========================================================")
    print("FASTQ augmentation done. Now run FastQC + MultiQC on each:")
    print("=========================================================\n")
    print("# Paste this into the shell (uses brew fastqc + conda multiqc env):")
    for src_dir, rep_dir in multiqc_jobs:
        print(f"mkdir -p '{rep_dir}/_fastqc'")
        print(f"fastqc -o '{rep_dir}/_fastqc/' -q {src_dir}/*.fastq.gz")
        print(f"conda run -n multiqc multiqc -f -q -o '{rep_dir}/' '{rep_dir}/_fastqc/' 2>&1 | tail -1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="Print plan, no writes")
    ap.add_argument("--do-json", action="store_true", help="Generate JSON-level augmentations")
    ap.add_argument("--do-fastq", action="store_true", help="Generate FASTQ-level augmentations")
    args = ap.parse_args()

    if not (args.plan or args.do_json or args.do_fastq):
        plan()
        return

    if args.plan:
        plan()
    if args.do_json:
        do_json()
    if args.do_fastq:
        do_fastq()


if __name__ == "__main__":
    main()
