"""Phase C augmentation strategies for MultiQC report fixtures.

Maps the 10 base reports (Cohort 1 + Cohort 2 from
`data/multiqc_reports/SRR1039513-522/`) to 50 augmented reports across 5
strategies. The classifier sees these as 50 distinct samples spread across
3 labels (include / exclude / manual-review).

Strategies (with execution cost):

  1. originals          (label = include, n=10)
     - copy the base report tree unchanged. ~1 ms / sample.

  2. module dropout     (label = manual-review, n=10)
     - JSON-level: randomly remove 1-2 FastQC modules from
       `report_plot_data` + `report_saved_raw_data` to simulate the
       upstream-tool-failure pattern where MultiQC was given a partial
       FastQC output. Fast (~10 ms / sample, no FASTQ regeneration).

  3. adapter injection  (label = exclude, n=10)
     - FASTQ-level: prepend a 33 bp Illumina TruSeq adapter to 10% of
       reads, then re-run FastQC + MultiQC. Slow (~10 s / sample) but
       produces a real MultiQC report with elevated
       `fastqc_adapter_content_plot` signal.

  4. quality degradation  (label = exclude, n=10)
     - FASTQ-level: truncate reads to 30 bp and append 10 low-quality
       bases (Phred '#' = 2), then re-run FastQC + MultiQC. Slow but
       produces a realistic "failed per-base-quality" report.

  5. mixed issue        (label = manual-review, n=10)
     - FASTQ-level: light adapter (3%) + slight read-length variation
       + JSON-level module drop. Realistic edge case.

Hour 3 (v0.1 first implementation) ships strategy 1 + 2 (the JSON-level
ones, ~20 reports). Strategies 3-5 (FASTQ-level, ~30 more reports) land
in Hour 4 after the feature extraction module is verified on the
20-report subset.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Illumina TruSeq adapter (33 bp), the most common adapter contamination
# pattern in real RNA-seq data. Appears at the 3' end of reads when the
# fragment is shorter than the sequenced read length.
TRUSEQ_ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"

# Phred-2 quality character (the lowest realistic "bad" base quality).
LOW_QUALITY_CHAR = "#"

# FastQC modules in the MultiQC v1.35 report. These are the keys under
# `report_plot_data` that we can drop to simulate "upstream tool failure"
# without breaking the rest of the JSON.
FASTQC_MODULES_DROPPABLE = (
    "fastqc_sequence_counts_plot",
    "fastqc_per_base_sequence_quality_plot",
    "fastqc_per_sequence_quality_scores_plot",
    "fastqc_per_sequence_gc_content_plot",
    "fastqc_per_base_n_content_plot",
    "fastqc_sequence_duplication_levels_plot",
    "fastqc_adapter_content_plot",
    "fastqc-status-check-heatmap",
)
# Note: `general_stats_table` is intentionally NOT droppable because
# the classifier's feature vector includes the general stats summary.


# ---------------------------------------------------------------------------
# Strategy 1: pass-through (originals)
# ---------------------------------------------------------------------------


def augment_passthrough(source_dir: Path, target_dir: Path) -> None:
    """Copy the source report tree to target unchanged.

    Used for the n=10 originals that anchor the `include` label class.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in source_dir.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, target_dir / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target_dir / entry.name)


# ---------------------------------------------------------------------------
# Strategy 2: module dropout (JSON-level)
# ---------------------------------------------------------------------------


def augment_module_dropout(
    source_dir: Path,
    target_dir: Path,
    n_drop: int = 1,
    seed: int = 42,
) -> list[str]:
    """Copy the report tree and remove `n_drop` random FastQC modules from
    the parsed JSON.

    Returns the list of dropped module names so the augmentation log can
    record which modules each augmented sample is missing. The feature
    extractor in `src/multiqc_gate/features.py` reads from
    `multiqc_data.json`, so removing modules there is sufficient to make
    the augmented sample look different to the classifier.

    Args:
        source_dir: Path to the per-sample report directory
            (contains `multiqc_data/multiqc_data.json`).
        target_dir: Where to write the augmented copy.
        n_drop: How many modules to remove (1 or 2 is realistic; 3+ would
            be an unusable report).
        seed: Random seed for reproducibility.

    Returns:
        List of dropped module names.
    """
    if n_drop < 1 or n_drop > len(FASTQC_MODULES_DROPPABLE) - 1:
        raise ValueError(
            f"n_drop must be in [1, {len(FASTQC_MODULES_DROPPABLE) - 1}], got {n_drop}"
        )

    augment_passthrough(source_dir, target_dir)

    json_path = target_dir / "multiqc_data" / "multiqc_data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"multiqc_data.json not found in {target_dir}")

    rng = random.Random(seed)
    to_drop = rng.sample(list(FASTQC_MODULES_DROPPABLE), n_drop)

    with json_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    # 1. Remove from report_plot_data (the per-module plot definitions).
    plot_data = data.get("report_plot_data", {})
    for module in to_drop:
        plot_data.pop(module, None)

    # 2. Remove from report_saved_raw_data (the underlying values).
    #    The raw-data keys use a different naming convention than the plot
    #    keys (e.g. `multiqc_fastqc` aggregates several plots), so the
    #    safest fingerprint is the plot key itself reduced to its FastQC
    #    "stem" (`fastqc_adapter_content_plot` -> `adapter_content`).
    raw_data = data.get("report_saved_raw_data", {})
    for module in to_drop:
        stem = module.replace("fastqc_", "").replace("_plot", "").replace("-", "_")
        for raw_key in list(raw_data.keys()):
            if stem in raw_key:
                raw_data.pop(raw_key, None)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    return to_drop


# ---------------------------------------------------------------------------
# FASTQ helpers (gzipped stream parsing + writing)
# ---------------------------------------------------------------------------


def _iter_fastq_gz(path: Path) -> Iterator[tuple[str, str, str, str]]:
    """Yield (header, seq, plus, qual) tuples from a gzipped FASTQ file."""
    with gzip.open(path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()
            if not qual:
                return
            yield header.rstrip("\n"), seq.rstrip("\n"), plus.rstrip("\n"), qual.rstrip("\n")


def _write_fastq_gz(path: Path, records: Iterable[tuple[str, str, str, str]]) -> int:
    """Write records to gzipped FASTQ; return record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt") as fh:
        for header, seq, plus, qual in records:
            fh.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Strategy 3: adapter injection (FASTQ-level)
# ---------------------------------------------------------------------------


def augment_adapter_injection(
    source_fastq: Path,
    target_fastq: Path,
    fraction: float = 0.10,
    seed: int = 42,
) -> int:
    """Inject the TruSeq adapter at the 3' end of `fraction` of reads.

    This simulates the most common real-world adapter-contamination
    pattern: short fragments where the sequencing read extends past the
    fragment's 3' end and into the adapter. FastQC's
    `adapter_content` module should flip from pass to warn/fail and
    `fastqc_adapter_content_plot` should show elevated signal at the
    3' positions.

    Quality string is preserved at the original length so the read
    length distribution does not change (the augmentation is sequence-
    only, not length).

    Args:
        source_fastq: Path to source gzipped FASTQ.
        target_fastq: Where to write the augmented gzipped FASTQ.
        fraction: Fraction of reads to contaminate (0.05-0.15 realistic).
        seed: RNG seed for reproducibility.

    Returns:
        Number of records written.
    """
    rng = random.Random(seed)
    adapter_len = len(TRUSEQ_ADAPTER)

    def _records() -> Iterator[tuple[str, str, str, str]]:
        for header, seq, plus, qual in _iter_fastq_gz(source_fastq):
            if rng.random() < fraction and len(seq) > adapter_len:
                # Replace the last `adapter_len` bases with the adapter.
                # Quality string stays the same length (adapter base quality
                # is whatever the original 3'-end quality was; this is what
                # FastQC sees in real adapter contamination too).
                pos = len(seq) - adapter_len
                new_seq = seq[:pos] + TRUSEQ_ADAPTER
                yield header, new_seq, plus, qual
            else:
                yield header, seq, plus, qual

    return _write_fastq_gz(target_fastq, _records())


# ---------------------------------------------------------------------------
# Strategy 4: quality degradation (FASTQ-level)
# ---------------------------------------------------------------------------


def augment_quality_degradation(
    source_fastq: Path,
    target_fastq: Path,
    truncate_to: int = 30,
    append_low_qual_bases: int = 10,
    seed: int = 42,
) -> int:
    """Truncate reads + append low-quality bases.

    Simulates a sequencing run where the final cycles failed: the read
    length distribution shifts down and the 3' end quality plummets.
    FastQC's `per_base_sequence_quality` and `sequence_length_distribution`
    modules should flip to warn/fail.

    The augmentation is applied to every read (not a fraction) so the
    distribution shift is clear.

    Args:
        source_fastq: Path to source gzipped FASTQ.
        target_fastq: Where to write the augmented gzipped FASTQ.
        truncate_to: Length to truncate each read to (30 bp = severe).
        append_low_qual_bases: How many 'N' bases (with Phred-2 quality)
            to append after truncation.
        seed: RNG seed (unused for deterministic ops, kept for API symmetry).

    Returns:
        Number of records written.
    """
    _ = seed  # API symmetry with the other augment functions.
    n_bad = append_low_qual_bases
    bad_qual = LOW_QUALITY_CHAR * n_bad
    bad_seq = "N" * n_bad

    def _records() -> Iterator[tuple[str, str, str, str]]:
        for header, seq, plus, qual in _iter_fastq_gz(source_fastq):
            new_seq = seq[:truncate_to] + bad_seq
            new_qual = qual[:truncate_to] + bad_qual
            yield header, new_seq, plus, new_qual

    return _write_fastq_gz(target_fastq, _records())


# ---------------------------------------------------------------------------
# Strategy 5: mixed-issue (FASTQ-level combination)
# ---------------------------------------------------------------------------


def augment_mixed_issue(
    source_fastq: Path,
    target_fastq: Path,
    adapter_fraction: float = 0.03,
    truncate_to: int = 60,
    seed: int = 42,
) -> int:
    """Combination: light adapter contamination + mild read truncation.

    Simulates a borderline-quality library: the issues are real but each
    is mild on its own. The classifier should land this on
    `manual-review` rather than the clear `exclude` of strategies 3/4.

    Args:
        source_fastq: Path to source gzipped FASTQ.
        target_fastq: Where to write the augmented gzipped FASTQ.
        adapter_fraction: Light adapter contamination (3% by default).
        truncate_to: Mild truncation (60 bp from 63 bp default — minor
            length-distribution change but the modal length still
            decreases).
        seed: RNG seed.

    Returns:
        Number of records written.
    """
    rng = random.Random(seed)
    adapter_len = len(TRUSEQ_ADAPTER)

    def _records() -> Iterator[tuple[str, str, str, str]]:
        for header, seq, plus, qual in _iter_fastq_gz(source_fastq):
            # 1. Truncate (mild).
            seq2 = seq[:truncate_to]
            qual2 = qual[:truncate_to]
            # 2. Maybe adapter inject (light).
            if rng.random() < adapter_fraction and len(seq2) > adapter_len:
                pos = len(seq2) - adapter_len
                seq2 = seq2[:pos] + TRUSEQ_ADAPTER
            yield header, seq2, plus, qual2

    return _write_fastq_gz(target_fastq, _records())
