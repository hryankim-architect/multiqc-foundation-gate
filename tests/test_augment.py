"""Unit tests for `src/multiqc_gate/augment.py`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from multiqc_gate import augment


@pytest.fixture
def source_report_dir() -> Path:
    """One of the committed Hour 2 fixtures (SRR1039513) used as input."""
    p = Path(__file__).parent.parent / "data" / "multiqc_reports" / "SRR1039513"
    if not p.exists():
        pytest.skip(f"fixture missing: {p}")
    return p


def test_passthrough_copies_tree(source_report_dir, tmp_path):
    target = tmp_path / "SRR1039513_passthrough"
    augment.augment_passthrough(source_report_dir, target)

    # The multiqc_data.json must exist and parse.
    json_path = target / "multiqc_data" / "multiqc_data.json"
    assert json_path.exists()
    with json_path.open() as fh:
        data = json.load(fh)
    assert "report_plot_data" in data


def test_module_dropout_removes_modules(source_report_dir, tmp_path):
    target = tmp_path / "SRR1039513_dropout"
    dropped = augment.augment_module_dropout(
        source_report_dir, target, n_drop=2, seed=42
    )

    assert len(dropped) == 2
    assert all(m in augment.FASTQC_MODULES_DROPPABLE for m in dropped)

    # The augmented JSON should not contain the dropped modules.
    json_path = target / "multiqc_data" / "multiqc_data.json"
    with json_path.open() as fh:
        data = json.load(fh)
    plot_keys = set(data.get("report_plot_data", {}).keys())
    for module in dropped:
        assert module not in plot_keys, f"{module} should have been removed"


def test_module_dropout_is_deterministic(source_report_dir, tmp_path):
    """Same seed -> same modules dropped (substrate reproducibility)."""
    target_a = tmp_path / "SRR1039513_a"
    target_b = tmp_path / "SRR1039513_b"

    dropped_a = augment.augment_module_dropout(source_report_dir, target_a, n_drop=1, seed=42)
    dropped_b = augment.augment_module_dropout(source_report_dir, target_b, n_drop=1, seed=42)

    assert dropped_a == dropped_b


def test_module_dropout_rejects_too_many(source_report_dir, tmp_path):
    target = tmp_path / "SRR1039513_too_many"
    with pytest.raises(ValueError):
        augment.augment_module_dropout(
            source_report_dir, target, n_drop=10, seed=42
        )


@pytest.fixture
def tiny_fastq(tmp_path) -> Path:
    """A 10-record gzipped FASTQ for fast FASTQ-level augmentation tests."""
    import gzip

    path = tmp_path / "tiny.fastq.gz"
    with gzip.open(path, "wt") as fh:
        for i in range(10):
            seq = "ACGT" * 16  # 64 bp
            qual = "I" * 64  # Phred 40 throughout
            fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")
    return path


def test_adapter_injection_modifies_fraction(tiny_fastq, tmp_path):
    """50% adapter injection should change ~5 of 10 reads at the 3' end."""
    target = tmp_path / "tiny_adapter.fastq.gz"
    n = augment.augment_adapter_injection(tiny_fastq, target, fraction=0.5, seed=42)
    assert n == 10  # All records written; only some are modified.

    # Count reads ending in the TruSeq adapter.
    import gzip

    n_with_adapter = 0
    with gzip.open(target, "rt") as fh:
        lines = fh.readlines()
    for i in range(0, len(lines), 4):
        seq = lines[i + 1].rstrip()
        if seq.endswith(augment.TRUSEQ_ADAPTER):
            n_with_adapter += 1
    assert 1 <= n_with_adapter <= 9, f"expected ~5 with adapter, got {n_with_adapter}"


def test_quality_degradation_truncates_and_appends(tiny_fastq, tmp_path):
    """Truncation should shorten every read; quality char appended is '#'."""
    target = tmp_path / "tiny_quality.fastq.gz"
    n = augment.augment_quality_degradation(
        tiny_fastq, target, truncate_to=20, append_low_qual_bases=5, seed=42
    )
    assert n == 10

    import gzip

    with gzip.open(target, "rt") as fh:
        lines = fh.readlines()
    for i in range(0, len(lines), 4):
        seq = lines[i + 1].rstrip()
        qual = lines[i + 3].rstrip()
        # Original length 64 -> truncate_to 20 + append 5 = 25 bp
        assert len(seq) == 25
        assert len(qual) == 25
        # Last 5 bases should be N + '#'
        assert seq[-5:] == "NNNNN"
        assert qual[-5:] == "#####"


def test_mixed_issue_truncates_lightly(tiny_fastq, tmp_path):
    """Mixed = mild truncation (~60bp) + light adapter (3%)."""
    target = tmp_path / "tiny_mixed.fastq.gz"
    n = augment.augment_mixed_issue(
        tiny_fastq, target, adapter_fraction=0.0, truncate_to=50, seed=42
    )
    assert n == 10

    import gzip

    with gzip.open(target, "rt") as fh:
        lines = fh.readlines()
    for i in range(0, len(lines), 4):
        seq = lines[i + 1].rstrip()
        assert len(seq) == 50  # Truncated, no adapter injected (fraction=0).


def test_fastq_io_roundtrip(tiny_fastq, tmp_path):
    """_write_fastq_gz writes the same records _iter_fastq_gz reads."""
    target = tmp_path / "roundtrip.fastq.gz"
    records = list(augment._iter_fastq_gz(tiny_fastq))
    n = augment._write_fastq_gz(target, records)
    assert n == 10
    records_back = list(augment._iter_fastq_gz(target))
    assert records == records_back
