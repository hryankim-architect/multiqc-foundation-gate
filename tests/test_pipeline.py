"""End-to-end smoke tests for the scaffold pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from bioscaffold import audit, pipeline


def test_pipeline_runs_and_produces_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    out_dir = tmp_path / "artifacts"
    result = pipeline.run_pipeline("smoke", out_dir)

    assert "job_id" in result
    assert result["metrics"]["body_elapsed_ms"] >= 0.0

    artifact_path = Path(result["artifact_path"])
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text())
    assert payload["run_name"] == "smoke"


def test_audit_chain_is_valid_after_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts")

    ok, n_entries, first_bad = audit.verify()
    assert ok, f"audit chain invalid at {first_bad}"
    assert n_entries >= 2  # at least pipeline_start and pipeline_end


def test_audit_chain_detects_tamper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts")
    ledger = audit.DEFAULT_LEDGER

    # Tamper: rewrite one entry in place.
    lines = ledger.read_text().splitlines()
    assert len(lines) >= 2
    tampered = json.loads(lines[0])
    tampered["fields"]["out_dir"] = "/etc/evil"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")

    ok, _, first_bad = audit.verify()
    assert not ok
    assert first_bad is not None
