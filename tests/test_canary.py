"""Canary smoke test invariants."""

from __future__ import annotations

from pathlib import Path

from bioscaffold import canary


def test_canary_passes_on_default_fixture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Copy fixture into tmp_path
    fixture_src = Path(__file__).parent / "fixtures" / "canary.json"
    fixture_dst = tmp_path / "tests" / "fixtures" / "canary.json"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    fixture_dst.write_text(fixture_src.read_text())

    monkeypatch.setenv("BIOSCAFFOLD_CANARY_FIXTURE", str(fixture_dst))

    result = canary.check()
    assert result["ok"], result
    assert result["fixture"]["tier"] == "tier-1"


def test_canary_fails_on_missing_keys(tmp_path: Path, monkeypatch) -> None:
    fixture_path = tmp_path / "bad.json"
    fixture_path.write_text('{"name": "x"}')
    monkeypatch.setenv("BIOSCAFFOLD_CANARY_FIXTURE", str(fixture_path))
    monkeypatch.chdir(tmp_path)

    result = canary.check()
    assert not result["ok"]
    assert "missing keys" in result["reason"]
