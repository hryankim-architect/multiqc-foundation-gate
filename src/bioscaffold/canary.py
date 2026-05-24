"""Deterministic canary smoke test.

Every capability-portrait repo exposes a canary entry point that the
Polish-Phase5 ``lab_semantic_check.py`` runner can probe daily. The canary
must:

1. Complete in under 30 seconds on a single workstation.
2. Produce deterministic output given a fixed fixture input.
3. Return an exit code of 0 on success, non-zero on any deviation.

The default fixture exercises the audit emit and tracking modules without
any external services. Override the fixture path via the
``BIOSCAFFOLD_CANARY_FIXTURE`` environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from bioscaffold import audit, tracking

DEFAULT_FIXTURE = Path("tests/fixtures/canary.json")

EXPECTED_KEYS = {"name", "tier", "expected_metric"}


def _load_fixture(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"canary fixture not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def check() -> dict[str, Any]:
    """Run the canary. Returns a structured result dict."""
    fixture_path = Path(
        os.environ.get("BIOSCAFFOLD_CANARY_FIXTURE", str(DEFAULT_FIXTURE))
    )
    fixture = _load_fixture(fixture_path)

    missing = EXPECTED_KEYS - set(fixture.keys())
    if missing:
        return {
            "ok": False,
            "reason": f"fixture missing keys: {sorted(missing)}",
        }

    job_id = f"canary-{fixture['name']}"
    audit.emit(
        action="canary_start",
        job_id=job_id,
        fields={"tier": fixture["tier"], "fixture_path": str(fixture_path)},
    )

    with tracking.run(name=job_id, experiment="canary"):
        tracking.log_params({"tier": fixture["tier"]})
        tracking.log_metric("expected_metric", float(fixture["expected_metric"]))

    audit.emit(
        action="canary_end",
        job_id=job_id,
        fields={"ok": True},
    )

    return {
        "ok": True,
        "job_id": job_id,
        "fixture": fixture,
    }


def main() -> int:
    result = check()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
