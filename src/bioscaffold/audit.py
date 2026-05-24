"""Hash-chained NDJSON audit emit.

A minimal substrate hook. Every pipeline run should emit at least one entry
to a local NDJSON ledger; if ``AUDIT_HOST`` is set, the entry is also POSTed
to the substrate audit-API at ``http://${AUDIT_HOST}/events``.

The chain is intentionally simple: each entry's ``prev_hash`` field is the
SHA-256 of the previous entry's canonical JSON encoding. Tampering is
detectable by replaying the chain.

The schema mirrors the Polish-Phase5 audit format::

    {
        "ts": "2026-05-23T17:00:00Z",
        "action": "pipeline_start",
        "actor": "bioscaffold@chi-mac-p",
        "job_id": "demo-2026-05-23-17",
        "fields": {...arbitrary payload...},
        "prev_hash": "...",
    }
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_LEDGER = Path("audit/local-demo.ndjson")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _actor() -> str:
    user = os.environ.get("USER", "anon")
    host = socket.gethostname()
    return f"{user}@{host}"


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _prev_hash(ledger_path: Path) -> str:
    if not ledger_path.exists():
        return "0" * 64
    last = ""
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                last = stripped
    if not last:
        return "0" * 64
    return hashlib.sha256(last.encode("utf-8")).hexdigest()


def emit(
    action: str,
    job_id: str,
    fields: dict[str, Any] | None = None,
    *,
    ledger_path: Path | None = None,
    post: bool | None = None,
) -> dict[str, Any]:
    """Append one audit entry. Returns the canonical entry (with prev_hash filled in).

    Args:
        action: short verb identifying what happened.
        job_id: stable identifier for the pipeline run.
        fields: arbitrary serializable payload.
        ledger_path: override the local NDJSON path (defaults to
            ``audit/local-demo.ndjson``).
        post: force-enable or force-disable posting to ``AUDIT_HOST``. By
            default, posting happens if ``AUDIT_HOST`` is set.
    """
    ledger_path = ledger_path or DEFAULT_LEDGER
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": _now_iso(),
        "action": action,
        "actor": _actor(),
        "job_id": job_id,
        "fields": fields or {},
        "prev_hash": _prev_hash(ledger_path),
    }

    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(_canonical(entry).decode("utf-8"))
        fh.write("\n")

    audit_host = os.environ.get("AUDIT_HOST")
    should_post = post if post is not None else bool(audit_host)
    if should_post and audit_host:
        # Substrate down is not pipeline-fatal. The local NDJSON is the source
        # of truth; the remote post is best-effort.
        with contextlib.suppress(requests.RequestException):
            requests.post(
                f"http://{audit_host}/events",
                json=entry,
                timeout=2.0,
            )

    return entry


def verify(ledger_path: Path | None = None) -> tuple[bool, int, str | None]:
    """Replay the chain. Returns (ok, n_entries, first_bad_ts)."""
    ledger_path = ledger_path or DEFAULT_LEDGER
    if not ledger_path.exists():
        return True, 0, None

    prev = "0" * 64
    n = 0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                return False, n, "malformed"
            if entry.get("prev_hash") != prev:
                return False, n, entry.get("ts", "unknown")
            prev = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
            n += 1

    return True, n, None
