"""End-to-end pipeline entry point.

This is the *pattern* that capability-portrait repos inherit. Each repo
replaces the body of ``run_pipeline`` with the actual bioinformatics work
(e.g. P3's VCF→HRD score, P1's Nextflow orchestration, P2's QC classifier,
P4's IHC + genomics calibration), but keeps the surrounding shape::

    audit_start  →  tracking_start  →  body  →  tracking_end  →  audit_end

The body must be deterministic enough that the canary smoke test exercises
the same code path with a fixture input.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from bioscaffold import audit, tracking


def _run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{name}-{stamp}"


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_manifest(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    """Download every entry in the manifest; verify SHA-256 checksums."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results: list[dict[str, Any]] = []
    for entry in manifest.get("inputs", []):
        url = entry["url"]
        rel = entry["path"]
        expected = entry.get("sha256")
        size_mb = entry.get("size_mb")
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and expected and _checksum(dest) == expected:
            results.append({"path": str(dest), "status": "cached"})
            continue

        urllib.request.urlretrieve(url, dest)
        actual = _checksum(dest)
        if expected and actual != expected:
            results.append({
                "path": str(dest),
                "status": "checksum_mismatch",
                "expected": expected,
                "actual": actual,
            })
            continue
        results.append({
            "path": str(dest),
            "status": "downloaded",
            "sha256": actual,
            "size_mb": size_mb,
        })

    return {"inputs": results}


def run_pipeline(run_name: str, out_dir: Path) -> dict[str, Any]:
    """Replace this body in your derived repo. The shape is the contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = _run_id(run_name)

    audit.emit(
        action="pipeline_start",
        job_id=job_id,
        fields={"out_dir": str(out_dir)},
    )

    metrics: dict[str, float] = {}

    with tracking.run(name=job_id, experiment="bioscaffold"):
        tracking.log_params({"run_name": run_name})

        # --- begin body (replace in derived repo) ---
        # Demo body: write a deterministic JSON artifact and emit one metric.
        t0 = time.time()
        artifact = {
            "run_name": run_name,
            "job_id": job_id,
            "message": "scaffold demo body",
        }
        artifact_path = out_dir / f"{run_name}.json"
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2, sort_keys=True)
        elapsed_ms = (time.time() - t0) * 1000.0
        metrics["body_elapsed_ms"] = elapsed_ms
        # --- end body ---

        tracking.log_metrics(metrics)

    audit.emit(
        action="pipeline_end",
        job_id=job_id,
        fields={"metrics": metrics, "artifact_path": str(artifact_path)},
    )

    return {
        "job_id": job_id,
        "metrics": metrics,
        "artifact_path": str(artifact_path),
    }


@click.group()
def cli() -> None:
    """bioscaffold capability-portrait pipeline."""


@cli.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/manifest.yaml"),
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data"),
)
def fetch(manifest: Path, out: Path) -> None:
    """Download public inputs declared in the manifest."""
    result = fetch_manifest(manifest, out)
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option("--name", default="demo")
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("artifacts"),
)
def run(name: str, out: Path) -> None:
    """Run the end-to-end pipeline."""
    result = run_pipeline(name, out)
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
