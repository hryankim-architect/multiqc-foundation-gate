"""End-to-end pipeline entry point.

This module provides the pipeline entry-point pattern used across repos. Each repo
replaces the body of ``run_pipeline`` with the actual bioinformatics work
(e.g. P3's VCF→HRD score, P1's Nextflow orchestration, P2's QC classifier,
P4's IHC + genomics calibration), but keeps the surrounding shape::

    audit_start  →  tracking_start  →  body  →  tracking_end  →  audit_end

The body must be deterministic enough that the canary smoke test exercises
the same code path with a fixture input.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from multiqc_gate import audit, tracking


def _run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{name}-{stamp}"


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


DOWNSAMPLE_READS = 100_000  # FASTQ records kept per file (see data/manifest.yaml)


def _gzip_deterministic(payload: bytes, dest: Path) -> None:
    """Write *payload* to *dest* as gzip with a zeroed mtime, so the bytes — and
    therefore the sha256 — are reproducible across machines and runs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # filename="" suppresses the gzip FNAME header field, so the bytes are
    # independent of the destination path (otherwise the basename leaks in).
    with open(dest, "wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6
    ) as gz:
        gz.write(payload)


def _fetch_fastq_downsampled(url: str, dest: Path, n_reads: int) -> None:
    """Stream a gzipped FASTQ from *url*, keep the first *n_reads* records, and
    write a reproducibly-gzipped FASTQ to *dest*. The transfer stops as soon as
    enough reads are decoded, so only a few MB cross the wire even when the
    source run is several GB."""
    want = n_reads * 4  # 4 lines per FASTQ record
    out: list[bytes] = []
    with urllib.request.urlopen(url, timeout=60) as resp, gzip.GzipFile(
        fileobj=resp, mode="rb"
    ) as gz:
        for line in gz:
            out.append(line)
            if len(out) >= want:
                break
    _gzip_deterministic(b"".join(out), dest)


def _https_for_ena(url: str) -> str:
    """ENA serves the same files over HTTPS; prefer it (FTP is slow and often
    firewall-blocked). The bytes are identical, so the sha256 is unchanged."""
    if url.startswith("ftp://ftp.sra.ebi.ac.uk/"):
        return "https://" + url[len("ftp://"):]
    return url


def fetch_manifest(
    manifest_path: Path, out_dir: Path, downsample_reads: int = DOWNSAMPLE_READS
) -> dict[str, Any]:
    """Download every entry in the manifest; verify SHA-256 checksums.

    FASTQ entries (``*.fastq.gz`` / ``*.fq.gz``) are streamed and downsampled to
    the first ``downsample_reads`` records (pass ``downsample_reads=0`` to keep
    the full file). The recorded sha256 is of the resulting on-disk file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results: list[dict[str, Any]] = []
    inputs = manifest.get("inputs", [])
    for n, entry in enumerate(inputs, 1):
        url = entry["url"]
        rel = entry["path"]
        expected = entry.get("sha256")
        size_mb = entry.get("size_mb")
        # A 'TBD...' placeholder is not a real expectation.
        want = expected if (expected and not str(expected).lower().startswith("tbd")) else None
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and want and _checksum(dest) == want:
            print(f"[{n}/{len(inputs)}] cached  {rel}", flush=True)
            results.append({"rel": rel, "path": str(dest), "status": "cached"})
            continue

        is_fastq = bool(downsample_reads) and rel.lower().endswith((".fastq.gz", ".fq.gz"))
        print(f"[{n}/{len(inputs)}] fetch   {rel} …", flush=True)
        try:
            src = _https_for_ena(url)
            if is_fastq:
                _fetch_fastq_downsampled(src, dest, downsample_reads)
            else:
                urllib.request.urlretrieve(src, dest)
        except Exception as exc:  # noqa: BLE001 — keep going on a single bad URL
            print(f"          ERROR {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            results.append({"rel": rel, "path": str(dest), "status": "error", "error": str(exc)})
            continue

        actual = _checksum(dest)
        if want and actual != want:
            print(f"          MISMATCH expected {want[:12]}… got {actual[:12]}…", flush=True)
            results.append({
                "rel": rel, "path": str(dest), "status": "checksum_mismatch",
                "expected": want, "actual": actual,
            })
            continue
        print(f"          ok {dest.stat().st_size / 1e6:.1f} MB  sha {actual[:12]}…", flush=True)
        results.append({
            "rel": rel, "path": str(dest), "status": "downloaded",
            "sha256": actual, "size_mb": size_mb,
        })

    return {"inputs": results}


def write_manifest_checksums(manifest_path: Path, result: dict[str, Any]) -> int:
    """Write freshly-computed sha256 values back into the manifest, matching
    entries by ``path``. Comment-preserving (line-based). Returns count filled."""
    by_rel = {
        r["rel"]: r["sha256"]
        for r in result.get("inputs", [])
        if r.get("status") == "downloaded" and r.get("rel") and r.get("sha256")
    }
    if not by_rel:
        return 0
    lines = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    url_re = re.compile(r"^\s*-\s*url:")
    path_re = re.compile(r"^\s*path:\s*(.+?)\s*$")
    sha_re = re.compile(r"^(\s*)sha256:\s*(.+?)\s*$")
    starts = [
        i for i, line in enumerate(lines)
        if url_re.match(line) and not line.lstrip().startswith("#")
    ]
    filled = 0
    for k, si in enumerate(starts):
        ei = starts[k + 1] if k + 1 < len(starts) else len(lines)
        rel = sha_i = None
        indent = ""
        for j in range(si, ei):
            if lines[j].lstrip().startswith("#"):
                continue
            mp, ms = path_re.match(lines[j]), sha_re.match(lines[j])
            if mp and rel is None:
                rel = mp.group(1).strip().strip('"').strip("'")
            if ms and sha_i is None:
                sha_i, indent = j, ms.group(1)
        if rel in by_rel and sha_i is not None:
            lines[sha_i] = f"{indent}sha256: {by_rel[rel]}\n"
            filled += 1
    if filled:
        manifest_path.write_text("".join(lines), encoding="utf-8")
    return filled


def run_pipeline(run_name: str, out_dir: Path) -> dict[str, Any]:
    """Run the MultiQC-gate classifier end-to-end.

    Wraps the substrate's audit + MLflow bracket around the P2 capability
    body:

        load features.npy
        load labels.csv
        train MLP (5-fold CV) + sklearn baseline (RF + LR) on the same folds
        aggregate per-fold metrics + confusion matrix + classification report
        drift detection (include vs everything-else baseline)
        save artifacts + log metrics

    Returns a dict with run-level summary that the canary smoke test can
    sanity-check. Substrate-aware: if MLflow / AUDIT_HOST are unset, the
    run degrades to local-only audit and silent tracking.
    """
    # Lazy imports so `multiqc_gate.pipeline` is importable even if torch
    # is not installed (e.g. in a CI job that only runs the scaffold tests).
    import numpy as np

    from multiqc_gate import baseline as baseline_mod
    from multiqc_gate import drift as drift_mod
    from multiqc_gate import eval as eval_mod
    from multiqc_gate import labels as labels_mod
    from multiqc_gate import train as train_mod

    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = _run_id(run_name)
    repo_root = Path.cwd()

    audit.emit(
        action="pipeline_start",
        job_id=job_id,
        fields={"out_dir": str(out_dir), "repo_root": str(repo_root)},
    )

    metrics: dict[str, float] = {}
    status = "unknown"

    try:
        with tracking.run(name=job_id, experiment="multiqc_gate"):
            tracking.log_params({"run_name": run_name})

            # ---- 1. Load features + labels --------------------------------
            features_path = repo_root / "data" / "features.npy"
            labels_path = repo_root / "data" / "labels.csv"
            sample_ids_path = repo_root / "data" / "feature_sample_ids.txt"

            if not features_path.exists():
                raise FileNotFoundError(
                    f"{features_path} missing. Run scripts/run_phase_c.py + "
                    f"feature extraction first."
                )
            X = np.load(features_path)
            sample_ids = sample_ids_path.read_text().splitlines() if sample_ids_path.exists() else []
            labels_df = labels_mod.load_labels(labels_path)

            # Align labels to features by sample_id order.
            label_map = dict(
                zip(labels_df["report_id"], labels_df["label"], strict=True)
            )
            y_strings = [label_map[sid] for sid in sample_ids]
            y = train_mod.labels_to_indices(y_strings)

            metrics["n_samples"] = float(X.shape[0])
            metrics["n_features"] = float(X.shape[1])
            audit.emit(
                action="dataset_loaded",
                job_id=job_id,
                fields={
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "class_distribution": {
                        labels_mod.LABELS[i]: int((y == i).sum())
                        for i in range(len(labels_mod.LABELS))
                    },
                },
            )

            # ---- 2. Train MLP (5-fold CV) ---------------------------------
            config = train_mod.TrainConfig(seed=42)
            tracking.log_params({
                "hidden_dims": str(config.hidden_dims),
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "max_epochs": config.max_epochs,
                "n_splits": 5,
            })
            mlp_results = train_mod.train_cv(X, y, config, job_id=job_id, n_splits=5)
            mlp_summary = eval_mod.aggregate_folds(
                "mlp",
                fold_y_true=[r.holdout_y_true for r in mlp_results],
                fold_y_pred=[r.holdout_y_pred for r in mlp_results],
            )
            eval_mod.write_summary_json(mlp_summary, out_dir / "eval_mlp.json")
            metrics["mlp_accuracy_mean"] = mlp_summary.accuracy_mean
            metrics["mlp_accuracy_std"] = mlp_summary.accuracy_std
            metrics["mlp_f1_macro_mean"] = mlp_summary.f1_macro_mean
            tracking.log_metric("mlp_accuracy_mean", mlp_summary.accuracy_mean)
            tracking.log_metric("mlp_f1_macro_mean", mlp_summary.f1_macro_mean)

            # ---- 3. Sklearn baselines on same folds -----------------------
            baseline_results = baseline_mod.run_all_baselines(X, y, job_id=job_id, n_splits=5, seed=42)
            baseline_summaries: list[eval_mod.EvalSummary] = []
            for method_name, fold_results in baseline_results.items():
                summary = eval_mod.aggregate_folds(
                    method_name,
                    fold_y_true=[fr.holdout_y_true for fr in fold_results],
                    fold_y_pred=[fr.holdout_y_pred for fr in fold_results],
                )
                baseline_summaries.append(summary)
                eval_mod.write_summary_json(summary, out_dir / f"eval_{method_name}.json")
                metrics[f"{method_name}_accuracy_mean"] = summary.accuracy_mean
                metrics[f"{method_name}_f1_macro_mean"] = summary.f1_macro_mean
                tracking.log_metric(f"{method_name}_accuracy_mean", summary.accuracy_mean)
                tracking.log_metric(f"{method_name}_f1_macro_mean", summary.f1_macro_mean)

            # ---- 4. Comparison table -------------------------------------
            comparison = eval_mod.compare_methods([mlp_summary] + baseline_summaries)
            with (out_dir / "comparison.json").open("w", encoding="utf-8") as fh:
                json.dump(comparison, fh, indent=2, sort_keys=True)
            audit.emit(
                action="comparison_table",
                job_id=job_id,
                fields={"comparison": comparison},
            )

            # ---- 5. Drift detection ---------------------------------------
            # Reference = include-class baseline (clean samples).
            # Test     = exclude + manual-review (augmented samples).
            include_mask = y == labels_mod.label_to_idx("include")
            reference = X[include_mask]
            test = X[~include_mask]
            drift_results = drift_mod.detect_drift(reference, test, alpha=0.05)
            drift_summary = drift_mod.summarize_drift(drift_results, alpha=0.05)
            with (out_dir / "drift.json").open("w", encoding="utf-8") as fh:
                json.dump(drift_summary, fh, indent=2, sort_keys=True)
            metrics["drift_n_drifted"] = float(drift_summary["n_drifted"])
            metrics["drift_fraction"] = float(drift_summary["fraction_drifted"])
            audit.emit(
                action="drift_summary",
                job_id=job_id,
                fields=drift_summary,
            )
            tracking.log_metric("drift_n_drifted", drift_summary["n_drifted"])

            tracking.log_metrics(metrics)
            status = "success"
    except Exception as exc:  # noqa: BLE001 — record & re-raise via finally below
        status = "failed"
        audit.emit(
            action="pipeline_error",
            job_id=job_id,
            fields={"error": repr(exc)},
        )
        raise
    finally:
        audit.emit(
            action="pipeline_end",
            job_id=job_id,
            fields={"status": status, "metrics": metrics},
        )

    return {
        "job_id": job_id,
        "status": status,
        "metrics": metrics,
        "out_dir": str(out_dir),
    }


@click.group()
def cli() -> None:
    """multiqc_gate demonstration pipeline."""


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
@click.option(
    "--downsample-reads",
    type=int,
    default=DOWNSAMPLE_READS,
    help="FASTQ records to keep per file (0 = full file).",
)
@click.option(
    "--write-checksums",
    is_flag=True,
    help="Write the computed sha256 values back into the manifest.",
)
def fetch(manifest: Path, out: Path, downsample_reads: int, write_checksums: bool) -> None:
    """Download public inputs declared in the manifest."""
    result = fetch_manifest(manifest, out, downsample_reads=downsample_reads)
    if write_checksums:
        result["checksums_written"] = write_manifest_checksums(manifest, result)
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
