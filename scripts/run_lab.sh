#!/usr/bin/env bash
# Execute the capability-portrait pipeline on a Polish-Phase5 lab node.
#
# This wraps `make run` with the substrate environment variables set to the
# lab defaults, so audit entries flow to chi-mac-m:8081 and MLflow runs are
# tracked at chi-mac-m:5050. On a fresh checkout without the substrate
# present, run `make run` directly instead.

set -euo pipefail

# --- substrate env (override per host if needed) ---
export AUDIT_HOST="${AUDIT_HOST:-chi-mac-m:8081}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://chi-mac-m:5050}"
export BIOSCAFFOLD_RUN_NAME="${BIOSCAFFOLD_RUN_NAME:-lab-$(date -u +%Y%m%d-%H%M%S)}"

# --- sanity check ---
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH" >&2
    exit 2
fi

# --- run ---
echo "[run_lab] AUDIT_HOST=${AUDIT_HOST}"
echo "[run_lab] MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "[run_lab] RUN_NAME=${BIOSCAFFOLD_RUN_NAME}"

uv run make run RUN_NAME="${BIOSCAFFOLD_RUN_NAME}"

# --- post-run: invoke canary for substrate registration ---
echo "[run_lab] canary check"
uv run python -m bioscaffold.canary

echo "[run_lab] done"
