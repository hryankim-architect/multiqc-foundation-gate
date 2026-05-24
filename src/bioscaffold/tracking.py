"""MLflow tracking wrapper with safe fallback.

If ``MLFLOW_TRACKING_URI`` is not set, the wrapper becomes a no-op so the
demo still runs on a fresh checkout without an MLflow server. When the
substrate is present, runs are tracked against the configured URI.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

_MLFLOW = None


def _mlflow():
    global _MLFLOW
    if _MLFLOW is None:
        try:
            import mlflow  # type: ignore

            _MLFLOW = mlflow
        except ImportError:
            _MLFLOW = False
    return _MLFLOW or None


def is_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI")) and _mlflow() is not None


@contextlib.contextmanager
def run(name: str, *, experiment: str | None = None) -> Iterator[Any]:
    """Context manager around an MLflow run. Yields the run object or None."""
    if not is_enabled():
        yield None
        return

    mlflow = _mlflow()
    if experiment:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as active:
        yield active


def log_params(params: dict[str, Any]) -> None:
    if not is_enabled():
        return
    _mlflow().log_params(params)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    if not is_enabled():
        return
    _mlflow().log_metric(key, value, step=step)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not is_enabled():
        return
    _mlflow().log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    if not is_enabled():
        return
    _mlflow().log_artifact(path)
