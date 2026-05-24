"""Training loop for the MultiQC gate MLP with per-epoch substrate hooks.

The trainer is deterministic (seeded random.RandomState + torch.Generator)
so the demo is byte-reproducible on `make run`. Each epoch emits one
audit entry and one MLflow metric for train / val loss / val accuracy,
giving the substrate a hash-chained ledger of the entire training run.

Class weighting compensates for the label imbalance in the canonical
50-sample dataset (10 include / 20 exclude / 20 manual-review).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from multiqc_gate import audit, tracking
from multiqc_gate.labels import LABELS, label_to_idx
from multiqc_gate.model import MultiQCGateMLP


@dataclass
class TrainConfig:
    """Hyperparameters for one MLP training run."""

    hidden_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 50
    patience: int = 10  # early-stopping window on val loss
    device: str = "auto"  # "auto" picks MPS if available, else CPU
    seed: int = 42


@dataclass
class FoldResult:
    """Per-fold training output."""

    fold: int
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    holdout_y_true: list[int]
    holdout_y_pred: list[int]
    holdout_y_proba: list[list[float]]  # per-class probabilities
    epoch_history: list[dict[str, float]] = field(default_factory=list)


def pick_device(requested: str) -> str:
    """Resolve 'auto' -> 'mps' if available, else 'cpu'."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div by zero
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    fold: int,
    job_id: str,
) -> FoldResult:
    """Train one MLP on one fold; return best-epoch metrics + val predictions."""
    _set_seed(config.seed + fold)  # per-fold seed for diversity
    device = pick_device(config.device)

    model = MultiQCGateMLP(
        n_features=X_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    weights = _class_weights(y_train, n_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(config.seed + fold),
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    epoch_history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss /= max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        epoch_history.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        audit.emit(
            action="epoch_end",
            job_id=job_id,
            fields={
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
        )
        tracking.log_metric(f"fold{fold}_train_loss", train_loss, step=epoch)
        tracking.log_metric(f"fold{fold}_val_loss", val_loss, step=epoch)
        tracking.log_metric(f"fold{fold}_val_acc", val_acc, step=epoch)

        # Early stopping on val loss.
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Restore best weights and produce holdout predictions.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_proba = torch.softmax(val_logits, dim=1).cpu().numpy().tolist()
        val_pred = val_logits.argmax(dim=1).cpu().numpy().tolist()

    return FoldResult(
        fold=fold,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        holdout_y_true=y_val.tolist(),
        holdout_y_pred=val_pred,
        holdout_y_proba=val_proba,
        epoch_history=epoch_history,
    )


def train_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    job_id: str,
    n_splits: int = 5,
) -> list[FoldResult]:
    """Stratified K-fold training. Returns one FoldResult per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    results: list[FoldResult] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        audit.emit(
            action="fold_start",
            job_id=job_id,
            fields={
                "fold": fold,
                "n_train": int(train_idx.shape[0]),
                "n_val": int(val_idx.shape[0]),
            },
        )
        t0 = time.time()
        fr = train_one_fold(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            config=config,
            fold=fold,
            job_id=job_id,
        )
        elapsed = time.time() - t0
        audit.emit(
            action="fold_end",
            job_id=job_id,
            fields={
                "fold": fold,
                "best_epoch": fr.best_epoch,
                "best_val_loss": fr.best_val_loss,
                "best_val_acc": fr.best_val_acc,
                "elapsed_s": elapsed,
            },
        )
        results.append(fr)
    return results


def labels_to_indices(label_strings: list[str]) -> np.ndarray:
    """Vectorised label-string -> integer-index mapping."""
    return np.array([label_to_idx(s) for s in label_strings], dtype=np.int64)
