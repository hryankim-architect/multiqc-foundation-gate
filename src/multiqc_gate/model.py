"""Tiny MLP classifier for the MultiQC gate.

Maps a 28-dim feature vector (from `features.py`) to one of 3 class logits
(include / exclude / manual-review). Architecture is intentionally small
(~2-3k parameters) so an n=50 training set does not immediately overfit.

Design notes:
    - Two hidden layers (32 -> 16) with ReLU + dropout 0.3 give enough
      non-linearity to learn the augmentation patterns
      (adapter / quality / dropout / mixed) without enough capacity to
      memorize the 50 training examples.
    - LayerNorm before the hidden layers stabilises training on the
      heterogeneous feature scale (Total Sequences in tens of thousands,
      status codes in [-1, 2], presence flags in {0, 1}).
    - The model is device-agnostic; pipeline.py picks MPS / CPU at
      training time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multiqc_gate.features import N_FEATURES
from multiqc_gate.labels import LABELS

N_CLASSES = len(LABELS)  # 3


class MultiQCGateMLP(nn.Module):
    """Small MLP for MultiQC report -> gate decision classification."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dims: tuple[int, ...] = (32, 16),
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout

        # LayerNorm on the input to handle heterogeneous feature scales
        # (Total Sequences ~1e5, status codes ~[-1, 2], presence ~[0, 1]).
        layers: list[nn.Module] = [nn.LayerNorm(n_features)]

        in_dim = n_features
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hdim

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, n_classes)."""
        return self.net(x)

    def n_parameters(self) -> int:
        """Total trainable-parameter count (for the README climax table)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
