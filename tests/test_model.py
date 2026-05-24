"""Unit tests for `src/multiqc_gate/model.py`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from multiqc_gate.features import N_FEATURES  # noqa: E402
from multiqc_gate.labels import LABELS  # noqa: E402
from multiqc_gate.model import MultiQCGateMLP  # noqa: E402


def test_mlp_forward_shape():
    model = MultiQCGateMLP()
    x = torch.zeros(4, N_FEATURES)
    y = model(x)
    assert y.shape == (4, len(LABELS))


def test_mlp_parameter_count_within_budget():
    """The README claim is ~2-3k params; verify within 5k as guard rail."""
    model = MultiQCGateMLP()
    n = model.n_parameters()
    assert 500 <= n <= 5000, f"unexpected MLP size: {n} params"


def test_mlp_gradient_flows():
    """A backward pass through the model should produce gradients."""
    model = MultiQCGateMLP()
    x = torch.randn(2, N_FEATURES)
    y = torch.tensor([0, 1], dtype=torch.long)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g.abs() > 0).any() for g in grads)


def test_mlp_handles_single_sample():
    """No-batch edge case (single-sample inference)."""
    model = MultiQCGateMLP()
    x = torch.zeros(1, N_FEATURES)
    y = model(x)
    assert y.shape == (1, len(LABELS))
