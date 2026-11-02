"""Direction diagnostics preserve the forward model and perturb only allowed weights."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from flymario.models.spiking_reservoir import LIFParams
from flymario.training.bptt import SurrogateBPTT

_spec = importlib.util.spec_from_file_location(
    'direction', Path(__file__).resolve().parents[2] / 'scripts/gates/gate_g_bptt_direction.py')
direction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(direction)


def test_relative_descent_and_projection():
    m0 = torch.tensor([1., 10., 100.])
    gradient = torch.tensor([3., -1., 0.02])
    d, _ = direction.relative_direction(m0, gradient)
    weights, record = direction.propose_weights(m0, -d, 0.001)
    assert float(gradient @ (weights - m0)) < 0
    assert record['realized_relative_rms'] == pytest.approx(0.001, rel=1e-4)
    clipped, _ = direction.propose_weights(m0, -d, 10)
    assert bool(((clipped >= 0) & (clipped <= 2*m0)).all())
    with pytest.raises(FloatingPointError):
        direction.relative_direction(m0, torch.zeros(3))


def test_surrogate_variants_change_backward_only():
    import scipy.sparse as sp
    torch.set_num_threads(1)
    W = sp.csr_matrix(np.array([[0, .5], [.7, 0]], dtype=np.float32))
    models = [SurrogateBPTT(W, np.array([1]), np.array([[.4, .6], [.6, .4]], np.float32),
                           np.array([[1.], [-1.]], np.float32),
                           LIFParams(gain=.5, noise_std=.02), seed=3, device='cpu', **variant)
              for variant in direction.VARIANTS.values()]
    noise = torch.randn(2, 8, 2, 1, generator=torch.Generator().manual_seed(12)) * .02
    metrics, grads = [], []
    for model in models:
        original = tuple(x.clone() for x in model.state)
        loss, values = direction.replay(model, model.state, model.m, [0, 1], [0, 1],
                                        noise, differentiate=True)
        grads.append(torch.autograd.grad(loss, model.m)[0])
        metrics.append(values)
        assert all(torch.equal(a,b) for a,b in zip(original, model.state))
    assert metrics[0] == metrics[1]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert not torch.equal(grads[0], grads[1])
