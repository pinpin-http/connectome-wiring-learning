"""The plastic GPU brain against the proven GPU brain, on a small random network."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFParams
from flymario.models.torch_lif import TorchLIF, gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs the GPU")

N, M = 400, 50


def _net(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((N, N)) < 0.08
    np.fill_diagonal(A, False)
    counts = 1.0 + np.floor(rng.lognormal(0.0, 1.0, (N, N)))
    sign = np.where(rng.random(N) < 0.7, 1.0, -1.0)
    W = sp.csr_matrix((A * counts * sign[None, :]).astype(np.float32))
    motor = np.sort(rng.choice(N, M, replace=False))
    cur = np.zeros(N, np.float32)
    cur[rng.choice(N, 60, replace=False)] = rng.uniform(0.1, 0.4, 60)
    return W, motor, cur


def _params():
    return LIFParams(gain=0.07, noise_std=0.02)


def _spikes(brain, cur, ticks):
    import torch
    c = torch.as_tensor(cur, device="cuda")
    return np.stack([brain.step(c).cpu().numpy() for _ in range(ticks)])


def test_matches_torchlif_at_initial_weights():
    from flymario.training.plastic import PlasticLIF
    W, motor, cur = _net()
    seeds = [1, 1, 2]
    ref = TorchLIF(W, _params(), batch=3, normalize=False, seeds=seeds, noise="torch")
    pl = PlasticLIF(W, motor, _params(), seeds=seeds, normalize=False)
    a, b = _spikes(ref, cur, 200), _spikes(pl, cur, 200)
    assert a.sum() > 1000
    assert np.array_equal(a, b)


def test_changed_weights_match_a_brain_built_from_the_changed_matrix():
    import torch
    from flymario.training.plastic import PlasticLIF
    W, motor, cur = _net(1)
    pl = PlasticLIF(W, motor, _params(), seeds=[3], normalize=False)
    rng = np.random.default_rng(5)
    new = pl.m0.cpu().numpy() * rng.uniform(0.0, 2.0, pl.P).astype(np.float32)
    pl.set_weights(torch.as_tensor(new[None, :], device="cuda"))
    W2 = W.tolil()
    Wm = sp.csr_matrix(W)[motor]
    Wm.sort_indices()
    rows = np.repeat(motor, np.diff(Wm.indptr))
    W2[rows, Wm.indices] = np.sign(Wm.data) * new
    ref = TorchLIF(sp.csr_matrix(W2), _params(), batch=1, normalize=False, seeds=[3], noise="torch")
    assert np.array_equal(_spikes(ref, cur, 150), _spikes(pl, cur, 150))


def test_bounds_hold():
    import torch
    from flymario.training.plastic import PlasticLIF
    W, motor, _ = _net(2)
    pl = PlasticLIF(W, motor, _params(), seeds=[1, 2], normalize=False)
    clip = pl.update(torch.randn((2, pl.P), device="cuda") * 100)
    assert clip["clipped_low"] > 0 and clip["clipped_high"] > 0
    assert float(pl.m.min()) >= 0.0
    assert bool((pl.m <= pl.hi[None, :]).all())


def test_checkpoint_resume_is_bit_identical():
    import torch
    from flymario.training.plastic import PlasticLIF
    W, motor, cur = _net(3)
    c = torch.as_tensor(cur, device="cuda")
    a = PlasticLIF(W, motor, _params(), seeds=[4, 5], normalize=False)
    for _ in range(40):
        a.step(c)
    snap = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in a.state_dict().items()}
    after_a = _spikes(a, cur, 60)
    b = PlasticLIF(W, motor, _params(), seeds=[9, 9], normalize=False)
    b.load_state_dict(snap)
    assert np.array_equal(after_a, _spikes(b, cur, 60))
    assert torch.equal(a.v, b.v)


def _fractional_brain(monkeypatch, mode, seeds=(4, 5, 6)):
    import torch
    from flymario.training.plastic import PlasticLIF
    monkeypatch.setenv("FLYMARIO_PLASTIC_PRODUCT", mode)
    W, motor, cur = _net(5)
    b = PlasticLIF(W, motor, _params(), seeds=list(seeds), normalize=False)
    g = torch.Generator(device=b.dev).manual_seed(1)
    b.set_weights(b.m * (1 + 0.3 * torch.rand(b.m.shape, device=b.dev, generator=g)))   # fractional
    return b, torch.as_tensor(cur, device="cuda")


@pytest.mark.parametrize("mode", ["spmm", "segment"])
def test_motor_input_paths_agree(monkeypatch, mode):
    """Both motor-input products (block SpMM; CUDA's segment reduction) give the same
    membrane up to summation order, from the same state (2026-10-09, NVIDIA deployment)."""
    import torch
    ref, c = _fractional_brain(monkeypatch, "spmm")
    other, _ = _fractional_brain(monkeypatch, mode)
    assert other.segment == (mode == "segment")
    for _ in range(5):
        ref.step(c)
    other.load_state_dict(ref.state_dict())
    ref.step(c)
    other.step(c)
    assert torch.allclose(ref.v, other.v, rtol=1e-5, atol=1e-4)


def test_auto_path_is_bit_repeatable_with_fractional_weights(monkeypatch):
    """The default path on this GPU repeats bit for bit once the magnitudes are
    fractional (NVIDIA's block SpMM did not: plastic.py docstring)."""
    import torch
    a, c = _fractional_brain(monkeypatch, "auto")
    b, _ = _fractional_brain(monkeypatch, "auto")
    for _ in range(30):
        a.step(c, observe=True)
        b.step(c, observe=True)
    assert torch.equal(a.v, b.v)
