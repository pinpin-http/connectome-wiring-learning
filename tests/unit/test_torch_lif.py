"""The GPU brain is the CPU brain: same equation, same noise, same spikes.

Small random networks, so these run in seconds; the full-connectome version is
scripts/diagnostics/gpu_equivalence.py. Skipped where no GPU is visible.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFBrain, LIFParams
from flymario.models.torch_lif import TorchLIF, gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="no GPU visible to torch")


def net(n=400, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    W = sp.random(n, n, density=density, format="csc", dtype=np.float32, random_state=rng)
    W.data *= rng.choice([-1.0, 1.0], size=W.nnz).astype(np.float32)
    return W


def test_same_trajectory_as_cpu_with_the_cpu_noise():
    W = net()
    p = LIFParams(gain=6.0, noise_std=0.05, seed=3)
    cpu = LIFBrain(W, p, normalize=True)
    gpu = TorchLIF(W, p, batch=1, normalize=True, seeds=[3], noise="numpy")
    drive = np.random.default_rng(1).uniform(0, 0.4, W.shape[0]).astype(np.float32)
    fired = 0
    for _ in range(200):
        a = cpu.step(drive)
        b = gpu.step(drive)[:, 0].cpu().numpy()
        assert np.array_equal(a, b)
        fired += a.sum()
    assert fired > 0, "a silent network would make this test vacuous"


def test_a_batch_is_the_same_brains_run_one_by_one():
    W = net(seed=4)
    p = LIFParams(gain=6.0, noise_std=0.05)
    seeds = [11, 12, 13]
    drive = np.random.default_rng(2).uniform(0, 0.4, W.shape[0]).astype(np.float32)
    batch = TorchLIF(W, p, batch=3, normalize=True, seeds=seeds, noise="numpy")
    singles = [LIFBrain(W, LIFParams(gain=6.0, noise_std=0.05, seed=s), normalize=True) for s in seeds]
    for _ in range(100):
        sb = batch.step(drive).cpu().numpy()
        for j, b in enumerate(singles):
            assert np.array_equal(sb[:, j], b.step(drive))


def test_bit_for_bit_repeatable():
    W = net(seed=5)
    p = LIFParams(gain=6.0, noise_std=0.05)
    runs = []
    for _ in range(2):
        b = TorchLIF(W, p, batch=4, normalize=True, seeds=[1, 2, 3, 4], noise="torch")
        runs.append(np.stack([b.step(0.3).cpu().numpy() for _ in range(80)]))
    assert np.array_equal(runs[0], runs[1])


def test_skipping_silent_brains_changes_nothing():
    # brain 0 is driven, brain 1 gets nothing and stays silent: the skip path
    # (SpMM on a column subset) must give brain 0 exactly its solo trajectory
    W = net(seed=6)
    p = LIFParams(gain=6.0, noise_std=0.0)
    n = W.shape[0]
    drive = np.zeros((n, 2), np.float32)
    drive[:, 0] = np.random.default_rng(3).uniform(0, 0.5, n)
    pair = TorchLIF(W, p, batch=2, normalize=True, seeds=[0, 1], noise="numpy")
    solo = LIFBrain(W, LIFParams(gain=6.0, noise_std=0.0), normalize=True)
    for _ in range(100):
        s = pair.step(drive).cpu().numpy()
        assert not s[:, 1].any()
        assert np.array_equal(s[:, 0], solo.step(drive[:, 0]))
