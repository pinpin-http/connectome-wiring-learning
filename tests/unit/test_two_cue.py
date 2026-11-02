"""The two-cue learner: learning changes weights, and a resumed run is bit-identical."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flymario.models.torch_lif import gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs the GPU")

N, M = 400, 50


def _setup():
    from flymario.training.two_cue import Learner
    rng = np.random.default_rng(0)
    A = rng.random((N, N)) < 0.08
    np.fill_diagonal(A, False)
    counts = 1.0 + np.floor(rng.lognormal(0.0, 1.0, (N, N)))
    sign = np.where(rng.random(N) < 0.7, 1.0, -1.0)
    W = sp.csr_matrix((A * counts * sign[None, :]).astype(np.float32))
    motor = np.sort(rng.choice(N, M, replace=False))
    cues = np.zeros((2, N), np.float32)
    idx = rng.choice(N, 60, replace=False)
    cues[:, idx] = rng.uniform(0.0, 0.3, (2, 60))
    learners = [Learner("np", 1, "node_perturbation", 0.05, "true", 0, 11),
                Learner("sr", 1, "spike_reinforce", 0.05, "true", 1, 12),
                Learner("shuf", 2, "node_perturbation", 0.05, "shuffled", 0, 11),
                Learner("eta0", 2, None, 0.0, "true", 0, 11)]
    return W, motor, cues, learners


def _make(W, motor, cues, learners):
    from flymario.training.two_cue import TwoCue
    return TwoCue(W, motor, cues, learners, "train", 40, record=True, gain=0.07)


def test_learning_moves_weights_and_controls_do_not():
    W, motor, cues, learners = _setup()
    run = _make(W, motor, cues, learners)
    m0 = run.brain.m.clone()
    run.run(20)
    moved = (run.brain.m - m0).abs().sum(1).cpu().numpy()
    assert moved[0] > 0 and moved[1] > 0 and moved[2] > 0
    assert moved[3] == 0.0
    assert run.log["rate_motor"][:20].mean() > 0


def test_resume_is_bit_identical():
    import torch
    W, motor, cues, learners = _setup()
    straight = _make(W, motor, cues, learners)
    straight.run(40)
    first = _make(W, motor, cues, learners)
    first.run(17)
    snap = first.state_dict()
    resumed = _make(W, motor, cues, learners)
    resumed.load_state_dict(snap)
    resumed.run(23)
    assert torch.equal(straight.brain.m, resumed.brain.m)
    assert torch.equal(straight.Rbar, resumed.Rbar)
    for k in ("reward", "correct", "margin", "update_rms"):
        assert np.array_equal(straight.log[k], resumed.log[k]), k
    assert np.array_equal(straight.features, resumed.features)
