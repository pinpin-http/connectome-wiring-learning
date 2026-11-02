"""The sweep-regime learner: history independence, exact eligibility, resume."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flymario.models.torch_lif import gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs the GPU")

N, M = 400, 50


def _net(cut_motor_outputs: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.random((N, N)) < 0.08
    np.fill_diagonal(A, False)
    counts = 1.0 + np.floor(rng.lognormal(0.0, 1.0, (N, N)))
    sign = np.where(rng.random(N) < 0.7, 1.0, -1.0)
    motor = np.sort(rng.choice(N, M, replace=False))
    if cut_motor_outputs:
        A[:, motor] = False                       # motor neurons project nowhere
    W = sp.csr_matrix((A * counts * sign[None, :]).astype(np.float32))
    cues = np.zeros((2, N), np.float32)
    idx = rng.choice(np.setdiff1d(np.arange(N), motor), 60, replace=False)
    cues[:, idx] = rng.uniform(0.2, 0.8, (2, 60))
    return W, motor, cues


def _learners():
    from flymario.training.sweep import SweepLearner
    return [SweepLearner("rule", 1, 0, 11, eta=0.05, reward="true"),
            SweepLearner("sup", 1, 1, 12, eta=0.05, reward="supervised"),
            SweepLearner("shuf", 2, 0, 11, eta=0.05, reward="shuffled"),
            SweepLearner("cent", 2, 1, 12, eta=0.05, reward="true", kappa=0.2),
            SweepLearner("frozen", 2, 0, 11, eta=0.0)]


def test_response_depends_on_the_cue_only():
    """With the noise off, every sweep of the same cue gives the same scores,
    whatever came before: nothing is carried across decisions."""
    from flymario.training.sweep import SweepTask
    W, motor, cues = _net()
    task = SweepTask(W, motor, cues, [_learners()[-1]], "explore", 30, gain=0.07,
                     noise_std=0.0)
    task.run(30, learn=False)
    d, c = task.log["d"][:, 0], task.cues[:30, 0].cpu().numpy()
    assert np.ptp(d[c == 0]) == 0 and np.ptp(d[c == 1]) == 0
    assert d[c == 0][0] != d[c == 1][0]
    assert task.log["rate_motor"].max() > 0


def test_eligibility_is_the_surrogate_gradient():
    """When motor neurons feed nothing back, sum_i u_i e_ij * gain * sign_ij is the
    autograd gradient of sum_i u_i c_i under the same surrogate and a detached reset."""
    import torch

    from flymario.training.sweep import SweepLearner, SweepTask
    W, motor, cues = _net(cut_motor_outputs=True)
    gain = 0.07
    task = SweepTask(W, motor, cues, [SweepLearner("x", 1, 0, 11, eta=1.0)], "explore", 4,
                     gain=gain, noise_std=0.0)
    cue = torch.zeros(1, dtype=torch.long, device=task.dev)
    c, e = task.sweep(cue, observe=True)
    u = torch.randn(M, device=task.dev)
    ours = (u.index_select(0, task.brain.post) * e[:, 0]) * gain * task.brain.sign

    p, b = task.params, task.brain
    w = task.rule.psi_width
    Wd = torch.as_tensor(W.toarray(), device=task.dev)
    keep = torch.ones(N, device=task.dev)
    keep[b.motor] = 0.0
    Wf = Wd * keep[:, None]                                  # frozen part (non-motor rows)
    m = b.m0.clone().requires_grad_(True)

    class Spike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0).float()

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g / (1.0 + x.abs() / w) ** 2

    Wm = torch.zeros((M, N), device=task.dev)
    Wm = Wm.index_put((b.post, b.pre), b.sign * m, accumulate=True)
    cur = task.cue_cur[:, 0]
    v = torch.zeros(N, device=task.dev)
    ref = torch.zeros(N, device=task.dev)
    s = torch.zeros(N, device=task.dev)
    count = torch.zeros(M, device=task.dev)
    for _ in range(task.K):
        rec = Wf @ s
        rec = rec.index_add(0, b.motor, Wm @ s)
        v = v * p.decay + gain * rec + cur
        free = (ref <= 0).float()
        s = Spike.apply(v - p.threshold) * free
        v = v * (1.0 - s.detach())
        ref = torch.clamp(ref - 1, min=0) * (1 - s.detach()) + p.refractory_steps * s.detach()
        count = count + s.index_select(0, b.motor)
    assert torch.equal(count, c[:, 0])                       # same forward
    (u * count).sum().backward()
    assert torch.allclose(m.grad, ours, rtol=1e-4, atol=1e-6)
    assert m.grad.abs().sum() > 0


def test_learners_move_and_the_frozen_control_does_not():
    from flymario.training.sweep import SweepTask
    W, motor, cues = _net()
    task = SweepTask(W, motor, cues, _learners(), "explore", 20, gain=0.07)
    m0 = task.weights()
    task.run(20)
    moved = (task.weights() - m0).abs().sum(1).cpu().numpy()
    assert (moved[:4] > 0).all() and moved[4] == 0.0
    b = task.brain
    assert bool((b.m >= 0).all()) and bool((b.m <= b.hi[None, :] + 1e-6).all())


def test_resume_is_bit_identical():
    import torch

    from flymario.training.sweep import RuleConfig, SweepTask
    W, motor, cues = _net()
    rule = RuleConfig(block=2, tau_adapt=0.5)
    make = lambda: SweepTask(W, motor, cues, _learners(), "explore", 24, rule=rule, gain=0.07)
    straight = make()
    straight.run(24)
    first = make()
    first.run(9)
    snap = first.state_dict()
    resumed = make()
    resumed.load_state_dict(snap)
    resumed.run(15)
    assert torch.equal(straight.brain.m, resumed.brain.m)
    for k in ("reward", "correct", "d", "update_rms"):
        assert np.array_equal(straight.log[k], resumed.log[k]), k


def test_delayed_reward_and_frame_pool_resume_bit_identical():
    """Delay > 0 (cross-decision trace) and per-trial frames: learners move only once
    rewards start arriving, frames stay in their pool, a resumed run is identical."""
    import torch

    from flymario.training.sweep import RuleConfig, SweepTask
    W, motor, cues = _net()
    rng = np.random.default_rng(3)
    frames = np.repeat(cues[:, None, :], 6, axis=1) * rng.uniform(0.7, 1.3, (2, 6, 1)).astype(np.float32)
    rule = RuleConfig(delay=2, trace_gamma=0.8)
    make = lambda: SweepTask(W, motor, cues, _learners(), "explore", 20, rule=rule, gain=0.07,
                             frames=frames, frame_pool=[1, 3, 4])
    t = make()
    assert set(t.fidx.cpu().numpy().ravel()) <= {1, 3, 4}
    m0 = t.weights()
    t.run(2)
    assert torch.equal(t.weights(), m0)                      # no reward has arrived yet
    t.run(18)
    moved = (t.weights() - m0).abs().sum(1).cpu().numpy()
    assert (moved[:4] > 0).all() and moved[4] == 0.0
    first = make()
    first.run(7)
    snap = first.state_dict()
    resumed = make()
    resumed.load_state_dict(snap)
    resumed.run(13)
    assert torch.equal(t.brain.m, resumed.brain.m)
    for k in ("reward", "correct", "d"):
        assert np.array_equal(t.log[k], resumed.log[k]), k
