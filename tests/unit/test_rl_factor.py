"""The actor's score factor and the episode-end TD error (docs/RL_STUDY.md §3).

Exact expectations on a one-state bandit, no sampling and no GPU: what each factor
estimates when actions come from mu = (1 - eps) pi + eps / A."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from flymario.training.mario_sweep import SweepMarioSettings, end_delta, pg_weight

A, EPS = 8, 0.1


def _expected_update(z, Q, mode, eps=EPS):
    """sum_a mu(a) * w(a) * (Q(a) - b) * (onehot(a) - pi), with the critic baseline b = mu . Q."""
    pi = torch.softmax(torch.as_tensor(z, dtype=torch.float64), 0)
    mu = (1 - eps) * pi + eps / A
    Qt = torch.as_tensor(Q, dtype=torch.float64)
    b = mu @ Qt
    g = torch.zeros(A, dtype=torch.float64)
    for a in range(A):
        w = pg_weight(pi[None, :], torch.tensor([a]), eps, mode)[0]
        g += mu[a] * w * (Qt[a] - b) * (torch.nn.functional.one_hot(torch.tensor(a), A).double() - pi)
    return pi, g


@pytest.mark.parametrize("scale", [0.5, 2.0, 8.0])
def test_behaviour_factor_is_the_gradient_of_J_mu(scale):
    rng = np.random.default_rng(1)
    z, Q = rng.normal(0, 1, A) * scale, rng.normal(0, 1, A)
    pi, g = _expected_update(z, Q, "behaviour")
    grad_J_pi = pi * (torch.as_tensor(Q) - pi @ torch.as_tensor(Q))
    assert torch.allclose(g, (1 - EPS) * grad_J_pi, atol=1e-12)


def test_pi_factor_does_not_vanish_at_saturation():
    """The v1/v2 factor keeps an eps/A x (Q - b) term when pi is saturated: the
    documented consequence of the declared approximation."""
    rng = np.random.default_rng(2)
    z, Q = rng.normal(0, 1, A) * 30.0, rng.normal(0, 1, A)
    pi, g_pi = _expected_update(z, Q, "pi")
    _, g_mu = _expected_update(z, Q, "behaviour")
    assert float(pi.max()) > 0.999
    assert g_pi.norm() > 100 * g_mu.norm()
    mu = (1 - EPS) * pi + EPS / A
    assert torch.allclose(g_pi, mu * (torch.as_tensor(Q) - mu @ torch.as_tensor(Q)), atol=1e-12)


def test_pi_mode_and_no_exploration_give_unit_weights():
    pi = torch.softmax(torch.randn(4, A), 1)
    a = torch.tensor([0, 3, 5, 7])
    assert torch.equal(pg_weight(pi, a, EPS, "pi"), torch.ones(4))
    assert torch.equal(pg_weight(pi, a, 0.0, "behaviour"), torch.ones(4))
    with pytest.raises(ValueError):
        pg_weight(pi, a, EPS, "other")


def test_end_delta_bootstraps_only_truncations():
    r, V = torch.tensor([1.0, 1.0, -10.0]), torch.tensor([5.0, 5.0, 5.0])
    trunc = torch.tensor([True, False, False])            # stall/time, flag, death
    assert torch.allclose(end_delta(r, V, trunc, 0.98, False), r - V)
    out = end_delta(r, V, trunc, 0.98, True)
    assert torch.allclose(out, torch.tensor([1.0 + 0.98 * 5.0 - 5.0, 1.0 - 5.0, -10.0 - 5.0]))


def test_defaults_reproduce_the_v2_procedure():
    s = SweepMarioSettings()
    assert s.pg_factor == "pi" and s.bootstrap_truncation is False and s.jump_cost == 0.0 and s.actor_warmup == 0 \
        and s.frozen is False


def test_frozen_is_an_evaluation_switch_not_a_run_identity():
    from dataclasses import asdict, replace

    from flymario.training.mario_sweep import MarioSweepTrainer
    base = asdict(SweepMarioSettings(sample_seed=20261015))
    assert MarioSweepTrainer._same_settings(base, asdict(replace(SweepMarioSettings(sample_seed=20261015), frozen=True)))
    assert not MarioSweepTrainer._same_settings(base, asdict(SweepMarioSettings(sample_seed=1)))


def test_stable_streams_do_not_depend_on_the_batch():
    """RL_STUDY §11 item 5: a group's draws are the same whatever other groups exist; the
    default reproduces the earlier position-keyed episode draws."""
    import numpy as np

    from flymario.training.mario_sweep import GroupStreams, episode_rng
    a = GroupStreams(20261020, [0, 1, 2], "cpu")
    b = GroupStreams(20261020, [0, 1, 2, 5], "cpu")
    da = torch.stack([a.rand() for _ in range(20)])
    db = torch.stack([b.rand() for _ in range(20)])
    assert torch.equal(da[:, :3], db[:, :3])
    st = a.get_state()
    x = a.rand()
    a.set_state(st)
    assert torch.equal(a.rand(), x)
    old = np.random.default_rng([20261015, 4, 77]).integers(0, 1000, 10)
    assert np.array_equal(episode_rng(20261015, 4, 1, False).integers(0, 1000, 10), old)
    twin1 = episode_rng(20261015, 0, 2, True).integers(0, 1000, 10)
    twin2 = episode_rng(20261015, 7, 2, True).integers(0, 1000, 10)
    other = episode_rng(20261015, 0, 3, True).integers(0, 1000, 10)
    assert np.array_equal(twin1, twin2) and not np.array_equal(twin1, other)
    assert SweepMarioSettings().stable_streams is False
