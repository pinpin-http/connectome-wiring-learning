"""EXPERIENCE_COMMUNE_V1 policy signal (mission §9, analytic bandit): the corrected signal's expectation equals the
exact gradient of the expected reward under the mixture policy, in mode N (actions from pi) and in mode U (uniform
actions with the importance ratio), for several baselines and score shapes; rho stays in [0.1, 7.3]; tau divides
once."""
from __future__ import annotations

import numpy as np
import pytest

from flymario.experience_commune import policy as P

TAU, EPS, K = 2.0, 0.1, 8


def expected_reward(z, R, tau=TAU, eps=EPS):
    return float((P.probs(z, tau, eps)[1] * R).sum())


def fd_grad(z, R, tau=TAU, eps=EPS, h=1e-6):
    g = np.zeros(K)
    for i in range(K):
        e = np.zeros(K)
        e[i] = h
        g[i] = (expected_reward(z + e, R, tau, eps) - expected_reward(z - e, R, tau, eps)) / (2 * h)
    return g


def expected_signal(z, R, b, mode, tau=TAU, eps=EPS):
    _u, pi = P.probs(z, tau, eps)
    total = np.zeros(K)
    for a in range(K):
        w = pi[a] if mode == "N" else 1.0 / K
        q = P.behaviour_prob(pi[None], np.array([a]), mode)
        total += w * P.signal(z[None], np.array([a]), np.array([R[a]]), np.array([b]), q, tau, eps)["v"][0]
    return total / tau


SCORES = {"near_uniform": np.linspace(-0.05, 0.05, K), "moderate": np.array([0.3, -1.2, 2.0, 0.1, -0.4, 1.1, -2.0, 0.6]),
          "concentrated": np.array([12.0, -3.0, 0.5, -1.0, 2.0, -6.0, 1.0, 0.0])}


@pytest.mark.parametrize("name", list(SCORES))
@pytest.mark.parametrize("b", [0.0, 0.125, 0.5, -0.3])
@pytest.mark.parametrize("R", [np.eye(K)[2], np.eye(K)[6], np.linspace(0, 1, K)])
def test_expected_signal_is_the_gradient_in_both_modes(name, b, R):
    z = SCORES[name]
    g = fd_grad(z, R)
    for mode in ("N", "U"):
        np.testing.assert_allclose(expected_signal(z, R, b, mode), g, atol=1e-7)


def test_tau_divides_once_and_epsilon_enters_the_score_function():
    z, R = SCORES["moderate"], np.eye(K)[2]
    for tau, eps in ((1.0, 0.1), (3.0, 0.1), (2.0, 0.0), (2.0, 0.3)):
        np.testing.assert_allclose(expected_signal(z, R, 0.2, "U", tau, eps), fd_grad(z, R, tau, eps), atol=1e-7)


def test_importance_ratio_bounds():
    rng = np.random.default_rng(0)
    rhos, prods = [], []
    for _ in range(2000):
        z = rng.normal(0, rng.choice([0.01, 1, 20]), K)
        _u, pi = P.probs(z, TAU, EPS)
        a = rng.integers(0, K, 1)
        s = P.signal(z[None], a, np.array([1.0]), np.array([0.0]), P.behaviour_prob(pi[None], a, "U"), TAU, EPS)
        rhos.append(s["rho"][0])
        prods.append(s["rho"][0] * s["mix_factor"][0])
    assert min(rhos) >= EPS - 1e-12 and max(rhos) <= K - (K - 1) * EPS + 1e-12       # [0.1, 7.3]
    assert min(prods) >= 0 and max(prods) <= K * (1 - EPS) + 1e-12                    # K (1 - eps) u_a <= 7.2


def test_mode_N_has_unit_ratio_and_U_simplifies():
    z, a = SCORES["moderate"], np.array([3])
    u, pi = P.probs(z, TAU, EPS)
    sN = P.signal(z[None], a, np.array([1.0]), np.array([0.25]), P.behaviour_prob(pi[None], a, "N"), TAU, EPS)
    assert sN["rho"][0] == pytest.approx(1.0)
    sU = P.signal(z[None], a, np.array([1.0]), np.array([0.25]), P.behaviour_prob(pi[None], a, "U"), TAU, EPS)
    np.testing.assert_allclose(sU["v"][0], K * (1 - EPS) * u[3] * 0.75 * (np.eye(K)[3] - u), atol=1e-12)


def test_historical_signal_is_not_the_mixture_gradient():
    """Recorded for the report: Study S's (onehot - u)(r - b) is the exact softmax score only when epsilon = 0."""
    z, R = SCORES["moderate"], np.eye(K)[1]
    _u, pi = P.probs(z, TAU, EPS)
    hist = sum(pi[a] * P.historical_signal(z[None], np.array([a]), np.array([R[a]]), np.array([0.0]), TAU)[0]
               for a in range(K)) / TAU
    assert not np.allclose(hist, fd_grad(z, R), atol=1e-4)
    _u0, pi0 = P.probs(z, TAU, 0.0)
    hist0 = sum(pi0[a] * P.historical_signal(z[None], np.array([a]), np.array([R[a]]), np.array([0.0]), TAU)[0]
                for a in range(K)) / TAU
    np.testing.assert_allclose(hist0, fd_grad(z, R, eps=0.0), atol=1e-7)


# --- analysis (mission §11-§12) --------------------------------------------------------------------------------------
from flymario.experience_commune import analysis as AN  # noqa: E402


def test_trapezoid_weights():
    w = AN.trapezoid_weights()
    assert w.sum() == pytest.approx(1.0)
    A = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    manual = sum((AN.CHECKPOINTS[j + 1] - AN.CHECKPOINTS[j]) * (A[j] + A[j + 1]) / 2 for j in range(6)) / 6000
    assert w @ A == pytest.approx(manual)


def synthetic(G_b=4, G_c=4, R=3, n=64, nf=128, seed=0, conn_bonus=0.1):
    rng = np.random.default_rng(seed)
    roles = np.array(["connectome"] + ["b"] * G_b + ["c"] * G_c)
    G = len(roles)
    p = np.linspace(0.15, 0.6, 7)
    prob = np.broadcast_to(p, (G, 2, 2, R, 7)).copy()
    prob[0] += conn_bonus
    prog = rng.random((G, 2, 2, R, 7, n)) < prob[..., None]
    final = rng.random((G, 2, 2, R, nf)) < prob[..., -1][..., None]
    cls_p = np.tile(np.repeat(np.arange(8), n // 8), (R, 2, 1))
    cls_f = np.tile(np.repeat(np.arange(8), nf // 8), (R, 2, 1))
    return prog, final, cls_p, cls_f, roles


def test_contrasts_and_bootstrap_recover_a_known_advantage():
    prog, final, cp, cf, roles = synthetic(n=512, nf=1024, conn_bonus=0.1)
    Ap = prog.mean(-1)
    Af = final.mean(-1)
    s = AN.stats(Ap, Af, roles)
    assert s["L_G_b_where_U"] == pytest.approx(0.1, abs=0.02)
    assert s["L_I_c_loom"] == pytest.approx(0.0, abs=0.03)
    bs = AN.bootstrap(prog, final, cp, cf, roles, 200, 1)
    lo, hi = AN.interval(bs["L_G_b_where_U"], 0.9875)
    assert lo < 0.1 < hi and lo > 0.03


def test_bootstrap_identical_graphs_gives_zero():
    prog, final, cp, cf, roles = synthetic(conn_bonus=0.0)
    prog[:] = prog[:1]
    final[:] = final[:1]
    bs = AN.bootstrap(prog, final, cp, cf, roles, 50, 2)
    for k in ("L_G_b_where_U", "L_G_c_loom_N", "L_I_b_where", "final_G_c_where_U"):
        assert np.all(np.abs(bs[k]) < 1e-6)


def test_reading_table():
    assert AN.reading(0.04, 0.08, 0.03).startswith("practical advantage")
    assert AN.reading(-0.02, 0.02, 0.03).startswith("compatible")
    assert AN.reading(-0.09, -0.04, 0.03).startswith("practical disadvantage")
    assert AN.reading(-0.01, 0.05, 0.03).startswith("practical importance not settled")
