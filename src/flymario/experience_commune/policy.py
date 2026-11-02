"""Target policy, behaviour distributions and the importance-corrected learning signal.

    u = softmax(z / tau);   pi_a = (1 - epsilon) u_a + epsilon / K (target);   q_a = pi_a (mode N) or 1 / K (mode U)
    rho = pi_a / q_a;   v = rho (r - b) (1 - epsilon) (u_a / pi_a) (onehot(a) - u);   L = R^T v / tau

With b fixed before the draw, E_{a ~ q}[v] / tau = d E_{a ~ pi}[r] / dz in both modes (checked in
tests/unit/test_experience_commune.py). `historical_signal` is Study S's (onehot(a) - u)(r - b), for comparison only.
"""
from __future__ import annotations

import numpy as np

K = 8


def probs(z: np.ndarray, tau: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, pi) for scores z (..., K)."""
    s = np.asarray(z, np.float64) / tau
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    u = e / e.sum(-1, keepdims=True)
    return u, (1.0 - eps) * u + eps / u.shape[-1]


def behaviour_prob(pi: np.ndarray, a: np.ndarray, mode: str) -> np.ndarray:
    """q_a for the drawn actions a (...,): pi_a in N, 1/K in U."""
    pa = np.take_along_axis(pi, np.asarray(a)[..., None], -1)[..., 0]
    if mode == "N":
        return pa
    if mode == "U":
        return np.full_like(pa, 1.0 / pi.shape[-1])
    raise ValueError(mode)


def signal(z: np.ndarray, a: np.ndarray, r: np.ndarray, b: np.ndarray, q_a: np.ndarray, tau: float,
           eps: float) -> dict:
    """The corrected signal v (..., K) and its factors, every probability taken before any weight change."""
    u, pi = probs(z, tau, eps)
    a = np.asarray(a)
    ua = np.take_along_axis(u, a[..., None], -1)[..., 0]
    pa = np.take_along_axis(pi, a[..., None], -1)[..., 0]
    rho = pa / q_a
    mix = (1.0 - eps) * ua / pa
    onehot = np.eye(u.shape[-1])[a]
    adv = np.asarray(r, np.float64) - b
    v = (rho * adv * mix)[..., None] * (onehot - u)
    return {"v": v, "rho": rho, "mix_factor": mix, "u": u, "pi": pi, "advantage": adv}


def historical_signal(z: np.ndarray, a: np.ndarray, r: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    """Study S: (onehot(a) - softmax(z / tau)) (r - b). Reference only."""
    u, _ = probs(z, tau, 0.0)
    return (np.eye(u.shape[-1])[np.asarray(a)] - u) * (np.asarray(r, np.float64) - b)[..., None]


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0)
    return -(p * np.log(p)).sum(-1)
