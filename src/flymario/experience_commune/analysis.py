"""Primary measure, contrasts and their bootstrap (docs/experience_commune/PROTOCOL_FROZEN.md §6-§7).

    prog[g, t, m, r, c, i]   per-example correctness: graph, task (where, loom), mode (N, U), block, checkpoint, image
    final[g, t, m, r, i]     the independent final bank after 6,000 decisions
    L         = sum_j (n_{j+1} - n_j) (A_j + A_{j+1}) / 2 / 6000,   A(n_c) = mean_i prog[..., c, i]
    G_m(t, f) = mean_r L(connectome, t, m, r) - mean_{g in f, r} L(g, t, m, r),   f in {b, c};   I = G_U - G_N

Bootstrap: blocks resampled (one draw for every graph, mode and task), nulls as whole graphs, images within their
8 classes. Percentile intervals: 98.75 % for the four primary G_U (Bonferroni over 4), 95 % for the rest.
"""
from __future__ import annotations

import numpy as np

CHECKPOINTS = np.array([0, 250, 500, 1000, 2000, 4000, 6000], float)
TASKS, MODES, FAMILIES = ("where", "loom"), ("N", "U"), ("b", "c")


def trapezoid_weights(n: np.ndarray = CHECKPOINTS) -> np.ndarray:
    """L = w . A for A at the checkpoints."""
    w = np.zeros(len(n))
    d = np.diff(n)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w / (n[-1] - n[0])


def stats(Ap: np.ndarray, Af: np.ndarray, roles: np.ndarray, gsel: dict | None = None,
          bsel: np.ndarray | None = None) -> dict:
    """All declared statistics from accuracies Ap[g, t, m, r, c] (progress) and Af[g, t, m, r] (final).

    roles: array of 'connectome' | 'b' | 'c'; gsel: family -> graph indices (with repetition) entering the family
    means; bsel: block indices (with repetition)."""
    w = trapezoid_weights()
    if bsel is None:
        bsel = np.arange(Ap.shape[3])
    if gsel is None:
        gsel = {f: np.flatnonzero(roles == f) for f in FAMILIES}
    L = Ap[:, :, :, bsel] @ w                                       # (G, T, M, R')
    A0 = Ap[:, :, :, bsel, 0]
    A6 = Ap[:, :, :, bsel, -1]
    fin = Af[:, :, :, bsel]
    g0 = int(np.flatnonzero(roles == "connectome")[0])
    out = {}
    per = {"L": L, "A0": A0, "gain": A6 - A0, "Lgain": L - A0, "final": fin}
    for name, X in per.items():
        Xg = X.mean(-1)                                             # mean over blocks: (G, T, M)
        for ti, t in enumerate(TASKS):
            for mi, m in enumerate(MODES):
                out[f"{name}_connectome_{t}_{m}"] = float(Xg[g0, ti, mi])
                for f in FAMILIES:
                    out[f"{name}_{f}_{t}_{m}"] = float(Xg[gsel[f], ti, mi].mean())
                    out[f"{name}_G_{f}_{t}_{m}"] = out[f"{name}_connectome_{t}_{m}"] - out[f"{name}_{f}_{t}_{m}"]
            for f in FAMILIES:
                out[f"{name}_I_{f}_{t}"] = out[f"{name}_G_{f}_{t}_U"] - out[f"{name}_G_{f}_{t}_N"]
    return out


def strat_weights(cls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.zeros(len(cls))
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1.0)
    return w / w.sum()


def bootstrap(prog: np.ndarray, final: np.ndarray, cls_prog: np.ndarray, cls_final: np.ndarray, roles: np.ndarray,
              n_rep: int, seed: int) -> dict[str, np.ndarray]:
    """prog (G, T, M, R, C, n) and final (G, T, M, R, n') boolean/uint8; cls_prog (R, T, n), cls_final (R, T, n')."""
    rng = np.random.default_rng(seed)
    G, T, M, R, C, n = prog.shape
    fam = {f: np.flatnonzero(roles == f) for f in FAMILIES}
    pf = prog.astype(np.float32)
    ff = final.astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for b in range(n_rep):
        bsel = rng.integers(0, R, R)
        gsel = {f: idx[rng.integers(0, len(idx), len(idx))] for f, idx in fam.items()}
        Ap = np.empty((G, T, M, R, C), np.float32)
        Af = np.empty((G, T, M, R), np.float32)
        for k, r in enumerate(bsel):
            for t in range(T):
                Ap[:, t, :, k] = pf[:, t, :, r] @ strat_weights(cls_prog[r, t], rng)
                Af[:, t, :, k] = ff[:, t, :, r] @ strat_weights(cls_final[r, t], rng)
        for key, v in stats(Ap, Af, roles, gsel, None).items():
            out.setdefault(key, np.empty(n_rep))[b] = v
    return out


def interval(x: np.ndarray, level: float) -> tuple[float, float]:
    q = (1 - level) / 2
    return float(np.quantile(x, q)), float(np.quantile(x, 1 - q))


def reading(lo: float, hi: float, delta: float) -> str:
    """The docs/experience_commune/PROTOCOL_FROZEN.md §8 reading of an interval of G_U."""
    if lo > delta:
        return "practical advantage under common experience"
    if -delta <= lo and hi <= delta:
        return "compatible with practical equivalence at this margin (subject to sensitivity)"
    if hi < -delta:
        return "practical disadvantage under common experience"
    return "practical importance not settled at this precision"
