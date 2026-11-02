"""Primary and secondary contrasts and their conditional bootstrap (docs/retino_align/PROTOCOL_FROZEN.md §1, §8).

g = 0 the connectome, g = 1..K the nulls; c = 0 ordered, 1..P the permutations; t = 0 orientation, 1 brightness.
    M(g, c, t) = mean_i cbar[g, c, t, i]   (cbar: per-example correctness averaged over seed, draw, budget)
    L(g, t) = M(g, 0, t) - mean_p M(g, p, t) ;  D_t = L(0, t) - mean_{g >= 1} L(g, t)
    G = M(0, 0, orientation) - mean_{g >= 1} M(g, 0, orientation) ;  S = D_orientation - D_brightness
Bootstrap: test images resampled within the 16 cells (same multiset everywhere), nulls resampled as whole
graphs, connectome kept; a replicate is the weighted mean M_b = (w_b · cbar) / n_test, no refitting.
"""
from __future__ import annotations

import numpy as np

ORIENT, BRIGHT = 0, 1


def scores(cbar: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """M (G, C, T) from cbar (G, C, T, n) and optional example weights w (n,) summing to n."""
    if w is None:
        return cbar.mean(-1)
    return (cbar @ w) / w.sum()


def contrasts(M: np.ndarray, nulls: np.ndarray | None = None) -> dict:
    """D_orientation, D_brightness, G, S and the per-graph losses L (G, T). `nulls`: indices (with repetition) of
    the nulls entering the null mean; default all of 1..G-1."""
    if nulls is None:
        nulls = np.arange(1, M.shape[0])
    L = M[:, 0, :] - M[:, 1:, :].mean(1)
    D = L[0] - L[nulls].mean(0)
    G = M[0, 0, ORIENT] - M[nulls, 0, ORIENT].mean()
    return {"L": L, "D_orientation": float(D[ORIENT]), "D_brightness": float(D[BRIGHT]), "G": float(G),
            "S": float(D[ORIENT] - D[BRIGHT])}


def stratified_weights(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Counts (n,) of a resample with replacement within each stratum, same stratum sizes."""
    w = np.zeros(len(strata), np.float64)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1.0)
    return w


def bootstrap(cbar: np.ndarray, strata: np.ndarray, n_rep: int, seed: int, chunk: int = 250) -> dict:
    """Distribution of D_orientation, D_brightness, G, S over `n_rep` paired replicates."""
    rng = np.random.default_rng(seed)
    G_, C, T, n = cbar.shape
    K = G_ - 1
    out = {k: np.empty(n_rep) for k in ("D_orientation", "D_brightness", "G", "S")}
    flat = cbar.reshape(-1, n)
    for a in range(0, n_rep, chunk):
        m = min(chunk, n_rep - a)
        Wb = np.stack([stratified_weights(strata, rng) for _ in range(m)], 1)       # (n, m)
        Mb = (flat @ Wb / Wb.sum(0)).reshape(G_, C, T, m)
        for j in range(m):
            nul = 1 + rng.integers(0, K, K)
            c = contrasts(Mb[..., j], nul)
            for k in out:
                out[k][a + j] = c[k]
    return out


def interval(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    q = (1 - level) / 2
    return float(np.quantile(x, q)), float(np.quantile(x, 1 - q))


def reading(lo: float, hi: float, delta: float) -> str:
    """The declared decision table, applied to a 95 % interval of D and the margin delta."""
    if lo > delta:
        return "positive differential effect of practical importance supported (conditional protocol)"
    if -delta <= lo and hi <= delta:
        return "compatible with practical equivalence at this margin (conditional)"
    if lo > 0 and hi >= delta:
        return "positive effect compatible with the data; practical importance uncertain"
    if hi < -delta:
        return "important differential effect in the opposite direction"
    if hi < delta:
        return "a positive effect of the targeted size is excluded; a negative difference remains possible"
    return "indeterminate at this precision"


def rank_of_connectome(L: np.ndarray) -> dict:
    """Descriptive rank of the connectome's loss among the K + 1 graphs."""
    K = len(L) - 1
    ge = int((L[1:] >= L[0]).sum())
    return {"n_nulls_ge": ge, "rank_value": (1 + ge) / (K + 1), "K": K,
            "status": "descriptive (no calibrated Monte-Carlo test claimed)"}
