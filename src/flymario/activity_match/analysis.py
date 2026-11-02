"""Contrasts of ACTIVITY_MATCH_V1 (docs/activity_match/PROTOCOL_FROZEN.md §8-§9) and their paired bootstrap.

g = 0 the connectome, g = 1..K the nulls; c the gain conditions (C0, CL, CM, CH); t = 0 orientation, 1 brightness.
M(g, c, t) = mean per-example correctness; G_c = M(0, c) - mean_k M(k, c); T_real = M(0, CM) - M(0, C0);
T_nulls = mean_k [M(k, CM) - M(k, C0)]; R = G_CM - G_C0 = T_real - T_nulls.
Bootstrap: test images resampled within the 16 label cells (same multiset everywhere), nulls resampled as whole
graphs, connectome kept; nothing refitted. 95 % percentile intervals.
"""
from __future__ import annotations

import numpy as np

from flymario.retino_align.analysis import interval, stratified_weights

ORIENT, BRIGHT = 0, 1
TASK_NAMES = ("orient", "lum")


def contrasts(M: np.ndarray, conds: list[str], nulls: np.ndarray | None = None, matched: str = "CM") -> dict:
    """Every declared statistic from M (G, C, T); `nulls`: indices (with repetition) entering the null means;
    `matched`: the calibrated condition compared with C0 in T_real, T_nulls and R (CM in V1, CN in V2)."""
    if nulls is None:
        nulls = np.arange(1, M.shape[0])
    out = {}
    for t, tn in enumerate(TASK_NAMES):
        for ci, c in enumerate(conds):
            out[f"M_connectome_{c}_{tn}"] = float(M[0, ci, t])
            out[f"M_nulls_{c}_{tn}"] = float(M[nulls, ci, t].mean())
            out[f"G_{c}_{tn}"] = out[f"M_connectome_{c}_{tn}"] - out[f"M_nulls_{c}_{tn}"]
        if "C0" in conds and matched in conds:
            i0, im = conds.index("C0"), conds.index(matched)
            out[f"T_real_{tn}"] = float(M[0, im, t] - M[0, i0, t])
            out[f"T_nulls_{tn}"] = float((M[nulls, im, t] - M[nulls, i0, t]).mean())
            out[f"R_{tn}"] = out[f"G_{matched}_{tn}"] - out[f"G_C0_{tn}"]
    return out


def bootstrap(cbar: np.ndarray, strata: np.ndarray, conds: list[str], n_rep: int, seed: int,
              chunk: int = 250, matched: str = "CM") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    G_, C, T, n = cbar.shape
    K = G_ - 1
    flat = cbar.reshape(-1, n)
    out: dict[str, np.ndarray] = {}
    for a in range(0, n_rep, chunk):
        m = min(chunk, n_rep - a)
        Wb = np.stack([stratified_weights(strata, rng) for _ in range(m)], 1)
        Mb = (flat @ Wb / Wb.sum(0)).reshape(G_, C, T, m)
        for j in range(m):
            nul = 1 + rng.integers(0, K, K)
            for k, v in contrasts(Mb[..., j], conds, nul, matched).items():
                out.setdefault(k, np.empty(n_rep))[a + j] = v
    return out


def reading_G(lo: float, hi: float, delta: float) -> str:
    """Declared reading of the interval of G_CM (PROTOCOL_FROZEN.md §9)."""
    if -delta <= lo and hi <= delta:
        return "compatible with practical equivalence, conditional on this margin"
    if hi < -delta:
        return "persistent deficit of practical importance at the intermediate target"
    if lo > delta:
        return "advantage of practical importance at the intermediate target"
    return "practical importance not settled at this precision"


def strong_criteria(iv: dict, delta: float, informative: float, matching_valid: bool) -> dict:
    """The joint conditions for a convincing reduction of the deficit (orientation; PROTOCOL_FROZEN.md §9)."""
    c = {"activity_matching_valid": bool(matching_valid),
         "initial_deficit_reproduced_G_C0_below_minus_delta": iv["G_C0_orient"][1] < -delta,
         "G_CM_within_margin": -delta <= iv["G_CM_orient"][0] and iv["G_CM_orient"][1] <= delta,
         "R_positive_interval_excludes_zero": iv["R_orient"][0] > 0,
         "T_real_positive_interval_excludes_zero": iv["T_real_orient"][0] > 0,
         "absolute_scores_informative_at_CM": (iv["M_connectome_CM_orient"][0] > informative
                                               and iv["M_nulls_CM_orient"][0] > informative)}
    c["all"] = all(c.values())
    return c


def criteria_v2(iv: dict, delta: float, informative: float, matching_valid: bool, t_real_zero: bool) -> dict:
    """ACTIVITY_MATCH_V2: the conditions under which the deficit at the common gain is read as accounted for
    by the activity level (connectome unchanged at gamma = 1, nulls brought to its rate)."""
    c = {"activity_matching_valid": bool(matching_valid),
         "connectome_identical_in_both_conditions": bool(t_real_zero),
         "deficit_at_C0_below_minus_delta": iv["G_C0_orient"][1] < -delta,
         "G_CN_within_margin": -delta <= iv["G_CN_orient"][0] and iv["G_CN_orient"][1] <= delta,
         "R_positive_interval_excludes_zero": iv["R_orient"][0] > 0,
         "absolute_scores_above_chance_plus_delta_at_CN": (iv["M_connectome_CN_orient"][0] > informative
                                                           and iv["M_nulls_CN_orient"][0] > informative)}
    c["all"] = all(c.values())
    return c


def intervals(bs: dict[str, np.ndarray], level: float = 0.95) -> dict[str, tuple[float, float]]:
    return {k: interval(v, level) for k, v in bs.items()}
