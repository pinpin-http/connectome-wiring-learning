"""Label-free activity calibration of the global gain (mission §7). Nothing here receives a label, an accuracy or a
decoder: the inputs are readout spike counts and the gains that produced them.

Output rate (mission §7.3), over every readout neuron including silent ones:

    r = total readout spikes / (images x readout neurons x window seconds),  window = ticks x dt

Validity of one (graph, gamma) point (mission §7.4, frozen in configs/activity_match/protocol.yaml), on the mean
over the calibration noise seeds:
    finite diagnostics; r > 0; mean over images of the fraction of readout neurons with >= 1 spike >= min_active;
    mean over images of the fraction with >= sat_frac x max_count spikes <= max_saturated.

The simulator's maximum count is derived from the coded refractoriness: a spike at tick t blocks ticks
t+1 .. t+refractory_steps, so at most ceil(ticks / (refractory_steps + 1)) spikes fit in the window
(16 ticks, 1 refractory tick: 8; checked by simulation in tests/unit/test_activity_match.py).

Common range: [max over graphs of their lowest valid rate, min over graphs of their highest valid rate] on the grid.
Targets: r_min + q (r_max - r_min), q in (0.25, 0.50, 0.75). The search for one graph (mission §7.5) brackets each
target between two adjacent evaluated valid points, interpolates in log gamma inside the bracket and keeps a
bracket only while the evaluated rates stay monotone in it; at most `budget` extra evaluations per graph in total.
Selection: among the evaluated valid points within `fit_tol` relative error of the target (equivalent), the one
closest to gamma = 1, then the smallest gamma; if none is within `fit_tol`, the smallest relative error (same ties).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def max_count(ticks: int, refractory_steps: int) -> int:
    return int(math.ceil(ticks / (refractory_steps + 1)))


def activity(counts: np.ndarray, window_s: float, cmax: int, sat_frac: float) -> dict:
    """Diagnostics of one (graph, gamma, seed) run: counts (images, readout neurons)."""
    C = np.asarray(counts)
    n_img, n_out = C.shape
    per_img_rate = C.mean(1) / window_s
    thr = sat_frac * cmax
    return {"rate_hz": float(C.sum(dtype=np.int64) / (n_img * n_out * window_s)),
            "rate_hz_sd_over_images": float(per_img_rate.std()),
            "active_frac_mean": float((C > 0).mean(1).mean()), "active_frac_min": float((C > 0).mean(1).min()),
            "saturated_frac_mean": float((C >= thr).mean(1).mean()),
            "saturated_frac_max": float((C >= thr).mean(1).max()),
            "ge7_frac_mean": float((C >= 7).mean()),
            "count_hist": np.bincount(C.ravel().astype(np.int64), minlength=cmax + 1).tolist(),
            "n_constant_features": int((C.std(0) == 0).sum()), "max_count": int(C.max())}


def combine(per_seed: list[dict], finite: bool, rules: dict) -> dict:
    """Mean over the calibration seeds, the per-seed values and the validity verdict with its reasons."""
    keys = ("rate_hz", "active_frac_mean", "saturated_frac_mean")
    out = {k: float(np.mean([d[k] for d in per_seed])) for k in keys}
    out["rate_hz_by_seed"] = [d["rate_hz"] for d in per_seed]
    reasons = []
    if not finite:
        reasons.append("non-finite diagnostics")
    if not out["rate_hz"] > 0:
        reasons.append("rate is zero")
    if out["active_frac_mean"] < rules["minimum_active_fraction"]:
        reasons.append("active fraction below minimum")
    if out["saturated_frac_mean"] > rules["maximum_saturation_fraction"]:
        reasons.append("saturated fraction above maximum")
    out["valid"] = not reasons
    out["invalid_reasons"] = reasons
    return out


def common_range(curves: dict[str, dict[float, dict]]) -> dict:
    """curves: graph -> {gamma: combined point}. Per-graph envelope of valid rates and their intersection."""
    env = {}
    for g, pts in curves.items():
        r = [p["rate_hz"] for p in pts.values() if p["valid"]]
        env[g] = (min(r), max(r)) if r else None
    if any(v is None for v in env.values()):
        return {"exists": False, "reason": "a graph has no valid point", "envelopes": env}
    lo = max(v[0] for v in env.values())
    hi = min(v[1] for v in env.values())
    return {"exists": bool(lo < hi), "r_min": lo, "r_max": hi, "envelopes": env,
            "reason": None if lo < hi else "empty intersection"}


def targets(r_min: float, r_max: float, positions: dict[str, float]) -> dict[str, float]:
    return {name: r_min + q * (r_max - r_min) for name, q in positions.items()}


def rel_err(r: float, t: float) -> float:
    return abs(r - t) / t


def _closest_to_one(gammas: list[float]) -> float:
    return sorted(gammas, key=lambda g: (abs(g - 1.0), g))[0]


def select(points: dict[float, dict], target: float, fit_tol: float) -> tuple[float, str]:
    """The chosen gamma among evaluated valid points and the rule that chose it."""
    valid = {g: p for g, p in points.items() if p["valid"]}
    if not valid:
        raise ValueError("no valid point")
    ok = [g for g, p in valid.items() if rel_err(p["rate_hz"], target) <= fit_tol]
    if ok:
        return _closest_to_one(ok), "within fit tolerance, closest to gamma = 1"
    best = min(rel_err(p["rate_hz"], target) for p in valid.values())
    tied = [g for g, p in valid.items() if rel_err(p["rate_hz"], target) <= best + 1e-12]
    return _closest_to_one(tied), "smallest relative error (none within fit tolerance)"


def brackets(points: dict[float, dict], target: float) -> list[tuple[float, float]]:
    """Adjacent evaluated points, both valid, whose rates straddle the target (an invalid point breaks a branch)."""
    gs = sorted(points)
    out = []
    for a, b in zip(gs[:-1], gs[1:]):
        pa, pb = points[a], points[b]
        if pa["valid"] and pb["valid"] and (pa["rate_hz"] - target) * (pb["rate_hz"] - target) <= 0:
            out.append((a, b))
    return out


def _bracket_distance(br: tuple[float, float]) -> tuple[float, float]:
    a, b = br
    return (0.0 if a <= 1.0 <= b else min(abs(a - 1.0), abs(b - 1.0)), a)


@dataclass
class SearchLog:
    evaluations: list = field(default_factory=list)     # (target name, gamma, rate) in order
    notes: list = field(default_factory=list)


def search(points: dict[float, dict], tgts: dict[str, float], evaluate, budget: int, per_target: int,
           fit_tol: float) -> SearchLog:
    """Refine `points` (gamma -> combined point, the grid already evaluated) in place for each target in order.
    `evaluate(gamma)` returns a combined point. Unused per-target evaluations carry over to the next target."""
    log = SearchLog()
    used = 0
    allowance = 0
    for name, t in tgts.items():
        allowance += per_target
        while used < budget and allowance > 0:
            if any(p["valid"] and rel_err(p["rate_hz"], t) <= fit_tol for p in points.values()):
                break
            brs = brackets(points, t)
            if not brs:
                log.notes.append(f"{name}: no bracket among evaluated valid points")
                break
            a, b = sorted(brs, key=_bracket_distance)[0]
            ra, rb = points[a]["rate_hz"], points[b]["rate_hz"]
            la, lb = math.log(a), math.log(b)
            f = 0.5 if ra == rb else (t - ra) / (rb - ra)
            f = min(max(f, 0.1), 0.9)                                    # guaranteed progress inside the bracket
            g = float(round(math.exp(la + f * (lb - la)), 6))
            if g in points or not (a < g < b):
                log.notes.append(f"{name}: bracket [{a}, {b}] cannot be split further")
                break
            p = evaluate(g)
            points[g] = p
            used += 1
            allowance -= 1
            log.evaluations.append((name, g, p["rate_hz"]))
            if p["valid"] and not (min(ra, rb) - 1e-9 <= p["rate_hz"] <= max(ra, rb) + 1e-9):
                log.notes.append(f"{name}: rate not monotone in [{a}, {b}] (at {g}: {p['rate_hz']:.3f}); "
                                 "bracketing stopped for this target")
                break
    log.notes.append(f"extra evaluations used: {used} of {budget}")
    return log
