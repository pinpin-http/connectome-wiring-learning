"""REVISION48 reading (configs/revision48/campaign.yaml `analysis`), built on EC's measure.

X[c][g, t, r, k, i]: greedy correctness (0/1) per condition, graph, task, main block, checkpoint, progress image.
    A(k)     = mean_i X ;  L = sum_k w_k A(k)  (trapezoid / horizon), EC's primary measure
    G_c(t,f) = mean_r L(connectome) - mean_{g in f, r} L(g)      f in {S, E}
    D(t,f)   = G_tuned(t,f) - G_reference(t,f)              the change of the gap after equal-budget tuning
Paired bootstrap: main blocks, nulls by position in each family, and images within their 8 classes resampled
identically across graphs and conditions; percentile intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAMILIES = ("S", "E")


def trapezoid_weights(checkpoints) -> np.ndarray:
    n = np.asarray(checkpoints, float)
    w = np.zeros(len(n))
    d = np.diff(n)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w / (n[-1] - n[0])


def roles_of(specs: list[str]) -> np.ndarray:
    out = []
    for s in specs:
        fam = s.split(":")[0]
        out.append("connectome" if s == "connectome" else ("S" if fam in ("nulllat", "nulllatL") else "E"))
    return np.array(out)


def load_condition(dirs: list[Path], specs: list[str], tasks, blocks, checkpoints) -> dict:
    """Correctness arrays of one condition from one run directory per graph (learners in block-then-task order).
    A directory holding reuse.json (tuned = reference, trial 1) is read from the reference directory it names."""
    T, R, C = len(tasks), len(blocks), len(checkpoints)
    X = F = None
    cls_p = cls_f = None
    for gi, (spec, d) in enumerate(zip(specs, dirs, strict=True)):
        if (d / "reuse.json").exists():
            d = d.parents[1] / "reference" / d.name
        z = np.load(d / "predictions.npz")
        lab_p, lab_f = z["labels_progress"], z["labels_final"]
        n, nf = lab_p.shape[1], lab_f.shape[1]
        if X is None:
            X = np.zeros((len(specs), T, R, C, n), np.uint8)
            F = np.zeros((len(specs), T, R, nf), np.uint8)
            cls_p = lab_p.reshape(R, T, n).transpose(1, 0, 2).transpose(1, 0, 2)        # (R, T, n)
            cls_f = lab_f.reshape(R, T, nf)
        for ki, k in enumerate(checkpoints):
            X[gi, :, :, ki] = (z[f"progress_{k}"] == lab_p).reshape(R, T, n).transpose(1, 0, 2)
        F[gi] = (z["final"] == lab_f).reshape(R, T, nf).transpose(1, 0, 2)
    return {"X": X, "F": F, "cls_p": cls_p, "cls_f": cls_f, "roles": roles_of(specs), "specs": list(specs)}


def stats(Ap: np.ndarray, Af: np.ndarray, roles: np.ndarray, w: np.ndarray, tasks, gsel=None) -> dict:
    """Ap (G, T, R', C) accuracies, Af (G, T, R') final accuracies (blocks already selected)."""
    if gsel is None:
        gsel = {f: np.flatnonzero(roles == f) for f in FAMILIES}
    g0 = int(np.flatnonzero(roles == "connectome")[0])
    per = {"L": Ap @ w, "A0": Ap[..., 0], "Aend": Ap[..., -1], "final": Af}
    out = {}
    for name, Y in per.items():
        Yg = Y.mean(-1)                                      # mean over blocks: (G, T)
        for ti, t in enumerate(tasks):
            out[f"{name}_connectome_{t}"] = float(Yg[g0, ti])
            for f in FAMILIES:
                out[f"{name}_{f}_{t}"] = float(Yg[gsel[f], ti].mean())
                out[f"{name}_G_{f}_{t}"] = out[f"{name}_connectome_{t}"] - out[f"{name}_{f}_{t}"]
    return out


def point(conds: dict, tasks, checkpoints) -> dict:
    w = trapezoid_weights(checkpoints)
    out = {}
    for c, d in conds.items():
        Ap = d["X"].mean(-1)
        Af = d["F"].mean(-1)
        for k, v in stats(Ap, Af, d["roles"], w, tasks).items():
            out[f"{c}:{k}"] = v
    _add_D(out, conds, tasks)
    return out


def _add_D(out: dict, conds: dict, tasks) -> None:
    if "tuned" in conds and "reference" in conds:
        for t in tasks:
            for f in FAMILIES:
                out[f"D_{f}_{t}"] = out[f"tuned:L_G_{f}_{t}"] - out[f"reference:L_G_{f}_{t}"]


def _strat(cls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.zeros(len(cls))
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1.0)
    return w / w.sum()


def bootstrap(conds: dict, tasks, checkpoints, n_rep: int, seed: int) -> dict[str, np.ndarray]:
    """Paired replicates of every statistic of every condition, and of D."""
    rng = np.random.default_rng(seed)
    w = trapezoid_weights(checkpoints)
    any_c = next(iter(conds.values()))
    R = any_c["X"].shape[2]
    T = len(tasks)
    for d in conds.values():                                # the same banks in every condition
        if not (np.array_equal(d["cls_p"], any_c["cls_p"]) and np.array_equal(d["cls_f"], any_c["cls_f"])):
            raise ValueError("conditions do not share the evaluation banks")
    fam_size = {f: int((any_c["roles"] == f).sum()) for f in FAMILIES}
    Xf = {c: d["X"].astype(np.float32) for c, d in conds.items()}
    Ff = {c: d["F"].astype(np.float32) for c, d in conds.items()}
    out: dict[str, np.ndarray] = {}
    for b in range(n_rep):
        bsel = rng.integers(0, R, R)
        pos = {f: rng.integers(0, fam_size[f], fam_size[f]) for f in FAMILIES}
        wp = [[_strat(any_c["cls_p"][r, t], rng) for t in range(T)] for r in bsel]
        wf = [[_strat(any_c["cls_f"][r, t], rng) for t in range(T)] for r in bsel]
        rep = {}
        for c, d in conds.items():
            G = d["X"].shape[0]
            Ap = np.empty((G, T, R, len(checkpoints)), np.float32)
            Af = np.empty((G, T, R), np.float32)
            for k, r in enumerate(bsel):
                for t in range(T):
                    Ap[:, t, k] = Xf[c][:, t, r] @ wp[k][t]
                    Af[:, t, k] = Ff[c][:, t, r] @ wf[k][t]
            gsel = {f: np.flatnonzero(d["roles"] == f)[pos[f]] for f in FAMILIES}
            for key, v in stats(Ap, Af, d["roles"], w, tasks, gsel).items():
                rep[f"{c}:{key}"] = v
        _add_D(rep, conds, tasks)
        for key, v in rep.items():
            out.setdefault(key, np.empty(n_rep))[b] = v
    return out


def interval(x: np.ndarray, level: float) -> tuple[float, float]:
    q = (1 - level) / 2
    return float(np.quantile(x, q)), float(np.quantile(x, 1 - q))


def reading(lo: float, hi: float, delta: float) -> str:
    """EC's decision table, in the revision's words."""
    if lo > delta:
        return "practical advantage in this design"
    if -delta <= lo and hi <= delta:
        return "practical equivalence at this margin"
    if hi < -delta:
        return "practical disadvantage in this design"
    return "not settled at this precision"


def report(conds: dict, tasks, checkpoints, n_rep: int, seed: int, margin: float, primary: str = "tuned") -> dict:
    pt = point(conds, tasks, checkpoints)
    bs = bootstrap(conds, tasks, checkpoints, n_rep, seed)
    rows = []
    for c in conds:
        for t in tasks:
            for f in FAMILIES:
                key = f"{c}:L_G_{f}_{t}"
                level = 0.9875 if c == primary else 0.95
                lo, hi = interval(bs[key], level)
                rows.append({"condition": c, "task": t, "family": f, "G": pt[key], "level": level, "lo": lo, "hi": hi,
                             "reading": reading(lo, hi, margin), "primary": c == primary,
                             "L_connectome": pt[f"{c}:L_connectome_{t}"], "L_family": pt[f"{c}:L_{f}_{t}"]})
    dtab = []
    for t in tasks:
        for f in FAMILIES:
            key = f"D_{f}_{t}"
            if key in bs:
                lo, hi = interval(bs[key], 0.95)
                dtab.append({"task": t, "family": f, "D": pt[key], "lo": lo, "hi": hi, "level": 0.95})
    return {"contrasts": rows, "change_after_tuning": dtab, "point": pt, "n_rep": n_rep, "seed": seed,
            "margin": margin}


def save_report(path: Path, rep: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.replace(path)
