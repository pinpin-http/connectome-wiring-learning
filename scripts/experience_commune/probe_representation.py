#!/usr/bin/env python
"""Exploratory: is the connectome's learning advantage already visible in its untrained readout?

The 41 campaign graphs, untrained (gain 1), on block b0's final bank of each Study S task (2,048 images).
Readouts: the 2,227 motor spike counts over one 16-tick sweep from rest, and the 8 block-b0 projection
scores. RETINO_ALIGN's ridge decoder, budgets 8-128 per class, 10 nested balanced draws.

    PYTHONPATH=src .venv/bin/python scripts/experience_commune/probe_representation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"
OUT = ART / "complements/probe_representation"
BUDGETS = (8, 16, 32, 64, 128)       # examples per class


def balanced_subsets(y: np.ndarray, seeds, budgets):
    out = np.full((len(seeds), len(budgets), 8 * budgets[-1]), -1, np.int64)
    for r, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        order = {c: np.flatnonzero(y == c)[rng.permutation(int((y == c).sum()))] for c in range(8)}
        for k, b in enumerate(budgets):
            sel = np.sort(np.concatenate([order[c][:b] for c in range(8)]))
            out[r, k, :len(sel)] = sel
    return out


def ridge_8(Ftr, ytr, Fva, yva, Fte, yte, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """The study's normalised-loss ridge (standardise on train, thin SVD, lambda on validation, ties to the largest),
    with 8 one-hot classes."""
    X = Ftr.astype(np.float64)
    n = len(X)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Y = np.eye(8)[ytr]
    b = Y.mean(0)
    A = U.T @ (Y - b)
    Zva, Zte = (Fva - mu) / sd, (Fte - mu) / sd
    best, bj, W_best = -1, -1, None
    for j, lam in enumerate(lams):
        W = Vt.T @ ((s / (s ** 2 + n * lam))[:, None] * A)
        va = ((Zva @ W + b).argmax(1) == yva).mean()
        if va >= best - 1e-12:
            best, bj, W_best = va, j, W
    return float(((Zte @ W_best + b).argmax(1) == yte).mean()), bj


def main() -> int:
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.policy.readout import FixedRandomProjection
    from flymario.retino_align import simulate
    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads((ART / "campaign_status.json").read_text())
    graphs = st["graphs"]
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    ticks = int(config.get("timing.ticks_per_decision"))
    data = {t: dict(np.load(ART / "data/b0" / f"{t}.npz")) for t in ("where", "loom")}
    idx = data["where"]["idx"]
    res_path = OUT / "probe.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {
        "label": "EXPLORATORY (decided after EXPERIENCE_COMMUNE_V1's result)", "graphs": {}}
    for g in graphs:
        if g in res["graphs"]:
            continue
        t0 = time.time()
        _g, W, motor = graphsel.load(g)
        net = simulate.FrozenNet(W, motor, idx, p, ticks, batch=128)
        rec = {}
        for t in ("where", "loom"):
            d = data[t]
            X, y, uid = d["final_x"], d["final_cls"].astype(int), d["final_uid"]
            o = net.run(X, uid, 31337)
            C = o.counts.astype(np.float64)
            R = np.asarray(FixedRandomProjection(len(motor), 8, int(d["readout_seed"])).W, np.float64)
            S8 = C @ R.T
            n = len(y)
            part = np.zeros(n, np.int8)
            for c in range(8):                                    # 128 train, 64 val, 64 test per class
                m = np.flatnonzero(y == c)
                part[m[128:192]], part[m[192:]] = 1, 2
            subs = balanced_subsets(y[part == 0], [9101 + r for r in range(10)], BUDGETS)
            for name, F in (("counts", C), ("scores8", S8)):
                acc = np.zeros((10, len(BUDGETS)))
                for r in range(10):
                    for k in range(len(BUDGETS)):
                        sel = subs[r, k][subs[r, k] >= 0]
                        acc[r, k], _ = ridge_8(F[part == 0][sel], y[part == 0][sel], F[part == 1], y[part == 1],
                                               F[part == 2], y[part == 2])
                rec[f"{t}_{name}"] = {"M": float(acc.mean()), "by_budget": acc.mean(0).tolist()}
            rec[f"{t}_rate_hz"] = float(C.mean() / (ticks * 3.33e-3))
        rec["seconds"] = round(time.time() - t0, 1)
        res["graphs"][g] = rec
        res_path.write_text(json.dumps(res, indent=1))
        print(g, {k: round(v["M"], 3) for k, v in rec.items() if isinstance(v, dict)}, rec["seconds"], "s", flush=True)
        del net
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
