#!/usr/bin/env python
"""Study S follow-up F2 (exploratory, docs/STUDY_S.md): how class-specific is the UNTRAINED readout layer?

Per graph and task, the 200 held-out test stimuli are swept once through the untrained network; reports eta2
(between-class share of readout-count variance), dec_scores (ridge decodability from readout 0's 8 scores),
dec_sub50 (from 50 random readout neurons, 5 draws) and rate. Appends to artifacts/runs/study_s/mechanism.json.

    .venv/bin/python scripts/diagnostics/study_s_mechanism.py --graphs connectome nulllat:0 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_s as S  # noqa: E402

OUT = ROOT / "artifacts/runs/study_s/mechanism.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=6000)
    a = ap.parse_args()
    import torch
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.model_selection import cross_val_score

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(4)
    g0, _w, _m = graphsel.load("connectome")
    del _w
    St, idx = S.build_stimuli(g0, a.trials)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    B = 20
    for gname in a.graphs:
        g, W, motor = graphsel.load(gname)
        b = PlasticLIF(W, motor, params, seeds=[8300 + i for i in range(B)], cache_key=None)
        R0 = np.asarray(FixedRandomProjection(b.M, 8, block_seeds(3)[0]).W)
        entry = {}
        for task in S.TASKS:
            X, cls = St[f"test_{task}_x"], St[f"test_{task}_cls"]
            F = []
            for q in range(0, X.shape[1], B):
                cols = X[:, q:q + B]
                cur = torch.zeros((b.n, B), device=b.dev)
                cur[torch.as_tensor(idx, device=b.dev), :cols.shape[1]] = torch.as_tensor(cols, device=b.dev)
                c, _ = sweep_once(b, params, cur, K, 0.5, False)
                F.append(c[:, :cols.shape[1]].T.cpu().numpy())
            F = np.concatenate(F)                                         # (200, M)
            mu = np.stack([F[cls == k].mean(0) for k in range(8)])
            between = ((mu - F.mean(0)) ** 2).mean(0).sum()
            total = F.var(0).sum()
            rc = RidgeClassifierCV(alphas=np.logspace(-2, 5, 15))
            rng = np.random.default_rng(7)
            sub = [cross_val_score(rc, F[:, rng.choice(F.shape[1], 50, replace=False)], cls, cv=5).mean() for _ in range(5)]
            entry[task] = {"eta2": round(float(between / total), 4),
                           "dec_scores": round(float(cross_val_score(rc, F @ R0.T, cls, cv=5).mean()), 4),
                           "dec_sub50": round(float(np.mean(sub)), 4),
                           "rate": round(float(F.sum(1).mean()), 1)}
        res[gname] = entry
        print(gname, json.dumps(entry), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
        del b, W


if __name__ == "__main__":
    main()
