#!/usr/bin/env python
"""Instrument check for PERCEPTION_CREDIT_STUDY (post hoc): is "danger" easier to read from the
untrained brain's motor code in the schematic view than in the game view?

Frames: the criterion frames of scripts/gates/criterion_e_check.py (5-1, 8-1), in both views; one
sweep per frame, 2 noise seeds; standardised C=.05 logistic decoder trained on one level and tested on
the other, also on the lamina current. The premise holds if the schematic's motor AUC >= the game's.

    .venv/bin/python scripts/diagnostics/oracle_premise_check.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from criterion_e_check import frames

from flymario import config
from flymario.connectome import datasets
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once


def auc_cross_level(X, y, lv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    out = {}
    for tr_l, te_l in (("Level5-1", "Level8-1"), ("Level8-1", "Level5-1")):
        tr, te = lv == tr_l, lv == te_l
        keep = X[tr].std(0) > 0
        sc = StandardScaler().fit(X[tr][:, keep])
        clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(X[tr][:, keep]), y[tr])
        out[f"train_{tr_l[5:]}_test_{te_l[5:]}"] = round(float(roc_auc_score(y[te], clf.decision_function(
            sc.transform(X[te][:, keep])))), 4)
    out["mean"] = round(float(np.mean(list(out.values()))), 4)
    return out


def main():
    import torch
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    W = g.to_sparse()
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    F = frames(g)
    y = np.array([f[1] for f in F])
    lv = np.array([f[0] for f in F])
    vis = RetinotopicVision(g)
    lam = np.concatenate([vis.on_idx, vis.off_idx])
    res = {}
    for view, k in (("pixels", 2), ("schematic", 3)):
        cur = np.stack([f[k] for f in F], 1).astype(np.float32)
        cur2 = np.repeat(cur, 2, axis=1)
        out = []
        for s0 in range(0, cur2.shape[1], 64):
            n = min(64, cur2.shape[1] - s0)
            b = PlasticLIF(W, motor, params, seeds=[90000 + s0 + i for i in range(n)], cache_key=("ctrl", 0))
            c, _ = sweep_once(b, params, torch.as_tensor(cur2[:, s0:s0 + n], device=b.dev), K, 0.5, False)
            out.append(c.t().cpu().numpy())
            del b
            torch.cuda.empty_cache()
        C = np.concatenate(out, 0).reshape(len(F), 2, -1).mean(1)
        res[view] = {"motor_code": auc_cross_level(C, y, lv), "lamina_input": auc_cross_level(cur[lam].T, y, lv),
                     "motor_spikes_per_frame": round(float(C.sum(1).mean()), 1)}
        print(view, res[view], flush=True)
    ok = res["schematic"]["motor_code"]["mean"] >= res["pixels"]["motor_code"]["mean"]
    rep = {"what": "oracle premise: danger decodability, schematic vs game (post hoc instrument check)",
           "n_frames": len(F), "results": res, "premise_holds": ok,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "oracle_premise_check-20261004.json").write_text(json.dumps(rep, indent=1))
    print("premise holds:", ok)


if __name__ == "__main__":
    main()
