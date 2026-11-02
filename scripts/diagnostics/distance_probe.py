#!/usr/bin/env python
"""Exploratory: is the *distance* of an enemy readable from the untrained brain?

Frames from scripted play on the training levels (fit) and unseen 5-1, 8-1 (test), labelled from RAM;
one sweep from rest, 2 noise seeds; the standardised C = .05 logistic decoder on motor counts and on
the lamina input. Tasks: distance (16-40 vs 48-96 px) and, for reference, presence (8-64 px vs none).

    .venv/bin/python scripts/diagnostics/distance_probe.py [--threat] [--fit L,..] [--test L,..]
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.connectome import datasets
from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.env.schematic import danger_ahead, threat_ahead
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
FIT = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")
TEST = ("Level5-1", "Level8-1")


def collect(g, levels, steps=1800, seed=20261014, label=danger_ahead):
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    rows = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        for n in range(steps):
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            dx = label(env.env.unwrapped.get_ram())
            if n % 2 == 0 and (dx <= 96 or dx > 160):
                vis.reset()
                vis.encode(prev)
                rows.append((lvl, dx, vis.encode(obs)))
            prev = obs
            if term or trunc:
                prev, _ = env.reset()
        env.close()
    return rows


def auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    keep = Xtr.std(0) > 0
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(Xtr[:, keep]), ytr)
    return round(float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte[:, keep])))), 4)


def main():
    import torch
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    W = g.to_sparse()
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    use_threat = "--threat" in sys.argv          # ground threats only (schematic.threat_ahead)
    fit_l = sys.argv[sys.argv.index("--fit") + 1].split(",") if "--fit" in sys.argv else list(FIT)
    test_l = sys.argv[sys.argv.index("--test") + 1].split(",") if "--test" in sys.argv else list(TEST)
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 1800
    rows = collect(g, tuple(fit_l) + tuple(test_l), steps=steps, label=threat_ahead if use_threat else danger_ahead)
    lv = np.array([r[0] for r in rows])
    dx = np.array([r[1] for r in rows])
    cur = np.stack([r[2] for r in rows], 1).astype(np.float32)
    cur2 = np.repeat(cur, 2, axis=1)
    out = []
    for s0 in range(0, cur2.shape[1], 64):
        n = min(64, cur2.shape[1] - s0)
        b = PlasticLIF(W, motor, params, seeds=[95000 + s0 + i for i in range(n)], cache_key=("ctrl", 0))
        c, _ = sweep_once(b, params, torch.as_tensor(cur2[:, s0:s0 + n], device=b.dev), K, 0.5, False)
        out.append(c.t().cpu().numpy())
        del b
        torch.cuda.empty_cache()
    C = np.concatenate(out, 0).reshape(len(rows), 2, -1).mean(1)
    vis = RetinotopicVision(g)
    L = cur[np.concatenate([vis.on_idx, vis.off_idx])].T
    fit, test = np.isin(lv, fit_l), np.isin(lv, test_l)
    tasks = {"distance: 16-40 vs 48-96 px": (((dx >= 16) & (dx <= 40)), ((dx >= 48) & (dx <= 96))),
             "presence: 8-64 px vs none within 160": (((dx >= 8) & (dx <= 64)), dx > 160)}
    res = {}
    for name, (pos, neg) in tasks.items():
        sel = pos | neg
        y = pos.astype(int)
        res[name] = {"n_fit": [int((fit & pos).sum()), int((fit & neg).sum())],
                     "n_test": [int((test & pos).sum()), int((test & neg).sum())],
                     "motor_code_auc": auc(C[fit & sel], y[fit & sel], C[test & sel], y[test & sel]),
                     "lamina_input_auc": auc(L[fit & sel], y[fit & sel], L[test & sel], y[test & sel])}
        print(name, res[name], flush=True)
    rep = {"what": "enemy distance vs presence, decodability from the untrained brain (exploratory)",
           "fit_levels": FIT, "test_levels": TEST, "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rep["label"] = "threat_ahead" if use_threat else "danger_ahead"
    rep["fit_levels"], rep["test_levels"], rep["steps"] = fit_l, test_l, steps
    tag = ("-threat" if use_threat else "") + ("-" + "_".join(x[5:] for x in fit_l) + "-to-" + "_".join(x[5:] for x in test_l)
                                               if "--fit" in sys.argv else "")
    (RUNS / f"distance_probe{tag}-20261005.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
