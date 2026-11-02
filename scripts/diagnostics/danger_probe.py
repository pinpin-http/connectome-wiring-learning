#!/usr/bin/env python
"""Does the untrained connectome carry a level-invariant "enemy just ahead" signal?

Exploratory, no learning. Scripted-run frames from six day levels, labelled from RAM: DANGER = an
active enemy 8-64 px ahead, CLEAR = none within 160 px. Logistic decoder (C=.05) on the 2,227 motor
spike counts of one frozen sweep, versus the lamina input. Train on 1-1 (or the four training
levels), test on unseen 5-1 and 8-1; shuffled-label reference for each.

    .venv/bin/python scripts/diagnostics/danger_probe.py
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
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.models.torch_lif import TorchLIF
from flymario.paths import RUNS

LEVELS = ("Level1-1", "Level2-1", "Level4-1", "Level7-1", "Level5-1", "Level8-1")
RUN, JUMP = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")


def collect(g, per_level=600, seed=20261003, levels=LEVELS):
    rng = np.random.default_rng(seed)
    vision = RetinotopicVision(g)
    rows = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        obs, info = env.reset()
        vision.reset()
        n = 0
        while n < per_level:
            a = JUMP if rng.random() < 0.25 else RUN
            obs, _r, term, trunc, info = env.step(a)
            cur = vision.encode(obs)
            dx = info["enemy_dx"]
            lab = 1 if 8 <= dx <= 64 else 0 if dx > 160 else -1
            if lab >= 0:
                rows.append((lvl, lab, cur))
            n += 1
            if term or trunc:
                obs, info = env.reset()
                vision.reset()
        env.close()
    return rows


def decode(Xtr, ytr, Xte, yte, shuffled=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64)
    keep = Xtr.std(0) > 0
    if shuffled:
        ytr = np.random.default_rng(20261003).permutation(ytr)
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(Xtr[:, keep]), ytr)
    p = clf.predict_proba(sc.transform(Xte[:, keep]))[:, 1]
    bal = 0.5 * (np.mean(p[yte == 1] > 0.5) + np.mean(p[yte == 0] <= 0.5))
    return {"balanced_accuracy": round(float(bal), 4), "auc": round(float(roc_auc_score(yte, p)), 4)}


def main():
    import argparse

    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", action="store_true", help="fit on 1-1, test on night levels 3-1 and 6-1")
    a = ap.parse_args()
    levels = ("Level1-1", "Level3-1", "Level6-1") if a.night else LEVELS
    test_levels = ["Level3-1", "Level6-1"] if a.night else ["Level5-1", "Level8-1"]
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    rows = collect(g, levels=levels)
    lv = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    cur = np.stack([r[2] for r in rows]).astype(np.float32)
    vision = RetinotopicVision(g)
    lam = np.concatenate([vision.on_idx, vision.off_idx])
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    mot = torch.as_tensor(motor, device="cuda", dtype=torch.long)
    X = np.zeros((len(rows), len(motor)), np.float32)
    K = int(config.get("timing.ticks_per_decision"))
    for s0 in range(0, len(rows), 64):
        idx = np.arange(s0, min(len(rows), s0 + 64))
        b = TorchLIF(g.to_sparse(), p, batch=len(idx), normalize=False,
                     seeds=[990000 + int(i) for i in idx], noise="torch", cache_key="malecns")
        c = torch.as_tensor(cur[idx].T.copy(), device="cuda")
        acc = torch.zeros((len(motor), len(idx)), device="cuda")
        for _ in range(K):
            acc += b.step(c).index_select(0, mot).float()
        X[idx] = acc.t().cpu().numpy()
        del b
    test = np.isin(lv, test_levels)
    splits = {"train_1-1": lv == "Level1-1"}
    if not a.night:
        splits["train_4_levels"] = np.isin(lv, ["Level1-1", "Level2-1", "Level4-1", "Level7-1"])
    res = {"counts": {L: {"danger": int(((lv == L) & (y == 1)).sum()), "clear": int(((lv == L) & (y == 0)).sum())}
                      for L in levels}}
    for name, tr in splits.items():
        res[name] = {}
        for feat, F in (("motor_counts", X), ("lamina_input", cur[:, lam])):
            res[name][feat] = {"heldout_" + "_".join(t[5:] for t in test_levels): decode(F[tr], y[tr], F[test], y[test]),
                               "shuffled": decode(F[tr], y[tr], F[test], y[test], shuffled=True)}
    rep = {"what": "untrained connectome: level-invariant enemy-ahead signal (exploratory)",
           "labels": "DANGER: enemy 8-64 px ahead; CLEAR: none within 160 px (RAM, analysis only)",
           "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"danger_probe{'_night' if a.night else ''}-20261003.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
