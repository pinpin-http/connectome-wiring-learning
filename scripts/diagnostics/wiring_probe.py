#!/usr/bin/env python
"""Does the wiring give the untrained motor layer a more usable game code than its nulls?

Exploratory, docs/PERCEPTION_CREDIT_STUDY.md §19 S4. Scripted-play frames (fit: 1-1, 4-1; test: 5-1, 8-1),
one sweep from rest per frame and graph; C = .05 logistic decoder on motor counts, AUC on the test levels,
for RAM labels (take-off v2, obstacle, ground threat); the lamina input is the reference. --matched uses
the nulls' rate-matched gains; --graphs writes wiring_probe-<first graph kind>-20261006.json.

    .venv/bin/python scripts/diagnostics/wiring_probe.py [--device cpu] [--matched] [--graphs nulllat:0 ...]
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
from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.env.schematic import _tile, takeoff_label_v2, threat_ahead
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
FIT, TEST = ("Level1-1", "Level4-1"), ("Level5-1", "Level8-1")


def wall_pit(ram):
    ram = ram.astype(int)
    if ram[0xB5] != 1:
        return False
    mx = ram[0x6D] * 256 + ram[0x86]
    feet = ram[0xCE] + 32
    top = ram[0xCE] + (0 if ram[0x754] == 0 else 16)
    body_rows = sorted({(y - 32) // 16 for y in range(top, feet, 8)})
    feet_row = (feet - 32) // 16
    wall = any(_tile(ram, mx + 16 + dx, r) for dx in (4, 12) for r in body_rows)
    pit = any(not any(_tile(ram, mx + 8 + dx, r) for r in range(feet_row, 13)) for dx in (8, 16))
    return bool(wall or pit)


def collect(g, steps=1500, every=5, seed=20261016):
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    rows = []
    for lvl in FIT + TEST:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        for n in range(steps):
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram()
            if n % every == 0:
                vis.reset()
                vis.encode(prev)
                rows.append((lvl, takeoff_label_v2(ram), wall_pit(ram), threat_ahead(ram), vis.encode(obs)))
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


def motor_code(W, motor, cur, gain, spec, dev):
    import torch
    params = LIFParams(gain=float(gain), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    out = []
    for s0 in range(0, cur.shape[1], 64):
        n = min(64, cur.shape[1] - s0)
        b = PlasticLIF(W, motor, params, seeds=[96000 + s0 + i for i in range(n)], cache_key=("probe", spec, dev),
                       device=dev)
        c, _ = sweep_once(b, params, torch.as_tensor(cur[:, s0:s0 + n], device=b.dev), K, 0.5, False)
        out.append(c.t().cpu().numpy())
        del b
    return np.concatenate(out, 0)


def main():
    from flymario.connectome import graphsel
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    g0 = float(config.get("dynamics.gain"))
    conds = [("connectome", "connectome", g0)] + [(f"null:{s}", f"null:{s}", g0) for s in (0, 1, 2)]
    if "--matched" in sys.argv:
        rm = json.loads((RUNS / "null_rate_match-20261006.json").read_text())["nulls"]
        conds = [(f"null:{s}@matched", f"null:{s}", rm[f"null:{s}"]["gain"]) for s in (0, 1, 2)]
    if "--graphs" in sys.argv:
        conds = [(x, x, g0) for x in sys.argv[sys.argv.index("--graphs") + 1:] if ":" in x or x == "connectome"]
    g, _W, motor = graphsel.load("connectome")
    del _W
    rows = collect(g)
    lv = np.array([r[0] for r in rows])
    tko, obst = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    thr = np.array([r[3] for r in rows])
    cur = np.stack([r[4] for r in rows], 1).astype(np.float32)
    fit, test = np.isin(lv, FIT), np.isin(lv, TEST)
    tasks = {"takeoff_v2": (tko, ~tko), "obstacle": (obst, ~obst),
             "threat_8_64_vs_none_160": ((thr >= 8) & (thr <= 64), thr > 160)}
    counts = {k: {"fit": [int((fit & p).sum()), int((fit & q).sum())], "test": [int((test & p).sum()), int((test & q).sum())]}
              for k, (p, q) in tasks.items()}
    print("frames", len(rows), counts, flush=True)
    vis = RetinotopicVision(g)
    lam = cur[np.concatenate([vis.on_idx, vis.off_idx])].T

    def score(X):
        return {k: auc(X[fit & (p | q)], p[fit & (p | q)].astype(int), X[test & (p | q)], p[test & (p | q)].astype(int))
                for k, (p, q) in tasks.items()}

    res = {} if ("--matched" in sys.argv or "--graphs" in sys.argv) else {"lamina_input": score(lam)}
    if res:
        print("lamina_input", res["lamina_input"], flush=True)
    for name, spec, gain in conds:
        _, W, _ = graphsel.load(spec)
        C = motor_code(W, motor, cur, gain, spec, dev)
        del W
        res[name] = score(C) | {"gain": gain, "motor_spikes_per_decision": round(float(C.sum(1).mean()), 1)}
        print(name, res[name], flush=True)
    rep = {"what": "§19 S4: decodability of game labels from the untrained motor code, connectome vs nulls (exploratory)",
           "fit_levels": FIT, "test_levels": TEST, "counts": counts, "device": dev, "results": res,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    sfx = "-matched" if "--matched" in sys.argv else f"-{conds[0][1].split(':')[0]}" if "--graphs" in sys.argv else ""
    out = RUNS / f"wiring_probe{sfx}-20261006.json"
    out.write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
