#!/usr/bin/env python
"""Closed-loop check: does a danger readout learnt on 1-1 steer Mario on unseen levels?

Exploratory and supervised. A logistic readout on the untrained connectome's motor spike counts
separates "enemy 8-64 px ahead" from "no enemy within 160 px" (RAM labels, 1-1 frames), then plays:
jump (RIGHT+A+B) when P(danger) > 0.5, else run (RIGHT+B). Controls: raw_input (readout on the lamina
current), always_run, random_jump (at the motor controller's jump rate). Levels 1-1, 5-1, 8-1 (day),
3-1, 6-1 (night); starts after 0, 5, 10 NOOPs. Measure: furthest x, episode end.

    .venv/bin/python scripts/diagnostics/danger_controller.py
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
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMP, NOOP = (ACTION_NAMES.index(a) for a in ("RIGHT+B", "RIGHT+A+B", "NOOP"))
TEST = ("Level1-1", "Level5-1", "Level8-1", "Level3-1", "Level6-1")
DELAYS = (0, 5, 10)
MAX_DEC, STALL = 600, 150


class Brain:
    def __init__(self, g, motor):
        self.p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
        self.K = int(config.get("timing.ticks_per_decision"))
        self.W, self.motor = g.to_sparse(), motor
        self.brains = {}

    def counts(self, cur, seed0):
        """Motor counts of one sweep from rest per column of `cur` (n, B); one
        persistent brain per batch width (its noise keeps running between sweeps)."""
        import torch
        B = 1 if cur.shape[1] == 1 else 64
        if B not in self.brains:
            self.brains[B] = PlasticLIF(self.W, self.motor, self.p, seeds=[seed0 + i for i in range(B)],
                                        cache_key=("ctrl", 0))
        b, out = self.brains[B], []
        for s0 in range(0, cur.shape[1], B):
            c = cur[:, s0:s0 + B]
            n = c.shape[1]
            if n < B:
                c = np.concatenate([c, np.zeros((c.shape[0], B - n), c.dtype)], 1)
            x, _ = sweep_once(b, self.p, torch.as_tensor(c, device=b.dev), self.K, 0.5, False)
            out.append(x.t().cpu().numpy()[:n])
        return np.concatenate(out, 0)


def collect_1_1(g, steps=2400, seed=20261003):
    """Scripted play on 1-1 (run, jump with p .25, random NOOP start): labelled currents."""
    rng = np.random.default_rng(seed)
    vision = RetinotopicVision(g)
    env = MarioEnv(EnvConfig(state="Level1-1"))
    rows, n = [], 0

    def restart():
        obs, info = env.reset()
        vision.reset()
        for _ in range(int(rng.integers(0, 26))):
            obs, _r, _t, _u, info = env.step(NOOP)
        return obs, info
    restart()
    while n < steps:
        obs, _r, term, trunc, info = env.step(JUMP if rng.random() < 0.25 else RUN)
        cur = vision.encode(obs)
        dx = info["enemy_dx"]
        lab = 1 if 8 <= dx <= 64 else 0 if dx > 160 else -1
        if lab >= 0:
            rows.append((lab, cur))
        n += 1
        if term or trunc:
            restart()
    env.close()
    return np.array([r[0] for r in rows]), np.stack([r[1] for r in rows], 1).astype(np.float32)


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    keep = X.std(0) > 0
    sc = StandardScaler().fit(X[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(X[:, keep]), y)
    return lambda Z: clf.predict_proba(sc.transform(Z[:, keep]))[:, 1]


def play(level, delay, policy, g, rng):
    """policy(cur, info) -> action. Returns the episode record."""
    env = MarioEnv(EnvConfig(state=level))
    vision = RetinotopicVision(g)
    obs, info = env.reset()
    for _ in range(delay):
        obs, _r, _t, _u, info = env.step(NOOP)
    best, since, jumps, end, n = info["mario_x"], 0, 0, None, 0
    cur = vision.encode(obs)
    while end is None:
        a = policy(cur, rng)
        jumps += a == JUMP
        obs, _r, term, trunc, info = env.step(a)
        cur = vision.encode(obs)
        n += 1
        if info["mario_x"] > best:
            best, since = info["mario_x"], 0
        else:
            since += 1
        end = ("flag" if info["cleared"] else "death" if info["died"] else "stall" if since >= STALL
               else "time" if (term or trunc or n >= MAX_DEC) else None)
    env.close()
    return {"level": level, "delay": delay, "best_x": int(best), "end": end, "decisions": n,
            "jump_rate": round(jumps / max(1, n), 3)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=",".join(TEST))
    ap.add_argument("--delays", default=",".join(map(str, DELAYS)))
    ap.add_argument("--collect-seed", type=int, default=20261003)
    ap.add_argument("--noise-seed", type=int, default=880000)
    ap.add_argument("--tag", default="20261003")
    a = ap.parse_args()
    levels = a.levels.split(",")
    delays = [int(x) for x in a.delays.split(",")]
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    brain = Brain(g, motor)
    lam = np.concatenate([RetinotopicVision(g).on_idx, RetinotopicVision(g).off_idx])
    y, cur = collect_1_1(g, seed=a.collect_seed)
    Xm = brain.counts(cur, a.noise_seed - 110000)
    p_motor, p_raw = fit(Xm, y), fit(cur[lam].T, y)
    print(f"fitted on {len(y)} 1-1 frames ({int(y.sum())} danger)", flush=True)
    def motor_policy(c, rng):
        x = brain.counts(c[:, None], a.noise_seed)
        return JUMP if p_motor(x)[0] > 0.5 else RUN

    def raw_policy(c, rng):
        return JUMP if p_raw(c[lam][None, :])[0] > 0.5 else RUN

    eps = []
    for level in levels:
        for d in delays:
            rng = np.random.default_rng(a.collect_seed + 1000 * levels.index(level) + d)
            m = play(level, d, motor_policy, g, rng)
            rate = m["jump_rate"]
            row = {"motor_readout": m,
                   "raw_input": play(level, d, raw_policy, g, rng),
                   "always_run": play(level, d, lambda c, r: RUN, g, rng),
                   "random_jump": play(level, d, lambda c, r, q=rate: JUMP if r.random() < q else RUN, g, rng)}
            eps.append(row)
            print(f"{level} delay {d:2d}: " + "  ".join(f"{k}={v['best_x']}({v['end'][0]})" for k, v in row.items()),
                  flush=True)
    summ = {}
    for k in ("motor_readout", "raw_input", "always_run", "random_jump"):
        summ[k] = {}
        for grp, lv in (("fit_level_1-1", ("Level1-1",)),
                        ("unseen_day", ("Level2-1", "Level4-1", "Level5-1", "Level7-1", "Level8-1")),
                        ("unseen_night", ("Level3-1", "Level6-1"))):
            x = np.array([e[k]["best_x"] for e in eps if e[k]["level"] in lv], float)
            if len(x) == 0:
                continue
            summ[k][grp] = {"mean_x": round(float(x.mean()), 1), "median_x": float(np.median(x)),
                            "se_x": round(float(x.std(ddof=1) / np.sqrt(len(x))), 1) if len(x) > 1 else None,
                            "n": len(x)}
    # paired tests, motor readout against each control, unseen day levels
    from scipy.stats import wilcoxon
    day = [e for e in eps if e["motor_readout"]["level"] not in ("Level1-1", "Level3-1", "Level6-1")]
    if len(day) > 5:
        m = np.array([e["motor_readout"]["best_x"] for e in day], float)
        summ["paired_unseen_day"] = {}
        for k in ("raw_input", "always_run", "random_jump"):
            o = np.array([e[k]["best_x"] for e in day], float)
            nz = m != o
            p = float(wilcoxon(m[nz], o[nz], alternative="greater").pvalue) if nz.sum() > 0 else 1.0
            summ["paired_unseen_day"][k] = {"n": len(m), "wins": int((m > o).sum()), "losses": int((m < o).sum()),
                                            "mean_diff": round(float((m - o).mean()), 1), "wilcoxon_p_greater": round(p, 5)}
    rep = {"what": "closed-loop danger readout on the frozen connectome, fitted on 1-1 (exploratory, supervised)",
           "n_fit_frames": len(y), "n_fit_danger": int(y.sum()), "summary": summ, "episodes": eps,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rep["levels"], rep["delays"], rep["collect_seed"], rep["noise_seed"] = levels, delays, a.collect_seed, a.noise_seed
    (RUNS / f"danger_controller-{a.tag}.json").write_text(json.dumps(rep, indent=1, default=int))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
