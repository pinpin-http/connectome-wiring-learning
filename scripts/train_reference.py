#!/usr/bin/env python
"""Task-accessibility references (declared in docs/RL_STUDY.md §5): linear softmax policies, no spiking network.

R1 sup-pixels: the eye's 3,575 lamina currents, cross-entropy on take-off label v2. R2 rl-pixels: same features,
actor-critic (v3 procedure). R3 rl-ram: privileged RAM features. Actor step c / d, c in {0.01, 0.03, 0.1}, 3 seeds.
--jump-costs (RL_STUDY §8): RL references with r_learn = (reward - cost x jumped - 75 died) / 10.

    .venv/bin/python scripts/train_reference.py train [--decisions 20000] [--tag 20261007]
    .venv/bin/python scripts/train_reference.py train --jump-costs 0,3,6 --tag jumpcost-20261008
    .venv/bin/python scripts/train_reference.py evaluate [--tag 20261007]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from flymario.env.retro_env import ACTION_NAMES
from flymario.env.schematic import _tile, takeoff_label_v2, threat_ahead
from flymario.paths import RUNS
from flymario.training.mario import EnvPool

TRAIN = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")
DEV = ("Level5-1", "Level8-1", "Level3-1", "Level6-1")
NA = len(ACTION_NAMES)
RUN, PRESS, NOOP = (ACTION_NAMES.index(a) for a in ("RIGHT+B", "RIGHT+A+B", "NOOP"))
JUMPS = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
GAMMA, LAM, EPS, SCALE, DEATH_EXTRA, CRITIC_LR = 0.98, 0.9, 0.1, 10.0, -75.0, 0.05
MAX_FRAMES, STALL = 6000, 150
OUT = RUNS / "reference"


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


def ram_features(ram):
    t = threat_ahead(ram)
    return np.array([1.0, float(takeoff_label_v2(ram)), float(wall_pit(ram)), float(16 <= t <= 40),
                     float(8 <= t <= 64), float(ram[0x1D] != 0)], np.float32)


class PixelFeatures:
    """The connectome's eye (RetinotopicVision), its 3,575 ON/OFF lamina currents."""

    def __init__(self, g):
        from flymario.env.vision import RetinotopicVision
        self.vis = RetinotopicVision(g)
        self.idx = np.concatenate([self.vis.on_idx, self.vis.off_idx])

    def reset(self):
        self.vis.reset()

    def __call__(self, frame):
        return self.vis.encode(frame)[self.idx].astype(np.float32)


class Learner:
    def __init__(self, name, kind, c, seed, d, jump_cost=0.0):
        self.name, self.kind, self.c, self.seed, self.jump_cost = name, kind, c, seed, jump_cost
        self.rng = np.random.default_rng(seed)
        self.d = d + 1 if kind != "rl-ram" else d               # pixels get a bias
        self.W = np.zeros((NA, self.d), np.float64)
        self.wv = np.zeros(self.d)
        self.E = np.zeros((NA, self.d))
        self.zv = np.zeros(self.d)
        self.mu_x, self.var_x, self.nx = np.zeros(d), np.ones(d), 0
        self.dvar, self.phi2 = 1.0, 1.0
        self.prev = None                                        # (V, r) awaiting the next state

    def feat(self, x):
        if self.kind == "rl-ram":
            return x.astype(np.float64)
        self.nx += 1
        a = max(1e-3, 1.0 / self.nx)
        self.mu_x += a * (x - self.mu_x)
        self.var_x += a * ((x - self.mu_x) ** 2 - self.var_x)
        return np.append((x - self.mu_x) / np.sqrt(self.var_x + 1e-6), 1.0)

    def feat_frozen(self, x):
        if self.kind == "rl-ram":
            return x.astype(np.float64)
        return np.append((x - self.mu_x) / np.sqrt(self.var_x + 1e-6), 1.0)

    def pi(self, xf):
        z = self.W @ xf
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, pi):
        mu = (1 - EPS) * pi + EPS / NA
        return int(self.rng.choice(NA, p=mu)), mu

    def reset_episode(self):
        self.E[:] = 0.0
        self.zv[:] = 0.0
        self.prev = None


def always_run_table(levels, delays):
    from oracle_criterion_check import play
    return {f"{lv}|{d}": play(lv, d, lambda ram: RUN)["best_x"] for lv in levels for d in delays}


def train(n_dec, tag, jump_costs=None):
    from flymario.connectome import datasets
    OUT.mkdir(parents=True, exist_ok=True)
    base_f = OUT / "always_run_train.json"
    if not base_f.exists():
        t0 = time.time()
        base_f.write_text(json.dumps(always_run_table(TRAIN, range(23))))
        print(f"always-run baseline computed in {time.time() - t0:.0f} s", flush=True)
    base = json.loads(base_f.read_text())
    g = datasets.load(datasets.DEFAULT)
    pix = PixelFeatures(g)
    dpix = len(pix.idx)
    learners = []
    if jump_costs is None:
        for kind in ("sup-pixels", "rl-pixels", "rl-ram"):
            for c in (0.01, 0.03, 0.1):
                for s in range(3):
                    learners.append(Learner(f"{kind}_c{c}_s{s}", kind, c, 20261070 + s,
                                            dpix if kind != "rl-ram" else 6))
    else:
        for kind in ("rl-ram", "rl-pixels"):
            for jc in jump_costs:
                for s in range(3):
                    learners.append(Learner(f"{kind}_c0.1_jc{jc:g}_s{s}", kind, 0.1, 20261070 + s,
                                            dpix if kind != "rl-ram" else 6, jump_cost=jc))
    B = len(learners)
    eyes = [PixelFeatures(g) if l.kind != "rl-ram" else None for l in learners]
    pool = EnvPool(B, {"state": "Level1-1", "frame_skip": 4, "max_frames": MAX_FRAMES})
    ep_rng = [np.random.default_rng([20261071, b]) for b in range(B)]
    obs, info, ep, pal = [None] * B, [None] * B, [None] * B, [None] * B

    def restart(b):
        lv = str(ep_rng[b].choice(TRAIN))
        dl = int(ep_rng[b].integers(0, 23))
        o, i = pool.reset_to(b, lv, dl)
        obs[b], info[b] = o, i
        pal[b] = (ep_rng[b].permutation(3), float(ep_rng[b].uniform(0.75, 1.25))) if learners[b].kind != "rl-ram" else None
        if eyes[b] is not None:
            eyes[b].reset()
        learners[b].reset_episode()
        ep[b] = {"level": lv, "delay": dl, "best_x": i.get("mario_x", 0.0), "best_p": i.get("progress", 0.0),
                 "since": 0, "n": 0, "jumps": 0, "ret": 0.0, "ent": 0.0}

    for b in range(B):
        restart(b)
    episodes, t0 = [], time.time()
    for k in range(n_dec):
        acts, cache = np.zeros(B, int), []
        for b, L in enumerate(learners):
            if L.kind == "rl-ram":
                xf = L.feat(ram_features(info[b]["ram"]))
            else:
                f = obs[b] if pal[b] is None else np.clip(obs[b][..., pal[b][0]].astype(np.float32) * pal[b][1],
                                                           0, 255).astype(np.uint8)
                xf = L.feat(eyes[b](f))
            p = L.pi(xf)
            a, mu = L.act(p)
            acts[b] = a
            cache.append((xf, p, mu))
        out = pool.step(acts)
        for b, L in enumerate(learners):
            xf, p, mu = cache[b]
            o, r, term, trunc, inf = out[b]
            e = ep[b]
            e["n"] += 1
            e["jumps"] += int(acts[b] in JUMPS)
            e["ret"] += r
            e["ent"] += float(-(p * np.log(p + 1e-12)).sum())
            e["best_x"] = max(e["best_x"], inf.get("mario_x", 0.0))
            if inf.get("progress", 0.0) > e["best_p"]:
                e["best_p"], e["since"] = inf["progress"], 0
            else:
                e["since"] += 1
            stalled = e["since"] >= STALL
            done = term or trunc or stalled
            died, cleared = bool(inf.get("died")), bool(inf.get("cleared"))
            truncated = done and not died and not cleared
            if L.kind == "sup-pixels":
                tgt = PRESS if takeoff_label_v2(info[b]["ram"]) else RUN
                L.W += (L.c / L.d) * np.outer(np.eye(NA)[tgt] - p, xf)
            else:
                # actor-critic, v3 procedure: TD error of the previous decision, now that its next state is seen
                V = L.wv @ xf
                if L.prev is not None:
                    Vp, rp = L.prev
                    delta = rp + GAMMA * V - Vp
                    L.dvar = 0.99 * L.dvar + 0.01 * delta * delta
                    L.W += (L.c / L.d) * (delta / np.sqrt(L.dvar + 1e-8)) * L.E
                    L.wv += CRITIC_LR * delta * L.zv / L.phi2
                    V = L.wv @ xf
                w = (1 - EPS) * p[acts[b]] / mu[acts[b]]
                L.E = GAMMA * LAM * L.E + w * np.outer(np.eye(NA)[acts[b]] - p, xf)
                L.zv = GAMMA * LAM * L.zv + xf
                L.phi2 = 0.99 * L.phi2 + 0.01 * float(xf @ xf) + 0.01
                r_l = (r - L.jump_cost * float(acts[b] in JUMPS) + DEATH_EXTRA * died) / SCALE
                if done:
                    delta = r_l + (GAMMA * V if truncated else 0.0) - V
                    L.dvar = 0.99 * L.dvar + 0.01 * delta * delta
                    L.W += (L.c / L.d) * (delta / np.sqrt(L.dvar + 1e-8)) * L.E
                    L.wv += CRITIC_LR * delta * L.zv / L.phi2
                else:
                    L.prev = (V, r_l)
            obs[b], info[b] = o, inf
            if done:
                end = "flag" if cleared else "death" if died else "stall" if stalled else "time"
                episodes.append({"learner": L.name, "at_decision": k + 1, "level": e["level"], "delay": e["delay"],
                                 "best_x": e["best_x"], "dx": e["best_x"] - base[f"{e['level']}|{e['delay']}"],
                                 "end": end, "decisions": e["n"], "jump_share": round(e["jumps"] / e["n"], 3),
                                 "entropy": round(e["ent"] / e["n"], 3), "return": round(e["ret"], 1)})
                restart(b)
        if (k + 1) % 2000 == 0 or k + 1 == n_dec:
            snap = {L.name: {"W": L.W, "wv": L.wv, "mu_x": L.mu_x, "var_x": L.var_x, "kind": L.kind, "c": L.c}
                    for L in learners}
            np.save(OUT / f"weights-{tag}-{k + 1}.npy", snap, allow_pickle=True)
            (OUT / f"train-{tag}.json").write_text(json.dumps({"decisions": k + 1, "episodes": episodes}))
            print(f"{k + 1} decisions, {time.time() - t0:.0f} s, {len(episodes)} episodes", flush=True)
    pool.close()


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "20261007"
    if sys.argv[1] == "train":
        n = int(sys.argv[sys.argv.index("--decisions") + 1]) if "--decisions" in sys.argv else 20000
        jc = [float(x) for x in sys.argv[sys.argv.index("--jump-costs") + 1].split(",")] if "--jump-costs" in sys.argv else None
        train(n, tag, jc)
    else:
        raise SystemExit("evaluate: added after training (RL_STUDY §5)")
    print(datetime.now(UTC).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
