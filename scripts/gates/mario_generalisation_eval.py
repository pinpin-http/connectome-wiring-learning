#!/usr/bin/env python
"""Generalisation tests of the `mario_gen` learners (docs/GENERALISATION_STUDY.md §4).

Weights frozen, greedy action; the six trained learners and, per readout, the untrained brain play at once.
Suites: trained 1-1, desynchronised 1-1, unseen day (5-1, 8-1) and night (3-1, 6-1) levels. Measures:
furthest x, episode ending, enemy encounters survived; occlusion: change in P(action) and P(jump) when
the enemy's box is erased versus two scenery boxes.

    .venv/bin/python scripts/gates/mario_generalisation_eval.py --ckpt artifacts/runs/checkpoints/mario_gen.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.training.mario import EnvPool
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

JUMP = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
TAU = 2.0
SUITES = {"trained": [("Level1-1", "fixed0")],
          "desynchronised": [("Level1-1", "desync")],
          "unseen_day": [("Level5-1", "spread"), ("Level8-1", "spread")],
          "unseen_night": [("Level3-1", "spread"), ("Level6-1", "spread")]}


def delays(kind, rng, n):
    if kind == "fixed0":
        return [0] * n
    if kind == "desync":
        return [int(x) for x in rng.integers(6, 26, n)]
    return [0, 5, 10][:n]


class Evaluator:
    def __init__(self, ckpt, episodes, max_dec, policy="greedy", graph="connectome"):
        import torch
        self.torch = torch
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.trained_names = [l["name"] for l in st["learners"]]
        from flymario.connectome import graphsel
        g, self.W, self.motor = graphsel.load(graph)          # the wiring the brains were trained on
        self.g = g
        self.params = LIFParams(gain=float(config.get("dynamics.gain")),
                                noise_std=float(config.get("dynamics.noise_std")))
        self.K = int(config.get("timing.ticks_per_decision"))
        ro = block_seeds(3)
        self.names = self.trained_names + [f"untrained_ro{i + 1}" for i in range(3)]
        self.readout = [l["readout_seed"] for l in st["learners"]] + ro
        self.seeds = [l["noise_seed"] for l in st["learners"]] + [7300, 7301, 7302]
        self.B = len(self.names)
        # each brain sees what it was trained on: the game, or the RAM schematic
        self.views = [l.get("view", "pixels") for l in st["learners"]] + ["pixels"] * 3
        self.brain = PlasticLIF(self.W, self.motor, self.params, seeds=self.seeds, cache_key=("eval", graph))
        m = self.brain.m0[None, :].repeat(self.B, 1).clone()
        m[:len(self.trained_names)] = st["brain"]["m"].to(self.brain.dev)
        self.brain.set_weights(m)
        self.m = m
        self.R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(self.brain.M, 8, r).W)
                                           for r in self.readout]), device=self.brain.dev)
        self.visions = [RetinotopicVision(g) for _ in range(self.B)]
        self.pool = EnvPool(self.B, {"state": "Level1-1", "frame_skip": 4, "max_frames": 6000})
        self.episodes, self.max_dec = episodes, max_dec
        self.policy = policy
        self.rng = np.random.default_rng(20261005)
        self.samples = []                                   # occlusion material

    def scores(self, frames):
        torch = self.torch
        cur = torch.as_tensor(np.stack([v.encode(f) for v, f in zip(self.visions, frames)], 1),
                              device=self.brain.dev)
        c, _ = sweep_once(self.brain, self.params, cur, self.K, 0.5, False)
        return torch.einsum("bam,mb->ba", self.R, c).cpu().numpy()

    def view_frame(self, b, obs, info):
        if self.views[b] == "schematic":
            from flymario.env.schematic import schematic_frame
            return schematic_frame(info["ram"], obs.shape)
        return obs

    def episode(self, suite, level, delay):
        B = self.B
        obs, info = [None] * B, [None] * B
        for b in range(B):
            obs[b], info[b] = self.pool.reset_to(b, level, delay)
            self.visions[b].reset()
        active = np.ones(B, bool)
        rec = [{"suite": suite, "level": level, "delay": delay, "learner": self.names[b],
                "best_x": info[b]["mario_x"], "decisions": 0, "end": None, "close_calls": 0,
                "died_near_enemy": False, "best_progress": info[b]["progress"], "since": 0,
                "was_close": False, "jumps": 0} for b in range(B)]
        frames = [self.view_frame(b, obs[b], info[b]) for b in range(B)]
        while active.any():
            sc = self.scores(frames)
            if self.policy == "greedy":
                act = sc.argmax(1)
            else:                                     # sample the learnt policy, no floor
                z = sc / TAU
                p = np.exp(z - z.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                act = np.array([self.rng.choice(len(row), p=row) for row in p])
            for b in np.flatnonzero(active):
                self.pool.conns[b].send(("step", int(act[b])))
                rec[b]["jumps"] += int(int(act[b]) in JUMP)
            for b in np.flatnonzero(active):
                o, _r, term, trunc, inf = self.pool.conns[b].recv()
                e = rec[b]
                e["decisions"] += 1
                close = inf.get("enemy_dx", 999) < 40
                if e["was_close"] and not close and not inf.get("died"):
                    e["close_calls"] += 1
                e["was_close"] = close
                e["best_x"] = max(e["best_x"], inf.get("mario_x", 0))
                if inf.get("progress", 0) > e["best_progress"]:
                    e["best_progress"], e["since"] = inf["progress"], 0
                else:
                    e["since"] += 1
                # occlusion material: a trained learner, an enemy ahead and on screen
                if (b < len(self.trained_names) and suite in ("desynchronised", "unseen_day")
                        and 24 <= inf.get("enemy_dx", 999) <= 72 and 12 < inf.get("enemy_sx", -1) < 228
                        and sum(s["b"] == b and s["suite"] == suite for s in self.samples) < 14):
                    self.samples.append({"b": int(b), "suite": suite, "frame": o.copy(),
                                         "sx": inf["enemy_sx"], "sy": inf["enemy_sy"],
                                         "mx": inf.get("mario_sx", 0), "my": inf.get("mario_sy", 0)})
                end = ("flag" if inf.get("cleared") else "death" if inf.get("died") else
                       "stall" if e["since"] >= 150 else "time" if (term or trunc or e["decisions"] >= self.max_dec) else None)
                if inf.get("died") and inf.get("enemy_dx", 999) < 40:
                    e["died_near_enemy"] = True
                if end:
                    e["end"], active[b] = end, False
                frames[b] = self.view_frame(b, o, inf)
        for e in rec:
            for k in ("best_progress", "since", "was_close"):
                e.pop(k)
        return rec

    def run_suites(self):
        rng = np.random.default_rng(20261003)
        out = []
        for suite, items in SUITES.items():
            for level, kind in items:
                for d in delays(kind, rng, self.episodes):
                    t0 = time.time()
                    rec = self.episode(suite, level, d)
                    out += rec
                    print(f"{suite:14s} {level} delay {d:2d}: " + " ".join(
                        f"{r['learner'][:4]}{r['learner'][-1]}={r['best_x']:.0f}" for r in rec)
                        + f"  ({time.time() - t0:.0f}s)", flush=True)
        return out

    def occlusion(self):
        """Paired: original / enemy erased / scenery above / scenery ahead, 2 noise seeds."""
        torch = self.torch
        res = []
        variants = ("original", "enemy", "scenery1", "scenery2", "swap")
        chunk = 5                                             # samples per batch
        for s0 in range(0, len(self.samples), chunk):
            S = self.samples[s0:s0 + chunk]
            frames, cols, seeds = [], [], []
            for i, smp in enumerate(S):
                for v in variants:
                    for k in range(2):
                        if v == "swap":           # positive reference: another sample's whole frame
                            other = self.samples[(s0 + i + len(self.samples) // 2) % len(self.samples)]
                            frames.append(other["frame"].copy())
                        else:
                            frames.append(paint(smp, v))
                        cols.append(smp["b"])
                        seeds.append(50000 + 10 * (s0 + i) + k)
            br = PlasticLIF(self.W, self.motor, self.params, seeds=seeds, cache_key=("eval", 0))
            br.set_weights(self.m[cols])
            vis = RetinotopicVision(self.g)
            cur = []
            for f in frames:
                vis.reset()
                cur.append(vis.encode(f))
            cur = torch.as_tensor(np.stack(cur, 1), device=br.dev)
            c, _ = sweep_once(br, self.params, cur, self.K, 0.5, False)
            R = self.R[cols]
            sc = torch.einsum("bam,mb->ba", R, c)
            pi = torch.softmax(sc / TAU, 1).cpu().numpy().reshape(len(S), len(variants), 2, 8).mean(2)
            z = (sc / TAU).cpu().numpy().reshape(len(S), len(variants), 2, 8)
            for i, smp in enumerate(S):
                p0 = pi[i, 0]
                a0 = int(p0.argmax())
                row = {"learner": self.names[smp["b"]], "suite": smp["suite"], "action": ACTION_NAMES[a0]}
                # log-odds of jumping against the chosen action: not flattened by a saturated softmax
                def lo(zz, a0=a0):
                    m = zz[:, JUMP].max(1, keepdims=True)
                    return (np.log(np.exp(zz[:, JUMP] - m).sum(1)) + m[:, 0] - zz[:, a0]).mean()
                lo0 = lo(z[i, 0])
                for j, v in enumerate(variants[1:], 1):
                    row[f"d_chosen_{v}"] = float(pi[i, j, a0] - p0[a0])
                    row[f"d_jump_{v}"] = float(pi[i, j, JUMP].sum() - p0[JUMP].sum())
                    row[f"d_logodds_{v}"] = float(lo(z[i, j]) - lo0)
                res.append(row)
            del br
            torch.cuda.empty_cache()
        return res


BOX = 20


def boxes(smp):
    """The enemy's box and the two scenery boxes with the most visual structure
    (pixel variance) among shifted candidates that overlap neither the enemy nor
    Mario -- a control that can actually change the picture."""
    f = smp["frame"]
    ex, ey = int(smp["sx"]) - 10, int(smp["sy"]) - 2
    mx, my = int(smp["mx"]) - 10, int(smp["my"]) + 6

    def ok(x, y):
        if x < 0 or y < 32 or x + BOX > f.shape[1] or y + BOX > f.shape[0]:
            return False
        for (ox, oy) in ((ex, ey), (mx, my)):
            if abs(x - ox) < BOX and abs(y - oy) < BOX:
                return False
        return True
    cands = [(ex + dx, ey + dy) for dx in (-112, -96, -80, -64, -48, 48, 64, 80, 96, 112) for dy in (-96, -64, -32, 0)]
    cands = [c for c in cands if ok(*c)]
    cands.sort(key=lambda c: -float(f[c[1]:c[1] + BOX, c[0]:c[0] + BOX].astype(float).var()))
    return {"enemy": (ex, ey), "scenery1": cands[0] if cands else None,
            "scenery2": cands[1] if len(cands) > 1 else None}


def paint(smp, which):
    """Erase a BOX x BOX region, filled with the median colour of its border."""
    f = smp["frame"].copy()
    if which == "original":
        return f
    pos = boxes(smp)[which]
    if pos is None:
        return f
    x0, y0 = max(0, pos[0]), max(0, pos[1])
    x1, y1 = min(f.shape[1], x0 + BOX), min(f.shape[0], y0 + BOX)
    if x1 - x0 < 4 or y1 - y0 < 4:                   # box (mostly) off screen
        return f
    border = np.concatenate([f[max(0, y0 - 1), x0:x1], f[y0:y1, max(0, x0 - 1)]], 0)
    f[y0:y1, x0:x1] = np.median(border, axis=0).astype(np.uint8)
    return f


def summarise(eps, occ):
    out = {}

    def role(name):                       # "specialist_ro1" -> "specialist", "oracleshuf_ro2" -> "oracleshuf"
        return name.rsplit("_ro", 1)[0]
    roles = sorted({role(e["learner"]) for e in eps})
    for suite in SUITES:
        out[suite] = {}
        for r in roles:
            E = [e for e in eps if e["suite"] == suite and role(e["learner"]) == r]
            if E:
                x = np.array([e["best_x"] for e in E], float)
                out[suite][r] = {"n": len(E), "mean_x": round(float(x.mean()), 1),
                                 "se_x": round(float(x.std(ddof=1) / np.sqrt(len(x))), 1) if len(x) > 1 else None,
                                 "median_x": float(np.median(x)),
                                 "close_calls_survived": float(np.mean([e["close_calls"] for e in E])),
                                 "deaths_near_enemy": int(sum(e["died_near_enemy"] for e in E)),
                                 "ends": {k: sum(e["end"] == k for e in E) for k in ("death", "stall", "time", "flag")}}
    occ_s = {}
    for r in sorted({role(o["learner"]) for o in occ}):
        O = [o for o in occ if role(o["learner"]) == r]
        if O:
            occ_s[r] = {"n": len(O)}
            for k in ("d_chosen_enemy", "d_chosen_scenery1", "d_chosen_scenery2", "d_chosen_swap",
                      "d_jump_enemy", "d_jump_scenery1", "d_jump_scenery2", "d_jump_swap",
                      "d_logodds_enemy", "d_logodds_scenery1", "d_logodds_scenery2", "d_logodds_swap"):
                v = np.array([o[k] for o in O])
                occ_s[r][k] = {"mean": round(float(v.mean()), 4), "mean_abs": round(float(np.abs(v).mean()), 4),
                               "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else None}
    return {"suites": out, "occlusion": occ_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "checkpoints" / "mario_gen.pt"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-decisions", type=int, default=450)
    ap.add_argument("--tag", default="20261003")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "sampled"])
    ap.add_argument("--no-occlusion", action="store_true")
    a = ap.parse_args()
    ev = Evaluator(a.ckpt, a.episodes, a.max_decisions, a.policy)
    t0 = time.time()
    eps = ev.run_suites()
    path = RUNS / f"mario_generalisation_eval-{a.tag}.json"
    # episodes first: a failure in the occlusion step must not lose them
    path.write_text(json.dumps({"episodes": eps, "partial": True, "policy": a.policy,
                                "summary": summarise(eps, [])}, indent=1, default=float))
    print(f"occlusion on {len(ev.samples)} frames", flush=True)
    occ = [] if a.no_occlusion else ev.occlusion()
    rep = {"what": "generalisation tests of mario_gen (docs/GENERALISATION_STUDY.md)", "policy": a.policy,
           "checkpoint": a.ckpt, "episodes_per_level": a.episodes, "max_decisions": a.max_decisions,
           "learners": ev.names, "summary": summarise(eps, occ), "episodes": eps, "occlusion": occ,
           "seconds": round(time.time() - t0), "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    path.write_text(json.dumps(rep, indent=1, default=float))
    ev.pool.close()
    print(json.dumps(rep["summary"], indent=1))


if __name__ == "__main__":
    main()
