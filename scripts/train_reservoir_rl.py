#!/usr/bin/env python
"""Readout ablation: the spiking network as a FIXED reservoir with a linear readout learnt by RL
(docs/RL_STUDY.md §10).

Per decision: frame -> eye -> one 16-tick sweep of the untrained network -> 2,227 standardised
motor spike counts -> linear softmax actor and linear critic, trained exactly as the linear
references R2/R3. Only the feature map differs from R2; only the wiring differs between arms.

    .venv/bin/python scripts/train_reservoir_rl.py --graph connectome [--seeds 3] [--decisions 10000] [--c 0.1]
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
sys.path.insert(0, str(ROOT / "scripts"))

from train_reference import (
    CRITIC_LR,
    DEATH_EXTRA,
    EPS,
    GAMMA,
    JUMPS,
    LAM,
    MAX_FRAMES,
    NA,
    SCALE,
    STALL,
    TRAIN,
    Learner,
)

from flymario import config
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.mario import EnvPool
from flymario.training.sweep import sweep_once

OUT = RUNS / "reference"


def main():
    import torch

    from flymario.connectome import graphsel
    from flymario.env.vision import RetinotopicVision
    from flymario.training.plastic import PlasticLIF
    a = sys.argv
    spec = a[a.index("--graph") + 1] if "--graph" in a else "connectome"
    n_seeds = int(a[a.index("--seeds") + 1]) if "--seeds" in a else 3
    n_dec = int(a[a.index("--decisions") + 1]) if "--decisions" in a else 10000
    c = float(a[a.index("--c") + 1]) if "--c" in a else 0.1
    tag = f"reservoir-{spec.replace(':', '')}-20261008"
    base = json.loads((OUT / "always_run_train.json").read_text())
    g, W, motor = graphsel.load(spec)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    brain = PlasticLIF(W, motor, params, seeds=[8400 + s for s in range(n_seeds)], cache_key=("reservoir", spec))
    learners = [Learner(f"res-{spec}_c{c}_s{s}", "rl-res", c, 20261070 + s, len(motor)) for s in range(n_seeds)]
    for L in learners:                    # standardised features + bias, as the pixel references
        L.kind = "rl-pixels"
        L.d = len(motor) + 1
        L.W = np.zeros((NA, L.d))
        L.wv = np.zeros(L.d)
        L.E = np.zeros((NA, L.d))
        L.zv = np.zeros(L.d)
    eyes = [RetinotopicVision(g) for _ in learners]
    pool = EnvPool(n_seeds, {"state": "Level1-1", "frame_skip": 4, "max_frames": MAX_FRAMES})
    ep_rng = [np.random.default_rng([20261071, b]) for b in range(n_seeds)]
    obs, info, ep, pal = [None] * n_seeds, [None] * n_seeds, [None] * n_seeds, [None] * n_seeds

    def restart(b):
        lv = str(ep_rng[b].choice(TRAIN))
        dl = int(ep_rng[b].integers(0, 23))
        o, i = pool.reset_to(b, lv, dl)
        obs[b], info[b] = o, i
        pal[b] = (ep_rng[b].permutation(3), float(ep_rng[b].uniform(0.75, 1.25)))
        eyes[b].reset()
        learners[b].reset_episode()
        ep[b] = {"level": lv, "delay": dl, "best_x": i.get("mario_x", 0.0), "best_p": i.get("progress", 0.0),
                 "since": 0, "n": 0, "jumps": 0, "ret": 0.0, "ent": 0.0, "spk": 0.0}

    for b in range(n_seeds):
        restart(b)
    episodes, t0 = [], time.time()
    for k in range(n_dec):
        frames = [np.clip(obs[b][..., pal[b][0]].astype(np.float32) * pal[b][1], 0, 255).astype(np.uint8)
                  for b in range(n_seeds)]
        cur = torch.as_tensor(np.stack([eyes[b].encode(frames[b]) for b in range(n_seeds)], 1), device=brain.dev)
        cnt, _ = sweep_once(brain, params, cur, K, 0.5, False)
        counts = cnt.t().cpu().numpy().astype(np.float64)                 # (B, M)
        acts, cache = np.zeros(n_seeds, int), []
        for b, L in enumerate(learners):
            xf = L.feat(counts[b])
            p = L.pi(xf)
            act, mu = L.act(p)
            acts[b] = act
            cache.append((xf, p, mu))
        out = pool.step(acts)
        for b, L in enumerate(learners):
            xf, p, mu = cache[b]
            o, r, term, trunc, inf = out[b]
            e = ep[b]
            e["n"] += 1
            e["jumps"] += int(acts[b] in JUMPS)
            e["ret"] += r
            e["spk"] += float(counts[b].sum())
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
            r_l = (r + DEATH_EXTRA * died) / SCALE
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
                                 "entropy": round(e["ent"] / e["n"], 3), "return": round(e["ret"], 1),
                                 "motor_spikes": round(e["spk"] / e["n"], 1)})
                restart(b)
        if (k + 1) % 1000 == 0 or k + 1 == n_dec:
            (OUT / f"train-{tag}.json").write_text(json.dumps({"decisions": k + 1, "graph": spec, "c": c,
                                                               "episodes": episodes}))
            np.save(OUT / f"weights-{tag}.npy", {L.name: {"W": L.W, "wv": L.wv, "mu_x": L.mu_x, "var_x": L.var_x}
                                                  for L in learners}, allow_pickle=True)
            print(f"{k + 1} decisions, {time.time() - t0:.0f} s, {len(episodes)} episodes", flush=True)
    pool.close()
    print(datetime.now(UTC).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
