#!/usr/bin/env python
"""RL_STUDY §30 (exploratory, speedrun stage S2): can the connectome imitate a level-clearing teacher?
G variant "G-planner-imitation-v1". Teacher = the §29 v2 planner (savestate lookahead, RAM only for the
teacher); brain input = pixels. Development levels 1-1 and 4-1 only.

    .venv/bin/python scripts/diagnostics/planner_imitation.py [--rounds 2] [--labels 800] [--updates 3000]

Stages (cached under artifacts/runs/mission/s2/, resumable): round 0 planner trajectories -> train ->
[DAgger: frozen play, planner labels of visited states, aggregate, retrain] x rounds -> frozen evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

OUT = ROOT / "artifacts/runs/mission/s2"
LEVELS = ("Level1-1", "Level4-1")
K_PLAN, H_PLAN, E_PLAN = 32, 40, 4
MAX_DEC, STALL = 600, 150

_W = {}


# ------------------------------------------------------------------ teacher (worker processes)
def _init():
    from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
    _W.update(RUN=ACTION_NAMES.index("RIGHT+B"), JUMP=ACTION_NAMES.index("RIGHT+A+B"),
              MarioEnv=MarioEnv, EnvConfig=EnvConfig, level=None, env=None)
    _use("Level1-1")


def _use(level):
    """One emulator per process (stable-retro): reopen only when the level changes."""
    if _W.get("level") == level:
        return
    if _W.get("env") is not None:
        _W["em"] = None                        # drop our reference first, or the emulator is never freed
        _W["env"].close()
        _W["env"] = None
        import gc
        gc.collect()
    env = _W["MarioEnv"](_W["EnvConfig"](state=level, max_frames=10 ** 9))
    env.reset()
    _W.update(env=env, em=env.env.unwrapped.em, level=level)


def _seq(rng):
    seq = []
    while len(seq) < H_PLAN:
        seq += [int(rng.choice([_W["RUN"], _W["JUMP"]], p=[0.6, 0.4]))] * int(rng.integers(1, 9))
    return seq[:H_PLAN]


def _plan(root, rng, keep=None):
    env, em = _W["env"], _W["em"]
    cands = [_seq(rng) for _ in range(K_PLAN)]
    if keep:
        cands[0] = (keep + _seq(rng))[:H_PLAN]
    best, bscore = None, -1e18
    for s in cands:
        em.set_state(root)
        sc = -1.0
        for t, a in enumerate(s):
            _o, _r, _te, _tr, info = env.step(a)
            sc = max(sc, float(info["mario_x"]))
            if info["cleared"]:
                sc = 1e6 - t
                break
            if info["died"]:
                sc -= 1e5
                break
        if sc > bscore:
            best, bscore = s, sc
    em.set_state(root)
    return best


def label_state(args):
    """One planning step from an emulator state: the teacher's first action."""
    state, seed = args
    rng = np.random.default_rng(seed)
    return int(_plan(state, rng)[0])


def trajectory(args):
    """A full planner episode from a level start (+ optional prefix): frames (prev, obs) and actions."""
    level, prefix, seed = args
    rng = np.random.default_rng(seed)
    _use(level)
    env = _W["env"]
    prev, _ = env.reset()
    for a in prefix:
        prev, _r, _te, _tr, _i = env.step(a)
    frames, acts, keep, n, end, best = [], [], None, 0, None, 0.0
    while end is None:
        root = _W["em"].get_state()
        plan = _plan(root, rng, keep)
        keep = plan[E_PLAN:]
        for a in plan[:E_PLAN]:
            obs, _r, _te, _tr, info = env.step(a)
            frames.append((prev, obs))
            acts.append(int(a))
            prev = obs
            n += 1
            best = max(best, float(info["mario_x"]))
            end = "flag" if info["cleared"] else "death" if info["died"] else "time" if n >= 900 else None
            if end:
                break
    return {"level": level, "end": end, "best_x": best, "frames": frames, "actions": acts}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--labels", type=int, default=800)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    import torch
    from common_stream import EBAR, ETA, JUMPS, KAPPA, PRESS, RD, RUN, TAU

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.mario import EnvPool
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"what": "RL_STUDY §30 S2 probe", "args": vars(a), "rounds": []}
    g, W, motor = graphsel.load("connectome")
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    vis0 = RetinotopicVision(g)
    idx = np.unique(np.r_[vis0.on_idx, vis0.off_idx]).astype(np.int64)
    RUNS_ = [ACTION_NAMES.index(x) for x in ("RIGHT+B", "RIGHT")]
    pool = ProcessPoolExecutor(max_workers=a.workers, initializer=_init)

    def encode_pair(prev, obs):
        vis0.reset()
        vis0.encode(prev)
        return vis0.encode(obs).astype(np.float32)[idx]

    # round 0: planner trajectories
    r0 = OUT / "round0.npz"
    if r0.exists():
        z = np.load(r0)
        X, Y = z["X"], z["Y"]
    else:
        jobs = []
        for li, lv in enumerate(LEVELS):
            jobs.append((lv, [], 20261900 + 100 * li))
            for i in range(12):
                rng = np.random.default_rng(20261910 + 100 * li + i)
                pre = [int(RUN if rng.random() < 0.6 else ACTION_NAMES.index("NOOP")) for _ in range(int(rng.integers(5, 41)))]
                jobs.append((lv, pre, 20261920 + 100 * li + i))
        t0 = time.time()
        trajs = list(pool.map(trajectory, jobs))
        X = np.stack([encode_pair(p, o) for t in trajs for p, o in t["frames"]], 1)
        Y = np.array([int(act == PRESS) for t in trajs for act in t["actions"]], np.int8)
        np.savez_compressed(r0, X=X, Y=Y)
        log["round0"] = {"trajectories": [{k: t[k] for k in ("level", "end", "best_x")} for t in trajs],
                         "frames": int(len(Y)), "jump_share": round(float(Y.mean()), 3), "seconds": round(time.time() - t0)}
        print("round 0", json.dumps(log["round0"]), flush=True)
    n = W.shape[0]

    def expand(c):
        out = np.zeros((n, c.shape[1]), np.float32)
        out[idx] = c
        return out

    ro = block_seeds(3)
    b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], cache_key=("s2", "connectome"))
    R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro]),
                        device=b.dev)
    B = 3
    agg_X = [[X] for _ in range(B)]
    agg_Y = [[Y] for _ in range(B)]
    st = {"ebar": None, "ms": torch.zeros(B, device=b.dev), "k": 0}

    def train(n_upd, seed):
        Ax = [np.concatenate(agg_X[j], 1) for j in range(B)]
        Ay = [np.concatenate(agg_Y[j]) for j in range(B)]
        rng = np.random.default_rng(seed)
        for _ in range(n_upd):
            pick = [int(rng.integers(0, len(Ay[j]))) for j in range(B)]
            cur = torch.as_tensor(np.stack([expand(Ax[j][:, p:p + 1])[:, 0] for j, p in enumerate(pick)], 1), device=b.dev)
            lab = np.array([Ay[j][p] for j, p in enumerate(pick)])
            c, e = sweep_once(b, params, cur, K, 0.5, True)
            if st["ebar"] is None:
                st["ebar"] = e.clone()
            ec = (e - st["ebar"]) + KAPPA * st["ebar"]
            st["ebar"].mul_(1 - EBAR).add_(e, alpha=EBAR)
            pi = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)
            tgt = torch.as_tensor(np.where(lab == 1, PRESS, RUN), device=b.dev, dtype=torch.long)
            oh = torch.nn.functional.one_hot(tgt, pi.shape[1]).float()
            L = torch.einsum("ba,bam->mb", oh - pi, R) / TAU
            U = ec * L.index_select(0, b.post) * b.sign[:, None]
            st["ms"] = RD * st["ms"] + (1 - RD) * (U * U).mean(0)
            st["k"] += 1
            U = U / torch.sqrt(st["ms"] / (1 - RD ** st["k"]) + 1e-30)[None, :]
            b.update((U * ETA).t() * b.m0[None, :])
        return {"aggregate": [int(len(y)) for y in Ay], "jump_share": [round(float(y.mean()), 3) for y in Ay]}

    envpool = EnvPool(B, {"state": "Level1-1", "frame_skip": 4, "max_frames": 6000})
    visions = [RetinotopicVision(g) for _ in range(B)]

    def play(starts, collect=None):
        """Frozen greedy two-class play from (level, prefix) starts; optionally collect (current, state)."""
        res = []
        for level, prefix in starts:
            obs, info = [None] * B, [None] * B
            for j in range(B):
                obs[j], info[j] = envpool.reset_to(j, level, prefix)
                visions[j].reset()
            active = np.ones(B, bool)
            best = np.array([float(i["mario_x"]) for i in info])
            since, ends, t = np.zeros(B, int), [None] * B, 0
            while active.any():
                cur_np = np.stack([v.encode(o) for v, o in zip(visions, obs)], 1).astype(np.float32)
                if collect is not None:
                    states = envpool.get_states()
                c, _ = sweep_once(b, params, torch.as_tensor(cur_np, device=b.dev), K, 0.5, False)
                p = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1).cpu().numpy()
                jump = p[:, JUMPS].sum(1) > p[:, RUNS_].sum(1)
                for j in np.flatnonzero(active):
                    if collect is not None and len(collect[j]) < a.labels:
                        collect[j].append((cur_np[idx, j], states[j][0]))
                    envpool.conns[j].send(("step", int(PRESS if jump[j] else RUN)))
                for j in np.flatnonzero(active):
                    o, _r, te, tr, inf = envpool.conns[j].recv()
                    obs[j] = o
                    if inf["mario_x"] > best[j]:
                        best[j], since[j] = inf["mario_x"], 0
                    else:
                        since[j] += 1
                    e = ("flag" if inf.get("cleared") else "death" if inf.get("died") else "stall" if since[j] >= STALL
                         else "time" if (te or tr or t + 1 >= MAX_DEC) else None)
                    if e:
                        ends[j], active[j] = e, False
                t += 1
            res.append({"level": level, "prefix_len": len(prefix), "best_x": best.tolist(), "end": ends})
        return res

    tr0 = train(a.updates, 20261990)
    log["rounds"].append({"round": 0, **tr0})
    print("trained round 0", tr0, flush=True)
    for r in range(1, a.rounds + 1):
        starts = []
        for li, lv in enumerate(LEVELS):
            for i in range(6):
                rng = np.random.default_rng(20261950 + 100 * li + 10 * r + i)
                starts.append((lv, [int(RUN if rng.random() < 0.6 else ACTION_NAMES.index("NOOP"))
                                    for _ in range(int(rng.integers(5, 41)))]))
        collect = [[] for _ in range(B)]
        k_start = 0
        while min(len(c) for c in collect) < a.labels and k_start < 60:
            play([starts[k_start % len(starts)]], collect)
            k_start += 1
        t0 = time.time()
        for j in range(B):
            labs = list(pool.map(label_state, [(s, 20262000 + 1000 * r + 10 * j + q) for q, (_c, s) in enumerate(collect[j])],
                                 chunksize=4))
            agg_X[j].append(np.stack([c for c, _s in collect[j]], 1))
            agg_Y[j].append(np.array([int(x == PRESS) for x in labs], np.int8))
        rec = {"round": r, "labelled": [len(c) for c in collect], "label_seconds": round(time.time() - t0),
               "new_jump_share": [round(float(agg_Y[j][-1].mean()), 3) for j in range(B)]}
        rec.update(train(a.updates, 20261990 + r))
        log["rounds"].append(rec)
        print("round", r, json.dumps(rec), flush=True)
    # frozen evaluation
    hs = json.loads((ROOT / "artifacts/runs/harness_v2/starts-20261009.json").read_text())["starts"]
    ev_starts = [(lv, []) for lv in LEVELS] + [(s["level"], s["prefix"]) for s in hs if s["level"] in LEVELS]
    ev = play(ev_starts)
    flags = {lv: [sum(e["end"][j] == "flag" for e in ev if e["level"] == lv) for j in range(B)] for lv in LEVELS}
    from_start = {lv: [{"best_x": e["best_x"][j], "end": e["end"][j]} for j in range(B)] for e in ev[:2] for lv in [e["level"]]}
    log["evaluation"] = {"from_level_start": from_start, "flags_over_17_starts": flags,
                         "mean_best_x": {lv: [round(float(np.mean([e["best_x"][j] for e in ev if e["level"] == lv])), 1)
                                              for j in range(B)] for lv in LEVELS}}
    log["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (OUT / "s2_probe.json").write_text(json.dumps(log, indent=1))
    torch.save({"m": b.m.cpu()}, OUT / "s2_brains.pt")
    print("evaluation", json.dumps(log["evaluation"]), flush=True)
    envpool.close()
    pool.shutdown()


if __name__ == "__main__":
    main()
