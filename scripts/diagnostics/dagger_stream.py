#!/usr/bin/env python
"""Learning variant G-DAgger-v1: dataset aggregation on the agent's own frames (declared in docs/RL_STUDY.md).

Round 1 is the expert stream. Each later round: calibrate each brain's Youden threshold as the
frozen-play evaluation does, collect frames from the brains' own play on the training levels,
aggregate them with the expert stream, then train with the Stage D supervised rule.
--control adds fresh expert-stream frames instead of own frames (exposure-matched control).
Writes checkpoints/common_stream_{dagger|daggerctl}_<graph>.pt and artifacts/runs/dagger/{dagger|daggerctl}-<graph>.json.

    .venv/bin/python scripts/diagnostics/dagger_stream.py --graph connectome [--control]
        [--rounds 2] [--collect 1500] [--train 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from common_stream import EBAR, ETA, JUMPS, KAPPA, PRESS, RD, RUN, TAU, stream  # noqa: E402

TRAIN = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")
NOOP = 0
MAX_DEC, STALL = 600, 150


def collection_start(level_idx, r, i):
    rng = np.random.default_rng(20261500 + 1000 * level_idx + 100 * r + i)
    n = int(rng.integers(5, 41))
    return [RUN if rng.random() < 0.6 else NOOP for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="connectome")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--collect", type=int, default=1500)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--source", default=None, help="tests only: another starting checkpoint")
    ap.add_argument("--tag", default="", help="tests only: isolates every output")
    a = ap.parse_args()
    import torch
    from criterion_e_relabel import frames as relabel_frames

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.env.schematic import takeoff_label_v2
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.paths import RUNS
    from flymario.policy.readout import FixedRandomProjection
    from flymario.training.mario import EnvPool
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(4)
    kind = "daggerctl" if a.control else "dagger"
    gt = a.graph.replace(":", "")
    CK = RUNS / "checkpoints"
    out_dir = RUNS / "dagger"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (a.source or a.tag) and not (a.source and a.tag):
        sys.exit("--source and --tag go together (tests only)")
    kind += f"-{a.tag}" if a.tag else ""
    src = torch.load(a.source or CK / f"common_stream_expert_{gt}.pt", map_location="cpu", weights_only=False)
    g, W, motor = graphsel.load(a.graph)
    g0, _W0, _m0 = graphsel.load("connectome")         # the stream is encoded on the connectome's eye map,
    del _W0                                           # as in common_stream.py (same node order)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    seeds = [l["noise_seed"] for l in src["learners"]]
    ro = [l["readout_seed"] for l in src["learners"]]
    b = PlasticLIF(W, motor, params, seeds=seeds, cache_key=("dagger", a.graph))
    b.set_weights(src["brain"]["m"].to(b.dev))
    R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro]),
                        device=b.dev)
    B = b.B
    vis0 = RetinotopicVision(g0)
    idx = np.unique(np.r_[vis0.on_idx, vis0.off_idx]).astype(np.int64)
    n = b.n

    def compact(cur):                                  # (n,) or (n, F) -> values at the eye's cells
        return np.asarray(cur, np.float32)[idx]

    def expand(c):                                     # (len(idx), F) -> (n, F)
        out = np.zeros((n, c.shape[1]), np.float32)
        out[idx] = c
        return out

    # round 1 = the expert stream
    ex_cur, ex_lab, _ = stream(g0, TRAIN, 2500, 20261073, kind="expert")
    agg_c = [[compact(ex_cur)] for _ in range(B)]
    agg_l = [[ex_lab.copy()] for _ in range(B)]
    del ex_cur
    log = {"what": f"G-DAgger-v1 ({kind}) on {a.graph}", "args": vars(a), "rounds": []}
    ebar, ms, k_upd = None, torch.zeros(B, device=b.dev), 0
    pool = EnvPool(B, {"state": "Level1-1", "frame_skip": 4, "max_frames": 6000})
    visions = [RetinotopicVision(g) for _ in range(B)]   # as the evaluator
    for r in range(1, a.rounds + 1):
        rec = {"round": r}
        # 1. calibration, exactly as the evaluation
        F = relabel_frames(g, levels=TRAIN, steps=2000, cap=60, seed=20261017)
        tko = np.array([f[3] for f in F])
        P = np.zeros((len(F), B))
        for i, f in enumerate(F):
            cur = torch.as_tensor(np.repeat(f[2][:, None], B, 1).astype(np.float32), device=b.dev)
            c, _ = sweep_once(b, params, cur, K, 0.5, False)
            P[i] = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)[:, JUMPS].sum(1).cpu().numpy()
        taus = []
        for j in range(B):
            cand = np.unique(P[:, j])
            yj = [(tko & (P[:, j] >= t)).sum() / max(1, tko.sum()) - (~tko & (P[:, j] >= t)).sum() / max(1, (~tko).sum())
                  for t in cand]
            taus.append(float(cand[int(np.argmax(yj))]))
        rec["tau"] = [round(t, 4) for t in taus]
        # 2. collection
        new_c, new_l = [[] for _ in range(B)], [[] for _ in range(B)]
        jumps, decs, blocked_on, episodes = np.zeros(B), np.zeros(B), [[] for _ in range(B)], 0
        if a.control:
            cc, cl, _ = stream(g0, TRAIN, 2500, 20261573 + r, kind="expert")
            pick = np.random.default_rng(20261580 + r).permutation(cc.shape[1])[:a.collect]
            for j in range(B):
                new_c[j], new_l[j] = [compact(cc[:, pick])], [cl[pick]]
            del cc
        else:
            got = np.zeros(B, int)
            i_start = 0
            while got.min() < a.collect:
                li = i_start % len(TRAIN)
                prefix = collection_start(li, r, i_start // len(TRAIN))
                i_start += 1
                episodes += 1
                obs, info = [None] * B, [None] * B
                for j in range(B):
                    obs[j], info[j] = pool.reset_to(j, TRAIN[li], prefix)
                    visions[j].reset()
                active = np.ones(B, bool)
                best = np.array([inf["mario_x"] for inf in info], float)
                since = np.zeros(B, int)
                xs = [[] for _ in range(B)]
                t = 0
                while active.any():
                    cur_np = np.stack([v.encode(o) for v, o in zip(visions, obs)], 1).astype(np.float32)
                    cur = torch.as_tensor(cur_np, device=b.dev)
                    c, _ = sweep_once(b, params, cur, K, 0.5, False)
                    pj = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)[:, JUMPS].sum(1).cpu().numpy()
                    jump = pj >= np.array(taus)
                    sc = torch.einsum("bam,mb->ba", R, c).cpu().numpy()
                    sc[:, JUMPS] = -np.inf
                    act = np.where(jump, PRESS, sc.argmax(1))
                    for j in np.flatnonzero(active):
                        ram = info[j]["ram"]
                        if t % 2 == 0 and ram[0xB5] == 1 and got[j] < a.collect:
                            lab = bool(takeoff_label_v2(ram))
                            new_c[j].append(compact(cur_np[:, j])[:, None])
                            new_l[j].append(np.array([lab]))
                            got[j] += 1
                            if lab:
                                blocked_on[j].append(len(xs[j]) >= 5 and xs[j][-1] - xs[j][-5] <= 2)
                        pool.conns[j].send(("step", int(act[j])))
                        jumps[j] += bool(jump[j])
                        decs[j] += 1
                    for j in np.flatnonzero(active):
                        o, _r, term, trunc, inf = pool.conns[j].recv()
                        obs[j], info[j] = o, inf
                        xs[j].append(inf.get("mario_x", 0.0))
                        if inf.get("mario_x", 0) > best[j]:
                            best[j], since[j] = inf["mario_x"], 0
                        else:
                            since[j] += 1
                        if inf.get("cleared") or inf.get("died") or since[j] >= STALL or term or trunc or t + 1 >= MAX_DEC:
                            active[j] = False
                    t += 1
            new_c = [[np.concatenate(x, 1)] for x in new_c]
            new_l = [[np.concatenate(x)] for x in new_l]
        for j in range(B):
            agg_c[j] += new_c[j]
            agg_l[j] += new_l[j]
        lab_new = [np.concatenate(new_l[j]) for j in range(B)]
        rec.update({"collected": [int(len(x)) for x in lab_new], "label_rate": [round(float(x.mean()), 3) for x in lab_new],
                    "blocked_among_label_on": [round(float(np.mean(x)), 3) if len(x) else None for x in blocked_on],
                    "jump_rate_in_collection": [round(float(jumps[j] / max(1, decs[j])), 3) for j in range(B)],
                    "episodes": episodes})
        print("round", r, json.dumps(rec), flush=True)
        # 3-4. aggregate and train
        Ac = [np.concatenate(agg_c[j], 1) for j in range(B)]
        Al = [np.concatenate(agg_l[j]) for j in range(B)]
        rng = np.random.default_rng(20261600 + r)
        for _ in range(a.train):
            pick = [int(rng.integers(0, Al[j].shape[0])) for j in range(B)]
            cur = torch.as_tensor(np.stack([expand(Ac[j][:, p:p + 1])[:, 0] for j, p in enumerate(pick)], 1),
                                  device=b.dev)
            lab = np.array([Al[j][p] for j, p in enumerate(pick)])
            c, e = sweep_once(b, params, cur, K, 0.5, True)
            if ebar is None:
                ebar = e.clone()
            ec = (e - ebar) + KAPPA * ebar
            ebar.mul_(1 - EBAR).add_(e, alpha=EBAR)
            pi = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)
            tgt = torch.as_tensor(np.where(lab, PRESS, RUN), device=b.dev, dtype=torch.long)
            oh = torch.nn.functional.one_hot(tgt, pi.shape[1]).float()
            L = torch.einsum("ba,bam->mb", oh - pi, R) / TAU
            U = ec * L.index_select(0, b.post) * b.sign[:, None]
            ms = RD * ms + (1 - RD) * (U * U).mean(0)
            k_upd += 1
            U = U / torch.sqrt(ms / (1 - RD ** k_upd) + 1e-30)[None, :]
            b.update((U * ETA).t() * b.m0[None, :])
        rec["aggregate_size"] = [int(x.shape[0]) for x in Al]
        rec["aggregate_label_rate"] = [round(float(x.mean()), 3) for x in Al]
        log["rounds"].append(rec)
    pool.close()
    pre = "csc" if a.control else "csd"
    st = {"learners": [{"name": f"{pre}_ro{i + 1}", "readout_seed": int(ro[i]), "noise_seed": int(seeds[i]),
                        "view": "pixels"} for i in range(B)],
          "brain": {"m": b.m.detach().cpu().clone()}, "graph": a.graph, "stream": kind,
          "what": f"G-DAgger-v1 {kind}: expert stream + {a.rounds} rounds", "from": f"common_stream_expert_{gt}.pt"}
    tmp = CK / f"common_stream_{kind}_{gt}.pt.part"
    torch.save(st, tmp)
    os.replace(tmp, CK / f"common_stream_{kind}_{gt}.pt")
    log["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out_dir / f"{kind}-{gt}.json").write_text(json.dumps(log, indent=1))
    print("saved", kind, gt, flush=True)


if __name__ == "__main__":
    main()
