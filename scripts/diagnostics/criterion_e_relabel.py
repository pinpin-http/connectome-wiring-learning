#!/usr/bin/env python
"""Exploratory: did the taught learners learn to jump at ground enemies or at obstacles?

Frames from scripted play on unseen 5-1 and 8-1 are split into THREAT (ground threat 8-64 px ahead),
OBSTACLE (wall or pit, no threat within 64 px) and CLEAR. Per learner vs its matched twin,
q = P(jump) difference; reports mean q per category and one-sided Mann-Whitney vs CLEAR.
With --pairs, per-frame data are also saved as .npz next to the JSON.

    .venv/bin/python scripts/diagnostics/criterion_e_relabel.py
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
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
JUMP = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
PAIRS = {
    "mario_stageB.pt": [(f"tfull_ro{i}", f"tfullshuf_ro{i}") for i in (1, 2, 3)],
    "mario_stageC_40k.pt": [(f"tko_ro{i}", f"tkoshuf_ro{i}") for i in (1, 2, 3)],
    "mario_ref2.pt": [(f"teacher2_ro{i}", f"tshufA_ro{i}") for i in (1, 2, 3)],
    "mario_oracle.pt": [(f"pixels_ro{i}", f"pixelsshuf_ro{i}") for i in (1, 2, 3)],
}


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


def frames(g, levels=("Level5-1", "Level8-1"), steps=3000, cap=120, seed=20261009):
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    out = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        got = {0: 0, 1: 0, 2: 0}
        for n in range(steps):
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram()
            t, wp = threat_ahead(ram), wall_pit(ram)
            cat = 0 if (8 <= t <= 64) else 1 if (t > 64 and wp) else 2 if (t > 160 and not wp) else -1
            if cat >= 0 and got[cat] < cap and n % 2 == 0:
                vis.reset()
                vis.encode(prev)
                out.append((lvl, cat, vis.encode(obs), takeoff_label_v2(ram)))
                got[cat] += 1
            prev = obs
            if term or trunc:
                prev, _ = env.reset()
        env.close()
    return out


def main():
    import torch
    from scipy.stats import mannwhitneyu

    from flymario.connectome import graphsel
    spec = sys.argv[sys.argv.index("--graph") + 1] if "--graph" in sys.argv else "connectome"
    g, W, motor = graphsel.load(spec)                    # the wiring the brains were trained on
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    F = frames(g)
    cat = np.array([f[1] for f in F])
    print("frames per category (threat, obstacle, clear):", [int((cat == c).sum()) for c in (0, 1, 2)], flush=True)
    cur = np.repeat(np.stack([f[2] for f in F], 1).astype(np.float32), 2, axis=1)

    def pjump(m_row, readout_seed):
        out = []
        for s0 in range(0, cur.shape[1], 64):
            n = min(64, cur.shape[1] - s0)
            b = PlasticLIF(W, motor, params, seeds=[90000 + s0 + i for i in range(n)], cache_key=("ctrl", spec))
            b.set_weights(m_row[None, :].repeat(n, 1).to(b.dev))
            c, _ = sweep_once(b, params, torch.as_tensor(cur[:, s0:s0 + n], device=b.dev), K, 0.5, False)
            out.append(c.t().cpu().numpy())
            del b
            torch.cuda.empty_cache()
        z = np.concatenate(out, 0) @ np.asarray(FixedRandomProjection(len(motor), 8, readout_seed).W).T / 2.0
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        spk = np.concatenate(out, 0).sum(1).reshape(len(F), 2).mean(1)
        return p[:, JUMP].sum(1).reshape(len(F), 2).mean(1), spk

    res, per_frame = {}, {}
    todo = PAIRS
    if "--stageD" in sys.argv:                   # Stage D: its own pairs, its own output
        todo = {sys.argv[sys.argv.index("--stageD") + 1]: [(f"tko2_ro{i}", f"tko2shuf_ro{i}") for i in (1, 2, 3)]}
    if "--pairs" in sys.argv:                    # any checkpoint: --pairs <ckpt> <learner prefix> <twin prefix>
        k = sys.argv.index("--pairs")
        todo = {sys.argv[k + 1]: [(f"{sys.argv[k + 2]}_ro{i}", f"{sys.argv[k + 3]}_ro{i}") for i in (1, 2, 3)]}
    for ck, pairs in todo.items():
        st = torch.load(RUNS / "checkpoints" / Path(ck).name, map_location="cpu", weights_only=False)
        idx = {d["name"]: (j, d) for j, d in enumerate(st["learners"])}
        for lname, tname in pairs:
            (jl, dl), (jt, _dt) = idx[lname], idx[tname]
            (pl, sl), (pt, stw) = pjump(st["brain"]["m"][jl], dl["readout_seed"]), pjump(st["brain"]["m"][jt], dl["readout_seed"])
            q = pl - pt
            per_frame.update({f"p_{lname}": pl, f"spk_{lname}": sl, f"p_{tname}": pt, f"spk_{tname}": stw})
            r = {"q_threat": round(float(q[cat == 0].mean()), 4), "q_obstacle": round(float(q[cat == 1].mean()), 4),
                 "q_clear": round(float(q[cat == 2].mean()), 4),
                 "p_threat_gt_clear": round(float(mannwhitneyu(q[cat == 0], q[cat == 2], alternative="greater").pvalue), 5),
                 "p_obstacle_gt_clear": round(float(mannwhitneyu(q[cat == 1], q[cat == 2], alternative="greater").pvalue), 5)}
            res[f"{ck[:-3]}:{lname}"] = r
            print(ck, lname, r, flush=True)
        del st
    rep = {"what": "criterion-E style re-analysis with ground-threat / obstacle / clear frames (exploratory)",
           "n_frames": {"threat": int((cat == 0).sum()), "obstacle": int((cat == 1).sum()), "clear": int((cat == 2).sum())},
           "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    out = (f"criterion_e_relabel-{Path(next(iter(todo))).stem}.json" if ("--stageD" in sys.argv or "--pairs" in sys.argv)
           else "criterion_e_relabel-20261005.json")
    (RUNS / out).write_text(json.dumps(rep, indent=1))
    if "--pairs" in sys.argv:
        np.savez_compressed((RUNS / out).with_suffix(".npz"), cat=cat, level=np.array([f[0] for f in F]),
                            tko=np.array([f[3] for f in F]), **per_frame)


if __name__ == "__main__":
    main()
