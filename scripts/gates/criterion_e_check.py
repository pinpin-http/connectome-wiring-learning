#!/usr/bin/env python
"""Criterion E of docs/PERCEPTION_CREDIT_STUDY.md: a learner against its matched non-contingent twin.

Criterion-D frames (5-1, 8-1, seed 20261009; DANGER = enemy 8-64 px ahead, CLEAR = none within 160 px)
in each brain's training view. q = P(jump)_learner - P(jump)_twin, P(jump) = softmax(scores / 2) over
RIGHT+A, RIGHT+A+B, A, mean of 2 shared noise seeds. E = mean q(DANGER) - mean q(CLEAR).
PASS: one-sided Mann-Whitney (q DANGER > q CLEAR) p < 0.05 and E >= 0.05.

    .venv/bin/python scripts/gates/criterion_e_check.py validate   # mario_ref2 references
    .venv/bin/python scripts/gates/criterion_e_check.py study      # mario_oracle arms
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
from flymario.env.schematic import danger_ahead, schematic_frame
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
JUMP = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
E_MIN = 0.05


def frames(g, levels=("Level5-1", "Level8-1"), steps=2500, cap=80, seed=20261009):
    """The criterion-D frames, encoded both as the game and as the schematic."""
    rng = np.random.default_rng(seed)
    vp, vs = RetinotopicVision(g), RetinotopicVision(g)
    out = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        prev_ram = env.env.unwrapped.get_ram().copy()
        n, got = 0, {0: 0, 1: 0}
        while n < steps and min(got.values()) < cap:
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram().copy()
            n += 1
            dx = danger_ahead(ram)
            lab = 1 if 8 <= dx <= 64 else 0 if dx > 160 else -1
            if lab >= 0 and got[lab] < cap and n % 2 == 0:
                vp.reset()
                vp.encode(prev)
                vs.reset()
                vs.encode(schematic_frame(prev_ram, prev.shape))
                out.append((lvl, lab, vp.encode(obs), vs.encode(schematic_frame(ram, obs.shape))))
                got[lab] += 1
            prev, prev_ram = obs, ram
            if term or trunc:
                prev, _ = env.reset()
                prev_ram = env.env.unwrapped.get_ram().copy()
        env.close()
    return out


def verdict(q, y):
    from scipy.stats import mannwhitneyu
    p = float(mannwhitneyu(q[y == 1], q[y == 0], alternative="greater").pvalue)
    E = float(q[y == 1].mean() - q[y == 0].mean())
    return {"E": round(E, 4), "p_greater": round(p, 6), "pass": bool(p < 0.05 and E >= E_MIN)}


def main():
    import torch
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    ck = {"validate": "mario_ref2.pt", "study": "mario_oracle.pt", "stageB": "mario_stageB.pt",
          "stageB2": "mario_stageB2.pt", "stageC": "mario_stageC.pt"}[mode]
    override = sys.argv[2] if len(sys.argv) > 2 else None    # a checkpoint copy (milestones)
    if override:
        ck = Path(override).name
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    W = g.to_sparse()
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    F = frames(g)
    y = np.array([f[1] for f in F])
    views = {"pixels": np.repeat(np.stack([f[2] for f in F], 1).astype(np.float32), 2, axis=1),
             "schematic": np.repeat(np.stack([f[3] for f in F], 1).astype(np.float32), 2, axis=1)}
    print(f"{len(F)} frames: {int(y.sum())} danger, {int((y == 0).sum())} clear", flush=True)
    st = torch.load(RUNS / "checkpoints" / ck, map_location="cpu", weights_only=False)
    L = {d["name"]: (j, d) for j, d in enumerate(st["learners"])}

    def pjump(name):
        j, d = L[name]
        cur = views[d.get("view", "pixels")]
        out = []
        for s0 in range(0, cur.shape[1], 64):
            n = min(64, cur.shape[1] - s0)
            b = PlasticLIF(W, motor, params, seeds=[90000 + s0 + i for i in range(n)], cache_key=("ctrl", 0))
            b.set_weights(st["brain"]["m"][j][None, :].repeat(n, 1).to(b.dev))
            c, _ = sweep_once(b, params, torch.as_tensor(cur[:, s0:s0 + n], device=b.dev), K, 0.5, False)
            out.append(c.t().cpu().numpy())
            del b
            torch.cuda.empty_cache()
        C = np.concatenate(out, 0)
        z = C @ np.asarray(FixedRandomProjection(len(motor), 8, d["readout_seed"]).W).T / 2.0
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return p[:, JUMP].sum(1).reshape(len(F), 2).mean(1)

    P = {n: pjump(n) for n in L}
    res = {}
    if mode == "validate":
        for i in (1, 2, 3):
            res[f"positive_teacher2_vs_tshufA_ro{i}"] = verdict(P[f"teacher2_ro{i}"] - P[f"tshufA_ro{i}"], y)
            res[f"negative_tshufB_vs_tshufA_ro{i}"] = verdict(P[f"tshufB_ro{i}"] - P[f"tshufA_ro{i}"], y)
            res[f"negative_tshufA_vs_tshufB_ro{i}"] = verdict(P[f"tshufA_ro{i}"] - P[f"tshufB_ro{i}"], y)
        pos = sum(v["pass"] for k, v in res.items() if k.startswith("positive"))
        neg = sum(v["pass"] for k, v in res.items() if k.startswith("negative"))
        summary = {"positives_passing": pos, "negatives_passing": neg, "criterion_validated": pos >= 2 and neg == 0}
    elif mode == "stageC":
        for i in (1, 2, 3):
            res[f"tko_ro{i}"] = verdict(P[f"tko_ro{i}"] - P[f"tkoshuf_ro{i}"], y)
        summary = {"tko_passing": sum(res[f"tko_ro{i}"]["pass"] for i in (1, 2, 3))}
    elif mode in ("stageB", "stageB2"):
        for i in (1, 2, 3):
            res[f"tfull_ro{i}"] = verdict(P[f"tfull_ro{i}"] - P[f"tfullshuf_ro{i}"], y)
        summary = {"tfull_passing": sum(res[f"tfull_ro{i}"]["pass"] for i in (1, 2, 3))}
    else:
        for i in (1, 2, 3):
            res[f"oracle_ro{i}"] = verdict(P[f"oracle_ro{i}"] - P[f"oracleshuf_ro{i}"], y)
            res[f"pixels_ro{i}"] = verdict(P[f"pixels_ro{i}"] - P[f"pixelsshuf_ro{i}"], y)
        summary = {"oracle_passing": sum(res[f"oracle_ro{i}"]["pass"] for i in (1, 2, 3)),
                   "pixels_passing": sum(res[f"pixels_ro{i}"]["pass"] for i in (1, 2, 3))}
    for k, v in res.items():
        print(k, v, flush=True)
    rep = {"what": f"criterion E ({mode}), PERCEPTION_CREDIT_STUDY §5", "E_min": E_MIN, "checkpoint": ck,
           "trained_decisions": int(st["decisions"]), "n_frames": len(F), "results": res, **summary,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    out = f"criterion_e_{mode}-{Path(ck).stem}.json" if override else f"criterion_e_{mode}-20261003.json"
    (RUNS / out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
