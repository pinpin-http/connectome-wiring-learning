#!/usr/bin/env python
"""Criterion D (docs/PERCEPTION_CREDIT_STUDY.md): did learning make the fly jump *because* an enemy is ahead?

Scripted-play frames on unseen 5-1 and 8-1, labelled from RAM (DANGER: enemy 8-64 px ahead, CLEAR: none
within 160 px). dp = P(jump)_trained - P(jump)_untrained twin; D = mean dp(DANGER) - mean dp(CLEAR);
PASS: one-sided Mann-Whitney p < 0.05 and D >= 0.10. Valid if >= 2 of 3 teachers pass and no negative does.

    .venv/bin/python scripts/gates/criterion_d_check.py
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
from flymario.env.schematic import danger_ahead
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
JUMP = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
D_MIN = 0.10


def frames(g, levels=("Level5-1", "Level8-1"), steps=2500, cap=80, seed=20261009):
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    out = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        n, got = 0, {0: 0, 1: 0}
        while n < steps and min(got.values()) < cap:
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            n += 1
            dx = danger_ahead(env.env.unwrapped.get_ram())
            lab = 1 if 8 <= dx <= 64 else 0 if dx > 160 else -1
            if lab >= 0 and got[lab] < cap and n % 2 == 0:
                vis.reset()
                vis.encode(prev)
                out.append((lvl, lab, vis.encode(obs)))
                got[lab] += 1
            prev = obs
            if term or trunc:
                prev, _ = env.reset()
        env.close()
    return out


def verdict(dp, y):
    from scipy.stats import mannwhitneyu
    d1, d0 = dp[y == 1], dp[y == 0]
    p = float(mannwhitneyu(d1, d0, alternative="greater").pvalue)
    D = float(d1.mean() - d0.mean())
    return {"D": round(D, 4), "p_greater": round(p, 6), "n_danger": len(d1), "n_clear": len(d0),
            "pass": bool(p < 0.05 and D >= D_MIN)}


def main():
    import torch
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    W = g.to_sparse()
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    F = frames(g)
    y = np.array([f[1] for f in F])
    cur = np.stack([f[2] for f in F], 1).astype(np.float32)
    cur2 = np.repeat(cur, 2, axis=1)                         # 2 noise seeds per frame
    print(f"{len(F)} frames: {int(y.sum())} danger, {int((y == 0).sum())} clear", flush=True)

    def counts(m_row, seed0):
        out = []
        for s0 in range(0, cur2.shape[1], 64):
            n = min(64, cur2.shape[1] - s0)
            b = PlasticLIF(W, motor, params, seeds=[seed0 + s0 + i for i in range(n)], cache_key=("ctrl", 0))
            if m_row is not None:
                b.set_weights(m_row[None, :].repeat(n, 1).to(b.dev))
            c, _ = sweep_once(b, params, torch.as_tensor(cur2[:, s0:s0 + n], device=b.dev), K, 0.5, False)
            out.append(c.t().cpu().numpy())
            del b
            torch.cuda.empty_cache()
        return np.concatenate(out, 0)

    def pjump(C, readout_seed):
        R = np.asarray(FixedRandomProjection(len(motor), 8, readout_seed).W)
        z = C @ R.T / 2.0
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return p[:, JUMP].sum(1).reshape(len(F), 2).mean(1)

    keep_dp, keep_lo = {}, {}

    def logodds(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    C0 = counts(None, 90000)                                 # the untrained twin of every learner
    C0b = counts(None, 190000)                               # the noise twin
    res, ro = {}, list(block_seeds(3))
    for i, r in enumerate(ro):
        res[f"noise_twin_ro{i + 1}"] = verdict(pjump(C0b, r) - pjump(C0, r), y)
        keep_dp[f"noise_twin_ro{i + 1}"] = pjump(C0b, r) - pjump(C0, r)
        keep_lo[f"noise_twin_ro{i + 1}"] = logodds(pjump(C0b, r)) - logodds(pjump(C0, r))
        print(f"noise_twin_ro{i + 1}", res[f"noise_twin_ro{i + 1}"], flush=True)
    runs = [("mario_teacher", "mario_teacher.pt"), ("mario_gen", "mario_gen_final_20505.pt")]
    if (RUNS / "checkpoints/mario_teacher_shuffled.pt").exists():
        runs.append(("mario_teacher_shuffled", "mario_teacher_shuffled.pt"))
    for _run, ck in runs:
        st = torch.load(RUNS / "checkpoints" / ck, map_location="cpu", weights_only=False)
        for j, d in enumerate(st["learners"]):
            pt, p0 = pjump(counts(st["brain"]["m"][j], 90000), d["readout_seed"]), pjump(C0, d["readout_seed"])
            dp = pt - p0
            keep_dp[d["name"]], keep_lo[d["name"]] = dp, logodds(pt) - logodds(p0)
            res[d["name"]] = verdict(dp, y)
            res[d["name"]]["trained_decisions"] = int(st["decisions"])
            print(d["name"], res[d["name"]], flush=True)
        del st
    pos = sum(v["pass"] for k, v in res.items() if k.startswith("teacher"))
    # revision 2 (declared before the shuffled-teacher run): negatives are brains known
    # not to have learnt danger-contingent jumping; mario_gen is measured, not a reference
    v2 = any(k.startswith("tshuffled") for k in res)
    negs = ("tshuffled", "noise_twin") if v2 else ("specialist", "generalist", "noise_twin")
    neg = sum(v["pass"] for k, v in res.items() if k.startswith(negs))
    n_neg = sum(1 for k in res if k.startswith(negs))
    ok = pos >= 2 and neg == 0
    rep = {"what": "criterion D references (PERCEPTION_CREDIT_STUDY §5)", "D_min": D_MIN,
           "n_frames": len(F), "results": res, "teachers_passing": pos, "negatives_passing": neg,
           "negatives": list(negs), "revision": 2 if v2 else 1,
           "criterion_validated": ok, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    # exploratory (after revision 2): each learner against the shuffled-target teacher with
    # the same readout, per frame; and D in log-odds
    from scipy.stats import mannwhitneyu
    matched = {}
    for k, dpk in keep_dp.items():
        if k.startswith(("teacher", "specialist", "generalist")) and v2:
            ref = "tshuffled_ro" + k[-1]
            diff = dpk - keep_dp[ref]
            matched[k] = {"vs": ref, "D_minus_ref": round(float(diff[y == 1].mean() - diff[y == 0].mean()), 4),
                          "p_greater": round(float(mannwhitneyu(diff[y == 1], diff[y == 0], alternative="greater").pvalue), 6)}
    D_logit = {k: round(float(v[y == 1].mean() - v[y == 0].mean()), 4) for k, v in keep_lo.items()}
    rep["exploratory_matched_vs_shuffled_teacher"] = matched
    rep["exploratory_D_logodds"] = D_logit
    print("matched vs shuffled teacher:", json.dumps(matched, indent=1))
    print("D in log-odds:", json.dumps(D_logit))
    np.savez_compressed(RUNS / "criterion_d_frames-20261003.npz", y=y, levels=np.array([f[0] for f in F]),
                        **{f"dp_{k}": v for k, v in keep_dp.items()}, **{f"lo_{k}": v for k, v in keep_lo.items()})
    name = "criterion_d_check-v2-20261003.json" if v2 else "criterion_d_check-20261003.json"
    (RUNS / name).write_text(json.dumps(rep, indent=1))
    print(f"teachers passing {pos}/3, negatives passing {neg}/{n_neg} -> criterion validated: {ok}")


if __name__ == "__main__":
    main()
