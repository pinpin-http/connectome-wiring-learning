#!/usr/bin/env python
"""Check the occlusion criterion of docs/PERCEPTION_CREDIT_STUDY.md on references.

Frames with an enemy 24-72 px ahead; d = |change in jump log-odds| when the enemy's box (d_enemy), the two
most textured scenery boxes (d_scen) or the whole frame (d_swap) is replaced. PASS if Wilcoxon
(d_enemy > d_scen) p < 0.05 and S = (mean d_enemy - mean d_scen) / mean d_swap >= 0.12.
Negatives (must fail): mario_gen learners, untrained brains; positive: scripts/diagnostics/danger_controller.py.

    .venv/bin/python scripts/gates/occlusion_criterion_check.py            # development
    .venv/bin/python scripts/gates/occlusion_criterion_check.py --fresh    # validation (fresh frames, refitted readout)
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from danger_controller import Brain, collect_1_1
from mario_generalisation_eval import paint

from flymario.connectome import datasets
from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.env.schematic import screen_enemies
from flymario.env.vision import RetinotopicVision
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMP = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
VARIANTS = ("original", "enemy", "scenery1", "scenery2", "swap")


def criterion(d_enemy, d_scen):
    from scipy.stats import wilcoxon
    e, s = np.abs(np.asarray(d_enemy, float)), np.abs(np.asarray(d_scen, float))
    diff = e - s
    p = float(wilcoxon(diff[diff != 0], alternative="greater").pvalue) if (diff != 0).any() else 1.0
    ratio = float(e.mean() / max(s.mean(), 1e-12))
    return {"n": len(e), "enemy": round(float(e.mean()), 4), "scenery": round(float(s.mean()), 4),
            "ratio": round(ratio, 2), "p_greater": round(p, 5), "pass": bool(p < 0.05 and ratio >= 2)}


S_MIN = 0.12


def specificity(d_enemy, d_scen, d_swap):
    """The validated-candidate criterion (see the module docstring)."""
    base = criterion(d_enemy, d_scen)
    S = (base["enemy"] - base["scenery"]) / max(float(np.mean(np.abs(d_swap))), 1e-12)
    return {**base, "swap": round(float(np.mean(np.abs(d_swap))), 4), "S": round(float(S), 3),
            "pass": bool(base["p_greater"] < 0.05 and S >= S_MIN)}


def negative_reference():
    d = json.loads((RUNS / "mario_generalisation_eval-20261003.json").read_text())
    out = {}
    for name in sorted({o["learner"] for o in d["occlusion"]}):
        R = [o for o in d["occlusion"] if o["learner"] == name]
        out[name] = criterion([o["d_logodds_enemy"] for o in R],
                              [(abs(o["d_logodds_scenery1"]) + abs(o["d_logodds_scenery2"])) / 2 for o in R])
    return out


def test_frames(levels=("Level5-1", "Level8-1"), steps=900, per_level=60, seed=20261007):
    rng = np.random.default_rng(seed)
    out = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        env.reset()
        n = k = 0
        while n < steps and k < per_level:
            obs, _r, term, trunc, _i = env.step(JUMP if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram().astype(int)
            n += 1
            mario = ram[0x6D] * 256 + ram[0x86]
            ahead = [(lx - mario, sx, sy) for lx, sx, sy, _ in screen_enemies(ram) if lx > mario]
            if ahead:
                dx, sx, sy = min(ahead)
                if 24 <= dx <= 72 and 4 < sx < 220 and n % 2 == 0:
                    # paint()/boxes() take the uncropped screen x, as MarioEnv reports it
                    out.append({"frame": obs.copy(), "sx": sx + 8, "sy": sy, "mx": int(ram[0x3AD]),
                                "my": int(ram[0xCE]), "level": lvl})
                    k += 1
            if term or trunc:
                env.reset()
        env.close()
    return out


def fresh():
    """Validation of the specificity criterion on fresh frames and references."""
    import torch

    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF as P
    st = torch.load(RUNS / "checkpoints/mario_gen_final_20505.pt", map_location="cpu", weights_only=False)
    names = [d["name"] for d in st["learners"]] + [f"untrained_ro{i + 1}" for i in range(3)]
    ro = [d["readout_seed"] for d in st["learners"]] + list(block_seeds(3))
    m_tr = st["brain"]["m"]
    del st
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    brain = Brain(g, motor)
    y, cur = collect_1_1(g, seed=20261005)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = brain.counts(cur, 771000)
    keep = X.std(0) > 0
    sc = StandardScaler().fit(X[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(X[:, keep]), y)
    S = test_frames(steps=1500, seed=20261008)
    print(f"readout refitted on {len(y)} 1-1 frames; {len(S)} fresh test frames", flush=True)
    vis = RetinotopicVision(g)
    cols, seeds = [], []
    for i, smp in enumerate(S):
        for v in VARIANTS:
            for k in range(2):
                f = S[(i + len(S) // 2) % len(S)]["frame"] if v == "swap" else paint(smp, v)
                vis.reset()
                cols.append(vis.encode(f))
                seeds.append(70000 + 10 * i + k)
    cur_all = np.stack(cols, 1)
    nV = len(VARIANTS)
    R = {n: np.asarray(FixedRandomProjection(len(motor), 8, r).W) for n, r in zip(names, ro)}

    def counts_for(m_row):
        out = []
        for s0 in range(0, cur_all.shape[1], 60):
            sd = seeds[s0:s0 + 60]
            b = P(brain.W, motor, brain.p, seeds=sd, cache_key=("ctrl", 0))
            if m_row is not None:
                b.set_weights(m_row[None, :].repeat(len(sd), 1).to(b.dev))
            c, _ = sweep_once(b, brain.p, torch.as_tensor(cur_all[:, s0:s0 + 60], device=b.dev), brain.K, 0.5, False)
            out.append(c.t().cpu().numpy())
            del b
            torch.cuda.empty_cache()
        return np.concatenate(out, 0)                       # (columns, M)

    def jump_logodds(C, Rw):
        z = (C @ Rw.T / 2.0).reshape(len(S), nV, 2, 8)      # tau 2
        a0 = z[:, 0].mean(1).argmax(1)
        J = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
        zj = z[..., J]
        mx = zj.max(-1, keepdims=True)
        lse = np.log(np.exp(zj - mx).sum(-1)) + mx[..., 0]
        lo = (lse - np.take_along_axis(z, a0[:, None, None, None].repeat(nV, 1).repeat(2, 2), -1)[..., 0]).mean(2)
        return lo - lo[:, :1]                                # change from the original frame

    def verdict(d):
        return specificity(d[:, 1], (np.abs(d[:, 2]) + np.abs(d[:, 3])) / 2, d[:, 4])

    res = {}
    C0 = counts_for(None)                                    # untrained: positive and untrained negatives
    lg = clf.decision_function(sc.transform(C0[:, keep])).reshape(len(S), nV, 2).mean(2)
    res["positive_danger_readout"] = verdict(lg - lg[:, :1])
    for i, n in enumerate(names):
        C = C0 if n.startswith("untrained") else counts_for(m_tr[i])
        res[n] = verdict(jump_logodds(C, R[n]))
        print(n, res[n], flush=True)
    ok = res["positive_danger_readout"]["pass"] and not any(v["pass"] for k, v in res.items() if k != "positive_danger_readout")
    rep = {"what": "fresh validation of the occlusion specificity criterion (PERCEPTION_CREDIT_STUDY)",
           "criterion": f"Wilcoxon p < 0.05 and S >= {S_MIN}", "n_frames": len(S), "results": res,
           "criterion_validated": ok, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "occlusion_criterion_check-fresh-20261003.json").write_text(json.dumps(rep, indent=1))
    print("positive:", res["positive_danger_readout"], "\ncriterion validated:", ok)


def main():
    import torch
    if "--fresh" in sys.argv:
        return fresh()
    neg = negative_reference()
    print("negative reference:", json.dumps(neg), flush=True)
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    brain = Brain(g, motor)
    y, cur = collect_1_1(g)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = brain.counts(cur, 770000)
    keep = X.std(0) > 0
    sc = StandardScaler().fit(X[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(X[:, keep]), y)
    S = test_frames()
    print(f"readout fitted on {len(y)} 1-1 frames; {len(S)} test frames", flush=True)
    vis = RetinotopicVision(g)
    rows = []
    for s0 in range(0, len(S), 6):
        chunk = S[s0:s0 + 6]
        frames, seeds = [], []
        for i, smp in enumerate(chunk):
            for v in VARIANTS:
                for k in range(2):
                    f = (S[(s0 + i + len(S) // 2) % len(S)]["frame"] if v == "swap" else paint(smp, v))
                    frames.append(f)
                    seeds.append(60000 + 10 * (s0 + i) + k)
        b = PlasticLIF(brain.W, motor, brain.p, seeds=seeds, cache_key=("ctrl", 0))
        cur_b = []
        for f in frames:
            vis.reset()
            cur_b.append(vis.encode(f))
        c, _ = sweep_once(b, brain.p, torch.as_tensor(np.stack(cur_b, 1), device=b.dev), brain.K, 0.5, False)
        logit = clf.decision_function(sc.transform(c.t().cpu().numpy()[:, keep]))
        logit = logit.reshape(len(chunk), len(VARIANTS), 2).mean(2)
        for i, smp in enumerate(chunk):
            rows.append({"level": smp["level"], **{f"d_{v}": float(logit[i, j] - logit[i, 0])
                                                   for j, v in enumerate(VARIANTS) if j}})
        del b
        torch.cuda.empty_cache()
    pos = criterion([r["d_enemy"] for r in rows], [(abs(r["d_scenery1"]) + abs(r["d_scenery2"])) / 2 for r in rows])
    pos["swap_mean_abs"] = round(float(np.mean([abs(r["d_swap"]) for r in rows])), 4)
    ok = pos["pass"] and not any(v["pass"] for v in neg.values())
    rep = {"what": "occlusion criterion references for docs/PERCEPTION_CREDIT_STUDY.md",
           "negative_mario_gen": neg, "positive_danger_readout": pos, "criterion_validated": ok,
           "rows": rows, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "occlusion_criterion_check-20261003.json").write_text(json.dumps(rep, indent=1))
    print("positive reference:", json.dumps(pos), "\ncriterion validated:", ok)


if __name__ == "__main__":
    main()
