#!/usr/bin/env python
"""Audit of the game trainer's reinforcement-learning procedure (docs/RL_STUDY.md §2). No training.

A. Policy factor under the exploration floor: actions ~ mu = (1 - eps) pi + eps / A, update
   onehot(a) - pi (docs/MARIO_PILOT_PLAN.md); on a one-state bandit, compared exactly with the
   gradients of J(pi) and J(mu) as the policy saturates, then plain ascent with each factor.
B. Episode ends in the RL runs: deaths vs truncations (time limit, 150-decision stall).
C. Saturation in the checkpoints: policy entropy over training, plastic weights at [0, 2 m0].

    .venv/bin/python scripts/diagnostics/rl_procedure_audit.py [--part A|B|C]
"""
from __future__ import annotations

import collections
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.paths import RUNS

A_N, EPS = 8, 0.1


def softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def expected_updates(z, Q, eps=EPS):
    pi = softmax(z)
    mu = (1 - eps) * pi + eps / len(z)
    b = mu @ Q
    eye = np.eye(len(z))
    impl = sum(mu[a] * (Q[a] - b) * (eye[a] - pi) for a in range(len(z)))
    behav = sum(mu[a] * (Q[a] - b) * (1 - eps) * pi[a] / mu[a] * (eye[a] - pi) for a in range(len(z)))
    true_pi = pi * (Q - pi @ Q)
    return pi, mu, impl, behav, true_pi


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def part_a():
    rng = np.random.default_rng(20261007)
    out = {"saturation": [], "ascent": {}}
    Q = rng.normal(0, 1, A_N)
    z0 = rng.normal(0, 1, A_N)
    for scale in (0.5, 1, 2, 4, 8, 16):
        z = z0 * scale
        pi, mu, impl, behav, tru = expected_updates(z, Q)
        H = float(-(pi * np.log(pi + 1e-300)).sum())
        out["saturation"].append({
            "logit_scale": scale, "entropy_pi": round(H, 3), "max_pi": round(float(pi.max()), 4),
            "cos(implemented, grad J(pi))": round(cos(impl, tru), 4),
            "cos(behaviour factor, grad J(pi))": round(cos(behav, tru), 4),
            "|implemented| / |grad J(pi)|": round(float(np.linalg.norm(impl) / (np.linalg.norm(tru) + 1e-300)), 2),
            "implemented: share of the eps/A term": round(float(
                np.linalg.norm(EPS / A_N * (Q - mu @ Q)) / (np.linalg.norm(impl) + 1e-300)), 3)})
    # ascent dynamics from the same start: the best action is not the initially preferred one
    for name in ("implemented", "behaviour", "exact"):
        z = z0.copy()
        traj = []
        for t in range(4001):
            pi, mu, impl, behav, tru = expected_updates(z, Q)
            g = {"implemented": impl, "behaviour": behav, "exact": (1 - EPS) * tru}[name]
            if t % 500 == 0:
                traj.append({"step": t, "entropy": round(float(-(pi * np.log(pi + 1e-300)).sum()), 4),
                             "logit_range": round(float(np.ptp(z)), 2), "pi_best": round(float(pi[np.argmax(Q)]), 4),
                             "J_mu": round(float(mu @ Q), 4)})
            z = z + 0.5 * g
        out["ascent"][name] = traj
    out["Q"] = Q.round(3).tolist()
    for r in out["saturation"]:
        print("A", r)
    for k, v in out["ascent"].items():
        print("A ascent", k, v[0], v[-1])
    return out


def part_b():
    res = {}
    for r in ("mario_gen", "mario_oracle", "mario_stageD", "mario_stageD_null0"):
        f = RUNS / f"{r}.json"
        if not f.exists():
            continue
        eps = json.loads(f.read_text())["episodes"]
        c = collections.Counter(e["end"] for e in eps)
        n = sum(c.values())
        dec = collections.Counter()
        for e in eps:
            dec[e["end"]] += e["decisions"]
        res[r] = {"episodes": n, "ends": dict(c), "truncated_share": round((c["stall"] + c["time"]) / n, 3),
                  "decisions_in_truncated_episodes_share": round((dec["stall"] + dec["time"]) / sum(dec.values()), 3)}
        print("B", r, res[r])
    return res


def part_c():
    import torch
    res = {}
    for ck in ("mario_gen_final_20505.pt", "mario_oracle.pt"):
        st = torch.load(RUNS / "checkpoints" / ck, map_location="cpu", weights_only=False)
        names = [l["name"] for l in st["learners"]]
        ent = np.asarray(st["stats"]["entropy"])            # (decisions, B)
        n = len(ent)
        q = [0, n // 10, n // 4, n // 2, n - 1]
        res[ck] = {"decisions": n, "entropy_at": {names[b]: [round(float(ent[max(0, i - 200):i + 1, b].mean()), 3)
                                                             for i in q] for b in range(len(names))},
                   "entropy_checkpoints_decision": q}
        m = st["brain"]["m"]
        res[ck]["weights_at_zero_share"] = {names[b]: round(float((m[b] == 0).float().mean()), 4)
                                            for b in range(len(names))}
        print("C", ck, json.dumps(res[ck])[:900])
        del st
    return res


def main():
    parts = sys.argv[sys.argv.index("--part") + 1].split(",") if "--part" in sys.argv else ["A", "B", "C"]
    out = {}
    for p in parts:
        out[p] = {"A": part_a, "B": part_b, "C": part_c}[p]()
    out["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (RUNS / f"rl_procedure_audit-{'_'.join(parts)}-20261007.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
