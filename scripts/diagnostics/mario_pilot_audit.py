#!/usr/bin/env python
"""Audit of the stopped Mario pilot (sweep regime): what the learners learnt and why.

Reads the pilot's final report and checkpoint (no simulation) and writes
artifacts/runs/mario_pilot_audit-<tag>.json: per-fly episode statistics and action mix, policy
entropy and update size (collapse), share past the first Goomba (x > 350), and the followed
learner's action mix per block of 10 episodes.

    .venv/bin/python scripts/diagnostics/mario_pilot_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.env.retro_env import ACTION_NAMES
from flymario.paths import RUNS

JUMPS = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mario_sweep_pilot")
    ap.add_argument("--tag", default="20261003")
    a = ap.parse_args()
    rep = json.loads((RUNS / f"{a.run}.json").read_text())
    ck = torch.load(RUNS / "checkpoints" / f"{a.run}.pt", map_location="cpu", weights_only=False)
    names = [l["name"] for l in rep["learners"]]
    eps = rep["episodes"]
    per = {}
    for n in names:
        E = [e for e in eps if e["learner"] == n]
        last = E[-50:]
        act = np.sum([e["actions"] for e in last], 0).astype(float)
        act /= max(1.0, act.sum())
        per[n] = {"episodes": len(E),
                  "median_x_first20": float(np.median([e["best_x"] for e in E[:20]])),
                  "median_x_last50": float(np.median([e["best_x"] for e in last])),
                  "max_x": float(max(e["best_x"] for e in E)),
                  "deaths": sum(e["end"] == "death" for e in E),
                  "action_mix_last50": dict(zip(ACTION_NAMES, act.round(3).tolist())),
                  "return_per_episode_last50": float(np.mean([e["reward"] for e in last])),
                  "decisions_per_episode_last50": float(np.mean([e["decisions"] for e in last]))}
    ent = np.asarray(ck["stats"]["entropy"])
    upd = np.asarray(ck["stats"]["update_rms"])
    blocks = range(0, len(ent), 1000)
    collapse = {n: {"entropy_per_1000": [round(float(ent[i:i + 1000, b].mean()), 4) for i in blocks],
                    "raw_update_rms_per_1000": [float(upd[i:i + 1000, b].mean()) for i in blocks]}
                for b, n in enumerate(names)}
    goomba = {r: float(np.mean([e["best_x"] > 350 for e in eps if e["learner"].startswith(r)]))
              for r in ("learner", "frozen", "shuffled")}
    chron = []
    E = [e for e in eps if e["learner"] == names[0]]
    for i in range(0, min(len(E), 80), 10):
        act = np.sum([e["actions"] for e in E[i:i + 10]], 0).astype(float)
        act /= act.sum()
        chron.append({"episodes": [i, i + 9], "at_decision": E[i]["at_decision"],
                      "run_right_b": round(float(act[ACTION_NAMES.index("RIGHT+B")]), 3),
                      "any_jump": round(float(act[JUMPS].sum()), 3),
                      "median_x": float(np.median([e["best_x"] for e in E[i:i + 10]]))})
    out = {"what": "audit of the stopped Mario pilot (sweep regime)", "run": a.run,
           "decisions": rep["decisions"], "episodes": len(eps), "per_fly": per,
           "collapse": collapse, "share_past_first_goomba": goomba, "chronology_learner_ro1": chron,
           "reward_definition": "per decision: forward progress (px) past the best point - 0.02; "
                                "-25 on death, +100 at the flag (src/flymario/env/retro_env.py)"}
    path = RUNS / f"mario_pilot_audit-{a.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"share_past_first_goomba": goomba, "chronology": chron}, indent=1))


if __name__ == "__main__":
    main()
