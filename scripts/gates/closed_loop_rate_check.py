#!/usr/bin/env python
"""Candidate A of docs/PERCEPTION_CREDIT_STUDY.md §13: does a frozen learner get further than
random jumping at its own jump rate?

Each brain plays greedily on 5-1, 8-1, 3-1, 6-1 after 0, 3, ..., 21 NOOP decisions (up to 600
decisions, stall 150), against a random jumper at that episode's jump rate (`oracle_criterion_check.play`).
PASS per brain: one-sided Wilcoxon (learner > rate-matched random) p < 0.05.

    .venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt artifacts/runs/checkpoints/mario_stageB.pt --tag stageB
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from mario_generalisation_eval import Evaluator
from oracle_criterion_check import JUMP, RUN, criterion, play

from flymario.paths import RUNS

LEVELS = ("Level5-1", "Level8-1", "Level3-1", "Level6-1")
DELAYS = tuple(range(0, 22, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=1, help="random jumpers averaged per start (5 since 2026-10-04 19:06)")
    ap.add_argument("--graph", default="connectome", help="the wiring the checkpoint was trained on")
    a = ap.parse_args()
    ev = Evaluator(a.ckpt, 1, 600, graph=a.graph)
    rows = []
    for level in LEVELS:
        for d in DELAYS:
            rec = ev.episode("rate", level, d)
            for b, e in enumerate(rec):
                q = e["jumps"] / max(1, e["decisions"])
                if e["learner"].startswith("untrained") and a.k > 1:
                    r = {"best_x": float("nan")}                # baselines: not needed for the verdict
                else:
                    gens = [np.random.default_rng(20261013 + 1000 * LEVELS.index(level) + 10 * d + b + 100000 * j)
                            for j in range(a.k)]
                    r = {"best_x": float(np.mean([play(level, d, lambda ram, g=g, q=q: JUMP if g.random() < q else RUN)
                                                  ["best_x"] for g in gens]))}
                rows.append({"learner": e["learner"], "level": level, "delay": d, "best_x": e["best_x"],
                             "jump_rate": round(q, 3), "random_best_x": r["best_x"], "end": e["end"]})
            print(f"{level} delay {d:2d}: " + " ".join(
                f"{r['learner'][:5]}{r['learner'][-1]}={r['best_x']:.0f}/{r['random_best_x']:.0f}"
                for r in rows[-len(rec):]), flush=True)
    res = {}
    for name in ev.names:
        R = [r for r in rows if r["learner"] == name]
        if np.isnan(R[0]["random_best_x"]):
            continue
        res[name] = {**criterion([r["best_x"] for r in R], [r["random_best_x"] for r in R]),
                     "mean_jump_rate": round(float(np.mean([r["jump_rate"] for r in R])), 3)}
        print(name, res[name], flush=True)
    rep = {"what": "candidate A on learners: frozen greedy play vs random jumping at the learner's own rate",
           "checkpoint": a.ckpt, "levels": LEVELS, "delays": DELAYS, "results": res, "rows": rows,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"closed_loop_rate_check-{a.tag}-20261004.json").write_text(json.dumps(rep, indent=1, default=float))
    ev.pool.close()


if __name__ == "__main__":
    main()
