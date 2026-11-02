#!/usr/bin/env python
"""Rate-matched random baseline on the training levels (docs/RL_STUDY.md §5,
amendment 18:27): mean furthest x of a random jumper (RIGHT+A+B with probability
q, else RIGHT+B) per level, start delay and jump rate q, K seeds each. Lets every
training episode be scored against random jumping at its own jump share.

    .venv/bin/python scripts/diagnostics/random_jumper_table.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

LEVELS = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")
DELAYS = tuple(range(0, 23, 2))
RATES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
K = 3


def one(args):
    from oracle_criterion_check import JUMP, RUN, play
    lv, d = args
    out = {}
    for q in RATES:
        xs = []
        for k in range(K):
            g = np.random.default_rng([20261072, LEVELS.index(lv), d, int(q * 100), k])
            xs.append(play(lv, d, lambda ram, g=g, q=q: JUMP if g.random() < q else RUN)["best_x"])
        out[str(q)] = float(np.mean(xs))
    return f"{lv}|{d}", out


def main():
    from flymario.paths import RUNS
    jobs = [(lv, d) for lv in LEVELS for d in DELAYS]
    with ProcessPoolExecutor(6) as ex:
        table = dict(ex.map(one, jobs))
    rep = {"what": "mean furthest x of random jumpers (K=3) per training level, delay and jump rate",
           "levels": LEVELS, "delays": DELAYS, "rates": RATES, "table": table}
    (RUNS / "reference" / "random_jumper_train.json").write_text(json.dumps(rep, indent=1))
    for lv in LEVELS:
        print(lv, {q: round(float(np.mean([table[f"{lv}|{d}"][str(q)] for d in DELAYS]))) for q in RATES})


if __name__ == "__main__":
    main()
