#!/usr/bin/env python
"""Frozen-checkpoint competence (docs/RL_STUDY.md §11.1 b).

Reads the frozen evaluations (`train_mario_sweep.py --version 19 --eval-from <ckpt>`,
runs `mario_stageD_eval[_<graph>]_frozeneval`). Per brain: episodes, mean Δx_rand, 95%
bootstrap interval; competent = at least 100 episodes and mean >= +50. Learners are read
only if every take-off v2 reference is competent and every shuffled-label twin is not.

    .venv/bin/python scripts/diagnostics/frozen_competence.py
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

import learning_curves as lc

from flymario.paths import RUNS

RUNS_ = {"connectome": "mario_stageD_eval_frozeneval", "null:0": "mario_stageD_eval_null0_frozeneval"}
MIN_EPISODES, THRESHOLD = 100, 50.0


def dx_rand(e):
    a = e["actions"]
    js = (a[3] + a[4] + a[5]) / max(1, sum(a))
    return e["best_x"] - lc.rand_base(e["level"], e["delay"], js)


def main():
    rng = np.random.default_rng(0)
    res = {}
    for arm, run in RUNS_.items():
        f = RUNS / f"{run}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        by = {}
        for e in d["episodes"]:
            if f"{e.get('level')}|{e.get('delay')}" in lc.BASE:
                by.setdefault(e["learner"], []).append(dx_rand(e))
        brains = {}
        for name, v in sorted(by.items()):
            v = np.asarray(v)
            boots = rng.choice(v, (5000, len(v))).mean(1)
            m = float(v.mean())
            brains[name] = {"episodes": len(v), "mean": round(m, 1),
                            "ci95": [round(float(np.percentile(boots, q)), 1) for q in (2.5, 97.5)],
                            "competent": bool(len(v) >= MIN_EPISODES and m >= THRESHOLD),
                            "judgeable": len(v) >= MIN_EPISODES}
        refs_ok = all(b["competent"] for n, b in brains.items() if n.startswith("script_"))
        twins_ok = all(not b["competent"] for n, b in brains.items() if n.startswith("tko2shuf_"))
        enough = all(b["episodes"] >= MIN_EPISODES for n, b in brains.items() if n.startswith("script_"))
        res[arm] = {"decisions": d["decisions"], "brains": brains, "enough_episodes": enough,
                    "positive_references_pass": refs_ok, "negative_twins_fail": twins_ok,
                    "criterion_validated": bool(refs_ok and twins_ok and enough)}
        print(arm, d["decisions"], "validated" if res[arm]["criterion_validated"] else "NOT validated",
              {"refs": refs_ok, "twins": twins_ok, "enough": enough}, flush=True)
        for n, b in brains.items():
            if res[arm]["criterion_validated"] or not n.startswith("tko2_"):
                print("  ", n, b, flush=True)
            else:
                print("  ", n, "(learner not read: criterion not validated)", flush=True)
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (RUNS / "frozen_competence-20261009.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
