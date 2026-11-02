#!/usr/bin/env python
"""RL pilot report (docs/RL_STUDY.md §6): per arm, learners and shuffled-reward
twins — whole-run mean Δx_rand and its 5,000-decision windows (the validated
metric, `learning_curves.py`), learner minus twin per readout, action frequencies
and jump share per window, policy entropy per window (from the checkpoint copies),
relative weight change at each copy.

    .venv/bin/python scripts/diagnostics/pilot_report.py [--srl]

--srl (RL_STUDY §9): the supervised + RL runs (`mario_srl*`, learners srl_ro<i>, twins
srlshuf_ro<i>), 2,500-decision windows.
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

from flymario.env.retro_env import ACTION_NAMES
from flymario.paths import RUNS

SRL = "--srl" in sys.argv
ARMS = ({"connectome": "mario_srl", "null:0": "mario_srl_null0", "null:1": "mario_srl_null1", "null:2": "mario_srl_null2"}
        if SRL else {"connectome": "mario_rl3", "nulllat:0": "mario_rl3_nulllat0", "nulllat:1": "mario_rl3_nulllat1"})
PRE, TWIN = ("srl_", "srlshuf_") if SRL else ("rl_", "rlshuf_")
if SRL:
    lc.WIN = 2500


def action_windows(eps, name, win=None):
    win = win or lc.WIN
    out = {}
    for w0 in range(0, 20000, win):
        a = np.zeros(len(ACTION_NAMES))
        for e in eps:
            if e["learner"] == name and w0 < e["at_decision"] <= w0 + win:
                a += np.asarray(e["actions"])
        if a.sum():
            out[f"{w0 // 1000}-{(w0 + win) // 1000}k"] = {ACTION_NAMES[i]: round(float(a[i] / a.sum()), 3)
                                                         for i in range(len(a)) if a[i] / a.sum() >= 0.02}
    return out


def entropy_windows(run):
    import torch
    out = {}
    for k in ((5, 10) if SRL else (5, 10, 20)):
        f = RUNS / "checkpoints" / f"{run}_{k}k.pt"
        if not f.exists():
            continue
        st = torch.load(f, map_location="cpu", weights_only=False)
        ent = np.asarray(st["stats"]["entropy"])
        names = [x["name"] for x in st["learners"]]
        out[f"{k}k"] = {n: round(float(ent[-1000:, b].mean()), 3) for b, n in enumerate(names)}
        del st
    return out


def main():
    res = {}
    for arm, run in ARMS.items():
        f = RUNS / f"{run}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        eps = [e for e in d["episodes"] if f"{e.get('level')}|{e.get('delay')}" in lc.BASE]
        cur = lc.curves(eps)
        learners = {k: v for k, v in cur.items() if not k.startswith(TWIN)}
        twins = {k: v for k, v in cur.items() if k.startswith(TWIN)}
        diff = {k: round(v["mean_dx_rand"] - twins.get(k.replace(PRE, TWIN), {}).get("mean_dx_rand", np.nan), 1)
                for k, v in learners.items()}
        res[arm] = {"decisions": d["decisions"],
                    "learners": {k: {"mean_dx_rand": v["mean_dx_rand"], "windows": v["windows_dx_rand"],
                                     "competent": bool(v["mean_dx_rand"] is not None and v["mean_dx_rand"] >= 50)}
                                 for k, v in learners.items()},
                    "twins": {k: {"mean_dx_rand": v["mean_dx_rand"], "windows": v["windows_dx_rand"]} for k, v in twins.items()},
                    "learner_minus_twin": diff,
                    "learners_above_twin": int(sum(x > 0 for x in diff.values() if not np.isnan(x))),
                    "relative_rms_weight_change": d.get("relative_rms_weight_change"),
                    "actions": {n: action_windows(d["episodes"], n) for n in cur},
                    "entropy_last1000_at_copies": entropy_windows(run)}
        print(arm, d["decisions"], "| learners", {k: v["mean_dx_rand"] for k, v in res[arm]["learners"].items()},
              "| twins", {k: v["mean_dx_rand"] for k, v in res[arm]["twins"].items()},
              "| learner-twin", diff, flush=True)
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (RUNS / ("srl_report-20261008.json" if SRL else "pilot_report-20261007.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
