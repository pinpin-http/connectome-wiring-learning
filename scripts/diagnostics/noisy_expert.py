#!/usr/bin/env python
"""RL_STUDY §21 (exploratory): a noisy take-off-label expert under hold2 on the harness-v2 training
suite. Trigger with probability h when the label is on, f when off (per eligible decision); play test
against K = 5 random hold2 jumpers at its own trigger rate (draw set 1 seeds). References only.

    .venv/bin/python scripts/diagnostics/noisy_expert.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from oracle_criterion_check import criterion  # noqa: E402
from play_harness_v2 import STARTS, SUITES, Hold6, _random_hold6, play  # noqa: E402

K, HOLD = 5, 2
HS, FS = (1.0, 0.7, 0.5, 0.3), (0.0, 0.05, 0.1, 0.2, 0.3)


def noisy(args):
    level, prefix, h, f, seed = args
    from flymario.env.schematic import takeoff_label_v2
    g = np.random.default_rng(seed)
    cnt = {"on_dec": 0, "off_dec": 0, "off_trig": 0, "onsets": 0, "missed": 0}
    state = {"prev": False, "hit": False}

    def trig(ram):
        lab = bool(takeoff_label_v2(ram))
        t = g.random() < (h if lab else f)
        if lab and not state["prev"]:
            cnt["onsets"] += 1
            if state["prev"] is False and state.get("open"):
                pass
        cnt["on_dec" if lab else "off_dec"] += 1
        cnt["off_trig"] += int(t and not lab)
        return t, lab

    pol = Hold6(lambda ram: None, HOLD)
    window = {"open": False, "hit": False}

    def trigger(ram):
        t, lab = trig(ram)
        if lab and not window["open"]:
            window.update(open=True, hit=False)
        if window["open"] and t and lab:
            window["hit"] = True
        if not lab and window["open"]:
            cnt["missed"] += int(not window["hit"])
            window["open"] = False
        state["prev"] = lab
        return t
    pol.trigger = trigger
    out = play(level, prefix, pol)
    if window["open"]:
        cnt["missed"] += int(not window["hit"])
    out.update(cnt)
    out["trigger_rate"] = pol.triggers / max(1, pol.eligible)
    return out


def main():
    st = [s for s in json.loads(STARTS.read_text())["starts"] if s["level"] in SUITES["training"]]
    allst = json.loads(STARTS.read_text())["starts"]
    res = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for h in HS:
            for f in FS:
                eps = list(pool.map(noisy, [(s["level"], s["prefix"], h, f, 20261700 + 100 * i)
                                            for i, s in enumerate(st)]))
                jobs = []
                for s, e in zip(st, eps):
                    si = allst.index(s)
                    for j in range(K):
                        jobs.append((s["level"], s["prefix"], e["trigger_rate"], 20261200 + 50 * si + j, HOLD))
                xs = list(pool.map(_random_hold6, jobs, chunksize=8))
                ref = [float(np.mean(xs[K * i:K * i + K])) for i in range(len(st))]
                c = criterion([e["best_x"] for e in eps], ref)
                lv = SUITES["training"]
                carried = any(np.mean([eps[i]["best_x"] - ref[i] for i in range(len(st)) if st[i]["level"] != L]) <= 0
                              for L in lv)
                ons = sum(e["onsets"] for e in eps)
                row = {"h": h, "f": f, "onset_miss_rate": round(sum(e["missed"] for e in eps) / max(1, ons), 3),
                       "off_label_trigger_rate": round(sum(e["off_trig"] for e in eps) / max(1, sum(e["off_dec"] for e in eps)), 3),
                       "trigger_rate": round(float(np.mean([e["trigger_rate"] for e in eps])), 3),
                       "mean_diff": c["mean_diff"], "p": c["p_greater"], "pass": bool(c["pass"]),
                       "counts": bool(c["pass"] and not carried)}
                res.append(row)
                print(row, flush=True)
    out = ROOT / "artifacts/runs/mission/noisy_expert-hold2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"what": "RL_STUDY §21 noisy expert under hold2, training suite", "rows": res,
                               "utc": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1))


if __name__ == "__main__":
    main()
