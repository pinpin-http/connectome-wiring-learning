#!/usr/bin/env python
"""Is the Stage D timing stored in the weights? (docs/RL_STUDY.md §9.2-9.3)

Compares the connectome Stage D checkpoint resumed with learning on (`mario_stageD_cont`) and
frozen (`mario_stageD_frozen`), episodes after decision 20,000: mean Δx_rand per learner, paired
on − off differences with a 95% bootstrap interval. --graph null0 uses the `mario_stageD_null0_*` runs.

    .venv/bin/python scripts/diagnostics/stageD_frozen_check.py [--graph null0]
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

START = 20000
TAUGHT = [f"tko2_ro{i}" for i in (1, 2, 3)]
TWINS = [f"tko2shuf_ro{i}" for i in (1, 2, 3)]
G = sys.argv[sys.argv.index("--graph") + 1] if "--graph" in sys.argv else ""
PRE = f"mario_stageD_{G}_" if G else "mario_stageD_"


def dx_rand(e):
    a = e["actions"]
    js = (a[3] + a[4] + a[5]) / max(1, sum(a))
    return e["best_x"] - lc.rand_base(e["level"], e["delay"], js)


def episodes(run):
    d = json.loads((RUNS / f"{run}.json").read_text())
    out = {}
    for e in d["episodes"]:
        if e["at_decision"] > START and f"{e.get('level')}|{e.get('delay')}" in lc.BASE:
            out.setdefault(e["learner"], []).append(e)
    return d["decisions"], out


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)]


def main():
    n_on, on = episodes(PRE + "cont")
    n_off, off = episodes(PRE + "frozen")
    res = {"decisions_on": n_on, "decisions_off": n_off, "learners": {}}
    pooled = {"taught": [], "twins": []}
    for k in TAUGHT + TWINS:
        a, b = on.get(k, []), off.get(k, [])
        m = min(len(a), len(b))
        same = [(x["level"], x["delay"]) == (y["level"], y["delay"]) for x, y in zip(a[:m], b[:m])]
        diffs = [dx_rand(x) - dx_rand(y) for x, y, s in zip(a[:m], b[:m], same) if s]
        pooled["taught" if k in TAUGHT else "twins"] += diffs
        res["learners"][k] = {
            "on_mean": round(float(np.mean([dx_rand(x) for x in a])), 1) if a else None,
            "off_mean": round(float(np.mean([dx_rand(y) for y in b])), 1) if b else None,
            "n_on": len(a), "n_off": len(b), "pairs_same_start": int(sum(same)), "pairs_checked": m,
            "paired_diff_mean": round(float(np.mean(diffs)), 1) if diffs else None,
            "jump_share_on": round(float(np.mean([x["jump_share"] for x in a])), 3) if a and "jump_share" in a[0] else None,
            "jump_share_off": round(float(np.mean([y["jump_share"] for y in b])), 3) if b and "jump_share" in b[0] else None}
        print(k, res["learners"][k], flush=True)
    for g, v in pooled.items():
        res[f"pooled_{g}"] = {"n_pairs": len(v), "mean": round(float(np.mean(v)), 1) if v else None,
                              "ci95": boot_ci(v) if len(v) > 2 else None}
        print(g, res[f"pooled_{g}"])
    L = res["learners"]
    ref_pass = sum((L[k]["on_mean"] or -1e9) >= 50 for k in TAUGHT) >= 2
    frozen_ok = sum((L[k]["off_mean"] or -1e9) >= 50 for k in TAUGHT)
    ci = res["pooled_taught"]["ci95"]
    if not ref_pass:
        verdict = "inconclusive: learning-on does not reproduce Stage D"
    elif frozen_ok >= 2:
        verdict = "stored timing"
    elif ci is not None and ci[0] > 0:
        verdict = "online steering"
    else:
        verdict = "inconclusive"
    res["declared"] = {"reference_on_passes": ref_pass, "frozen_readouts_ge_50": frozen_ok, "verdict": verdict}
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    print("declared:", res["declared"])
    (RUNS / f"stageD_frozen_check{'-' + G if G else ''}-20261008.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
