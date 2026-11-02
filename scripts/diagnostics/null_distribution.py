#!/usr/bin/env python
"""RL_STUDY §24 reading: the connectome's expert-stream learning statistic within the null distribution.

    .venv/bin/python scripts/diagnostics/null_distribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/runs"


def S(path, key):
    d = json.loads(path.read_text())
    A = np.array([c["auc"] for c in d["results"][key] if c["frames"] > 0])
    return float(A.mean()), A.mean(0).tolist()


def main():
    import sys
    rm = "--matched" in sys.argv          # §25: nulls at the matched gain; the connectome's run is unchanged
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "nulllat"   # §26: nullin
    scripted = "--scripted" in sys.argv     # §27: the scripted stream
    st = "" if scripted else "expert"
    c, c_ro = S(RUNS / f"common_stream-{st + '-' if st else ''}connectome-20261009.json", "connectome")
    nulls = {}
    for s in range(40):
        p = RUNS / (f"common_stream-{kind}{s}-20261009.json" if scripted
                    else f"common_stream-expert{'rm' if rm else ''}-{kind}{s}-20261009.json")
        if p.exists():
            nulls[s] = S(p, f"{kind}:{s}")[0]
    v = np.array(list(nulls.values()))
    K = len(v)
    lo, hi = (1 + (v <= c).sum()) / (K + 1), (1 + (v >= c).sum()) / (K + 1)
    out = {"K": K, "connectome_S": c, "connectome_readouts": c_ro, "null_S": nulls,
           "rank_from_bottom": int((v < c).sum()) + 1, "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
           "z": float((c - v.mean()) / v.std(ddof=1)), "p_two_sided": float(min(1.0, 2 * min(lo, hi))),
           "complete": K == 40}
    b1 = RUNS / "common_stream-expertb1-connectome-20261009.json"
    if b1.exists() and not scripted:
        cb = S(b1, "connectome")[0]
        out["connectome_block1_S"] = cb
        out["block1_rank_from_bottom"] = int((v < cb).sum()) + 1
    act = RUNS / "mission/graph_activity.json"
    if act.exists() and not rm:
        a = json.loads(act.read_text())
        xs = [(np.log(a[f"{kind}:{s}"]["motor_spikes_per_decision"]), nulls[s]) for s in nulls if f"{kind}:{s}" in a]
        if len(xs) >= 5 and "connectome" in a:
            x, y = np.array(xs).T
            lr = stats.linregress(x, y)
            xc = np.log(a["connectome"]["motor_spikes_per_decision"])
            n = len(x)
            se = np.sqrt(np.sum((y - (lr.intercept + lr.slope * x)) ** 2) / (n - 2))
            pi = stats.t.ppf(0.975, n - 2) * se * np.sqrt(1 + 1 / n + (xc - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
            pred = lr.intercept + lr.slope * xc
            out["activity"] = {"n_nulls": n, "slope": lr.slope, "r": lr.rvalue, "p_slope": lr.pvalue,
                               "connectome_log_activity": float(xc), "null_log_activity_range": [float(x.min()), float(x.max())],
                               "predicted_S": float(pred), "pi95": [float(pred - pi), float(pred + pi)],
                               "connectome_inside_pi": bool(pred - pi <= c <= pred + pi),
                               "extrapolated": bool(xc < x.min() or xc > x.max())}
    out["regime"] = "matched activity (§25)" if rm else "shared parameters (§24)"
    out["null_kind"] = kind
    tagn = "27" if scripted else ("25" if rm else ("26" if kind == "nullin" else "24"))
    (RUNS / f"mission/readings-{tagn}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "null_S"}, indent=1))


if __name__ == "__main__":
    main()
