#!/usr/bin/env python
"""Study S reading (docs/STUDY_S.md §6), applied exactly as declared. Reads artifacts/runs/study_s/<graph>.json
for the connectome, nulllat0-39 (rung b) and nullin0-39 (rung c); writes artifacts/runs/study_s/reading.json.

    .venv/bin/python scripts/diagnostics/study_s_reading.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "artifacts/runs/study_s"
TASKS = ("where", "loom")
PRIMARY = [("E1", t) for t in TASKS]
SECONDARY = [(e, t) for t in TASKS for e in ("A_test", "A_gen", "E4", "E5")]
SIGN = {"E1": 1, "A_test": 1, "A_gen": 1, "E4": -1, "E5": -1}     # +1: higher is better for the learner


def value(d, e, t):
    return d["summary"][f"true/{t}"][e]


def load(name):
    p = D / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def family(conn, nulls, eps):
    """Per-endpoint rank, z, raw two-sided p and Westfall-Young max-|z| adjusted p (leave-one-out nulls)."""
    K = len(nulls)
    V = np.array([[value(n, e, t) for (e, t) in eps] for n in nulls])        # (K, m)
    c = np.array([value(conn, e, t) for (e, t) in eps])
    mu, sd = V.mean(0), V.std(0, ddof=1)
    zc = (c - mu) / np.where(sd > 0, sd, np.nan)
    maxz = []
    for i in range(K):
        o = np.delete(V, i, 0)
        z = (V[i] - o.mean(0)) / np.where(o.std(0, ddof=1) > 0, o.std(0, ddof=1), np.nan)
        maxz.append(np.nanmax(np.abs(z)))
    maxz = np.array(maxz)
    out = {}
    for j, (e, t) in enumerate(eps):
        lo, hi = (1 + (V[:, j] <= c[j]).sum()) / (K + 1), (1 + (V[:, j] >= c[j]).sum()) / (K + 1)
        better = SIGN[e] * (c[j] - mu[j]) > 0
        out[f"{e}/{t}"] = {"connectome": round(float(c[j]), 4), "null_mean": round(float(mu[j]), 4),
                           "null_sd": round(float(sd[j]), 4), "null_range": [round(float(V[:, j].min()), 4), round(float(V[:, j].max()), 4)],
                           "rank_from_bottom": int((V[:, j] < c[j]).sum()) + 1, "z": round(float(zc[j]), 3),
                           "p_raw_two_sided": round(float(min(1.0, 2 * min(lo, hi))), 4),
                           "p_adjusted_maxz": round(float((1 + (maxz >= abs(zc[j])).sum()) / (K + 1)), 4),
                           "direction": "connectome better" if better else "connectome worse",
                           "percentile": round(float((V[:, j] < c[j]).mean() * 100), 1)}
    return out


def main():
    conn = load("connectome")
    res = {"what": "Study S reading (docs/STUDY_S.md §6)"}
    rungs = {"b": [load(f"nulllat{s}") for s in range(40)], "c": [load(f"nullin{s}") for s in range(40)]}
    for rung, ns in rungs.items():
        ns = [n for n in ns if n is not None]
        res[f"rung_{rung}"] = {"K": len(ns), "primary": family(conn, ns, PRIMARY), "secondary": family(conn, ns, SECONDARY)}
    verdict = {}
    for fam in ("primary", "secondary"):
        for key in res["rung_b"][fam]:
            b_, c_ = res["rung_b"][fam][key], res["rung_c"][fam][key]
            sig = b_["p_adjusted_maxz"] < 0.05 and c_["p_adjusted_maxz"] < 0.05 and b_["direction"] == c_["direction"]
            typical = 10 <= b_["percentile"] <= 90 and 10 <= c_["percentile"] <= 90
            verdict[key] = ("ADVANTAGE" if sig and b_["direction"] == "connectome better" else
                            "DISADVANTAGE" if sig else "no difference shown" + (" (typical)" if typical else ""))
    res["verdict"] = verdict
    # instrument summary: learnability of the analysed graphs (descriptive)
    allg = [conn] + [n for ns in rungs.values() for n in ns if n is not None]
    res["learnability"] = {t: {"A_test_range": [round(min(value(x, "A_test", t) for x in allg), 3), round(max(value(x, "A_test", t) for x in allg), 3)],
                               "E1_range": [round(min(value(x, "E1", t) for x in allg), 3), round(max(value(x, "E1", t) for x in allg), 3)]}
                           for t in TASKS}
    (D / "reading.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"K": {r: res[f"rung_{r}"]["K"] for r in "bc"}, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
