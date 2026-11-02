#!/usr/bin/env python
"""Declared readings of RL_STUDY §18 (replication of the §13.1 drop pattern) and §18.1 (supervised
sample efficiency on identical data), applied exactly as declared. Writes
`artifacts/runs/mission/readings-18.json`.

    .venv/bin/python scripts/diagnostics/mission_readings.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/runs"
OUT = RUNS / "mission"
NULLS = ("nulllat0", "nulllat1", "nulllat2")


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def curve_means(prefix):
    """Per graph, per brain: mean held-out AUC over the scoring points after 0."""
    out = {}
    for gt in ("connectome",) + NULLS:
        d = load(RUNS / f"common_stream-{prefix}{gt}-20261009.json")
        if d is None:
            continue
        curve = d["results"][gt if gt == "connectome" else gt.replace("nulllat", "nulllat:")]
        A = np.array([c["auc"] for c in curve if c["frames"] > 0])
        out[gt] = {"curve_mean": A.mean(0).round(4).tolist(), "points": int(A.shape[0]),
                   "final": curve[-1]["auc"], "frames_final": curve[-1]["frames"]}
    return out


def mw(a, b, alternative):
    if len(a) == 0 or len(b) == 0:
        return None
    r = mannwhitneyu(a, b, alternative=alternative, method="exact")
    return {"U": float(r.statistic), "p": float(r.pvalue)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    # §18.1
    for name, prefix in (("expert (primary)", "expert-"), ("scripted (secondary)", "")):
        cm = curve_means(prefix)
        conn = cm.get("connectome", {}).get("curve_mean", [])
        nulls = [v for g in NULLS if g in cm for v in cm[g]["curve_mean"]]
        null_means = [float(np.mean(cm[g]["curve_mean"])) for g in NULLS if g in cm]
        res[f"18.1 {name}"] = {"per_graph": cm, "connectome": conn, "nulls": nulls,
                               "diff_of_means": round(float(np.mean(conn) - np.mean(nulls)), 4) if conn and nulls else None,
                               "per_null_means": [round(x, 4) for x in null_means],
                               "mann_whitney_two_sided": mw(conn, nulls, "two-sided"),
                               "complete": len(conn) == 3 and len(nulls) == 9}
        t = res[f"18.1 {name}"]
        t["verdict"] = (None if not t["complete"] else
                        ("difference: connectome " + ("lower" if t["diff_of_means"] < 0 else "higher"))
                        if t["mann_whitney_two_sided"]["p"] < 0.05 else "no difference shown at this power")
    # §18 replication
    rep = load(RUNS / "frozen_probe/reportv2-training-20261009.json")
    if rep:
        brains = rep.get("brains", {})
        drops = {}
        for key, b in brains.items():
            gt, name = key.split("/")
            if name.startswith("cs_ro") and b.get("s_fixed") is not None and b.get("s_own") is not None:
                drops.setdefault(gt, []).append(round(b["s_fixed"] - b["s_own"], 4))
        conn = drops.get("connectome", [])
        n01 = drops.get("nulllat0", []) + drops.get("nulllat1", [])
        n012 = n01 + drops.get("nulllat2", [])
        prim = mw(conn, n01, "less")
        res["18 replication"] = {"drops": drops, "primary_conn_vs_nulls01": prim,
                                 "replicated": None if prim is None or len(conn) < 3 or len(n01) < 6 else prim["p"] < 0.05,
                                 "secondary_conn_vs_nulls012": mw(conn, n012, "less"),
                                 "classes": {k: v.get("class") for k, v in brains.items()}}
    (OUT / "readings-18.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
