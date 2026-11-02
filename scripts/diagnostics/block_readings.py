#!/usr/bin/env python
"""RL_STUDY §23 readings: hold2 + expert stream, blocks 0 and 1. Block 0 = the §20.3 batched plays
(connectome, nulls 0-1) and the separate play for nulllat:2; block 1 = the expertb1 plays.

    .venv/bin/python scripts/diagnostics/block_readings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gates"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
sys.path.insert(0, str(ROOT / "src"))
from frozen_play_probe import play_test  # noqa: E402

FP = ROOT / "artifacts/runs/frozen_probe"
G = ("connectome", "nulllat0", "nulllat1", "nulllat2")


def brains(path, prefix):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for name in dict.fromkeys(r["learner"] for r in d["rows"]):
        if name.startswith(prefix):
            t = play_test([r for r in d["rows"] if r["learner"] == name])
            out[name] = {"gain": t["mean_diff"], "C": t["counts"], "p": t["p_greater"]}
    return out


def main():
    res = {"block0": {}, "block1": {}}
    for g in G:
        b0 = (brains(FP / f"playv2a-training-hold2-{g}-20261009.json", "csx_")
              or brains(FP / f"playv2x-training-hold2-{g}-20261009.json", "csx_"))
        res["block0"][g] = b0
        res["block1"][g] = brains(FP / f"playv2y-training-hold2-{g}-20261009.json", "csxb_")
    conn = [v for blk in ("block0", "block1") for v in (res[blk]["connectome"] or {}).values()]
    nulls = [v for blk in ("block0", "block1") for g in G[1:] for v in (res[blk][g] or {}).values()]
    cc, nc = sum(v["C"] for v in conn), sum(v["C"] for v in nulls)
    res["counts"] = {g: {blk: (sum(v["C"] for v in res[blk][g].values()) if res[blk][g] else None)
                         for blk in ("block0", "block1")} for g in G}
    res["replication_null0_block1"] = (None if not res["block1"]["nulllat0"] else
                                       sum(v["C"] for v in res["block1"]["nulllat0"].values()) >= 2)
    res["pooled"] = {"n_connectome": len(conn), "n_nulls": len(nulls), "C_connectome": cc, "C_nulls": nc,
                     "fisher_two_sided_p": float(fisher_exact([[cc, len(conn) - cc], [nc, len(nulls) - nc]])[1])
                     if conn and nulls else None,
                     "mw_gain_two_sided": (lambda r: {"U": float(r.statistic), "p": float(r.pvalue)})(
                         mannwhitneyu([v["gain"] for v in conn], [v["gain"] for v in nulls], alternative="two-sided"))
                     if conn and nulls else None,
                     "mean_gain_connectome": float(np.mean([v["gain"] for v in conn])) if conn else None,
                     "mean_gain_nulls": float(np.mean([v["gain"] for v in nulls])) if nulls else None,
                     "complete": len(conn) == 6 and len(nulls) == 18}
    out = ROOT / "artifacts/runs/mission/readings-23.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("counts", "replication_null0_block1", "pooled")}, indent=1))


if __name__ == "__main__":
    main()
