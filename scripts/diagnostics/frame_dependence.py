#!/usr/bin/env python
"""RL_STUDY §4.2 (declared 2026-10-07 18:18): how many independent observations do
the contingency frames carry?

Per learner (learner minus its coin twin, q per frame, from the relabel `.npz`):
the rises q(threat) - q(clear) and q(obstacle) - q(clear), their naive 95%
interval (frames independent) and a moving-block bootstrap interval (blocks of
consecutive frames in collection order, within level; block 10 and 20), plus the
lag-1 autocorrelation of q between consecutive frames of the same category.

    .venv/bin/python scripts/diagnostics/frame_dependence.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.paths import RUNS

RUNS_NPZ = ("mario_stageD", "mario_stageD_null0", "mario_stageD_null1", "mario_stageD_null2")


def rises(q, cat):
    c = q[cat == 2].mean()
    return q[cat == 0].mean() - c, q[cat == 1].mean() - c


def block_boot(q, cat, lv, block, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    starts = [i for i in range(len(q)) if i + block <= len(q) and lv[i] == lv[i + block - 1]]
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, len(q) // block + 1)])[:len(q)]
        if min((cat[idx] == k).sum() for k in (0, 1, 2)) == 0:
            continue
        out.append(rises(q[idx], cat[idx]))
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5], axis=0)


def naive(q, cat):
    c = q[cat == 2]
    res = []
    for k in (0, 1):
        x = q[cat == k]
        se = np.sqrt(x.var(ddof=1) / len(x) + c.var(ddof=1) / len(c))
        m = x.mean() - c.mean()
        res.append((m - 1.96 * se, m + 1.96 * se))
    return np.array(res).T


def main():
    res = {}
    for run in RUNS_NPZ:
        f = RUNS / f"criterion_e_relabel-{run}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cat, lv = z["cat"], z["level"]
        for i in (1, 2, 3):
            q = z[f"p_tko2_ro{i}"] - z[f"p_tko2shuf_ro{i}"]
            rt, ro = rises(q, cat)
            ac = []
            for k in (0, 1, 2):
                for L in np.unique(lv):
                    x = q[(cat == k) & (lv == L)]
                    if len(x) > 3:
                        ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            nv = naive(q, cat)
            b10, b20 = block_boot(q, cat, lv, 10), block_boot(q, cat, lv, 20)
            r = {"threat_rise": round(float(rt), 4), "obstacle_rise": round(float(ro), 4),
                 "lag1_autocorrelation_mean": round(float(np.nanmean(ac)), 3),
                 "ci95_naive": [[round(float(v), 4) for v in nv[:, j]] for j in (0, 1)],
                 "ci95_block10": [[round(float(v), 4) for v in b10[:, j]] for j in (0, 1)],
                 "ci95_block20": [[round(float(v), 4) for v in b20[:, j]] for j in (0, 1)]}
            r["excludes_zero_block20"] = [bool(b20[0, j] > 0) for j in (0, 1)]
            res[f"{run}:tko2_ro{i}"] = r
            print(run, i, r)
    rep = {"what": "contingency rises with naive vs block-bootstrap intervals (RL_STUDY §4.2)",
           "frames": len(cat), "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "frame_dependence-20261007.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
