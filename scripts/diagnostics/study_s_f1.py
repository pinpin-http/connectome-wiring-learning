#!/usr/bin/env python
"""Study S follow-up F1 reading (docs/STUDY_S.md, declared 2026-10-12 06:10 UTC): replication with fresh seeds.

E1 replicates on a task when (i) the replicated connectome's E1 is above every replicated null and (ii) it lies
> 3 SD above the mean of the ORIGINAL null distribution at both rungs. Also reports each replicated null's shift
from its original E1. Writes artifacts/runs/study_s/f1_reading.json.
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "artifacts/runs/study_s"
NULLS = ["nulllat0", "nulllat1", "nulllat2", "nulllat3", "nullin0", "nullin1", "nullin2", "nullin3"]


def e1(name, t):
    return json.loads((D / f"{name}.json").read_text())["summary"][f"true/{t}"]["E1"]


def main():
    orig = json.loads((D / "reading-K20c.json").read_text())
    out = {"what": "Study S F1 reading (declared 2026-10-12 06:10 UTC)", "tasks": {}}
    for t in ("where", "loom"):
        c = e1("connectome-rep1", t)
        reps = {n: e1(f"{n}-rep1", t) for n in NULLS if (D / f"{n}-rep1.json").exists()}
        shifts = {n: round(v - e1(n, t), 4) for n, v in reps.items()}
        z = {r: (c - orig[f"rung_{r}"]["primary"][f"E1/{t}"]["null_mean"]) / orig[f"rung_{r}"]["primary"][f"E1/{t}"]["null_sd"] for r in "bc"}
        above_all = all(c > v for v in reps.values())
        out["tasks"][t] = {"connectome_rep1": c, "connectome_orig": e1("connectome", t), "nulls_rep1": reps,
                           "null_shift_rep1_minus_orig": shifts, "mean_abs_shift": round(float(np.mean(np.abs(list(shifts.values())))), 4),
                           "max_null_rep1": max(reps.values()), "z_vs_original_nulls": {r: round(v, 2) for r, v in z.items()},
                           "n_null_reps": len(reps), "replicates": bool(above_all and min(z.values()) > 3 and len(reps) == len(NULLS))}
    (D / "f1_reading.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
