#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 pilot summary against the criteria declared in docs/experience_commune/PILOT.md.

    .venv/bin/python scripts/experience_commune/pilot_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experience_commune_v1"
CK = (0, 250, 500, 1000, 2000, 4000, 6000)


def main() -> int:
    runs = {}
    for g in ("nulllat100", "nullin100"):
        d = ART / "runs_pilot" / g
        r = json.loads((d / "run.json").read_text())
        z = np.load(d / "predictions.npz")
        diag = np.load(d / "diagnostics.npz")
        keys = [f"{l['task']}-{l['mode']}-{l['block']}-{l['kind']}" for l in r["learners"]]
        res = {}
        for j, k in enumerate(keys):
            prog = [float((z[f"progress_{c}"][j] == z["labels_progress"][j]).mean()) for c in CK]
            res[k] = {"progress": prog, "final": float((z["final"][j] == z["labels_final"][j]).mean()),
                      "ood": float((z["ood"][j] == z["labels_ood"][j]).mean()),
                      "frozen_predictions_identical_all_checkpoints":
                          bool(all(np.array_equal(z[f"progress_{c}"][j], z["progress_0"][j]) for c in CK)),
                      "reward_rate_training": float(diag["reward"][:, j].mean()),
                      "rho_mean": float(diag["rho"][:, j].mean()), "rho_max": float(diag["rho"][:, j].max()),
                      "entropy_pi_first_last": [float(diag["entropy_pi"][:100, j].mean()),
                                                float(diag["entropy_pi"][-100:, j].mean())]}
        runs[g] = {"learners": res, "timing": {k: r[k] for k in ("train_step_s_median", "train_s_total", "eval_s",
                                                                "gpu_peak_mem_gb", "wall_s")}}
    crit = {}
    for t in ("where", "loom"):
        mean = lambda kind, mode: float(np.mean([runs[g]["learners"][f"{t}-{mode}-d0-{kind}"]["final"] for g in runs]))  # noqa: E731
        crit[t] = {"U_final_mean": mean("main", "U"), "N_final_mean": mean("main", "N"),
                   "supervised_final_mean": mean("supervised", "N"), "shuffled_final_mean": mean("shuffled", "U"),
                   "frozen_no_progression": all(runs[g]["learners"][f"{t}-U-d0-frozen"]
                                                ["frozen_predictions_identical_all_checkpoints"] for g in runs)}
        c = crit[t]
        c["pass_U_learns"] = c["U_final_mean"] >= 0.25
        c["pass_supervised"] = c["supervised_final_mean"] >= 0.80
        c["pass_shuffled"] = c["shuffled_final_mean"] <= 0.20
    allpass = all(c["pass_U_learns"] and c["pass_supervised"] and c["pass_shuffled"] and c["frozen_no_progression"]
                  for c in crit.values())
    out = {"criteria": crit, "all_pass": allpass, "runs": runs}
    (ART / "manifests/pilot_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"criteria": crit, "all_pass": allpass}, indent=1))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
