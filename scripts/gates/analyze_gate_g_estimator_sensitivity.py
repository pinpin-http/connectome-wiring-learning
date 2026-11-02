#!/usr/bin/env python
"""Re-fit saved Gate G v2 regressions with selected directions separated.

This is an offline sensitivity analysis. It does not replace or rewrite the
pre-registered v2 verdict and does not run the simulator or use the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
FIRST = RUNS / "gate_g_estimator_check_v2.json"
POWER = RUNS / "gate_g_estimator_power_v2.json"
VERDICT = RUNS / "gate_g_estimator_verdict_v2.json"
OUT = RUNS / "gate_g_estimator_sensitivity_v2.json"


def fit(x, y):
    result = stats.linregress(np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64))
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    return {
        "n": int(len(x)),
        "slope": float(result.slope),
        "slope_ci95": [float(result.slope - tcrit * result.stderr),
                       float(result.slope + tcrit * result.stderr)],
        "r": float(result.rvalue),
        "intercept": float(result.intercept),
    }


def compare_random_to_mixed(x_random, y_random, x_selected, y_selected):
    return {
        "random_only": fit(x_random, y_random),
        "with_selected_direction": fit(
            [*x_random, float(x_selected)], [*y_random, float(y_selected)]),
    }


def main():
    first = json.loads(FIRST.read_text())
    power = json.loads(POWER.read_text()) if POWER.exists() else {}
    verdict = json.loads(VERDICT.read_text())
    result = {
        "what": "Exploratory sensitivity analysis: v2 calibration with selected direction separated",
        "interpretation": "Not a replacement verdict; descriptive OLS intervals treat x as fixed and omit Monte Carlo uncertainty in both axes.",
        "sources": [str(FIRST.relative_to(ROOT)), str(POWER.relative_to(ROOT)),
                    str(VERDICT.relative_to(ROOT))],
        "configurations": {},
    }
    for key in verdict["configurations"]:
        src = power.get(key, first[key])
        fd1 = src["fd_set1"]
        fd2 = src["fd_set2"]
        fd_random = compare_random_to_mixed(
            [x[0] for x in fd2], [x[0] for x in fd1],
            src["own_fd_set2_node_perturbation"][0],
            src["own_fd"]["node_perturbation"][0],
        )
        rules = {}
        for name in ("node_perturbation", "node_perturbation_full_trajectory"):
            row = src["rules"][name]
            rules[name] = compare_random_to_mixed(
                row["estimates"][:-1], row["true"][:-1],
                row["estimates"][-1], row["true"][-1],
            )
        result["configurations"][key] = {
            "source": "power rerun" if key in power else "first pass",
            "finite_difference_reference": fd_random,
            "rules": rules,
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
