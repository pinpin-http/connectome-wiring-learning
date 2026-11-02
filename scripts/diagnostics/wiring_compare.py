#!/usr/bin/env python
"""First wiring comparison (docs/PERCEPTION_CREDIT_STUDY.md §18-19): connectome vs depth-stratified nulls.

Rule: a difference is claimed only if the connectome's mean over its three learners lies outside the range of
the null means AND differs from their grand mean by more than 2 SD (ddof = 1). Quantities: threat/obstacle
rise, px and passes vs rate-matched random jumpers, motor spikes, AUCs; X (post hoc, exploratory).
Inputs: criterion_e_relabel-<run>.json/.npz, closed_loop_rate_check-<tag>-20261004.json.

    .venv/bin/python scripts/diagnostics/wiring_compare.py [--tag <suffix>] [--arms nulllat]

--arms nulllat (§21): the laterality-preserving nulls instead of the primary ones.
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

KIND = sys.argv[sys.argv.index("--arms") + 1] if "--arms" in sys.argv else "null"
ARMS = {"connectome": ("mario_stageD", "stageD")} | {
    f"{KIND}:{s}": (f"mario_stageD_{KIND}{s}", f"stageD-{KIND}{s}") for s in (0, 1, 2)}
L, T = "tko2", "tko2shuf"


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def arm_values(run, tag):
    v = {}
    f = RUNS / f"criterion_e_relabel-{run}.json"
    if f.exists():
        r = json.loads(f.read_text())["results"]
        for i in (1, 2, 3):
            x = r[f"{run}:{L}_ro{i}"]
            v.setdefault("1a_threat_rise", []).append(x["q_threat"] - x["q_clear"])
            v.setdefault("1b_obstacle_rise", []).append(x["q_obstacle"] - x["q_clear"])
    f = RUNS / f"closed_loop_rate_check-{tag}-20261004.json"
    if f.exists():
        r = json.loads(f.read_text())["results"]
        v["2a_play_px_vs_random"] = [r[f"{L}_ro{i}"]["mean_diff"] for i in (1, 2, 3)]
        v["2b_play_passes"] = [float(r[f"{L}_ro{i}"]["pass"]) for i in (1, 2, 3)]
        v["play_jump_rate"] = [r[f"{L}_ro{i}"]["mean_jump_rate"] for i in (1, 2, 3)]
    f = RUNS / f"criterion_e_relabel-{run}.npz"
    if f.exists():
        z = np.load(f)
        cat, tko = z["cat"], z["tko"].astype(bool)
        for i in (1, 2, 3):
            pl, pt = z[f"p_{L}_ro{i}"], z[f"p_{T}_ro{i}"]
            q = pl - pt
            v.setdefault("S1_motor_spikes_learner", []).append(float(z[f"spk_{L}_ro{i}"].mean()))
            v.setdefault("S1_motor_spikes_twin", []).append(float(z[f"spk_{T}_ro{i}"].mean()))
            v.setdefault("S2_auc_q_object_vs_clear", []).append(auc(q[cat < 2], q[cat == 2]))
            v.setdefault("S3_auc_takeoff_learner_minus_twin", []).append(
                auc(pl[tko], pl[~tko]) - auc(pt[tko], pt[~tko]))
            v.setdefault("S3_takeoff_share", []).append(float(tko.mean()))
            dl = np.log(pl / (1 - pl)) - np.log(pt / (1 - pt))
            v.setdefault("X_posthoc_logit_common", []).append(float(dl.mean()))
            v.setdefault("X_posthoc_logit_specific_sd", []).append(float(dl.std()))
    return v


def rule(c, nulls):
    cm, nm = float(np.mean(c)), [float(np.mean(x)) for x in nulls]
    gm, sd = float(np.mean(nm)), float(np.std(nm, ddof=1))
    outside = cm < min(nm) or cm > max(nm)
    beyond = abs(cm - gm) > 2 * sd
    return {"connectome_mean": round(cm, 4), "connectome_values": [round(x, 4) for x in c],
            "null_means": [round(x, 4) for x in nm], "null_grand_mean": round(gm, 4), "null_sd": round(sd, 4),
            "outside_range": bool(outside), "beyond_2sd": bool(beyond),
            "difference_claimed": bool(outside and beyond),
            "direction": "connectome higher" if cm > gm else "connectome lower"}


def detectable(sd, n_nulls=(3, 5, 10, 20), alpha=0.05, power=0.8):
    """Planning only: the connectome-minus-null difference a prediction-interval t test
    against n null realisations would detect (two-sided alpha, given power), if the
    null means spread as in this pilot (SD sd)."""
    from scipy.stats import t
    return {n: round(float((t.ppf(1 - alpha / 2, n - 1) + t.ppf(power, n - 1)) * sd * np.sqrt(1 + 1 / n)), 4)
            for n in n_nulls}


def main():
    vals = {a: arm_values(*ARMS[a]) for a in ARMS}
    out = {}
    keys = sorted({k for v in vals.values() for k in v})
    for k in keys:
        nulls = [vals[a][k] for a in ARMS if a != "connectome" and k in vals[a]]
        if k not in vals["connectome"] or len(nulls) < 3:
            out[k] = {"status": f"incomplete: {len(nulls)}/3 null arms",
                      "values": {a: vals[a].get(k) for a in ARMS}}
            continue
        out[k] = rule(vals["connectome"][k], nulls)
        out[k]["detectable_difference_by_n_nulls"] = detectable(out[k]["null_sd"])
    for k, r in out.items():
        print(k, json.dumps(r))
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else (
        "20261006" if KIND == "null" else f"{KIND}-20261006")
    rep = {"what": "first wiring comparison, PILOT (§18 reading rule; §19 secondary analyses exploratory)",
           "arms": ARMS, "per_arm": vals, "comparison": out, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"wiring_compare-{tag}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
