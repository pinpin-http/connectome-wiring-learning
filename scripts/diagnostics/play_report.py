#!/usr/bin/env python
"""Per-level play report and its validation (declared in docs/RL_STUDY.md §4.1).

d_level = mean over a level's 8 starts of (furthest x - mean furthest x of the brain's
rate-matched random jumpers). PASS = d_level > 0 on >= 3 of 4 levels and mean d_level > 0.
Also reports the one-sided Wilcoxon over the 32 starts and the distinct furthest-x count.

    .venv/bin/python scripts/diagnostics/play_report.py --validate
    .venv/bin/python scripts/diagnostics/play_report.py artifacts/runs/<closed_loop or threshold json> [...]
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from flymario.paths import RUNS


def wilcoxon_old(x, r):
    from oracle_criterion_check import criterion
    return criterion(x, r)


def report(rows):
    """rows: dicts with level, delay, best_x, random_best_x (one brain)."""
    levels = sorted({r["level"] for r in rows})
    per = {}
    for lv in levels:
        R = [r for r in rows if r["level"] == lv]
        d = [r["best_x"] - r["random_best_x"] for r in R]
        per[lv] = {"d": round(float(np.mean(d)), 1), "starts_won": int(sum(x > 0 for x in d)),
                   "starts_lost": int(sum(x < 0 for x in d)), "n": len(d),
                   "distinct_trajectories": len({round(r["best_x"]) for r in R})}
    dl = [v["d"] for v in per.values()]
    old = wilcoxon_old([r["best_x"] for r in rows], [r["random_best_x"] for r in rows])
    jaj = [r["jump_after_jump"] for r in rows if r.get("jump_after_jump") is not None]
    jr = [r["jump_rate"] for r in rows if "jump_rate" in r]
    return {"per_level": per, "levels_positive": int(sum(x > 0 for x in dl)), "n_levels": len(dl),
            "mean_jump_rate": round(float(np.mean(jr)), 3) if jr else None,
            "mean_jump_after_jump": round(float(np.mean(jaj)), 3) if jaj else None,
            "mean_d": round(float(np.mean(dl)), 1),
            "pass": bool(sum(x > 0 for x in dl) >= 3 and np.mean(dl) > 0),
            "old_rule": {"wins": old["wins"], "losses": old["losses"], "p": old["p_greater"], "pass": old["pass"]}}


def scripted_rows(file, who, ref="random_a"):
    eps = json.loads((RUNS / file).read_text())["episodes"]
    return [{"level": e["level"], "delay": e["delay"], "best_x": e[who]["best_x"],
             "random_best_x": e[ref]["best_x"]} for e in eps]


def brain_rows(file):
    rows = json.loads((RUNS / file).read_text())["rows"]
    out = {}
    for r in rows:
        if np.isnan(r["random_best_x"]):
            continue
        out.setdefault(r["learner"], []).append(r)
    return out


def validate():
    pos = {f"expert {k}": scripted_rows(f, "ram_jumper") for k, f in
           (("take-off v1", "oracle_criterion_check-takeoff-k5-20261004.json"),
            ("take-off v2", "oracle_criterion_check-takeoff2-k5-20261004.json"),
            ("full label", "oracle_criterion_check-full-k5-20261004.json"))}
    neg = {"random draw b (v2 file)": scripted_rows("oracle_criterion_check-takeoff2-k5-20261004.json", "random_b"),
           "always-run (v2 file)": scripted_rows("oracle_criterion_check-takeoff2-k5-20261004.json", "always_run")}
    for tag, f in (("connectome", "closed_loop_rate_check-stageD-20261004.json"),
                   ("null0", "closed_loop_rate_check-stageD-null0-20261004.json"),
                   ("null2", "closed_loop_rate_check-stageD-null2-20261004.json")):
        for name, rows in brain_rows(f).items():
            if "shuf" in name:
                neg[f"coin twin {tag} {name}"] = rows
    extra = {f"{who} ({k} file)": scripted_rows(f, who) for k, f in
             (("v1", "oracle_criterion_check-takeoff-k5-20261004.json"),
              ("full", "oracle_criterion_check-full-k5-20261004.json")) for who in ("random_b", "always_run")}
    res = {"positives": {k: report(v) for k, v in pos.items()},
           "negatives": {k: report(v) for k, v in neg.items()},
           "extra_negatives": {k: report(v) for k, v in extra.items()}}
    fp_new = sum(r["pass"] for r in res["negatives"].values())
    fp_old = sum(r["old_rule"]["pass"] for r in res["negatives"].values())
    tp_new = sum(r["pass"] for r in res["positives"].values())
    tp_old = sum(r["old_rule"]["pass"] for r in res["positives"].values())
    res["summary"] = {"new_rule": {"positives_passed": f"{tp_new}/{len(pos)}", "negatives_passed": f"{fp_new}/{len(neg)}",
                                   "extra_negatives_passed": f"{sum(r['pass'] for r in res['extra_negatives'].values())}/{len(extra)}"},
                      "old_rule": {"positives_passed": f"{tp_old}/{len(pos)}", "negatives_passed": f"{fp_old}/{len(neg)}",
                                   "extra_negatives_passed": f"{sum(r['old_rule']['pass'] for r in res['extra_negatives'].values())}/{len(extra)}"},
                      "adopted": bool(tp_new == len(pos) and fp_new <= 1)}
    for grp in ("positives", "negatives", "extra_negatives"):
        for k, r in res[grp].items():
            print(grp[:3], k, "| levels+", r["levels_positive"], "mean d", r["mean_d"], "PASS" if r["pass"] else "fail",
                  "| old", r["old_rule"]["wins"], "/", r["old_rule"]["losses"], "p", r["old_rule"]["p"],
                  "| per level", {lv[5:]: v["d"] for lv, v in r["per_level"].items()})
    print("summary", json.dumps(res["summary"]))
    return res


def main():
    if "--validate" in sys.argv:
        res = validate()
        out = RUNS / "play_report-validation-20261007.json"
    else:
        res = {}
        for f in [a for a in sys.argv[1:] if a.endswith(".json")]:
            res[Path(f).name] = {name: report(rows) for name, rows in brain_rows(Path(f).name).items()}
            for name, r in res[Path(f).name].items():
                print(Path(f).name, name, "| levels+", r["levels_positive"], "mean d", r["mean_d"],
                      "PASS" if r["pass"] else "fail", "| old", r["old_rule"]["wins"], "/", r["old_rule"]["losses"],
                      "p", r["old_rule"]["p"], "| jump", r["mean_jump_rate"], "held", r["mean_jump_after_jump"],
                      "| per level", {lv[5:]: v["d"] for lv, v in r["per_level"].items()},
                      "| distinct", [v["distinct_trajectories"] for v in r["per_level"].values()])
        out = RUNS / f"play_report-{'_'.join(Path(f).stem for f in res)[:80]}.json"
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
