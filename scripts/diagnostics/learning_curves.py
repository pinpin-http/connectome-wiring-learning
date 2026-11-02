#!/usr/bin/env python
"""Learning curves for every learner trained on the four training levels (docs/RL_STUDY.md §5).

Primary metric Δx_rand = furthest x - mean furthest x of random jumpers at the episode's jump share,
same level and start (`artifacts/runs/reference/random_jumper_train.json`). Also reported: Δx
against always-run (competence at +100 px). Per learner: mean per 5,000-decision window, decisions
to competence (rolling 50-episode mean Δx_rand >= +50 px, censored), area under the curve.
Positive reference: the take-off v2 expert.

    .venv/bin/python scripts/diagnostics/learning_curves.py
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

BASE = json.loads((RUNS / "reference" / "always_run_train.json").read_text())
_RJ = RUNS / "reference" / "random_jumper_train.json"
RJ = json.loads(_RJ.read_text()) if _RJ.exists() else None
WIN = 5000


def rand_base(level, delay, q):
    """Mean furthest x of random jumpers at rate q from (level, nearest delay)."""
    d = min(RJ["delays"], key=lambda x: abs(x - delay))
    row = RJ["table"][f"{level}|{d}"]
    rates = np.array(RJ["rates"])
    return float(np.interp(q, rates, [row[str(r)] for r in RJ["rates"]]))


def scripted_reference():
    """The take-off v2 expert on the training levels (positive reference, cached)."""
    f = RUNS / "reference" / "scripted_takeoff2_train.json"
    if not f.exists():
        sys.path.insert(0, str(ROOT / "scripts/gates"))
        from oracle_criterion_check import JUMP, RUN, play

        from flymario.env.schematic import takeoff_label_v2
        eps = []
        for lv in ("Level1-1", "Level2-1", "Level4-1", "Level7-1"):
            for d in range(0, 23, 2):
                r = play(lv, d, lambda ram: JUMP if takeoff_label_v2(ram) else RUN)
                eps.append({"level": lv, "delay": d, "best_x": r["best_x"], "jump_share": r["jump_rate"]})
        f.write_text(json.dumps(eps))
    eps = json.loads(f.read_text())
    dxr = [e["best_x"] - rand_base(e["level"], e["delay"], e["jump_share"]) for e in eps]
    return {"episodes": len(eps), "mean_dx_rand": round(float(np.mean(dxr)), 1),
            "per_level": {lv: round(float(np.mean([x for x, e in zip(dxr, eps) if e["level"] == lv])), 1)
                          for lv in ("Level1-1", "Level2-1", "Level4-1", "Level7-1")},
            "mean_jump_share": round(float(np.mean([e["jump_share"] for e in eps])), 3)}


def curves(episodes, key="learner"):
    out = {}
    for e in episodes:
        k = f"{e['level']}|{e['delay']}"
        if k not in BASE:
            continue
        dx = e.get("dx", e["best_x"] - BASE[k])
        acts = e.get("actions")
        js = e.get("jump_share", (acts[3] + acts[4] + acts[5]) / max(1, sum(acts)) if acts else None)
        dxr = e["best_x"] - rand_base(e["level"], e["delay"], js) if (RJ and js is not None) else None
        out.setdefault(e[key], []).append((e["at_decision"], dx, js, e["decisions"], dxr))
    res = {}
    for name, rows in out.items():
        rows.sort()
        at = np.array([r[0] for r in rows])
        dx = np.array([r[1] for r in rows], float)
        n = np.array([r[3] for r in rows], float)
        roll = np.convolve(dx, np.ones(50) / 50, mode="valid") if len(dx) >= 50 else np.array([])
        hit = np.flatnonzero(roll >= 100)
        windows = {}
        for w0 in range(0, int(at.max()) + 1, WIN):
            m = (at > w0) & (at <= w0 + WIN)
            if m.any():
                js = [r[2] for r, mm in zip(rows, m) if mm and r[2] is not None]
                windows[f"{w0 // 1000}-{(w0 + WIN) // 1000}k"] = {
                    "mean_dx": round(float(dx[m].mean()), 1), "episodes": int(m.sum()),
                    "jump_share": round(float(np.mean(js)), 3) if js else None}
        dr = np.array([r[4] for r in rows if r[4] is not None], float)
        atr = np.array([r[0] for r in rows if r[4] is not None])
        rr = np.convolve(dr, np.ones(50) / 50, mode="valid") if len(dr) >= 50 else np.array([])
        hr = np.flatnonzero(rr >= 50)
        rand = {"mean_dx_rand": round(float(dr.mean()), 1) if len(dr) else None,
                "decisions_to_competence_rand": int(atr[hr[0] + 49]) if len(hr) else None,
                "windows_dx_rand": {f"{w0 // 1000}-{(w0 + WIN) // 1000}k": round(float(dr[(atr > w0) & (atr <= w0 + WIN)].mean()), 1)
                                    for w0 in range(0, int(at.max()) + 1, WIN) if ((atr > w0) & (atr <= w0 + WIN)).any()}}
        res[name] = {**rand, "episodes": len(rows), "last_decision": int(at.max()),
                     "decisions_to_competence": int(at[hit[0] + 49]) if len(hit) else None,
                     "auc_mean_dx": round(float((dx * n).sum() / n.sum()), 1), "windows": windows}
    return res


def main():
    src = {}
    ref = RUNS / "reference" / "train-20261007.json"
    if ref.exists():
        src["reference"] = json.loads(ref.read_text())["episodes"]
    jc = RUNS / "reference" / "train-jumpcost-20261008.json"
    if jc.exists():
        src["reference_jumpcost"] = json.loads(jc.read_text())["episodes"]
    for run in ("mario_gen", "mario_oracle", "mario_stageD", "mario_stageD_null0", "mario_stageD_null1",
                "mario_stageD_null2", "mario_rl3", "mario_rl3_nulllat0", "mario_rl3_nulllat1"):
        f = RUNS / f"{run}.json"
        if f.exists():
            eps = [e for e in json.loads(f.read_text())["episodes"] if e.get("level") in
                   ("Level1-1", "Level2-1", "Level4-1", "Level7-1") and e.get("delay") is not None]
            src[run] = eps
    res = {k: curves(v) for k, v in src.items()}
    for k, v in res.items():
        for name, r in sorted(v.items()):
            print(k, name, "| ep", r["episodes"], "| dx_rand", r["mean_dx_rand"], "to comp.",
                  r["decisions_to_competence_rand"], r["windows_dx_rand"],
                  "| (vs always-run: AUC", r["auc_mean_dx"], ")")
    if RJ:
        res["scripted_takeoff2_reference"] = scripted_reference()
        print("scripted take-off v2 reference", res["scripted_takeoff2_reference"])
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    res["metric"] = "dx = furthest x - always-run furthest x, same level and start; competence = rolling-50 mean dx >= 100"
    (RUNS / "learning_curves-20261007.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
