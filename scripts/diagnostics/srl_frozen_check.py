#!/usr/bin/env python
"""Reference check of the RL_STUDY §9 reading rule (§9.1): the Stage D policy, frozen,
in the §9 harness (`mario_srl_frozen`, version 17).

Reports per readout: the frozen learners' whole-run mean Δx_rand and 2,500-decision
windows (validated metric, `learning_curves.py`), the declared pass test (≥ +50 in
≥ 2 of 3), the determinism check (twin rows = learner rows: same weights, noise and
draws), and the paired `mario_srl` learners and twins over the same first decisions.

    .venv/bin/python scripts/diagnostics/srl_frozen_check.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

import learning_curves as lc

from flymario.paths import RUNS

lc.WIN = 2500
STAGE_D_REF = {"srl_ro1": 82, "srl_ro2": 83, "srl_ro3": 126}


def episodes(run, upto=None):
    d = json.loads((RUNS / f"{run}.json").read_text())
    eps = [e for e in d["episodes"] if f"{e.get('level')}|{e.get('delay')}" in lc.BASE
           and (upto is None or e["at_decision"] <= upto)]
    return d["decisions"], eps


def main():
    n, fz = episodes("mario_srl_frozen")
    cf = lc.curves(fz)
    _, sr = episodes("mario_srl", upto=n)
    cs = lc.curves(sr)
    out = {"decisions": n, "readouts": {}}
    for k in sorted(STAGE_D_REF):
        tw = k.replace("srl_", "srlshuf_")
        a = [e["best_x"] for e in fz if e["learner"] == k]
        b = [e["best_x"] for e in fz if e["learner"] == tw]
        same = a == b
        out["readouts"][k] = {
            "frozen_whole_run": cf[k]["mean_dx_rand"], "frozen_windows": cf[k]["windows_dx_rand"],
            "frozen_twin_rows_identical": same, "n_episodes": len(a),
            "stage_d_reference": STAGE_D_REF[k],
            "srl_learner_same_decisions": cs[k]["mean_dx_rand"], "srl_twin_same_decisions": cs[tw]["mean_dx_rand"]}
        print(k, out["readouts"][k], flush=True)
    passes = sum(r["frozen_whole_run"] is not None and r["frozen_whole_run"] >= 50 for r in out["readouts"].values())
    out["declared_test"] = {"passing_readouts": passes, "pass": passes >= 2}
    out["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    print("declared test:", out["declared_test"])
    (RUNS / "srl_frozen_check-20261008.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
