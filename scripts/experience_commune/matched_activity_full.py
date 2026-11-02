#!/usr/bin/env python
"""EXPLORATORY X3 (decided 2026-10-21 04:40Z, after X2's result): X2 with the campaign's full learner set. The 20
P2/P3 nulls at ACTIVITY_MATCH's matched gains, all 24 learners of the campaign (2 tasks x modes N and U x blocks
b0-b5), final bank only. With the campaign's batch composition, the only difference from the campaign's own null runs
is the gain. Stops launching new graphs after 10:15Z (the analysis then uses the complete graphs, b/c interleaved).

    PYTHONPATH=src .venv/bin/python scripts/experience_commune/matched_activity_full.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"
OUT = ART / "complements/matched_activity_full"
STOP_LAUNCH = datetime(2026, 10, 21, 10, 15, tzinfo=UTC)


def main() -> int:
    from flymario.experience_commune import learn
    from flymario.experience_commune.learn import Learner
    plan = json.loads((ART / "complements/matched_activity/plan.json").read_text())
    rc = json.loads((ART / "runs/connectome/run.json").read_text())
    ls = [Learner(l["task"], l["mode"], l["block"], l.get("kind", "main")) for l in rc["learners"]]   # campaign order
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps({**plan, "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                               "learners": [l.key for l in ls], "stop_launch_utc": STOP_LAUNCH.isoformat(),
                                               "label": "EXPLORATORY X3"}, indent=1))
    for g in plan["graphs"]:
        d = OUT / g.replace(":", "")
        if (d / "run.json").exists() and json.loads((d / "run.json").read_text()).get("complete"):
            continue
        if datetime.now(UTC) >= STOP_LAUNCH:
            print("stop: launch deadline reached before", g, flush=True)
            break
        r = learn.run(g, ls, ART / "data", d, banks_final=("final",), gamma=plan["gammas"][g],
                      log=lambda m: print(datetime.now(UTC).isoformat(timespec="seconds"), m, flush=True))
        print(g, "gamma", plan["gammas"][g], "wall", r["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
