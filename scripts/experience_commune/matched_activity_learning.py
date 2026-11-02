#!/usr/bin/env python
"""EXPLORATORY complement to EXPERIENCE_COMMUNE_V1: does the connectome's advantage persist at matched null activity?

Mode N, blocks b0-b2, both tasks, the P2/P3 subset (10 family-b + 10 family-c nulls), each null at its
ACTIVITY_MATCH gain (calibrated on gratings, so matching is approximate). Connectome runs are reused from V1.

    PYTHONPATH=src .venv/bin/python scripts/experience_commune/matched_activity_learning.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"
OUT = ART / "complements/matched_activity"
AM = Path("/workspace/SMB_Flywire_Speedrun/artifacts/activity_match")


def main() -> int:
    from flymario.experience_commune import learn
    from flymario.experience_commune.learn import Learner
    sel = json.loads((ART / "manifests/graph_selection.json").read_text())
    gc = json.loads((AM / "calibration_v2/selection.json").read_text())["gammas"]
    gb = json.loads((AM / "calibration_v2b/selection.json").read_text())["gammas"]
    graphs = [g for pair in zip(sel["families"]["b"]["P2_P3"], sel["families"]["c"]["P2_P3"]) for g in pair]
    gammas = {g: (gb if g.startswith("nulllat") else gc)[g]["CN"] for g in graphs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps({"label": "EXPLORATORY", "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                               "graphs": graphs, "gammas": gammas, "blocks": ["b0", "b1", "b2"],
                                               "mode": "N"}, indent=1))
    ls = [Learner(t, "N", b) for b in ("b0", "b1", "b2") for t in ("where", "loom")]
    for g in graphs:
        d = OUT / g.replace(":", "")
        if (d / "run.json").exists() and json.loads((d / "run.json").read_text()).get("complete"):
            continue
        r = learn.run(g, ls, ART / "data", d, banks_final=("final",), gamma=gammas[g],
                      log=lambda m: print(datetime.now(UTC).isoformat(timespec="seconds"), m, flush=True))
        print(g, "gamma", gammas[g], "wall", r["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
