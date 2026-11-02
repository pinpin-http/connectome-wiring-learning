#!/usr/bin/env python
"""Frozen network: orientation and brightness decoding against readout activity, every condition of ACTIVITY_MATCH
V1 (C0, CL, CM, CH), V2 (C0, CN; family E) and V2B (C0, CN; family S). Exploratory, descriptive; no simulation, it
only reads the saved analyses.

    python scripts/diagnostics/synthesis_activity.py      # writes artifacts/complements/synthesis_activity.json

Input of fig_activite.pdf (scripts/figures/make_figures_campagne.py). One row per (run, graph, condition):
[run, graph, condition, readout rate (Hz), orientation accuracy, brightness accuracy].
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/complements/synthesis_activity.json"


def load(run):
    return json.loads((ROOT / f"artifacts/activity_match/{run}/metrics/analysis.json").read_text())


def main():
    runs = {"V1": load("main_am"), "V2": load("main_am2"), "V2B": load("main_am2b")}
    rows = []
    for name, a in runs.items():
        for g in a["graphs"]:
            for c in a["conditions"]:
                rows.append((name, g, c, a["activity_test"][g][c]["rate_hz"], a["M"][g][c]["orient"],
                             a["M"][g][c]["lum"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"{len(rows)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
