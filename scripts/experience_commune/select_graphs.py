#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 graph selection (mission §5), fixed before any run and independent of every score.

For each null family the 40 available seeds (0-39) are put in a seeded random order; plan P1 takes the first 20,
P2 and P3 the first 10 of the same order (nested). Family b = `nulllat` (depth_stratified_lateral: degrees, signs,
weight multisets, depth strata, sides and every neuron's out-strength kept); family c = `nullin`
(depth_stratified_lateral_instrength: the same with in-strength instead of out-strength). Development graphs,
never in the campaign: nulllat:100 (b) and nullin:100 (c).

    .venv/bin/python scripts/experience_commune/select_graphs.py
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = {"b": 20261020, "c": 20261021}
PREFIX = {"b": "nulllat", "c": "nullin"}


def main() -> int:
    out = {"utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "rule": "per family, numpy default_rng(seed).permutation(40) of the seeds 0-39; P1 = first 20, P2/P3 = first 10",
           "selection_seeds": SEEDS, "families": {}, "pilot_graphs": ["nulllat:100", "nullin:100"]}
    for f, s in SEEDS.items():
        order = np.random.default_rng(s).permutation(40).tolist()
        out["families"][f] = {"kind": PREFIX[f], "order": order,
                              "P1": [f"{PREFIX[f]}:{i}" for i in order[:20]],
                              "P2_P3": [f"{PREFIX[f]}:{i}" for i in order[:10]]}
    out["P1_graphs"] = ["connectome"] + out["families"]["b"]["P1"] + out["families"]["c"]["P1"]
    out["P2_P3_graphs"] = ["connectome"] + out["families"]["b"]["P2_P3"] + out["families"]["c"]["P2_P3"]
    p = ROOT / "artifacts/experience_commune_v1/manifests/graph_selection.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = json.loads(p.read_text())
        if old["P1_graphs"] != out["P1_graphs"]:
            print("refused: an existing selection differs", file=sys.stderr)
            return 2
        print("selection already recorded, identical")
        return 0
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out["families"][k]["P1"] for k in out["families"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
