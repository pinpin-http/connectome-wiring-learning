#!/usr/bin/env python
"""Null validation (PROTOCOL §8.3), independent of any task: what each cached null
realisation preserves and how much of the wiring it changed, against the connectome.

Per null: `randomize.mixing_report` (degrees, edge count, sign and weight multisets,
sampled edge overlap), the side-pair edge counts (C / L / M / R of source and
target: kept exactly by `depth_stratified_lateral`, not by `depth_stratified`), the
ipsilateral edge share, and the depth-stratum edge-class histogram (kept by both).

    .venv/bin/python scripts/diagnostics/null_validate.py null:0 nulllat:0 nulllat:1 nulllat:2
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.connectome import datasets, nullcache, randomize, strata
from flymario.paths import RUNS


def side_pairs(g):
    s = np.asarray(g.node_side)
    u = np.unique(s)
    return {f"{a}->{b}": int(((s[g.src] == a) & (s[g.dst] == b)).sum()) for a in u for b in u}


def main():
    specs = sys.argv[1:] or ["null:0", "nulllat:0", "nulllat:1", "nulllat:2"]
    g = datasets.load(datasets.DEFAULT)
    sensory = datasets.by_types(g, "malecns", "lamina_types")
    st = strata.compute(g, sensory, datasets.motor_index(g, "malecns"), "both")
    cls0 = np.bincount(st.edge_class(g.src, g.dst, g.sign))
    sp0 = side_pairs(g)
    ipsi = lambda h: float((np.asarray(h.node_side)[h.src] == np.asarray(h.node_side)[h.dst]).mean())
    res = {"connectome": {"ipsilateral_share": round(ipsi(g), 4), "side_pairs": sp0}}
    for spec in specs:
        kind, seed = spec.split(":")
        kind = {"null": "depth_stratified", "nulllat": "depth_stratified_lateral",
                "nullin": "depth_stratified_lateral_instrength"}[kind]
        null, status = nullcache.load_or_build(kind, g, int(seed), datasets.DEFAULT, build=False)
        if null is None:
            res[spec] = {"status": status}
            continue
        cls = np.bincount(st.edge_class(null.src, null.dst, null.sign), minlength=len(cls0))
        sp = side_pairs(null)
        res[spec] = {**randomize.mixing_report(g, null),
                     "strata_classes_equal": bool(np.array_equal(cls[:len(cls0)], cls0) and cls[len(cls0):].sum() == 0),
                     "side_pairs_equal": sp == sp0,
                     "ipsilateral_share": round(ipsi(null), 4),
                     "n_strata_classes_meta": null.meta.get("n_strata")}
        print(spec, {k: v for k, v in res[spec].items() if k != "edge_overlap_note"}, flush=True)
        del null
    rep = {"what": "null validation against the connectome (PROTOCOL §8.3), no task involved",
           "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    out = os.environ.get("NULL_VALIDATE_OUT")      # 2026-10-10: §24 writes elsewhere, never over the 10-06 file
    (Path(out) if out else RUNS / "null_validate-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
