#!/usr/bin/env python
"""Exploratory: is the excitatory wiring convergent -- do co-active neurons share their targets?

Co-active sets come from the untrained connectome (16 training-level frames). For the same sets in
each graph, convergence = excitatory edges to the next depth stratum / distinct neurons they reach.

    .venv/bin/python scripts/diagnostics/convergence.py [--device cpu] connectome nulllat:0 null:0
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from null_rate_match import frames

from flymario import config
from flymario.connectome import datasets, graphsel, nullcache, strata
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import reset_network


def main():
    import torch
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    specs = [a for a in sys.argv[1:] if ":" in a or a == "connectome"] or ["connectome", "nulllat:0", "null:0"]
    g, W, motor = graphsel.load("connectome")
    st = strata.compute(g, datasets.by_types(g, "malecns", "lamina_types"), motor, "both")
    depth = np.asarray(st.from_input)
    cur = frames(g)[:, ::12][:, :16]
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    b = PlasticLIF(W, motor, params, seeds=[98000 + i for i in range(cur.shape[1])], cache_key=("conv", dev), device=dev)
    reset_network(b)
    c = torch.as_tensor(cur, device=b.dev)
    fired = np.zeros((W.shape[0], cur.shape[1]), bool)
    for _t in range(int(config.get("timing.ticks_per_decision"))):
        fired |= b.step(c).cpu().numpy().astype(bool)
    del b, W
    res = {}
    for spec in specs:
        if spec == "connectome":
            src, dst, sign = g.src, g.dst, g.sign
        else:
            kind, s = spec.split(":")
            kind = {"null": "depth_stratified", "nulllat": "depth_stratified_lateral"}[kind]
            nl, _ = nullcache.load_or_build(kind, g, int(s), datasets.DEFAULT, build=False)
            src, dst, sign = nl.src, nl.dst, nl.sign
        exc = sign > 0
        out = {}
        for d in (0, 1, 2, 3):
            m = exc & (depth[src] == d) & (depth[dst] == d + 1)
            s_, t_ = src[m], dst[m]
            ratios, edges, targets = [], [], []
            for f in range(fired.shape[1]):
                act = fired[:, f] & (depth == d)
                sel = act[s_]
                e, u = int(sel.sum()), len(np.unique(t_[sel]))
                edges.append(e)
                targets.append(u)
                ratios.append(e / max(1, u))
            out[f"depth {d}->{d + 1}"] = {"edges": round(float(np.mean(edges)), 1),
                                         "distinct_targets": round(float(np.mean(targets)), 1),
                                         "convergence": round(float(np.mean(ratios)), 3)}
        res[spec] = out
        print(spec, json.dumps(out), flush=True)
    rep = {"what": "excitatory convergence of the connectome's co-active sets, in each graph (exploratory)",
           "frames": int(cur.shape[1]), "device": dev, "results": res,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "convergence-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
