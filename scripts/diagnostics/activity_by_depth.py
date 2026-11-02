#!/usr/bin/env python
"""Exploratory (2026-10-06): where does the connectome's sparser motor response arise?

Every null so far (depth-stratified, and laterality-preserving with the eye's
two-hop drive matched exactly) fires 1.7-2x the connectome's motor spikes per
decision. Here: untrained brains, the same 64 training-level frames
(null_rate_match.frames), one sweep from rest (16 ticks, frozen gain); spikes
counted per sensorimotor depth stratum (distance from the lamina, measured on
the connectome; the strata every null keeps) and per tick.

    .venv/bin/python scripts/diagnostics/activity_by_depth.py [--device cpu] connectome nulllat:0 null:0
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
from flymario.connectome import datasets, graphsel, strata
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import reset_network


def main():
    import torch
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    specs = [a for a in sys.argv[1:] if ":" in a or a == "connectome"] or ["connectome", "nulllat:0", "null:0"]
    g, _W, motor = graphsel.load("connectome")
    del _W
    st = strata.compute(g, datasets.by_types(g, "malecns", "lamina_types"), motor, "both")
    depth = np.asarray(st.from_input)
    groups = {f"depth {d}": depth == d for d in sorted(set(depth.tolist())) if d >= 0}
    groups["unreached"] = depth < 0
    groups["motor"] = np.zeros(len(depth), bool)
    groups["motor"][motor] = True
    cur = frames(g)[:, :64]
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    res = {}
    for spec in specs:
        _, W, _ = graphsel.load(spec)
        b = PlasticLIF(W, motor, params, seeds=[97000 + i for i in range(cur.shape[1])], cache_key=("depth", spec, dev),
                       device=dev)
        reset_network(b)
        c = torch.as_tensor(cur, device=b.dev)
        per_tick = []
        for _t in range(K):
            spk = b.step(c).float().cpu().numpy()          # (n, B)
            per_tick.append({k: float(spk[m].sum(0).mean()) for k, m in groups.items()})
        res[spec] = {"per_tick": per_tick,
                     "total": {k: round(sum(t[k] for t in per_tick), 1) for k in groups},
                     "neurons": {k: int(m.sum()) for k, m in groups.items()}}
        print(spec, res[spec]["total"], flush=True)
        del b, W
    rep = {"what": "spikes per decision by depth stratum, untrained, same frames (exploratory)",
           "frames": int(cur.shape[1]), "device": dev, "results": res,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "activity_by_depth-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
