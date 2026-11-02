#!/usr/bin/env python
"""RL_STUDY §24 secondary (a): untrained motor activity per graph on the common stream's 300 held-out
frames (5-1 / 8-1, seeds as common_stream.py): mean motor spikes per decision (16-tick sweep from rest),
three brains with the cs-expert noise seeds 8300 + i, plastic weights at their initial counts.
Writes artifacts/runs/mission/graph_activity.json after every graph (resumable: done graphs skipped).

    .venv/bin/python scripts/diagnostics/graph_activity.py connectome nulllat:0 nulllat:1 ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

OUT = ROOT / "artifacts/runs/mission/graph_activity.json"


def main():
    import torch
    from common_stream import stream

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    specs = sys.argv[1:]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    g, _W, motor = graphsel.load("connectome")
    del _W
    te_cur, _lab, _ = stream(g, ("Level5-1", "Level8-1"), 1500, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])
    te_cur = te_cur[:, keep]
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    for spec in specs:
        if spec in res:
            continue
        _, W, _ = graphsel.load(spec)
        b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], cache_key=None)   # no operator cache: 41 graphs would pin ~8 GB of GPU memory
        tot = []
        for s0 in range(te_cur.shape[1]):
            c, _ = sweep_once(b, params, torch.as_tensor(np.repeat(te_cur[:, s0:s0 + 1], 3, 1), device=b.dev),
                              K, 0.5, False)
            tot.append(c.sum(0).cpu().numpy())
        tot = np.array(tot)
        res[spec] = {"motor_spikes_per_decision": round(float(tot.mean()), 2),
                     "per_brain": [round(float(x), 2) for x in tot.mean(0)]}
        print(spec, res[spec], flush=True)
        tmp = OUT.with_name(OUT.name + ".part")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, OUT)
        del b, W
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
