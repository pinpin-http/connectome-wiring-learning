#!/usr/bin/env python
"""RL_STUDY §25.2 (exploratory): untrained motor spikes per decision vs global gain, on the common stream's
300 held-out frames, for a few graphs. Writes artifacts/runs/mission/regime_map.json.

    .venv/bin/python scripts/diagnostics/regime_map.py connectome nulllat:0 nulllat:1 nulllat:2
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
MULT = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3)


def main():
    import torch
    from common_stream import stream

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(2)
    g, _W, motor = graphsel.load("connectome")
    del _W
    te_cur, _lab, _ = stream(g, ("Level5-1", "Level8-1"), 1500, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])
    te = torch.as_tensor(te_cur[:, keep])
    base, noise = float(config.get("dynamics.gain")), float(config.get("dynamics.noise_std"))
    K = int(config.get("timing.ticks_per_decision"))
    out = {}
    for spec in sys.argv[1:]:
        _, W, _ = graphsel.load(spec)
        out[spec] = {}
        for m in MULT:
            params = LIFParams(gain=base * m, noise_std=noise)
            b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], cache_key=None)
            mot = tot = 0.0
            for s0 in range(te.shape[1]):
                b.reset() if False else None
                c, _ = sweep_once(b, params, te[:, s0:s0 + 1].repeat(1, 3).to(b.dev), K, 0.5, False)
                mot += float(c.sum()) / 3
            out[spec][str(m)] = round(mot / te.shape[1], 1)
            del b
            torch.cuda.empty_cache()
            print(spec, m, out[spec][str(m)], flush=True)
        (ROOT / "artifacts/runs/mission/regime_map.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
