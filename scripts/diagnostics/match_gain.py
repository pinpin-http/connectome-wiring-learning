#!/usr/bin/env python
"""RL_STUDY §25 (matched-operating-regime comparison, PROTOCOL §9.5): for each null, the global gain
at which its untrained motor activity on the common stream's 300 held-out frames equals the
connectome's at the frozen operating point (gain from configs/base.yaml). Bisection on log(gain);
activity measured exactly as `graph_activity.py` (three brains, noise seeds 8300 + i, 16-tick sweeps
from rest). Writes artifacts/runs/mission/matched_gains.json after every graph (resumable).

    .venv/bin/python scripts/diagnostics/match_gain.py nulllat:0 nulllat:1 ...
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

OUT = ROOT / "artifacts/runs/mission/matched_gains.json"
TOL = 0.01          # relative tolerance on mean motor spikes per decision


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
    g, Wc, motor = graphsel.load("connectome")
    te_cur, _lab, _ = stream(g, ("Level5-1", "Level8-1"), 1500, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])
    te = torch.as_tensor(te_cur[:, keep])
    base = float(config.get("dynamics.gain"))
    noise = float(config.get("dynamics.noise_std"))
    K = int(config.get("timing.ticks_per_decision"))

    def activity(W, gain, key):
        params = LIFParams(gain=gain, noise_std=noise)
        b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], cache_key=None)   # gain is applied per step; no cache leak
        tot = 0.0
        for s0 in range(te.shape[1]):
            c, _ = sweep_once(b, params, te[:, s0:s0 + 1].repeat(1, 3).to(b.dev), K, 0.5, False)
            tot += float(c.sum()) / 3
        del b
        torch.cuda.empty_cache()
        return tot / te.shape[1]

    if "connectome" not in res:
        res["connectome"] = {"gain": base, "activity": round(activity(Wc, base, ("act", "connectome")), 2)}
    del Wc
    target = res["connectome"]["activity"]
    print("target", target, flush=True)
    for spec in specs:
        if spec in res:
            continue
        _, W, _ = graphsel.load(spec)
        lo, hi = math.log(base * 0.3), math.log(base)
        a_hi = activity(W, base, ("act", spec, "base"))
        steps = [(base, a_hi)]
        a_lo = activity(W, math.exp(lo), ("act", spec, 0))
        steps.append((math.exp(lo), a_lo))
        if not (a_lo <= target <= a_hi):
            res[spec] = {"gain": None, "steps": steps, "note": "target outside [0.3, 1] x base"}
        else:
            for it in range(12):
                mid = (lo + hi) / 2
                a = activity(W, math.exp(mid), ("act", spec, it + 1))
                steps.append((math.exp(mid), a))
                if abs(a - target) / target <= TOL:
                    break
                lo, hi = (mid, hi) if a < target else (lo, mid)
            gm, am = min(steps, key=lambda s: abs(s[1] - target))
            res[spec] = {"gain": gm, "activity": round(am, 2), "base_activity": round(a_hi, 2),
                         "rel_error": round(abs(am - target) / target, 4), "steps": steps}
        print(spec, {k: v for k, v in res[spec].items() if k != "steps"}, flush=True)
        tmp = OUT.with_name(OUT.name + ".part")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, OUT)
        del W


if __name__ == "__main__":
    main()
