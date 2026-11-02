#!/usr/bin/env python
"""Rate matching of the depth-stratified nulls (docs/PROTOCOL.md §9.5; docs/PERCEPTION_CREDIT_STUDY.md §19).

Each null s gets its own global gain g_s so that its untrained motor layer fires as many spikes per
decision as the connectome at the frozen gain, on scripted-play frames of training levels 1-1 and 4-1.
Coarse log grid (gain / 2^k, k = 0-4), then bisection on log gain, 2% tolerance.
On the CPU the noise samples differ from the GPU generator's (a mean-rate calibration, not a replay).

    .venv/bin/python scripts/diagnostics/null_rate_match.py [--device cpu] [--nulls 0 1 2]
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, JUMPA = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")


def frames(g, levels=("Level1-1", "Level4-1"), steps=1200, every=12, seed=20261006):
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    out = []
    for lvl in levels:
        env = MarioEnv(EnvConfig(state=lvl))
        prev, _ = env.reset()
        for n in range(steps):
            obs, _r, term, trunc, _i = env.step(JUMPA if rng.random() < 0.25 else RUN)
            if n % every == 0:
                vis.reset()
                vis.encode(prev)
                out.append(vis.encode(obs))
            prev = obs
            if term or trunc:
                prev, _ = env.reset()
        env.close()
    return np.stack(out, 1).astype(np.float32)


def motor_rate(W, motor, cur, gain, spec, device="cuda"):
    import torch
    params = LIFParams(gain=float(gain), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    tot = []
    for s0 in range(0, cur.shape[1], 64):
        n = min(64, cur.shape[1] - s0)
        b = PlasticLIF(W, motor, params, seeds=[95000 + s0 + i for i in range(n)], cache_key=("ratematch", spec, device), device=device)
        c, _ = sweep_once(b, params, torch.as_tensor(cur[:, s0:s0 + n], device=b.dev), K, 0.5, False)
        tot.append(c.sum(0).cpu().numpy())
        del b
        torch.cuda.empty_cache()
    return float(np.concatenate(tot).mean())


def main():
    import torch

    from flymario.connectome import graphsel
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    nulls = [int(x) for x in sys.argv[sys.argv.index("--nulls") + 1:] if x.isdigit()] if "--nulls" in sys.argv else [0, 1, 2]
    g0 = float(config.get("dynamics.gain"))
    g, W, motor = graphsel.load("connectome")
    cur = frames(g)
    print("calibration frames:", cur.shape[1], flush=True)
    target = motor_rate(W, motor, cur, g0, "connectome", dev)
    print(f"connectome gain {g0}: {target:.1f} motor spikes per decision", flush=True)
    # coarse log grid for every graph (the connectome's own curve included), then
    # bisection inside the bracketing interval of each null
    grid = [g0 / 2 ** k for k in range(5)]
    curve = {"connectome": [[round(x, 5), round(motor_rate(W, motor, cur, x, "connectome", dev) if x != g0 else target, 1)]
                            for x in grid]}
    print("connectome curve", curve["connectome"], flush=True)
    del W
    torch.cuda.empty_cache()
    res = {}
    for s in nulls:
        spec = f"null:{s}"
        _, Wn, _ = graphsel.load(spec)
        trace = [(x, motor_rate(Wn, motor, cur, x, spec, dev)) for x in grid]
        curve[spec] = [[round(a, 5), round(b, 1)] for a, b in trace]
        print(spec, "curve", curve[spec], flush=True)
        above = [x for x, r in trace if r > target]
        below = [x for x, r in trace if r <= target]
        if above and below and max(below) < min(above):
            lo, hi = np.log(max(below)), np.log(min(above))
            for _ in range(7):
                mid = (lo + hi) / 2
                r = motor_rate(Wn, motor, cur, float(np.exp(mid)), spec, dev)
                trace.append((float(np.exp(mid)), r))
                print(spec, f"gain {np.exp(mid):.5f}: {r:.1f}", flush=True)
                if abs(r / target - 1) < 0.02:
                    break
                lo, hi = (mid, hi) if r < target else (lo, mid)
        best = min(trace, key=lambda t: abs(t[1] / target - 1))
        res[spec] = {"gain": round(best[0], 5), "gain_ratio": round(best[0] / g0, 4), "motor_spikes": round(best[1], 1),
                     "ratio_at_shared_gain": round(trace[0][1] / target, 3),
                     "matched": bool(abs(best[1] / target - 1) < 0.02),
                     "trace": [[round(a, 5), round(b, 1)] for a, b in trace]}
        print(spec, res[spec], flush=True)
        del Wn
        torch.cuda.empty_cache()
    rep = {"what": "global gain per null matching the connectome's untrained motor spikes per decision "
                   "(PROTOCOL §9.5, matched-operating-regime arm; training-level frames only)",
           "frames": int(cur.shape[1]), "device": dev, "connectome": {"gain": g0, "motor_spikes": round(target, 1)},
           "curves": curve, "nulls": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "null_rate_match-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
