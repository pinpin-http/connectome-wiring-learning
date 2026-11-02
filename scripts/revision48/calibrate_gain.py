#!/usr/bin/env python
"""Common gain of the log-weight operator (reviewer R2), calibrated on development graphs only.

    calibrate_gain.py [--graphs nulllat:100,nullin:100] [--lo 2 --hi 200 --tol 0.02]
        -> artifacts/revision48/calibration/lognorm_gain.json

Target: mean untrained readout activity of the development graphs with raw weights at gain 0.18452
(configs/revision48/campaign.yaml `lognorm.base_gain`). The gain giving the same activity with log
weights is found by bisection on log(gain); no analysed graph or learning score is read.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/revision48"


def activity(spec: str, weights: str, gain: float, X_by_task: dict, seed: int, cols: int = 64, slow: bool = False):
    """Mean readout spikes per image and neuron, untrained magnitudes, keyed evaluation noise."""
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.experience_commune import learn
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import reset_network
    _g, W, motor = graphsel.load(spec)
    if weights == "lognorm":
        W = learn.lognorm_operator(W)
    params = LIFParams(gain=float(gain), noise_std=float(config.get("dynamics.noise_std")))
    ticks = int(config.get("timing.ticks_per_decision"))
    cls = PlasticLIF if slow else learn.fast_plastic_class()
    b = cls(W, motor, params, seeds=[1] * cols, normalize=False)
    idx = None
    tot, n_img = 0.0, 0
    for task, (X, idx) in X_by_task.items():
        Xt = torch.as_tensor(X, device=b.dev)
        for c0 in range(0, len(X), cols):
            sub = Xt[c0:c0 + cols]
            k = sub.shape[0]
            gen = torch.Generator(device=b.dev)
            gen.manual_seed(learn.eval_key(seed, 900_000 + c0 // cols + (0 if task == "where" else 50_000)))
            noise = torch.randn((ticks, b.n, cols), generator=gen, device=b.dev) * params.noise_std
            cur = torch.zeros((b.n, cols), device=b.dev)
            cur[torch.as_tensor(idx, device=b.dev)] = torch.cat([sub, sub[:1].expand(cols - k, -1)]).t() if k < cols \
                else sub.t()
            reset_network(b)
            cnt = torch.zeros((b.M, cols), device=b.dev)
            for t in range(ticks):
                spk = b.step(cur, observe=False, noise=noise[t])
                cnt.add_(spk.index_select(0, b.motor).float())
            tot += float(cnt[:, :k].sum())
            n_img += k
    del b
    return tot / (n_img * len(motor))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default="nulllat:100,nullin:100")
    ap.add_argument("--lo", type=float, default=2.0)
    ap.add_argument("--hi", type=float, default=200.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--slow", action="store_true")
    a = ap.parse_args()
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    graphs = a.graphs.split(",")
    if any(not g.endswith(("100", "101")) for g in graphs):
        raise SystemExit("calibration uses development graphs only")
    dev = c["blocks"]["dev"][0]
    X_by_task, seed = {}, None
    for t in c["tasks"]:
        z = np.load(ART / "data" / dev / f"{t}.npz")
        X_by_task[t] = (z["progress_x"], z["idx"])
        seed = int(z["eval_noise_seed"])

    def mean_act(weights, gain):
        return float(np.mean([activity(g, weights, gain, X_by_task, seed, slow=a.slow) for g in graphs]))
    from flymario import config
    g_raw = float(config.get("dynamics.gain"))
    target = mean_act("raw", g_raw)
    rec = {"graphs": graphs, "block": dev, "raw_gain": g_raw, "target_activity": target, "iterations": []}
    rec["activity_at_historical_lognorm_gain_20"] = mean_act("lognorm", 20.0)
    lo, hi = a.lo, a.hi
    f_lo, f_hi = mean_act("lognorm", lo) - target, mean_act("lognorm", hi) - target
    rec["bracket"] = {"lo": [lo, f_lo], "hi": [hi, f_hi]}
    if f_lo * f_hi > 0:
        rec["result"] = None
        rec["note"] = "target not bracketed in [lo, hi]: nothing to freeze; report and widen only with a dated reason"
    else:
        for _ in range(a.max_iter):
            mid = math.exp((math.log(lo) + math.log(hi)) / 2)
            f = mean_act("lognorm", mid) - target
            rec["iterations"].append([mid, f + target])
            if abs(f) <= a.tol * target:
                break
            if (f < 0) == (f_lo < 0):
                lo, f_lo = mid, f
            else:
                hi = mid
        rec["result"] = mid
        rec["relative_error"] = abs(f) / target
    p = ART / "calibration/lognorm_gain.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("target_activity", "activity_at_historical_lognorm_gain_20", "result")}))
    return 0 if rec["result"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
