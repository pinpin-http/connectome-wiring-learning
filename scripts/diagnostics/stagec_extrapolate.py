#!/usr/bin/env python
"""Exploratory (no training): is the Stage C gap a matter of size?

Scales each take-off-taught learner's learnt motor-incoming change, m(k) = clip(m0 + k (m - m0), 0, 2 m0),
k = 0, 1, 2, 4, 8, and reports on the criterion frames (5-1, 8-1; DANGER vs CLEAR, shared noise) the share
of greedy jumps and the mean P(jump).

    .venv/bin/python scripts/diagnostics/stagec_extrapolate.py artifacts/runs/checkpoints/mario_stageC_40k.pt
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from criterion_e_check import JUMP, frames

from flymario import config
from flymario.connectome import datasets
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

KS = (0, 1, 2, 4, 8)


def main():
    import torch
    ck = Path(sys.argv[1])
    st = torch.load(ck, map_location="cpu", weights_only=False)
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    W = g.to_sparse()
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    F = frames(g)
    y = np.array([f[1] for f in F])
    cur = np.repeat(np.stack([f[2] for f in F], 1).astype(np.float32), 2, axis=1)
    res = {}
    probe = PlasticLIF(W, motor, params, seeds=[0], cache_key=("ctrl", 0))
    m0 = probe.m0.cpu()
    del probe
    for j, d in enumerate(st["learners"]):
        if not d["name"].startswith("tko_"):
            continue
        R = np.asarray(FixedRandomProjection(len(motor), 8, d["readout_seed"]).W)
        dm = st["brain"]["m"][j] - m0
        res[d["name"]] = {}
        for k in KS:
            m = torch.clamp(m0 + k * dm, min=0.0)
            m = torch.minimum(m, 2 * m0)
            out = []
            for s0 in range(0, cur.shape[1], 64):
                n = min(64, cur.shape[1] - s0)
                b = PlasticLIF(W, motor, params, seeds=[90000 + s0 + i for i in range(n)], cache_key=("ctrl", 0))
                b.set_weights(m[None, :].repeat(n, 1).to(b.dev))
                c, _ = sweep_once(b, params, torch.as_tensor(cur[:, s0:s0 + n], device=b.dev), K, 0.5, False)
                out.append(c.t().cpu().numpy())
                del b
                torch.cuda.empty_cache()
            z = np.concatenate(out, 0) @ R.T / 2.0
            greedy_jump = np.isin(z.argmax(1), JUMP).reshape(len(F), 2).mean(1)
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            pj = p[:, JUMP].sum(1).reshape(len(F), 2).mean(1)
            res[d["name"]][f"k{k}"] = {
                "greedy_jump_danger": round(float(greedy_jump[y == 1].mean()), 3),
                "greedy_jump_clear": round(float(greedy_jump[y == 0].mean()), 3),
                "p_jump_danger": round(float(pj[y == 1].mean()), 3),
                "p_jump_clear": round(float(pj[y == 0].mean()), 3)}
            print(d["name"], f"k={k}", res[d["name"]][f"k{k}"], flush=True)
    rep = {"what": "Stage C: scaling the learnt weight change (exploratory, no training)", "checkpoint": ck.name,
           "trained_decisions": int(st["decisions"]), "n_frames": len(F), "ks": KS, "results": res,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"stagec_extrapolate-{ck.stem}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
