#!/usr/bin/env python
"""Exploratory interface check (2026-10-10, literature review): does the ON channel (current into
L1/L5) reach the rest of the network, given that L1 is glutamatergic (inhibitory here) and the
network rests at 0 Hz? Untrained connectome, CPU, 60 held-out frames: spikes outside the lamina and
motor spikes per sweep with the full input, ON-only and OFF-only.

    .venv/bin/python scripts/diagnostics/on_off_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))


def main():
    import torch
    from common_stream import stream

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import reset_network
    torch.set_num_threads(6)
    g, W, motor = graphsel.load("connectome")
    vis = RetinotopicVision(g)
    cur, _lab, _ = stream(g, ("Level5-1", "Level8-1"), 400, 20261075, every=3)
    cur = cur[:, np.random.default_rng(1).permutation(cur.shape[1])[:60]]
    lam = np.zeros(g.n_nodes, bool)
    lam[vis.on_idx] = lam[vis.off_idx] = True
    on_only, off_only = cur.copy(), cur.copy()
    on_only[vis.off_idx] = 0
    off_only[vis.on_idx] = 0
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], device="cpu")
    nt = np.asarray(g.node_type)
    l1 = np.isin(np.arange(g.n_nodes), vis.on_idx) & (nt == "L1")
    res = {}
    for name, C in (("full", cur), ("on_only", on_only), ("off_only", off_only)):
        non_lam, mot, l1s, inj = [], [], [], []
        for f in range(C.shape[1]):
            reset_network(b)
            x = torch.as_tensor(np.repeat(C[:, f:f + 1], 3, 1))
            tot = torch.zeros(b.n)
            for _ in range(K):
                tot += b.step(x).float().sum(1)
            tot = tot.numpy() / 3
            non_lam.append(tot[~lam].sum())
            mot.append(tot[motor].sum())
            l1s.append(tot[l1].sum())
            inj.append(float((C[:, f] > 0).sum()))
        res[name] = {"spikes_outside_lamina": round(float(np.mean(non_lam)), 1), "motor_spikes": round(float(np.mean(mot)), 1),
                     "L1_spikes": round(float(np.mean(l1s)), 1), "driven_cells": round(float(np.mean(inj)), 1)}
        print(name, res[name], flush=True)
    res["what"] = "untrained connectome, 60 held-out frames, mean per sweep (3 noise seeds)"
    (ROOT / "artifacts/runs/mission/on_off_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
