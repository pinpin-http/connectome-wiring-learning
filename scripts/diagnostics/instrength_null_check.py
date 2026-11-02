#!/usr/bin/env python
"""Exploratory (2026-10-06): is the connectome's sparser response an in-strength
property? Builds lateral null 0 with `weights_by="dst"` (each target gets its own
incoming weights back: in-strength kept, out-strength not; seed 0, in memory only)
and measures spikes per decision by depth stratum exactly as activity_by_depth.py
(same 64 frames, same noise seeds, CPU). If it fires like the connectome, the
sparseness follows from the in-strength sequence, not from who connects to whom.

    .venv/bin/python scripts/diagnostics/instrength_null_check.py [--seed 0]
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
from flymario.connectome import datasets, randomize, strata
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import reset_network


def main():
    import torch
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    null = randomize.depth_stratified(g, seed=seed, lateral=True, weights_by="dst")
    ins = lambda h: np.bincount(h.dst, weights=h.weight, minlength=h.n_nodes)
    outs = lambda h: np.bincount(h.src, weights=h.weight, minlength=h.n_nodes)
    check = {"in_strength_equal": bool(np.allclose(ins(g), ins(null))),
             "out_strength_equal": bool(np.allclose(outs(g), outs(null))),
             "same_side_share": round(float((np.asarray(null.node_side)[null.src]
                                              == np.asarray(null.node_side)[null.dst]).mean()), 4)}
    print(check, flush=True)
    st = strata.compute(g, datasets.by_types(g, "malecns", "lamina_types"), motor, "both")
    depth = np.asarray(st.from_input)
    groups = {f"depth {d}": depth == d for d in range(5)} | {"motor": np.isin(np.arange(len(depth)), motor)}
    cur = frames(g)[:, :64]
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    b = PlasticLIF(null.to_sparse(), motor, params, seeds=[97000 + i for i in range(cur.shape[1])],
                   cache_key=("instr", 0), device="cpu")
    reset_network(b)
    c = torch.as_tensor(cur, device=b.dev)
    tot = dict.fromkeys(groups, 0.0)
    for _t in range(int(config.get("timing.ticks_per_decision"))):
        spk = b.step(c).float().cpu().numpy()
        for k, m in groups.items():
            tot[k] += float(spk[m].sum(0).mean())
    res = {k: round(v, 1) for k, v in tot.items()}
    print(f"nulllat{seed} weights_by=dst", res, flush=True)
    rep = {"what": "lateral null 0 with in-strength kept (weights_by=dst): spikes per decision by depth (exploratory)",
           "check": check, "results": res, "reference": "activity_by_depth-20261006.json",
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rep["seed"] = seed
    (RUNS / f"instrength_null_check{'' if seed == 0 else f'-s{seed}'}-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
