#!/usr/bin/env python
"""Exploratory: does the measured inhibitory wiring keep the connectome's response sparse?

Hybrid graphs built per edge class (degrees per sign exact): E_null+I_con (excitatory and sign-0
edges from lateral null 0, inhibitory edges from the connectome) and E_con+I_null (the reverse).
Same frames and measure as activity_by_depth.py (untrained, spikes per decision per depth stratum).

    .venv/bin/python scripts/diagnostics/sign_hybrids.py [--device cpu]
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
from flymario.connectome import datasets, nullcache, strata
from flymario.connectome.schema import ConnectomeGraph
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import reset_network


def hybrid(exc, inh, name):
    me, mi = exc.sign >= 0, inh.sign < 0
    return ConnectomeGraph(node_ids=exc.node_ids, src=np.concatenate([exc.src[me], inh.src[mi]]),
                           dst=np.concatenate([exc.dst[me], inh.dst[mi]]),
                           weight=np.concatenate([exc.weight[me], inh.weight[mi]]),
                           sign=np.concatenate([exc.sign[me], inh.sign[mi]]),
                           node_type=exc.node_type, node_class=exc.node_class, node_side=exc.node_side,
                           node_pos=exc.node_pos, source=name, meta={})


def main():
    import torch
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    lat, _ = nullcache.load_or_build("depth_stratified_lateral", g, 0, datasets.DEFAULT, build=False)
    st = strata.compute(g, datasets.by_types(g, "malecns", "lamina_types"), motor, "both")
    depth = np.asarray(st.from_input)
    groups = {f"depth {d}": depth == d for d in range(5)} | {"motor": np.isin(np.arange(len(depth)), motor)}
    cur = frames(g)[:, :64]
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    graphs = {"E_null+I_con": hybrid(lat, g, "E_null+I_con"), "E_con+I_null": hybrid(g, lat, "E_con+I_null")}
    res = {}
    for name, h in graphs.items():
        W = h.to_sparse()
        b = PlasticLIF(W, motor, params, seeds=[97000 + i for i in range(cur.shape[1])], cache_key=("hyb", name, dev),
                       device=dev)
        reset_network(b)
        c = torch.as_tensor(cur, device=b.dev)
        tot = dict.fromkeys(groups, 0.0)
        for _t in range(K):
            spk = b.step(c).float().cpu().numpy()
            for k, m in groups.items():
                tot[k] += float(spk[m].sum(0).mean())
        res[name] = {k: round(v, 1) for k, v in tot.items()}
        print(name, res[name], flush=True)
        del b, W
    rep = {"what": "sign hybrids of the connectome and lateral null 0: spikes per decision by depth (exploratory)",
           "frames": int(cur.shape[1]), "device": dev, "results": res,
           "reference": "activity_by_depth-20261006.json (connectome, nulllat:0, null:0; same frames and seeds)",
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "sign_hybrids-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
