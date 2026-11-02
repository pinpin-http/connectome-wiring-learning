#!/usr/bin/env python
"""RL_STUDY §26 validation of the in-strength nulls: per null, signed in-strength (sum of sign x weight over
incoming edges) of every neuron equal to the connectome's, degrees exact, sign and weight multisets equal,
side pairs and depth classes as null_validate.py (run separately). Writes mission/nulls/instrength_validate.json.

    .venv/bin/python scripts/diagnostics/instrength_validate.py 0 1 2 ...
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from flymario.connectome import datasets, nullcache  # noqa: E402


def ins(h):
    return np.bincount(h.dst, weights=h.weight * h.sign, minlength=h.n_nodes)


def outs(h):
    return np.bincount(h.src, weights=h.weight * h.sign, minlength=h.n_nodes)


g = datasets.load(datasets.DEFAULT)
i0, o0 = ins(g), outs(g)
res = {}
for s in map(int, sys.argv[1:]):
    n, st = nullcache.load_or_build("depth_stratified_lateral_instrength", g, s, datasets.DEFAULT, build=False)
    if n is None:
        res[s] = {"status": st}
        continue
    res[s] = {"in_strength_exact": bool(np.allclose(ins(n), i0)),
              "out_strength_max_abs_change": float(np.abs(outs(n) - o0).max()),
              "in_degree_exact": bool(np.array_equal(np.bincount(n.dst, minlength=g.n_nodes), np.bincount(g.dst, minlength=g.n_nodes))),
              "out_degree_exact": bool(np.array_equal(np.bincount(n.src, minlength=g.n_nodes), np.bincount(g.src, minlength=g.n_nodes))),
              "weight_multiset_equal": bool(np.array_equal(np.sort(n.weight), np.sort(g.weight))),
              "sign_multiset_equal": bool(np.array_equal(np.sort(n.sign), np.sort(g.sign)))}
    print(s, res[s], flush=True)
(ROOT / "artifacts/runs/mission/nulls/instrength_validate.json").write_text(json.dumps(res, indent=1))
