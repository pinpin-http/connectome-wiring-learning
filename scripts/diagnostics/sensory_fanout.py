#!/usr/bin/env python
"""Why does the null's motor layer fire ~2x the connectome's, nearly independently of gain?

On the simulated operator W (|weights|), from the eye's ON/OFF input neurons:
    reached_1hop   neurons receiving at least one edge from an input neuron
    motor_1hop     summed weight onto the 2,227 motor neurons straight from input
    motor_2hop     summed weight onto motor neurons from the 1-hop-reached set
    motor_in       total in-strength of the motor neurons (should be preserved)

    .venv/bin/python scripts/diagnostics/sensory_fanout.py [connectome null:0 ...]
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.connectome import graphsel
from flymario.env.vision import RetinotopicVision
from flymario.paths import RUNS


def main():
    specs = sys.argv[1:] or ["connectome", "null:0", "null:1", "null:2"]
    res = {}
    for spec in specs:
        g, W, motor = graphsel.load(spec)
        A = abs(W.tocsr())
        vis = RetinotopicVision(g)
        x = np.zeros(A.shape[0])
        x[np.unique(np.concatenate([vis.on_idx, vis.off_idx]))] = 1.0
        h1 = A @ x
        r1 = (h1 > 0).astype(float)
        h2 = A @ r1
        res[spec] = {"input_neurons": int(x.sum()), "reached_1hop": int(r1.sum()),
                     "motor_1hop_weight": round(float(h1[motor].sum()), 2),
                     "motor_2hop_weight": round(float(h2[motor].sum()), 1),
                     "motor_in_strength": round(float(A[motor].sum()), 1),
                     "motor_reached_2hop": int((h2[motor] > 0).sum())}
        print(spec, res[spec], flush=True)
        del W, A
    rep = {"what": "feed-forward reach of the eye's input neurons onto the motor layer, per graph (exploratory)",
           "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    sfx = "" if not sys.argv[1:] else "-" + specs[0].split(":")[0]
    (RUNS / f"sensory_fanout{sfx}-20261006.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
