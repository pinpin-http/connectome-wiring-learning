#!/usr/bin/env python
"""Gate G1 — fix the neural clock against the policy clock.

At dt_ms = 3.33, 2 ticks per decision give 6.66 ms of neural time against 66.7 ms of game time
(frame_skip 4), which handicaps the connectome's multi-hop path to the readout over rewired shortcuts.
Measures how many ticks a lamina pulse needs to reach the motor readout and reports the smallest tick
budget; run it for every arm and set the budget by the slowest.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flymario.connectome import datasets, randomize
from flymario.connectome.schema import ConnectomeGraph
from flymario.dynamics.calibrate import probe_current, regions
from flymario.models.spiking_reservoir import LIFBrain, LIFParams
from flymario.paths import CONNECTOME, RUNS

AUDIBLE = 0.002        # motor fraction that counts as "the readout heard it"


def latency(g, dataset, gain, noise, seed=0, warm=40, window=24, amp=2.0):
    """Ticks from a lamina pulse to an audible motor response, over baseline."""
    brain = LIFBrain(g.to_sparse(), LIFParams(gain=gain, bias=0.0,
                                              noise_std=noise, seed=seed))
    cur = probe_current(g, dataset, amplitude=amp)
    reg = regions(g, dataset)
    mot = reg["motor"]

    # Settle without drive, then record the undriven baseline.
    for _ in range(warm):
        brain.step(None)
    base = np.mean([float(brain.step(None)[mot].mean()) for _ in range(20)])

    # Single pulse on the lamina, then silence. Watch the motor population.
    resp = []
    for k in range(window):
        s = brain.step(cur if k == 0 else None)
        resp.append(float(s[mot].mean()))

    over = [r - base for r in resp]
    first = next((i + 1 for i, v in enumerate(over) if v > AUDIBLE), None)
    peak = int(np.argmax(over)) + 1 if over else 0
    return {"baseline": float(base), "response": resp,
            "first_audible_tick": first, "peak_tick": peak,
            "peak_over_baseline": float(max(over)) if over else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default="flywire.npz")
    p.add_argument("--dataset", default="flywire")
    p.add_argument("--gain", type=float, default=14.0)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--arms", nargs="+",
                   default=["connectome", "degree_preserving"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=4)
    args = p.parse_args()

    base = ConnectomeGraph.load(CONNECTOME / args.graph)
    print(f"graph {base.n_nodes:,} nodes / {base.n_edges:,} pairs\n")

    results = {}
    for arm in args.arms:
        print(f"{arm} ...", flush=True)
        g = randomize.make(arm, base, seed=args.seed)
        r = latency(g, args.dataset, args.gain, args.noise, seed=args.seed)
        results[arm] = r
        fa = r["first_audible_tick"]
        print(f"  baseline motor       {r['baseline']*100:6.3f} %")
        print(f"  first audible tick   {fa if fa else 'never'}")
        print(f"  peak at tick         {r['peak_tick']}  "
              f"(+{r['peak_over_baseline']*100:.3f} pp)\n")

    audible = [r["first_audible_tick"] for r in results.values()
               if r["first_audible_tick"]]
    required = max(audible) if audible else None

    game_ms = args.frame_skip * (1000.0 / 60.0)
    print("=" * 62)
    if required is None:
        print("G1 FAIL — no arm produced an audible motor response.")
        print("Fix the operating point (G2) before setting the clock.")
    else:
        print(f"G1  minimum ticks_per_frame = {required}  (slowest arm)")
        print(f"    current setting         = 2")
        print(f"    game time per decision  = {game_ms:.1f} ms "
              f"(frame_skip={args.frame_skip})")
        for n in (required, 2 * required, 4 * required):
            print(f"    at {n:>3} ticks: dt_ms must be {game_ms / n:6.2f} "
                  f"for neural and game time to agree")
        if required > 2:
            print(f"\n    -> ticks_per_frame=2 is BELOW the requirement. "
                  f"The comparison is currently decided by the clock.")
    print("=" * 62)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "gate_g1.json"
    out.write_text(json.dumps({"graph": args.graph, "gain": args.gain,
                               "frame_skip": args.frame_skip,
                               "required_ticks": required,
                               "arms": results}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
