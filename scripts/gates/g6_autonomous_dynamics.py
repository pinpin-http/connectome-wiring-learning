#!/usr/bin/env python
"""Gate G6: does the recurrent network contribute state, or is it a driven filter?

  A  persistence   drive for `warm` ticks, cut it; PASS if activity stays above the noise
                   floor `PERSIST_TICKS` ticks after the cut.
  B  history       identical input, two histories; PASS if the motor trace differs by more
                   than the seed-only baseline.
  C  drive lag     cross-correlate motor activity with drive; a pure filter peaks at lag 0-1.

Read-only: loads the cached graph and writes a JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np

from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.dynamics.calibrate import probe_current, regions
from flymario.models.spiking_reservoir import LIFBrain, LIFParams
from flymario.paths import CONNECTOME, RUNS

PERSIST_TICKS = 20          # activity must survive this long after the cut
NOISE_FLOOR = 1e-4          # fraction of neurons spiking; below this is silence


def _brain(g: ConnectomeGraph, gain: float, noise: float, seed: int) -> LIFBrain:
    return LIFBrain(g.to_sparse(), LIFParams(gain=gain, bias=0.0,
                                             noise_std=noise, seed=seed))


def probe_a_persistence(g, dataset, gain, noise, warm=60, after=60, seed=0):
    """Drive, cut, and measure what survives."""
    brain = _brain(g, gain, noise, seed)
    cur = probe_current(g, dataset)
    reg = regions(g, dataset)
    for _ in range(warm):
        brain.step(cur)
    driven = float(brain.spikes.mean())

    trace = []
    motor = []
    for _ in range(after):
        s = brain.step(None)                      # drive removed
        trace.append(float(s.mean()))
        motor.append(float(s[reg["motor"]].mean()) if reg["motor"].size else 0.0)

    at = lambda k: trace[k - 1] if len(trace) >= k else 0.0
    return {
        "driven_rate": driven,
        "after_1": at(1), "after_5": at(5),
        f"after_{PERSIST_TICKS}": at(PERSIST_TICKS),
        "motor_after_1": motor[0] if motor else 0.0,
        f"motor_after_{PERSIST_TICKS}": motor[PERSIST_TICKS - 1] if len(motor) >= PERSIST_TICKS else 0.0,
        "trace": trace,
        "pass": at(PERSIST_TICKS) > NOISE_FLOOR,
    }


def probe_b_history(g, dataset, gain, noise, seed=0, ticks=40):
    """Same input, two histories. A memoryless filter cannot tell them apart."""
    cur = probe_current(g, dataset)
    reg = regions(g, dataset)

    def run(history_ticks, history_current):
        brain = _brain(g, gain, noise, seed)     # identical noise stream
        for _ in range(history_ticks):
            brain.step(history_current)
        out = []
        trace = np.zeros(len(reg["motor"]), np.float32)
        for _ in range(ticks):
            s = brain.step(cur)                  # identical probe from here on
            # Keep population patterns; equal mean rates can encode different
            # histories. The observation trace starts at zero in every arm.
            trace = 0.8 * trace + 0.2 * s[reg["motor"]]
            out.append(trace.copy())
        return np.asarray(out)

    hot = run(60, cur)            # history: driven
    cold = run(60, None)          # history: silent
    # Baseline: the same history, different noise seed only.
    brain = _brain(g, gain, noise, seed + 1)
    for _ in range(60):
        brain.step(cur)
    trace = np.zeros(len(reg["motor"]), np.float32)
    base = []
    for _ in range(ticks):
        trace = 0.8 * trace + 0.2 * brain.step(cur)[reg["motor"]]
        base.append(trace.copy())
    base = np.asarray(base)

    d_hist = float(np.abs(hot - cold).mean())
    d_seed = float(np.abs(hot - base).mean())
    return {
        "mean_abs_diff_history": d_hist,
        "mean_abs_diff_seed_only": d_seed,
        "ratio": d_hist / max(d_seed, 1e-12),
        "pass": d_hist > 2.0 * d_seed,
        "observation": "per-neuron motor trace, decay=0.8/tick; reset after conditioning",
        "history_distance_by_tick": np.abs(hot - cold).mean(axis=1).tolist(),
        "seed_distance_by_tick": np.abs(hot - base).mean(axis=1).tolist(),
        "interpretation": "Descriptive one-pair diagnostic; failure does not establish absence of memory.",
    }


def probe_c_lag(g, dataset, gain, noise, seed=0, ticks=200, max_lag=8):
    """Cross-correlate motor activity with a randomly modulated drive."""
    rng = np.random.default_rng(seed)
    brain = _brain(g, gain, noise, seed)
    base = probe_current(g, dataset)
    reg = regions(g, dataset)

    amps, mot = [], []
    for _ in range(ticks):
        a = float(rng.uniform(0.5, 2.0))
        s = brain.step(base * a)
        amps.append(a)
        mot.append(float(s[reg["motor"]].mean()) if reg["motor"].size else 0.0)
    amps = np.asarray(amps); mot = np.asarray(mot)

    def corr(lag):
        x, y = (amps[: len(amps) - lag], mot[lag:]) if lag else (amps, mot)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    lags = {f"lag_{k}": corr(k) for k in range(max_lag + 1)}
    peak = max(lags, key=lambda k: abs(lags[k]))
    return {**lags, "peak_lag": int(peak.split("_")[1]),
            "peak_corr": lags[peak],
            "note": "Peak lag is descriptive; filtering, inhibition and recurrence can all affect it."}


def classify_dynamics(persistent: bool, history: bool) -> str:
    """Do not require self-sustained activity for a memory-bearing reservoir."""
    if history:
        return "persistent_history" if persistent else "fading_memory"
    return "persistent_activity" if persistent else "memory_not_detected"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default="flywire.npz")
    p.add_argument("--dataset", default="flywire")
    p.add_argument("--gain", type=float, default=14.0, help="frozen operating point")
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    g = ConnectomeGraph.load(CONNECTOME / args.graph)
    print(f"graph {g.n_nodes:,} nodes / {g.n_edges:,} pairs   "
          f"gain={args.gain} noise={args.noise}\n")

    a = probe_a_persistence(g, args.dataset, args.gain, args.noise, seed=args.seed)
    print("A  persistence after drive removal")
    print(f"     driven               {a['driven_rate']*100:7.3f} %")
    print(f"     +1 tick              {a['after_1']*100:7.3f} %")
    print(f"     +5 ticks             {a['after_5']*100:7.3f} %")
    print(f"     +{PERSIST_TICKS} ticks            {a[f'after_{PERSIST_TICKS}']*100:7.3f} %")
    print(f"     motor +{PERSIST_TICKS}            {a[f'motor_after_{PERSIST_TICKS}']*100:7.3f} %")
    print(f"     -> {'PASS' if a['pass'] else 'FAIL'}\n")

    b = probe_b_history(g, args.dataset, args.gain, args.noise, seed=args.seed)
    print("B  history dependence at identical input")
    print(f"     |hot - cold|         {b['mean_abs_diff_history']:.6f}")
    print(f"     |seed only|          {b['mean_abs_diff_seed_only']:.6f}")
    print(f"     ratio                {b['ratio']:.2f}x")
    print(f"     -> {'PASS' if b['pass'] else 'FAIL'}\n")

    c = probe_c_lag(g, args.dataset, args.gain, args.noise, seed=args.seed)
    print("C  motor-vs-drive cross-correlation")
    for k in sorted([k for k in c if k.startswith("lag_")],
                    key=lambda s: int(s.split("_")[1])):
        print(f"     {k:<10} {c[k]:+.4f}")
    print(f"     peak at lag {c['peak_lag']}  ({c['peak_corr']:+.4f})\n")

    state = classify_dynamics(a["pass"], b["pass"])
    line = state
    note = ("Activity persistence and history dependence are separate observations.\n"
            "A failed probe does not establish feed-forward dynamics, absence of memory,\n"
            "or a fixed point. Confirm useful temporal information with held-out decoding,\n"
            "several seeds and controls without recurrent transmission.")

    print("=" * 62)
    print(f"G6 {line}")
    print("=" * 62)
    print(note)
    if c["peak_lag"] >= 2:
        print(f"\nProbe C peaks at lag {c['peak_lag']}, consistent with multi-hop "
              f"propagation\nrather than an instantaneous filter (compare the "
              f"measured readout\ndepth in scripts/gates/inventory.py).")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / (args.output or f"gate_g6_s{args.seed}.json")
    out.write_text(json.dumps(
        {"graph": args.graph, "gain": args.gain, "noise": args.noise,
         "seed": args.seed, "persistence": a, "history": b, "lag": c,
         "state": state, "verdict": bool(b["pass"]),
         "drive": "uniform bilateral lamina probe; not real image encoding",
         "provenance": {
             "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "graph_sha256": hashlib.sha256((CONNECTOME / args.graph).read_bytes()).hexdigest(),
         }}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
