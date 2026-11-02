#!/usr/bin/env python
"""Topology ablation with the firing rate held constant across conditions.

With one shared `gain`, the randomised graphs fire at 30-34% while the connectome fires at
4.6%, so differences measure dynamical regime, not topology (cf. arXiv:2604.04033). Each
topology's gain is therefore bisected to a common target rate before rollout.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flymario.agent import AgentConfig, FlyAgent, MotorPopulationReadout
from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.env.retro_env import EnvConfig, MarioEnv
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import CONNECTOME, RUNS

#: Primary contrast first. `connectome` is the identity transform.
TOPOLOGIES = ("connectome", "degree_preserving", "shuffled_weights", "shuffled_edges")


def _probe_frame(t: int) -> np.ndarray:
    f = np.zeros((224, 240, 3), np.uint8)
    f[:, :] = 110
    x = int((t * 7) % 220)
    f[60:200, x : x + 18] = 240
    return f


def measure_rate(graph, topo: str, gain: float, seed: int = 0) -> float:
    a = FlyAgent(graph, AgentConfig(topology=topo, topology_seed=seed,
                                    lif=LIFParams(gain=gain, noise_std=0.02, seed=seed)))
    a.reset()
    rates = []
    for t in range(40):
        a.features(_probe_frame(t))
        if t >= 25:
            rates.append(a.brain.firing_rate)
    return float(np.mean(rates))


def calibrate(graph, topo: str, target: float, iters: int = 11) -> float:
    """Bisect `gain` until this topology fires at `target`."""
    lo, hi = 0.05, 40.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if measure_rate(graph, topo, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default=datasets.default_cache_name())
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--target-rate", type=float, default=0.045)
    p.add_argument("--no-rate-match", action="store_true",
                   help="reproduce the confounded comparison on purpose")
    p.add_argument("--gain", type=float, default=18.0)
    args = p.parse_args()

    graph = ConnectomeGraph.load(CONNECTOME / args.graph)
    env = MarioEnv(EnvConfig())
    # Each null is generated once per seed and reused for calibration and rollout, so a
    # condition plays on the graph it was calibrated on.

    gains = {t: args.gain for t in TOPOLOGIES}
    if not args.no_rate_match:
        print(f"calibrating each topology to {args.target_rate * 100:.2f}% firing ...")
        for t in TOPOLOGIES:
            gains[t] = calibrate(graph, t, args.target_rate)
            print(f"  {t:<20} gain={gains[t]:7.3f}  rate={measure_rate(graph, t, gains[t]) * 100:5.2f}%")

    results = {}
    print(f"\n{'topology':<20}{'median x':>10}{'mean x':>9}{'std':>7}{'rate':>8}  seeds")
    for topo in TOPOLOGIES:
        xs, rates = [], []
        for seed in range(args.seeds):
            a = FlyAgent(graph, AgentConfig(
                topology=topo, topology_seed=seed,
                lif=LIFParams(gain=gains[topo], noise_std=0.02, seed=seed)))
            a.readout = MotorPopulationReadout(a.feature_dim, env.n_actions)
            obs, _ = env.reset()
            a.reset()
            best = 0.0
            for _ in range(args.steps):
                obs, _, term, trunc, info = env.step(a.readout.predict(a.features(obs)))
                best = max(best, info["mario_x"])
                if term or trunc:
                    break
            xs.append(best)
            rates.append(a.brain.firing_rate)
        results[topo] = {"x": xs, "gain": gains[topo], "rate": float(np.mean(rates))}
        print(f"{topo:<20}{np.median(xs):>10.0f}{np.mean(xs):>9.0f}{np.std(xs):>7.0f}"
              f"{np.mean(rates) * 100:>7.2f}%  {[int(x) for x in xs]}")

    env.close()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / ("ablation_naive.json" if args.no_rate_match else "ablation_rate_matched.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out}")
    print("\nNOTE: the readout is UNTRAINED. These numbers show the loop is closed and\n"
          "that the comparison is rate-sensitive; they do not measure learning yet.")


if __name__ == "__main__":
    main()
