#!/usr/bin/env python
"""Run the fly in Mario, with the live dashboard.

    python scripts/play.py                          # untrained, connectome-driven
    python scripts/play.py --topology degree_preserving   # the control condition
    python scripts/play.py --readout random         # pure noise, for comparison
    python scripts/play.py --no-ui --episodes 5     # headless benchmark

With no trained readout this plays badly on purpose. The point of this script
is the loop, not the score: pixels reach the connectome, spikes propagate
through real Drosophila wiring, and descending activity chooses the buttons.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from flymario.agent import AgentConfig, FlyAgent, MotorPopulationReadout, RandomReadout
from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import CONNECTOME


def build(args) -> tuple[MarioEnv, FlyAgent]:
    graph = ConnectomeGraph.load(CONNECTOME / args.graph)
    cfg = AgentConfig(
        topology=args.topology,
        topology_seed=args.seed,
        ticks_per_decision=args.ticks,
        lif=LIFParams(gain=args.gain, noise_std=0.02, seed=args.seed),
    )
    env = MarioEnv(EnvConfig(state=args.state, frame_skip=args.frame_skip))
    agent = FlyAgent(graph, cfg)
    agent.readout = (RandomReadout(env.n_actions, args.seed) if args.readout == "random"
                     else MotorPopulationReadout(agent.feature_dim, env.n_actions))
    return env, agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default=datasets.default_cache_name())
    p.add_argument("--topology", default="connectome",
                   help="connectome | shuffled_weights | degree_preserving | shuffled_edges")
    p.add_argument("--readout", default="motor", choices=["motor", "random"])
    p.add_argument("--state", default="Level1-1")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--ticks", type=int, default=AgentConfig().ticks_per_decision,
                   help="neural ticks per policy decision (config timing.ticks_per_decision)")
    p.add_argument("--gain", type=float, default=LIFParams().gain,
                   help="frozen operating point from configs/base.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--max-steps", type=int, default=100_000,
                   help="cap on decisions PER EPISODE")
    args = p.parse_args()

    env, agent = build(args)
    info_desc = agent.describe()
    print({**info_desc, "actions": env.n_actions})

    ui = None
    if not args.no_ui:
        from flymario.telemetry.dashboard import Dashboard

        ui = Dashboard(agent.graph, agent.motor_idx, ACTION_NAMES,
                       title=f"FlyMario — {info_desc['source']} / {args.topology}")

    for ep in range(args.episodes):
        obs, info = env.reset()
        agent.reset()
        steps = 0
        done, ep_r, t0, fps = False, 0.0, time.perf_counter(), 0.0
        while not done and steps < args.max_steps:
            if ui is not None and not ui.pump():
                done = True
                break
            trace = agent.features(obs)
            action = agent.readout.predict(trace)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            steps += 1
            done = term or trunc

            if ui is not None:
                ui.render(obs, agent.last_spikes, trace, action, {
                    "source": info_desc["source"], "topology": args.topology,
                    "neurons": info_desc["neurons"],
                    "pop_rate": agent.brain.firing_rate,
                    "motor_rate": float(agent.last_spikes[agent.motor_idx].mean()),
                    "mario_x": info["mario_x"], "reward": r, "episode_reward": ep_r,
                    "frame": info["frames"], "fps": fps,
                })
                fps = ui.tick(args.fps)

        dt = time.perf_counter() - t0
        outcome = ("CLEARED" if info.get("cleared") else
                   "died" if info.get("died") else "timeout")
        print(f"ep {ep}: x={info.get('mario_x', 0):.0f} reward={ep_r:.1f} "
              f"steps={steps} {outcome}  ({dt:.1f}s, {steps / max(dt, 1e-9):.0f} dec/s)")
        if ui is not None and not ui.running:
            break

    env.close()
    if ui is not None:
        ui.close()


if __name__ == "__main__":
    main()
