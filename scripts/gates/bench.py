#!/usr/bin/env python
"""Throughput benchmark: decisions/s, split between brain, emulator and readout.

Also times torch sparse CSR on the accelerator against the CPU recurrent step, to decide whether
the step must move to the GPU. `--no-env` skips the emulator if no ROM is present.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from flymario.agent import AgentConfig
from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import CONNECTOME, RUNS


def bench_brain(g, ticks_per_decision, gain, noise, steps, seed=0):
    from flymario.agent import AgentConfig, FlyAgent
    a = FlyAgent(g, AgentConfig(ticks_per_decision=ticks_per_decision,
                                lif=LIFParams(gain=gain, noise_std=noise, seed=seed)))
    a.reset()
    frame = np.full((224, 240, 3), 110, np.uint8)
    frame[60:200, 40:58] = 240
    for _ in range(10):
        a.features(frame)
    t0 = time.perf_counter()
    for _ in range(steps):
        a.features(frame)
    dt = time.perf_counter() - t0
    return {"decisions_per_s": steps / dt, "ms_per_decision": dt / steps * 1000,
            "ms_per_tick": dt / steps / ticks_per_decision * 1000}


def bench_gpu(g, ticks, gain):
    """torch sparse CSR on the accelerator, if one is present."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None)
           and torch.backends.mps.is_available() else None)
    if dev is None:
        return {"available": False, "reason": "no accelerator detected"}

    W = g.to_sparse().tocsr()
    Wt = torch.sparse_csr_tensor(
        torch.tensor(W.indptr, dtype=torch.int64),
        torch.tensor(W.indices, dtype=torch.int64),
        torch.tensor(W.data, dtype=torch.float32),
        size=W.shape).to(dev)
    s = (torch.rand(W.shape[0], device=dev) < 0.08).float()

    for _ in range(5):
        _ = Wt @ s
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ticks):
        _ = Wt @ s
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"available": True, "device": dev, "backend": torch.__version__,
            "ms_per_tick": dt / ticks * 1000, "ticks_per_s": ticks / dt}


def bench_loop(g, ticks_per_decision, gain, noise, steps, seed=0):
    from flymario.agent import AgentConfig, FlyAgent, MotorPopulationReadout
    from flymario.env.retro_env import EnvConfig, MarioEnv
    a = FlyAgent(g, AgentConfig(ticks_per_decision=ticks_per_decision,
                                lif=LIFParams(gain=gain, noise_std=noise, seed=seed)))
    env = MarioEnv(EnvConfig())
    a.readout = MotorPopulationReadout(a.feature_dim, env.n_actions)
    obs, _ = env.reset(); a.reset()
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(a.readout.predict(a.features(obs)))
        if term or trunc:
            obs, _ = env.reset(); a.reset()

    t0 = time.perf_counter()
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(a.readout.predict(a.features(obs)))
        if term or trunc:
            obs, _ = env.reset(); a.reset()
    dt_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(2)
    dt_env = time.perf_counter() - t0
    env.close()
    return {"decisions_per_s": steps / dt_loop,
            "ms_per_decision": dt_loop / steps * 1000,
            "emulator_ms_per_step": dt_env / steps * 1000,
            "emulator_share": dt_env / dt_loop}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default=datasets.default_cache_name())
    p.add_argument("--gain", type=float, default=14.0)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--ticks", type=int, default=AgentConfig().ticks_per_decision,
                   help="neural ticks per policy decision")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-env", action="store_true", help="skip the emulator")
    args = p.parse_args()

    g = ConnectomeGraph.load(CONNECTOME / args.graph)
    print(f"graph {g.n_nodes:,} nodes / {g.n_edges:,} pairs   "
          f"ticks_per_decision={args.ticks}\n")

    rep = {"graph": args.graph, "ticks_per_decision": args.ticks, "gain": args.gain}

    b = bench_brain(g, args.ticks, args.gain, args.noise, args.steps)
    rep["brain_cpu"] = b
    print(f"brain (CPU)     {b['decisions_per_s']:8.1f} dec/s   "
          f"{b['ms_per_decision']:6.1f} ms/dec   {b['ms_per_tick']:6.2f} ms/tick")

    gpu = bench_gpu(g, ticks=200, gain=args.gain)
    rep["gpu_spmv"] = gpu
    if gpu["available"]:
        print(f"spMV  ({gpu['device']})  {gpu['ticks_per_s']:8.1f} tick/s  "
              f"{gpu['ms_per_tick']:20.2f} ms/tick")
        speedup = b["ms_per_tick"] / max(gpu["ms_per_tick"], 1e-9)
        print(f"                -> {speedup:.1f}x available on the dominant term")
        rep["gpu_speedup_on_recurrent_step"] = speedup
    else:
        print(f"spMV  (GPU)     unavailable: {gpu['reason']}")

    if not args.no_env:
        try:
            lo = bench_loop(g, args.ticks, args.gain, args.noise, args.steps)
            rep["closed_loop"] = lo
            print(f"\nclosed loop     {lo['decisions_per_s']:8.1f} dec/s   "
                  f"{lo['ms_per_decision']:6.1f} ms/dec")
            print(f"emulator        {1000/lo['emulator_ms_per_step']:8.1f} step/s  "
                  f"{lo['emulator_ms_per_step']:6.2f} ms/step  "
                  f"({lo['emulator_share']*100:.1f}% of the loop)")
            brain_share = b["ms_per_decision"] / lo["ms_per_decision"]
            print(f"\nbottleneck      brain {brain_share*100:.1f}%   "
                  f"emulator {lo['emulator_share']*100:.1f}%")
            rep["brain_share"] = brain_share

            print("\nbudget, per Protocol III §10.2 "
                  "(E is UNMEASURED — these are illustrations, not estimates)")
            for E in (300, 1000, 2000):
                h = E * 300 / lo["decisions_per_s"] / 3600
                print(f"  E={E:>5}  {h:6.2f} h/run   {h*150:8.1f} h for 150 runs")
        except Exception as exc:                       # noqa: BLE001
            print(f"\nclosed loop unavailable: {exc}")
            print("(ROM missing? run `make import-rom`, or pass --no-env)")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "bench.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
