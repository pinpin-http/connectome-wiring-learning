#!/usr/bin/env python
"""Gate E — does the calibration hold under real sensory drive?

Declared in docs/PROTOCOL.md §29 and §9.4. The frozen point was calibrated on `calibrate.probe_current`
(uniform, still); game frames arrive through `RetinotopicVision.encode`. Four probes:
  A  operating point   rates under real frames; PASS if global is within tolerance and the readout is audible.
  B  decodability      the Gate D decoder with real frames as stimuli (only the stimulus changes).
  C  drive comparison  what the probe injects versus what the game injects.
  D  §9.3 in-loop      history dependence, response to drive removal, latency, under real pixels.
Needs a ROM. Read-only with respect to the project; writes one JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.decode import decode  # noqa: E402  the Gate D decoder, unchanged

from flymario import config  # noqa: E402
from flymario.agent import AgentConfig, FlyAgent  # noqa: E402
from flymario.connectome import datasets  # noqa: E402
from flymario.dynamics.calibrate import probe_current, regions  # noqa: E402
from flymario.env.retro_env import EnvConfig, MarioEnv  # noqa: E402
from flymario.models.spiking_reservoir import LIFBrain, LIFParams  # noqa: E402
from flymario.paths import ROOT, RUNS  # noqa: E402

#: The readout must stay above this to be worth decoding. Same floor
#: `calibrate.Operating.alive` uses.
AUDIBLE = 0.002
#: Tolerance on the global rate, in percentage points of the frozen target.
RATE_TOL = 0.02


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sample_frames(env: MarioEnv, n: int, stride: int, seed: int = 0) -> list[np.ndarray]:
    """Distinct frames from a real episode, `stride` decisions apart.

    Held still afterwards, as Gate D holds its synthetic stimulus, so the numbers compare.
    """
    obs, _ = env.reset()
    frames, k = [], 0
    while len(frames) < n:
        obs, _, term, trunc, _ = env.step(2)          # RIGHT+B, the level advances
        k += 1
        if k % stride == 0:
            frames.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
    return frames


# --- probe A: the operating point under real drive ------------------------
def probe_operating_point(agent: FlyAgent, env: MarioEnv, dataset: str,
                          decisions: int, warm_decisions: int) -> dict:
    reg = regions(agent.graph, dataset)
    obs, _ = env.reset()
    agent.reset()

    glob, per_region, motor, drive_mag = [], {k: [] for k in reg}, [], []
    for i in range(warm_decisions + decisions):
        cur = agent.vision.encode(obs)
        drive_mag.append(float(np.abs(cur).sum()))
        spikes = None
        for _ in range(agent.cfg.ticks_per_decision):
            spikes = agent.brain.step(cur)
            if i >= warm_decisions:
                glob.append(float(spikes.mean()))
                for k, idx in reg.items():
                    per_region[k].append(float(spikes[idx].mean()) if idx.size else np.nan)
        agent.last_spikes = spikes
        obs, _, term, trunc, _ = env.step(2)
        if term or trunc:
            obs, _ = env.reset()

    rates = {k: float(np.nanmean(v)) for k, v in per_region.items()}
    live = [v for v in rates.values() if np.isfinite(v) and v > 0]
    return {
        "global_rate": float(np.mean(glob)),
        "region_rates": rates,
        "spread": float(max(live) / min(live)) if len(live) > 1 else float("inf"),
        "readout_rate": rates.get("motor", 0.0),
        "alive": rates.get("motor", 0.0) > AUDIBLE,
        "saturated": any(v > 0.30 for v in rates.values() if np.isfinite(v)),
        "mean_abs_drive": float(np.mean(drive_mag)),
        "decisions_measured": decisions,
    }


# --- probe B: decodability with real frames as the stimuli ----------------
def collect_frames(agent: FlyAgent, frames, seeds, ticks, warmup, drive=True):
    """Readout counts per (frame, noise seed). Mirrors decode.collect exactly,
    but the current vector comes from the engineered visual interface."""
    W = agent.wired.to_sparse()
    read = agent.motor_idx
    p = agent.cfg.lif
    zero = np.zeros(agent.graph.n_nodes, np.float32)

    X, y, grp = [], [], []
    for ci, frame in enumerate(frames):
        agent.vision.reset()
        cur = agent.vision.encode(frame) if drive else zero
        for sd in seeds:
            b = LIFBrain(W, LIFParams(gain=p.gain, noise_std=p.noise_std, seed=int(sd)))
            for _ in range(warmup):
                b.step(zero)
            counts = np.zeros(len(read), np.float32)
            for _ in range(ticks):
                counts += b.step(cur)[read]
            X.append(counts)
            y.append(ci)
            grp.append(sd)
    return np.array(X, np.float32), np.array(y), np.array(grp)


def probe_decodability(agent, frames, seeds, ticks, warmup, rng) -> dict:
    X, y, grp = collect_frames(agent, frames, seeds, ticks, warmup)
    Xn, yn, gn = collect_frames(agent, frames, seeds, ticks, warmup, drive=False)
    real = decode(X, y, grp)
    shuffled = decode(X, rng.permutation(y), grp)
    no_input = decode(Xn, yn, gn)
    return {
        "ticks": ticks, "n_frames": len(frames), "n_seeds": len(seeds),
        "real": real, "shuffled": shuffled, "no_input": no_input,
        "active_units": int((X.sum(axis=0) > 0).sum()),
        "readout_units": int(X.shape[1]),
        "pass": bool(real["accuracy"] > 0.6
                     and shuffled["accuracy"] < 0.45
                     and no_input["accuracy"] < 0.45),
    }


# --- probe C: does the calibration stimulus resemble the real one? --------
def probe_drive_comparison(agent: FlyAgent, dataset: str, frames) -> dict:
    """§9.4, measured. The frozen point was calibrated on the left column."""
    probe = probe_current(agent.graph, dataset)
    lam = datasets.by_types(agent.graph, dataset, "lamina_types")
    injected = np.union1d(agent.vision.on_idx, agent.vision.off_idx)

    real = []
    for f in frames:
        agent.vision.reset()
        real.append(agent.vision.encode(f))
    real = np.stack(real)

    nz = real[:, injected]
    return {
        "probe": {
            "neurons_driven": int((probe != 0).sum()),
            "amplitude": float(probe[probe != 0][0]) if (probe != 0).any() else 0.0,
            "total_current": float(np.abs(probe).sum()),
            "spatially_uniform": True, "changes_over_time": False,
        },
        "game": {
            "neurons_driven_max": int((np.abs(real) > 0).any(axis=0).sum()),
            "neurons_driven_mean": float((np.abs(real) > 0).sum(axis=1).mean()),
            "amplitude_mean": float(np.abs(nz).mean()),
            "amplitude_p99": float(np.percentile(np.abs(nz), 99)),
            "amplitude_max": float(np.abs(nz).max()),
            "total_current_mean": float(np.abs(real).sum(axis=1).mean()),
            "spatially_uniform": False, "changes_over_time": True,
        },
        "lamina_population": int(len(lam)),
        "injected_population": int(len(injected)),
        "current_ratio_game_over_probe": float(
            np.abs(real).sum(axis=1).mean() / max(np.abs(probe).sum(), 1e-9)),
    }


# --- probe D: §9.3's three questions, in the loop -------------------------
def probe_dynamics(agent: FlyAgent, env: MarioEnv, frames, after: int = 40) -> dict:
    """History dependence, response to drive removal, and latency."""
    W = agent.wired.to_sparse()
    p = agent.cfg.lif
    read = agent.motor_idx
    zero = np.zeros(agent.graph.n_nodes, np.float32)
    agent.vision.reset()
    probe_frame = agent.vision.encode(frames[0])
    agent.vision.reset()
    other = agent.vision.encode(frames[len(frames) // 2])

    def run(history, ticks):
        b = LIFBrain(W, LIFParams(gain=p.gain, noise_std=p.noise_std, seed=0))
        for _ in range(150):
            b.step(history)
        return np.stack([b.step(probe_frame)[read].astype(np.float32)
                         for _ in range(ticks)])

    hot, cold = run(other, 40), run(zero, 40)
    b2 = LIFBrain(W, LIFParams(gain=p.gain, noise_std=p.noise_std, seed=1))
    for _ in range(150):
        b2.step(other)
    base = np.stack([b2.step(probe_frame)[read].astype(np.float32) for _ in range(40)])

    d_hist = float(np.abs(hot - cold).mean())
    d_seed = float(np.abs(hot - base).mean())

    # drive removal, mid-episode, on a warmed network
    b3 = LIFBrain(W, LIFParams(gain=p.gain, noise_std=p.noise_std, seed=0))
    for _ in range(150):
        b3.step(probe_frame)
    driven = float(b3.spikes.mean())
    decay = [float(b3.step(zero).mean()) for _ in range(after)]

    # latency: first tick at which the readout responds to a frame change
    b4 = LIFBrain(W, LIFParams(gain=p.gain, noise_std=p.noise_std, seed=0))
    for _ in range(150):
        b4.step(probe_frame)
    ref = float(np.mean([b4.step(probe_frame)[read].mean() for _ in range(20)]))
    resp = [float(b4.step(other)[read].mean()) for _ in range(24)]
    first = next((i + 1 for i, v in enumerate(resp) if abs(v - ref) > AUDIBLE), None)

    return {
        "history": {"mean_abs_diff_history": d_hist,
                    "mean_abs_diff_seed_only": d_seed,
                    "ratio": d_hist / max(d_seed, 1e-12),
                    "pass": d_hist > 2.0 * d_seed},
        "drive_removal": {"driven_rate": driven, "after_1": decay[0],
                          "after_5": decay[4], "after_20": decay[19],
                          "trace": decay},
        "latency": {"baseline_readout": ref, "response": resp,
                    "first_responding_tick": first,
                    "ticks_per_decision": agent.cfg.ticks_per_decision,
                    "fits_in_one_decision": bool(
                        first is not None and first <= agent.cfg.ticks_per_decision)},
    }


# --- recalibration under the real drive -----------------------------------
def sweep_under_real_drive(agent: FlyAgent, dataset: str, frames, gains,
                           warmup: int, ticks: int = 60) -> list[dict]:
    """What gain would put the network on target with the game as its stimulus?

    Reported, never written back (changing a frozen point is a protocol amendment, docs/PROTOCOL.md §21).
    The sweep holds one real frame still, as the probe does.
    """
    W = agent.wired.to_sparse()
    p = agent.cfg.lif
    reg = regions(agent.graph, dataset)
    agent.vision.reset()
    cur = agent.vision.encode(frames[0])

    out = []
    for gain in gains:
        b = LIFBrain(W, LIFParams(gain=float(gain), noise_std=p.noise_std, seed=0))
        for _ in range(warmup):
            b.step(cur)
        acc = {k: [] for k in reg}
        glob = []
        for _ in range(ticks):
            sp = b.step(cur)
            glob.append(float(sp.mean()))
            for k, idx in reg.items():
                acc[k].append(float(sp[idx].mean()) if idx.size else np.nan)
        rates = {k: float(np.nanmean(v)) for k, v in acc.items()}
        live = [v for v in rates.values() if np.isfinite(v) and v > 0]
        out.append({
            "gain": float(gain), "global_rate": float(np.mean(glob)),
            "region_rates": rates,
            "spread": float(max(live) / min(live)) if len(live) > 1 else float("inf"),
            "alive": rates.get("motor", 0.0) > AUDIBLE,
            "saturated": any(v > 0.30 for v in rates.values() if np.isfinite(v)),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--topology", default="connectome")
    p.add_argument("--decisions", type=int, default=60, help="measured, after warm-up")
    p.add_argument("--warm-decisions", type=int, default=10)
    p.add_argument("--frames", type=int, default=4, help="stimulus classes for probe B")
    p.add_argument("--stride", type=int, default=40, help="decisions between frames")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--ticks", type=int, default=None, help="decode window")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.add_argument("--skip", nargs="*", default=[], choices=list("ABCD"))
    p.add_argument("--recalibrate", nargs="*", type=float, default=None,
                   metavar="GAIN",
                   help="sweep these gains under the real drive and report the "
                        "operating point each gives. Reported, never written back: "
                        "re-freezing is a protocol amendment.")
    args = p.parse_args()

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    target = config.get("dynamics.target_rate")
    warmup = args.warmup if args.warmup is not None else config.get("dynamics.warmup_ticks")

    g = datasets.load(args.dataset)
    cfg = AgentConfig(topology=args.topology, topology_seed=args.seed)
    ticks = args.ticks if args.ticks is not None else cfg.ticks_per_decision
    agent = FlyAgent(g, cfg)
    env = MarioEnv(EnvConfig())

    print(f"{g.source} · {args.topology} · gain {cfg.lif.gain} · "
          f"noise {cfg.lif.noise_std} · {cfg.ticks_per_decision} ticks/decision")
    print(f"readout {len(agent.motor_idx):,} units · "
          f"injected {agent.vision.n_injected:,} lamina cells · "
          f"eye {agent.vision.coverage()['eye']!r}")
    print(f"frozen target {target:.3f}\n")

    rep: dict = {"dataset": args.dataset, "topology": args.topology,
                 "gain": cfg.lif.gain, "noise_std": cfg.lif.noise_std,
                 "ticks_per_decision": cfg.ticks_per_decision,
                 "target_rate": target, "warmup_ticks": warmup,
                 "coverage": agent.vision.coverage()}

    frames = sample_frames(env, max(args.frames, 2), args.stride, args.seed)
    rep["frames_sampled"] = len(frames)

    if "A" not in args.skip:
        a = probe_operating_point(agent, env, args.dataset,
                                  args.decisions, args.warm_decisions)
        rep["operating_point"] = a
        print("A  operating point under real frames")
        print(f"     global               {a['global_rate']*100:7.3f} %   "
              f"(frozen target {target*100:.1f} %)")
        for k, v in a["region_rates"].items():
            print(f"     {k:<20} {v*100:7.3f} %")
        print(f"     spread               {a['spread']:7.2f}x")
        print(f"     readout audible      {a['alive']}   saturated {a['saturated']}")
        on_target = abs(a["global_rate"] - target) <= RATE_TOL
        a["on_target"] = bool(on_target)
        a["pass"] = bool(on_target and a["alive"] and not a["saturated"])
        print(f"     -> {'PASS' if a['pass'] else 'FAIL'}\n")

    if "C" not in args.skip:
        c = probe_drive_comparison(agent, args.dataset, frames)
        rep["drive_comparison"] = c
        print("C  calibration stimulus vs the real one (§9.4)")
        print(f"     probe: {c['probe']['neurons_driven']:,} cells at "
              f"amplitude {c['probe']['amplitude']:.2f}, uniform, held still")
        print(f"     game : {c['game']['neurons_driven_mean']:.0f} cells on average, "
              f"mean |I| {c['game']['amplitude_mean']:.3f}, "
              f"p99 {c['game']['amplitude_p99']:.2f}, max {c['game']['amplitude_max']:.2f}")
        print(f"     total current, game / probe: "
              f"{c['current_ratio_game_over_probe']:.2f}x\n")

    if "B" not in args.skip:
        b = probe_decodability(agent, frames, np.arange(1, args.seeds + 1),
                               ticks, warmup, rng)
        rep["decodability"] = b
        print(f"B  decodability with real frames ({b['n_frames']} classes, "
              f"chance {b['real']['chance']:.3f}, {ticks} ticks)")
        print(f"     real                 {b['real']['accuracy']:.3f} "
              f"+/- {b['real']['std']:.3f}")
        print(f"     shuffled labels      {b['shuffled']['accuracy']:.3f}")
        print(f"     no input             {b['no_input']['accuracy']:.3f}")
        print(f"     active units         {b['active_units']:,} / {b['readout_units']:,}")
        print(f"     -> {'PASS' if b['pass'] else 'FAIL'}\n")

    if "D" not in args.skip:
        d = probe_dynamics(agent, env, frames)
        rep["dynamics"] = d
        print("D  §9.3 in the loop")
        h, dr, la = d["history"], d["drive_removal"], d["latency"]
        print(f"     history ratio        {h['ratio']:.2f}x   "
              f"{'PASS' if h['pass'] else 'FAIL'}")
        print(f"     drive removed        {dr['driven_rate']*100:.2f} % -> "
              f"{dr['after_1']*100:.2f} / {dr['after_5']*100:.2f} / "
              f"{dr['after_20']*100:.2f} % at +1/+5/+20")
        print(f"     readout responds at  tick {la['first_responding_tick']} "
              f"of {la['ticks_per_decision']}   "
              f"{'fits' if la['fits_in_one_decision'] else 'DOES NOT FIT'}\n")

    if args.recalibrate is not None:
        gains = args.recalibrate or [8, 12, 16, 20, 24, 30, 40]
        sw = sweep_under_real_drive(agent, args.dataset, frames, gains, warmup)
        rep["recalibration_sweep"] = sw
        print("R  gain sweep under the real drive (reported, not frozen)")
        head = f"{'gain':>6}{'global':>9}" + "".join(
            f"{k:>11}" for k in sw[0]["region_rates"])
        print("   " + head + "   spread  verdict")
        for o in sw:
            row = f"{o['gain']:6.1f}{o['global_rate']*100:8.2f}%"
            row += "".join(f"{v*100:10.2f}%" for v in o["region_rates"].values())
            verdict_s = ("saturated" if o["saturated"] else
                         "readout silent" if not o["alive"] else "usable")
            print(f"   {row}{o['spread']:9.1f}x  {verdict_s}")
        usable = [o for o in sw if o["alive"] and not o["saturated"]]
        if usable:
            best = min(usable, key=lambda o: (abs(o["global_rate"] - target), o["spread"]))
            rep["recalibration_suggestion"] = best
            print(f"\n     closest to target {target:.3f}: gain {best['gain']:.1f} "
                  f"-> global {best['global_rate']*100:.2f} %, "
                  f"readout {best['region_rates'].get('motor', 0)*100:.2f} %, "
                  f"spread {best['spread']:.1f}x")
            print("     This is a proposal. Freezing it is a §9.4 amendment.\n")
        else:
            print("\n     no usable gain in this range\n")

    verdict = all(rep.get(k, {}).get("pass", True)
                  for k in ("operating_point", "decodability"))
    rep["verdict"] = bool(verdict)
    print("=" * 66)
    print(f"GATE E {'PASS' if verdict else 'FAIL'} — calibration "
          f"{'holds' if verdict else 'does not hold'} under real sensory drive")
    print("=" * 66)
    if not verdict:
        print("§22.6 lists failed calibration as a predefined exclusion category.")
        print("Recalibrate against the real drive before anything downstream.")

    graph_path = datasets.cache_path(args.dataset)
    rep["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": git_sha(), "python": platform.python_version(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "argv": vars(args),
    }
    env.close()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RUNS / f"gate_e_{args.dataset}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
