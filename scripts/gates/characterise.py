#!/usr/bin/env python
"""Characterise the model over a parameter range, under realistic input (real frames -> encode -> connectome).

Not a calibration: selects nothing, writes nothing back; no accuracy, reward or score enters. Sequences come
from `env.stimulus_bank`. The default gain sweep spans the equivalent-gain distribution of Shiu et al. 2024
(see docs/CALIBRATION.md). Reported per value (docs/PROTOCOL.md §9.4): global, per-region and readout activity,
active fraction, temporal stability, saturation or extinction, between-sequence variability, sensory
information in the readout, history dependence.
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
from diagnostics.decode import decode  # noqa: E402

from flymario import config  # noqa: E402
from flymario.agent import AgentConfig, FlyAgent  # noqa: E402
from flymario.connectome import datasets  # noqa: E402
from flymario.dynamics.calibrate import regions  # noqa: E402
from flymario.env import stimulus_bank  # noqa: E402
from flymario.models.spiking_reservoir import LIFBrain, LIFParams  # noqa: E402
from flymario.paths import ROOT, RUNS  # noqa: E402

#: Shiu et al. 2024, Nature 634:210-219. Table in `docs/CALIBRATION.md`.
SHIU = {"v_rest_mV": -52.0, "v_threshold_mV": -45.0, "w_syn_mV": 0.275,
        "tau_membrane_ms": 20.0, "tau_synapse_ms": 5.0, "t_refractory_ms": 2.2,
        "t_delay_ms": 1.8}
SYNAPSES_TO_FIRE = (SHIU["v_threshold_mV"] - SHIU["v_rest_mV"]) / SHIU["w_syn_mV"]

#: Log-spaced over the literature-derived equivalent-gain distribution.
DEFAULT_GAINS = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)

#: A readout below this cannot be decoded from; a region above this is saturated.
EXTINCT = 0.002
SATURATED = 0.30


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def equivalent_gain_distribution(g) -> dict:
    """What gain reproduces the reference model's coincidence requirement.

    Fractional threshold (ours) vs absolute synapse count (reference) agree for one
    in-degree only, so the answer is a distribution, not a value.
    """
    S = np.bincount(g.dst, weights=g.weight.astype(np.float64),
                    minlength=g.n_nodes)
    live = S[S > 0]
    q = np.percentile(live / SYNAPSES_TO_FIRE, [5, 25, 50, 75, 95])
    return {
        "synapses_to_fire": round(float(SYNAPSES_TO_FIRE), 1),
        "in_strength_median": float(np.median(live)),
        "equivalent_gain": {"p5": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
                            "median": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
                            "p95": round(float(q[4]), 2)},
        "note": ("Our operator row-normalises and log1p-transforms weights; the "
                 "reference does neither. The comparison bounds a plausible "
                 "range, it does not transfer a value."),
    }


def run_sequence(W, lif: LIFParams, vision, frames, reg, motor_idx,
                 warmup: int, ticks_per_decision: int) -> dict:
    """One sequence at one parameter value. Returns per-tick series, not means."""
    brain = LIFBrain(W, lif)
    # Warm up ON THE SEQUENCE, not on silence: the network does not ignite from rest,
    # so warming on zero current would report the ignition transient as the regime.
    vision.reset()
    warmed = 0
    while warmed < warmup:
        for frame in frames:
            cur = vision.encode(frame)
            for _ in range(ticks_per_decision):
                brain.step(cur)
                warmed += 1
                if warmed >= warmup:
                    break
            if warmed >= warmup:
                break

    glob, per_region, motor = [], {k: [] for k in reg}, []
    ever = np.zeros(W.shape[0], bool)
    vision.reset()
    for frame in frames:
        cur = vision.encode(frame)
        for _ in range(ticks_per_decision):
            sp = brain.step(cur)
            ever |= sp
            glob.append(float(sp.mean()))
            motor.append(float(sp[motor_idx].mean()) if motor_idx.size else 0.0)
            for k, idx in reg.items():
                per_region[k].append(float(sp[idx].mean()) if idx.size else np.nan)

    glob = np.asarray(glob)
    return {
        "global_rate": float(glob.mean()),
        "global_cv": float(glob.std() / max(glob.mean(), 1e-12)),
        "global_series": [round(float(v), 5) for v in glob],
        "region_rates": {k: float(np.nanmean(v)) for k, v in per_region.items()},
        "readout_rate": float(np.mean(motor)),
        "active_fraction": float(ever.mean()),
        "readout_active_fraction": float(ever[motor_idx].mean()) if motor_idx.size else 0.0,
    }


def stability(series: np.ndarray) -> dict:
    """Is the regime a steady state, a drift, or an oscillation?"""
    n = len(series)
    first, last = series[: n // 4], series[-n // 4:]
    ac1 = float(np.corrcoef(series[:-1], series[1:])[0, 1]) if series.std() > 1e-12 else 0.0
    return {
        "coefficient_of_variation": float(series.std() / max(series.mean(), 1e-12)),
        "drift_last_minus_first": float(last.mean() - first.mean()),
        "drift_relative": float((last.mean() - first.mean()) / max(series.mean(), 1e-12)),
        "autocorrelation_lag1": ac1,
    }


def information_in_readout(W, lif, vision, bank, motor_idx, seeds, warmup,
                           ticks_per_decision, rng) -> dict:
    """Can the readout say which sequence is being shown?

    Classes are the recorded scenarios (chance 1/len(bank)); no reward, score or action enters.
    """
    zero = np.zeros(W.shape[0], np.float32)

    def collect(drive: bool):
        X, y, grp = [], [], []
        for ci, name in enumerate(sorted(bank.sequences)):
            frames = bank.sequences[name]
            for sd in seeds:
                b = LIFBrain(W, LIFParams(gain=lif.gain, noise_std=lif.noise_std,
                                          seed=int(sd)))
                for _ in range(warmup):
                    b.step(zero)
                vision.reset()
                counts = np.zeros(len(motor_idx), np.float32)
                # One decision's worth of the sequence, mid-way, so the trace
                # reflects a moving input rather than an onset transient.
                for frame in frames[len(frames) // 2: len(frames) // 2 + 2]:
                    cur = vision.encode(frame) if drive else zero
                    for _ in range(ticks_per_decision):
                        counts += b.step(cur)[motor_idx]
                X.append(counts)
                y.append(ci)
                grp.append(sd)
        return np.array(X, np.float32), np.array(y), np.array(grp)

    X, y, grp = collect(True)
    Xn, yn, gn = collect(False)
    real, shuf, none = decode(X, y, grp), decode(X, rng.permutation(y), grp), decode(Xn, yn, gn)
    return {"real": real, "shuffled": shuf, "no_input": none,
            "active_units": int((X.sum(axis=0) > 0).sum()),
            "readout_units": int(X.shape[1])}


def history_dependence(W, lif, vision, bank, motor_idx, warmup,
                       ticks_per_decision) -> dict:
    """Same input, two different preceding sequences. Does the state remember?"""
    names = sorted(bank.sequences)
    probe = bank.sequences[names[0]][: 2]
    other = bank.sequences[names[len(names) // 2]][: 8]
    zero = np.zeros(W.shape[0], np.float32)

    def run(history, seed):
        b = LIFBrain(W, LIFParams(gain=lif.gain, noise_std=lif.noise_std, seed=seed))
        for _ in range(warmup):
            b.step(zero)
        vision.reset()
        if history is not None:
            for f in history:
                cur = vision.encode(f)
                for _ in range(ticks_per_decision):
                    b.step(cur)
        vision.reset()
        out = []
        for f in probe:
            cur = vision.encode(f)
            for _ in range(ticks_per_decision):
                out.append(b.step(cur)[motor_idx].astype(np.float32))
        return np.stack(out)

    hot, cold, other_seed = run(other, 0), run(None, 0), run(other, 1)
    d_hist = float(np.abs(hot - cold).mean())
    d_seed = float(np.abs(hot - other_seed).mean())
    return {"mean_abs_diff_history": d_hist, "mean_abs_diff_seed_only": d_seed,
            "ratio": d_hist / max(d_seed, 1e-12),
            "note": "ratio > 2 would indicate the state carries its history"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--topology", default="connectome")
    p.add_argument("--gains", type=float, nargs="+", default=list(DEFAULT_GAINS))
    p.add_argument("--length", type=int, default=16, help="decisions per sequence")
    p.add_argument("--seeds", type=int, default=4, help="noise seeds for decoding")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    warmup = args.warmup if args.warmup is not None else config.get("dynamics.warmup_ticks")

    g = datasets.load(args.dataset)
    cfg = AgentConfig(topology=args.topology)
    agent = FlyAgent(g, cfg)
    W = agent.wired.to_sparse()
    reg = regions(g, args.dataset)
    tpd = cfg.ticks_per_decision

    print("building the stimulus bank ...")
    bank = stimulus_bank.build(length=args.length, verbose=True)
    eq = equivalent_gain_distribution(g)
    print(f"\n{g.source} · {args.topology} · {tpd} ticks/decision · warmup {warmup}")
    print(f"reference equivalent gain: p5 {eq['equivalent_gain']['p5']} · "
          f"median {eq['equivalent_gain']['median']} · "
          f"p95 {eq['equivalent_gain']['p95']}")
    print(f"frozen gain (not used for selection here): {cfg.lif.gain}\n")

    rep = {"dataset": args.dataset, "topology": args.topology,
           "ticks_per_decision": tpd, "warmup_ticks": warmup,
           "noise_std": cfg.lif.noise_std, "frozen_gain": cfg.lif.gain,
           "reference": SHIU, "equivalent_gain": eq,
           "bank": bank.describe(), "results": []}

    head = (f"{'gain':>6}{'global':>9}{'readout':>9}{'active':>8}"
            f"{'CV':>7}{'drift':>8}{'seq var':>9}{'spread':>8}  verdict")
    print(head)
    for gain in args.gains:
        lif = LIFParams(gain=float(gain), noise_std=cfg.lif.noise_std, seed=0)
        per_seq = {}
        for name in sorted(bank.sequences):
            per_seq[name] = run_sequence(W, lif, agent.vision, bank.sequences[name],
                                         reg, agent.motor_idx, warmup, tpd)

        glob = np.array([v["global_rate"] for v in per_seq.values()])
        read = np.array([v["readout_rate"] for v in per_seq.values()])
        regm = {k: float(np.mean([v["region_rates"][k] for v in per_seq.values()]))
                for k in reg}
        series = np.concatenate([np.asarray(v["global_series"]) for v in per_seq.values()])
        st = stability(series)
        live = [v for v in regm.values() if np.isfinite(v) and v > 0]

        entry = {
            "gain": float(gain),
            "global_rate": float(glob.mean()),
            "global_rate_hz": float(glob.mean() / (LIFParams().dt_ms / 1000.0)),
            "readout_rate": float(read.mean()),
            "region_rates": regm,
            "regional_spread": float(max(live) / min(live)) if len(live) > 1 else float("inf"),
            "active_fraction": float(np.mean([v["active_fraction"] for v in per_seq.values()])),
            "readout_active_fraction": float(np.mean(
                [v["readout_active_fraction"] for v in per_seq.values()])),
            "stability": st,
            "between_sequence_cv": float(glob.std() / max(glob.mean(), 1e-12)),
            "extinct": bool(read.mean() <= EXTINCT),
            "saturated": bool(any(v > SATURATED for v in regm.values() if np.isfinite(v))),
            "per_sequence": {k: {kk: vv for kk, vv in v.items() if kk != "global_series"}
                             for k, v in per_seq.items()},
        }
        if not args.no_decode:
            entry["information"] = information_in_readout(
                W, lif, agent.vision, bank, agent.motor_idx,
                np.arange(1, args.seeds + 1), warmup, tpd, rng)
            entry["history"] = history_dependence(
                W, lif, agent.vision, bank, agent.motor_idx, warmup, tpd)

        rep["results"].append(entry)
        verdict = ("extinct" if entry["extinct"] else
                   "saturated" if entry["saturated"] else "alive")
        print(f"{gain:6.1f}{entry['global_rate']*100:8.2f}%{entry['readout_rate']*100:8.2f}%"
              f"{entry['active_fraction']*100:7.1f}%{st['coefficient_of_variation']:7.2f}"
              f"{st['drift_relative']:+8.2f}{entry['between_sequence_cv']:9.2f}"
              f"{entry['regional_spread']:7.1f}x  {verdict}")

    if not args.no_decode:
        print(f"\n{'gain':>6}{'decode':>9}{'shuffled':>10}{'no input':>10}"
              f"{'chance':>8}{'units':>8}{'history':>9}")
        for e in rep["results"]:
            i, h = e["information"], e["history"]
            print(f"{e['gain']:6.1f}{i['real']['accuracy']:9.3f}"
                  f"{i['shuffled']['accuracy']:10.3f}{i['no_input']['accuracy']:10.3f}"
                  f"{i['real']['chance']:8.3f}{i['active_units']:8d}{h['ratio']:9.2f}x")

    print("\nNo value is selected here. Regimes are reported; the choice and its")
    print("justification belong in docs/CALIBRATION.md and a protocol amendment.")

    graph_path = datasets.cache_path(args.dataset)
    rep["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": git_sha(), "python": platform.python_version(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "bank_digest": bank.digest(),
        "wall_seconds": round(time.perf_counter() - t0, 1), "argv": vars(args),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RUNS / f"characterise_{args.dataset}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
