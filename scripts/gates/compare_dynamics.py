#!/usr/bin/env python
"""Model A vs Model B: how much of the behaviour is the connectome, how much the dynamics?

Model A (the project's): synapse counts -> log1p -> in-degree normalisation -> `gain` -> LIF, no
synaptic filter, no delay. Model B (after Shiu et al., Nature 634:210-219, 2024): counts x sign x
`w_syn` (mV) -> LIF with a 5 ms synaptic time constant and a 1.8 ms delay. Each free scalar is swept.
Reads no reward or outcome, selects nothing, writes nothing back. Banks: `calibration` (what choices
may see) and `heldout` (reproduction check only; a disagreement is reported, never acted on).
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

from flymario.agent import AgentConfig, FlyAgent  # noqa: E402
from flymario.connectome import datasets  # noqa: E402
from flymario.dynamics.calibrate import regions  # noqa: E402
from flymario.env import stimulus_bank  # noqa: E402
from flymario.models import reference_lif as ref  # noqa: E402
from flymario.models.spiking_reservoir import LIFBrain, LIFParams  # noqa: E402
from flymario.paths import ROOT, RUNS  # noqa: E402

EXTINCT, SATURATED = 0.002, 0.30

#: Model A, as a coincidence fraction. 0.083 is what the project's gain 12.0
#: implies; the rest brackets it. See docs/CALIBRATION.md §3.3 for why this is a
#: correspondence between models and not a conversion.
FRACTIONS_A = (0.25, 0.15, 0.083, 0.04, 0.02)

#: The 2x2 of weight transforms, LIF held fixed.
VARIANTS_A = ((True, True), (False, True), (True, False), (False, False))

#: Model B. Centred on the reference's published 0.275 mV and opened upward,
#: because on this graph that value sits near the lower edge of the live range.
WSYN_B = (0.275, 1.0, 3.0, 10.0, 30.0)

#: Noise as the same fraction of threshold in both models, so it is not a
#: hidden difference between them.
NOISE_FRACTION = 0.02


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class ModelA:
    """The project's LIF, with log1p and in-degree normalisation as a 2x2 of factors.

    The free scalar is a coincidence fraction (share of a median neuron's input mass that must be
    co-active to reach threshold), comparable across variants; gain 12.0 = fraction 0.083.
    """

    def __init__(self, graph, fraction: float, log_weights: bool = True,
                 normalise: bool = True, seed: int = 0):
        self.fraction = float(fraction)
        self.log_weights, self.normalise = log_weights, normalise
        W = graph.to_sparse(log_weights=log_weights)
        # Reference input mass: 1.0 after normalisation, else the median row.
        rows = np.abs(W).sum(axis=1).A.ravel()
        live = rows[rows > 0]
        self.row_mass = 1.0 if normalise else float(np.median(live))
        gain = 1.0 / (self.fraction * self.row_mass)
        self.value = self.fraction
        self.gain = gain
        self.W = W
        self.params = LIFParams(gain=gain, noise_std=NOISE_FRACTION, seed=seed)
        self.name = f"A_{'log' if log_weights else 'lin'}_{'norm' if normalise else 'raw'}"
        self.scalar = "fraction"

    def brain(self, seed: int | None = None):
        p = self.params if seed is None else LIFParams(
            gain=self.params.gain, noise_std=self.params.noise_std, seed=int(seed))
        return LIFBrain(self.W, p, normalize=self.normalise)

    def describe(self) -> dict:
        return {"model": self.name, "scalar": self.scalar, "value": self.value,
                "gain_implied": round(self.gain, 6),
                "median_row_mass": round(self.row_mass, 4),
                "log_weights": self.log_weights, "normalised": self.normalise,
                "synaptic_filtering": False, "transmission_delay": False,
                "threshold_units": "normalised (threshold = 1.0)"}


class ModelB:
    """Reference-style dynamics. Free scalar: w_syn in mV."""

    name = "B_reference"
    scalar = "w_syn_mV"

    def __init__(self, graph, value: float, seed: int = 0, **_):
        self.value = value
        self.W = ref.operator(graph, value)
        gap = ref.V_THRESHOLD_MV - ref.V_REST_MV
        self.params = ref.ReferenceParams(w_syn_mV=float(value),
                                          noise_mV=NOISE_FRACTION * gap, seed=seed)

    def brain(self, seed: int | None = None):
        p = self.params
        if seed is not None:
            p = ref.ReferenceParams(w_syn_mV=p.w_syn_mV, noise_mV=p.noise_mV,
                                    seed=int(seed))
        return ref.ReferenceBrain(self.W, p)

    def describe(self) -> dict:
        d = {"model": self.name, "scalar": self.scalar, "value": self.value,
             "log_weights": False, "normalised": False,
             "synaptic_filtering": True, "transmission_delay": True,
             "threshold_units": "millivolts (7 mV gap)",
             "synapses_to_fire": round(self.params.synapses_to_fire, 2)}
        return d


def run_sequence(model, vision, frames, reg, motor_idx, warmup, tpd, seed=0):
    """Warm up on the sequence itself, then measure. Never on silence: the
    network does not ignite from rest, so a zero warm-up reports the ignition
    transient rather than the regime."""
    b = model.brain(seed)
    vision.reset()
    warmed = 0
    while warmed < warmup:
        for f in frames:
            cur = vision.encode(f)
            for _ in range(tpd):
                b.step(cur)
                warmed += 1
                if warmed >= warmup:
                    break
            if warmed >= warmup:
                break

    glob, per_region, motor = [], {k: [] for k in reg}, []
    ever = np.zeros(model.W.shape[0], bool)
    vision.reset()
    for f in frames:
        cur = vision.encode(f)
        for _ in range(tpd):
            sp = b.step(cur)
            ever |= sp
            glob.append(float(sp.mean()))
            motor.append(float(sp[motor_idx].mean()) if motor_idx.size else 0.0)
            for rk, idx in reg.items():
                per_region[rk].append(float(sp[idx].mean()) if idx.size else np.nan)

    glob = np.asarray(glob)
    return {"global": glob,
            "region_rates": {k: float(np.nanmean(v)) for k, v in per_region.items()},
            "readout_rate": float(np.mean(motor)),
            "active_fraction": float(ever.mean())}


def propagation(model, vision, frames, reg, motor_idx, warmup, tpd) -> dict:
    """Sensorimotor propagation: the first tick at which the readout departs from its
    baseline after the input switches to a different scene."""
    b = model.brain(0)
    vision.reset()
    a_frames, b_frames = frames[: len(frames) // 2], frames[len(frames) // 2:]
    for _ in range(max(1, warmup // (tpd * max(len(a_frames), 1)))):
        for f in a_frames:
            cur = vision.encode(f)
            for _ in range(tpd):
                b.step(cur)
    cur_a = vision.encode(a_frames[0])
    base = np.mean([b.step(cur_a)[motor_idx].mean() for _ in range(20)])
    vision.reset()
    cur_b = vision.encode(b_frames[0])
    resp = [float(b.step(cur_b)[motor_idx].mean()) for _ in range(24)]
    first = next((i + 1 for i, v in enumerate(resp) if abs(v - base) > EXTINCT), None)
    return {"baseline_readout": float(base), "response": resp,
            "first_responding_tick": first, "ticks_per_decision": tpd}


def information(model, vision, bank, motor_idx, seeds, warmup, tpd, rng) -> dict:
    """Can the readout say which sequence is on screen? No reward, no action."""
    n = model.W.shape[0]
    zero = np.zeros(n, np.float32)

    def collect(drive: bool):
        X, y, grp = [], [], []
        for ci, name in enumerate(sorted(bank.sequences)):
            fr = bank.sequences[name]
            mid = fr[len(fr) // 2: len(fr) // 2 + 2]
            for sd in seeds:
                b = model.brain(sd)
                for _ in range(warmup):
                    b.step(zero)
                vision.reset()
                counts = np.zeros(len(motor_idx), np.float32)
                for f in mid:
                    cur = vision.encode(f) if drive else zero
                    for _ in range(tpd):
                        counts += b.step(cur)[motor_idx]
                X.append(counts); y.append(ci); grp.append(sd)
        return np.array(X, np.float32), np.array(y), np.array(grp)

    X, y, grp = collect(True)
    Xn, yn, gn = collect(False)
    return {"real": decode(X, y, grp),
            "shuffled": decode(X, rng.permutation(y), grp),
            "no_input": decode(Xn, yn, gn),
            "active_units": int((X.sum(axis=0) > 0).sum()),
            "readout_units": int(X.shape[1])}


def history(model, vision, bank, motor_idx, warmup, tpd) -> dict:
    names = sorted(bank.sequences)
    probe = bank.sequences[names[0]][:2]
    other = bank.sequences[names[len(names) // 2]][:6]
    zero = np.zeros(model.W.shape[0], np.float32)

    def run(hist, seed):
        b = model.brain(seed)
        for _ in range(warmup):
            b.step(zero)
        vision.reset()
        if hist is not None:
            for f in hist:
                cur = vision.encode(f)
                for _ in range(tpd):
                    b.step(cur)
        vision.reset()
        out = []
        for f in probe:
            cur = vision.encode(f)
            for _ in range(tpd):
                out.append(b.step(cur)[motor_idx].astype(np.float32))
        return np.stack(out)

    hot, cold, seeded = run(other, 0), run(None, 0), run(other, 1)
    d_h = float(np.abs(hot - cold).mean())
    d_s = float(np.abs(hot - seeded).mean())
    return {"mean_abs_diff_history": d_h, "mean_abs_diff_seed_only": d_s,
            "ratio": d_h / max(d_s, 1e-12)}


def evaluate(model, vision, bank, reg, motor_idx, warmup, tpd, seeds, rng,
             do_decode: bool) -> dict:
    per_seq = {name: run_sequence(model, vision, frames, reg, motor_idx, warmup, tpd)
               for name, frames in bank.sequences.items()}
    glob = np.array([v["global"].mean() for v in per_seq.values()])
    read = np.array([v["readout_rate"] for v in per_seq.values()])
    regm = {k: float(np.mean([v["region_rates"][k] for v in per_seq.values()]))
            for k in reg}
    series = np.concatenate([v["global"] for v in per_seq.values()])
    live = [v for v in regm.values() if np.isfinite(v) and v > 0]
    n = len(series)

    out = {
        **model.describe(),
        "bank": bank.which,
        "global_rate": float(glob.mean()),
        "readout_rate": float(read.mean()),
        "region_rates": regm,
        "regional_spread": float(max(live) / min(live)) if len(live) > 1 else float("inf"),
        "active_fraction": float(np.mean([v["active_fraction"] for v in per_seq.values()])),
        "stability": {
            "coefficient_of_variation": float(series.std() / max(series.mean(), 1e-12)),
            "drift_relative": float((series[-n // 4:].mean() - series[: n // 4].mean())
                                    / max(series.mean(), 1e-12)),
            "autocorrelation_lag1": float(np.corrcoef(series[:-1], series[1:])[0, 1])
            if series.std() > 1e-12 else 0.0,
        },
        "between_sequence_cv": float(glob.std() / max(glob.mean(), 1e-12)),
        "extinct": bool(read.mean() <= EXTINCT),
        "saturated": bool(any(v > SATURATED for v in regm.values() if np.isfinite(v))),
        "per_sequence_global": {k: round(float(v["global"].mean()), 5)
                                for k, v in per_seq.items()},
    }
    first = sorted(bank.sequences)[0]
    out["propagation"] = propagation(model, vision, bank.sequences[first],
                                     reg, motor_idx, warmup, tpd)
    if do_decode:
        out["information"] = information(model, vision, bank, motor_idx, seeds,
                                         warmup, tpd, rng)
        out["history"] = history(model, vision, bank, motor_idx, warmup, tpd)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS_A))
    p.add_argument("--variants", type=int, default=4,
                   help="how many of the 2x2 weight-transform cells to run")
    p.add_argument("--wsyn", type=float, nargs="+", default=list(WSYN_B))
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--banks", nargs="+", default=["calibration", "heldout"])
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    g = datasets.load(args.dataset)
    cfg = AgentConfig()
    agent = FlyAgent(g, cfg)
    reg = regions(g, args.dataset)
    tpd = cfg.ticks_per_decision
    seeds = np.arange(1, args.seeds + 1)

    banks = {}
    for which in args.banks:
        print(f"building the {which} bank ...")
        banks[which] = stimulus_bank.build(which=which, length=args.length)

    print(f"\n{g.source} · {tpd} ticks/decision · warmup {warmup_note(args.warmup)}")
    print("no reward, score, completion or distance enters anything below\n")

    rep = {"dataset": args.dataset, "ticks_per_decision": tpd,
           "warmup_ticks": args.warmup, "noise_fraction_of_threshold": NOISE_FRACTION,
           "banks": {k: b.describe() for k, b in banks.items()},
           "reference_parameters": {
               "source": "Shiu et al., Nature 634:210-219 (2024)",
               "v_rest_mV": ref.V_REST_MV, "v_threshold_mV": ref.V_THRESHOLD_MV,
               "tau_membrane_ms": ref.TAU_MEMBRANE_MS,
               "tau_synapse_ms": ref.TAU_SYNAPSE_MS,
               "refractory_ms": ref.REFRACTORY_MS, "delay_ms": ref.DELAY_MS,
               "w_syn_published_mV": ref.W_SYN_REFERENCE_MV},
           "results": []}

    header = (f"{'model':<13}{'scalar':>10}{'bank':>13}{'global':>9}{'readout':>9}"
              f"{'active':>8}{'CV':>7}{'seqCV':>7}{'spread':>8}{'prop':>6}  verdict")
    print(header)
    configs = [(ModelA, v, dict(log_weights=lw, normalise=nz))
               for lw, nz in VARIANTS_A[: args.variants] for v in args.fractions]
    configs += [(ModelB, v, {}) for v in args.wsyn]
    for Model, value, kw in configs:
            m = Model(g, value, **kw)
            for which, bank in banks.items():
                r = evaluate(m, agent.vision, bank, reg, agent.motor_idx,
                             args.warmup, tpd, seeds, rng, not args.no_decode)
                rep["results"].append(r)
                verdict = ("extinct" if r["extinct"] else
                           "saturated" if r["saturated"] else "alive")
                prop = r["propagation"]["first_responding_tick"]
                print(f"{m.name:<13}{value:>10.3f}{which:>13}"
                      f"{r['global_rate']*100:8.2f}%{r['readout_rate']*100:8.2f}%"
                      f"{r['active_fraction']*100:7.1f}%"
                      f"{r['stability']['coefficient_of_variation']:7.2f}"
                      f"{r['between_sequence_cv']:7.2f}"
                      f"{r['regional_spread']:7.1f}x{str(prop):>6}  {verdict}")

    if not args.no_decode:
        print(f"\n{'model':<13}{'scalar':>10}{'bank':>13}{'decode':>9}{'shuf':>7}"
              f"{'none':>7}{'chance':>8}{'units':>7}{'history':>9}")
        for r in rep["results"]:
            i, h = r.get("information"), r.get("history")
            if not i:
                continue
            print(f"{r['model']:<13}{r['value']:>10.3f}{r['bank']:>13}"
                  f"{i['real']['accuracy']:9.3f}{i['shuffled']['accuracy']:7.3f}"
                  f"{i['no_input']['accuracy']:7.3f}{i['real']['chance']:8.3f}"
                  f"{i['active_units']:7d}{h['ratio']:9.2f}x")

    print("\nNo model and no parameter is selected here. The calibration bank is")
    print("what choices may see; the held-out bank only checks that a regime")
    print("reproduces. A disagreement between them is a finding, not a reason")
    print("to revise anything.")

    graph_path = datasets.cache_path(args.dataset)
    rep["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": git_sha(), "python": platform.python_version(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "wall_seconds": round(time.perf_counter() - t0, 1), "argv": vars(args),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RUNS / f"compare_dynamics_{args.dataset}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"-> {out}")


def warmup_note(w: int) -> str:
    return f"{w} ticks, on the sequence itself"


if __name__ == "__main__":
    main()
