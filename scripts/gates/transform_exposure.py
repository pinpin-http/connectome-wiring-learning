#!/usr/bin/env python
"""Does the connectome-minus-null contrast survive the choice of weight transform?

For each of the 2x2 weight transforms, measure the connectome and the `depth_stratified` null
(rewiring within sensorimotor depth strata) and report the difference. The outcome is sensory
information reaching the readout plus the regional activity profile; no reward or score enters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
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
from flymario.connectome import datasets, randomize  # noqa: E402
from flymario.connectome.schema import ConnectomeGraph  # noqa: E402
from flymario.dynamics.calibrate import regions  # noqa: E402
from flymario.env import stimulus_bank  # noqa: E402
from flymario.models.spiking_reservoir import (  # noqa: E402
    LIFBrain, LIFParams, normalize_in_degree)
from flymario.paths import ROOT, RUNS  # noqa: E402

#: Null realisations (deterministic in their seed, ~4 min each on MaleCNS), cached and
#: keyed on the hash of the graph they were drawn from.
NULL_CACHE = ROOT / "artifacts" / "cache" / "nulls"
#: The regime sweep: coincidence fractions from 0.005 to 0.5 (wide: the lin_raw null's
#: cliff is near 0.29). 0.05 is the frozen operating point (gain 20 on the normalised operator).
REGIME_FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08,
                    0.10, 0.15, 0.20, 0.30, 0.50)

#: The 2x2 of weight transforms, exactly as in compare_dynamics.py.
VARIANTS = (("log_norm", True, True), ("lin_norm", False, True),
            ("log_raw", True, False), ("lin_raw", False, False))
NOISE_FRACTION = 0.02
#: docs/PROTOCOL.md §9.5. Both are run and every result is labelled with its own.
COMPARISONS = ("shared_parameter", "matched_regime")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_operator(graph, log_weights: bool, normalise: bool, fraction: float):
    """Operator plus the gain that puts a median neuron at `fraction` to fire.

    Each arm is scaled by its own median row mass, so the null's degree structure does not
    change the effective threshold.
    """
    W = graph.to_sparse(log_weights=log_weights)
    rows = np.abs(W).sum(axis=1).A.ravel()
    live = rows[rows > 0]
    mass = 1.0 if normalise else float(np.median(live))
    return W, 1.0 / (fraction * mass), mass


def measure(graph, log_weights, normalise, fraction, vision, bank, motor_idx,
            reg, warmup, tpd, seeds, rng) -> dict:
    W, gain, mass = build_operator(graph, log_weights, normalise, fraction)
    # Build the effective operator once, not per (class, seed) pair.
    W = normalize_in_degree(W) if normalise else W.tocsc()
    n = W.shape[0]
    zero = np.zeros(n, np.float32)
    shared = LIFBrain(W, LIFParams(gain=gain, noise_std=NOISE_FRACTION, seed=0),
                      normalize=False)

    def brain(seed):
        # Identical to a freshly constructed brain with this seed: zero state
        # and a noise stream restarted from `seed` (see LIFBrain.reset).
        shared.reset(seed=int(seed))
        return shared

    # --- sensory information reaching the readout -------------------------
    def collect(drive: bool):
        X, y, grp = [], [], []
        for ci, name in enumerate(sorted(bank.sequences)):
            fr = bank.sequences[name]
            mid = fr[len(fr) // 2: len(fr) // 2 + 2]
            for sd in seeds:
                b = brain(sd)
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
    info = {"real": decode(X, y, grp),
            "shuffled": decode(X, rng.permutation(y), grp),
            "no_input": decode(Xn, yn, gn),
            "active_units": int((X.sum(axis=0) > 0).sum())}

    # --- regional activity profile ----------------------------------------
    b = brain(0)
    first = bank.sequences[sorted(bank.sequences)[0]]
    vision.reset()
    warmed = 0
    while warmed < warmup:
        for f in first:
            cur = vision.encode(f)
            for _ in range(tpd):
                b.step(cur); warmed += 1
                if warmed >= warmup:
                    break
            if warmed >= warmup:
                break
    glob, per_region = [], {k: [] for k in reg}
    vision.reset()
    for f in first:
        cur = vision.encode(f)
        for _ in range(tpd):
            sp = b.step(cur)
            glob.append(float(sp.mean()))
            for k, idx in reg.items():
                per_region[k].append(float(sp[idx].mean()) if idx.size else np.nan)

    return {
        "gain_implied": round(gain, 6), "median_row_mass": round(mass, 4),
        "decode": info["real"]["accuracy"],
        # decode() omits `chance` when it cannot score (fewer than two seeds to
        # hold out); chance is 1/classes either way.
        "decode_chance": info["real"].get("chance", 1.0 / len(np.unique(y))),
        "shuffled": info["shuffled"]["accuracy"],
        "no_input": info["no_input"]["accuracy"],
        "active_units": info["active_units"],
        "global_rate": float(np.mean(glob)),
        "region_rates": {k: float(np.nanmean(v)) for k, v in per_region.items()},
    }


def measure_gpu(graph, log_weights, normalise, fraction, vision, bank, motor_idx,
                reg, warmup, tpd, seeds, rng) -> dict:
    """`measure`, with all (scene, seed) brains in one GPU batch.

    Same design and noise streams as the CPU; only the summation order of the recurrent
    input differs (see models/torch_lif.py), so a run uses one backend throughout.
    """
    import torch
    from flymario.models.torch_lif import TorchLIF

    W, gain, mass = build_operator(graph, log_weights, normalise, fraction)
    names = sorted(bank.sequences)
    C, S = len(names), len(seeds)
    brain_seeds = [int(sd) for _ in range(C) for sd in seeds]       # CPU row order
    key = (id(graph), bool(log_weights))
    b = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE_FRACTION), batch=C * S,
                 normalize=normalise, seeds=brain_seeds, noise="numpy", cache_key=key)
    dev = b.device
    motor = torch.as_tensor(np.asarray(motor_idx), device=dev, dtype=torch.long)
    # The interface is stateful (it sees change): each scene is encoded from a reset, as on CPU.
    per_frame = []
    for k in range(2):
        cols = []
        for name in names:
            fr = bank.sequences[name]
            mid = fr[len(fr) // 2: len(fr) // 2 + 2]
            vision.reset()
            for j in range(k + 1):
                c = vision.encode(mid[j])
            cols.append(c)
        per_frame.append(torch.from_numpy(np.stack(cols, 1)).to(dev)
                         .repeat_interleave(S, dim=1))
    y = np.repeat(np.arange(C), S)
    grp = np.tile(np.asarray(seeds), C)

    def collect(drive: bool):
        b.reset(seeds=brain_seeds)
        for _ in range(warmup):
            b.step(None)
        counts = torch.zeros((len(motor_idx), C * S), device=dev)
        for k in range(2):
            for _ in range(tpd):
                sp_ = b.step(per_frame[k] if drive else None)
                counts += sp_.index_select(0, motor).float()
        return counts.t().cpu().numpy().astype(np.float32)

    X, Xn = collect(True), collect(False)
    info = {"real": decode(X, y, grp), "shuffled": decode(X, rng.permutation(y), grp),
            "no_input": decode(Xn, y, grp), "active_units": int((X.sum(axis=0) > 0).sum())}

    # regional profile: one brain, seed 0, the first scene cycling -- as on CPU
    one = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE_FRACTION), batch=1,
                   normalize=normalise, seeds=[0], noise="numpy", cache_key=key)
    first = bank.sequences[names[0]]
    vision.reset()
    warmed = 0
    while warmed < warmup:
        for f in first:
            cur = vision.encode(f)
            for _ in range(tpd):
                one.step(cur); warmed += 1
                if warmed >= warmup:
                    break
            if warmed >= warmup:
                break
    glob, per_region = [], {k: [] for k in reg}
    reg_t = {k: torch.as_tensor(v, device=dev, dtype=torch.long) for k, v in reg.items()}
    vision.reset()
    for f in first:
        cur = vision.encode(f)
        for _ in range(tpd):
            sp_ = one.step(cur)[:, 0]
            glob.append(float(sp_.float().mean()))
            for k, idx in reg_t.items():
                per_region[k].append(float(sp_[idx].float().mean()) if idx.numel() else np.nan)
    return {
        "gain_implied": round(gain, 6), "median_row_mass": round(mass, 4),
        "decode": info["real"]["accuracy"],
        "decode_chance": info["real"].get("chance", 1.0 / C),
        "shuffled": info["shuffled"]["accuracy"], "no_input": info["no_input"]["accuracy"],
        "active_units": info["active_units"], "global_rate": float(np.mean(glob)),
        "region_rates": {k: float(np.nanmean(v)) for k, v in per_region.items()},
    }


def regime_gpu(graph, logw, norm, vision, bank, warmup, tpd, seed) -> list[float]:
    """The whole regime curve as ONE batch: a brain per coupling setting, all
    sharing the seed's noise stream, each with its own gain."""
    from flymario.models.torch_lif import TorchLIF

    W, _, mass = build_operator(graph, logw, norm, 1.0)
    gains = np.array([1.0 / (f * mass) for f in REGIME_FRACTIONS], np.float32)
    k = len(REGIME_FRACTIONS)
    b = TorchLIF(W, LIFParams(noise_std=NOISE_FRACTION), batch=k, normalize=norm,
                 seeds=[int(seed)] * k, noise="numpy", gains=gains,
                 cache_key=(id(graph), bool(logw)))
    frames = bank.sequences[sorted(bank.sequences)[0]]
    vision.reset()
    t = 0
    while t < warmup:
        for f in frames:
            cur = vision.encode(f)
            for _ in range(tpd):
                b.step(cur); t += 1
    vision.reset()
    acc = None
    n = 0
    for f in frames:
        cur = vision.encode(f)
        for _ in range(tpd):
            m = b.step(cur).float().mean(0)
            acc = m if acc is None else acc + m
            n += 1
    return (acc / n).cpu().numpy().astype(float).tolist()


def drive_rate(b: LIFBrain, vision, frames, warmup: int, tpd: int) -> float:
    """Global firing rate under real frames, measured after the warm-up."""
    vision.reset()
    rates, t = [], 0
    while t < warmup:                      # warm up on the same frames
        for f in frames:
            cur = vision.encode(f)
            for _ in range(tpd):
                b.step(cur); t += 1
    vision.reset()
    for f in frames:                       # then measure one full pass
        cur = vision.encode(f)
        for _ in range(tpd):
            rates.append(float(b.step(cur).mean()))
    return float(np.mean(rates))


def calibrate_fraction(graph, logw: bool, norm: bool, vision, bank, warmup: int,
                       tpd: int, target: float, tol: float = 0.005,
                       lo: float = 0.005, hi: float = 0.5, iters: int = 12,
                       seed: int = 1, backend: str = "cpu") -> dict:
    """Find the coincidence fraction at which this arm fires at `target` (docs/PROTOCOL.md §9.5).

    Bisects on log(fraction); rate falls as the fraction rises. If the target is out of reach
    in [lo, hi], returns `matched: False` with the closest rate found.
    """
    frames = bank.sequences[sorted(bank.sequences)[0]]
    W, _, mass = build_operator(graph, logw, norm, 1.0)
    W = normalize_in_degree(W) if norm else W.tocsc()

    def rate_at(f: float) -> float:
        if backend == "gpu":
            from flymario.models.torch_lif import TorchBrain
            b = TorchBrain(W, LIFParams(gain=1.0 / (f * mass), noise_std=NOISE_FRACTION,
                                        seed=seed), normalize=False)
        else:
            b = LIFBrain(W, LIFParams(gain=1.0 / (f * mass), noise_std=NOISE_FRACTION,
                                      seed=seed), normalize=False)
        return drive_rate(b, vision, frames, warmup, tpd)

    trace = []
    r_lo, r_hi = rate_at(lo), rate_at(hi)
    trace += [(lo, r_lo), (hi, r_hi)]
    if not (r_hi <= target <= r_lo):
        f, r = min(trace, key=lambda t: abs(t[1] - target))
        return {"fraction": f, "rate": r, "matched": False, "trace": trace,
                "note": f"target {target:.3f} outside reachable [{r_hi:.3f}, {r_lo:.3f}]"}
    a, b_ = np.log(lo), np.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b_)
        f = float(np.exp(m))
        r = rate_at(f)
        trace.append((f, r))
        if abs(r - target) <= tol:
            break
        if r > target:
            a = m          # too active: raise the fraction (lower the gain)
        else:
            b_ = m
    f, r = min(trace, key=lambda t: abs(t[1] - target))
    return {"fraction": f, "rate": r, "matched": bool(abs(r - target) <= tol),
            "trace": trace}


def load_or_build_null(kind: str, g, seed: int, dataset: str) -> tuple:
    """A null realisation from the shared cache, or built, saved and verified."""
    from flymario.connectome import nullcache
    return nullcache.load_or_build(kind, g, seed, dataset)


# ---------------------------------------------------------------------------
# Parallel execution: every task is fully determined by its key, so parallelism changes no
# number. Workers inherit state by fork and never write; only the main process saves.
# ---------------------------------------------------------------------------
_W: dict = {}


def _worker_init() -> None:
    # Workers inherit RunControl's handlers: reset them so pool.terminate() kills a worker
    # outright and Ctrl-C / SIGUSR1 belong to the main process only.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def run_task(task: tuple) -> tuple:
    t0 = time.perf_counter()
    if _W.get("started") is not None:
        _W["started"].put(task_key(task))   # lights the task up on the run map
    kind = task[0]
    V = {v[0]: v for v in VARIANTS}
    if kind == "regime":
        _, tname, arm, seed = task
        _, logw, norm = V[tname]
        if _W.get("backend") == "gpu":
            rates = regime_gpu(_W["arms"][arm], logw, norm, _W["vision"], _W["bank"],
                               _W["warmup"], _W["tpd"], seed)
            return task, {"fractions": list(REGIME_FRACTIONS), "rates": rates}, \
                time.perf_counter() - t0
        frames = _W["bank"].sequences[sorted(_W["bank"].sequences)[0]]
        W, _, mass = build_operator(_W["arms"][arm], logw, norm, 1.0)
        W = normalize_in_degree(W) if norm else W.tocsc()
        rates = []
        for f in REGIME_FRACTIONS:
            b = LIFBrain(W, LIFParams(gain=1.0 / (f * mass), noise_std=NOISE_FRACTION,
                                      seed=int(seed)), normalize=False)
            rates.append(drive_rate(b, _W["vision"], frames, _W["warmup"], _W["tpd"]))
        return task, {"fractions": list(REGIME_FRACTIONS), "rates": rates}, \
            time.perf_counter() - t0

    _, comparison, tname, arm = task
    _, logw, norm = V[tname]
    ci = COMPARISONS.index(comparison)
    vi = [v[0] for v in VARIANTS].index(tname)
    ai = list(_W["arms"]).index(arm)
    graph = _W["arms"][arm]
    if comparison == "matched_regime":
        cal = calibrate_fraction(graph, logw, norm, _W["vision"], _W["bank"],
                                 _W["warmup"], _W["tpd"], _W["target"],
                                 backend=_W.get("backend", "cpu"))
        frac = cal["fraction"]
        if not cal["matched"]:
            # No state at the target rate exists for this arm: nothing is measured.
            return task, {"transform": tname, "arm": arm, "comparison": comparison,
                          "fraction_used": frac, "calibration": cal,
                          "infeasible": True, "decode": None,
                          "log_weights": logw, "normalised": norm}, \
                time.perf_counter() - t0
    else:
        cal, frac = None, _W["fraction"]
    # One generator per measurement, derived from its place in the design, so the
    # shuffled control does not depend on what ran before or beside it.
    rng = np.random.default_rng([ci, vi, ai])
    fn = measure_gpu if _W.get("backend") == "gpu" else measure
    r = fn(graph, logw, norm, frac, _W["vision"], _W["bank"], _W["motor_idx"],
           _W["reg"], _W["warmup"], _W["tpd"], _W["seeds"], rng)
    r.update(transform=tname, arm=arm, comparison=comparison, fraction_used=frac,
             calibration=cal, log_weights=logw, normalised=norm)
    return task, r, time.perf_counter() - t0


def scene_png(frame: np.ndarray, vision, title: str) -> bytes:
    """The game frame, then what the eye's ON and OFF cells receive from it (env/vision.py).

    Each dot is one injected cell at its measured eye position; ON and OFF are drawn in
    separate panels because they share positions.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import LinearSegmentedColormap

    # From the panel colour to a vivid one: no current = invisible, strong current = bright.
    on_map = LinearSegmentedColormap.from_list("on", ["#0f1116", "#ffab3c", "#fff1d6"])
    off_map = LinearSegmentedColormap.from_list("off", ["#0f1116", "#3987e5", "#d6e8ff"])
    vision.reset()
    cur = vision.encode(frame)           # first frame after a reset: no change term
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.9), dpi=110, facecolor="#0f1116")
    axes[0].imshow(frame); axes[0].set_axis_off()
    for ax, idx, uv, cmap, name in ((axes[1], vision.on_idx, vision.on_uv, on_map, "ON · light"),
                                    (axes[2], vision.off_idx, vision.off_uv, off_map, "OFF · dark")):
        c = cur[idx]
        ax.set_facecolor("#0f1116"); ax.set_axis_off()
        ax.scatter(uv[:, 0], 1 - uv[:, 1], c=c, cmap=cmap, s=4, linewidths=0,
                   vmin=0, vmax=max(float(c.max()), 1e-6))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(name, color="#b4bccb", fontsize=6.5, pad=2)
    fig.suptitle(title, color="#b4bccb", fontsize=7, y=0.99)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.84, 0.04, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#0f1116")
    plt.close(fig)
    return buf.getvalue()


def task_key(task: tuple) -> str:
    return "|".join(str(x) for x in task)


def curve_summary(rates: list[float], target: float) -> dict:
    """What a rate-versus-fraction curve says about the operating regime.

    "Usable" (declared before the powered run): at least three consecutive grid points
    within a factor two of the target rate.
    """
    lo, hi = target / 2, target * 2
    best = run = 0
    for r in rates:
        run = run + 1 if lo <= r <= hi else 0
        best = max(best, run)
    drops = [rates[i] - rates[i + 1] for i in range(len(rates) - 1)]
    k = int(np.argmax(drops))
    return {"consecutive_in_band": best, "usable_regime": bool(best >= 3),
            # the largest fall between two neighbouring grid points: a cliff
            # shows up as one step carrying most of the whole curve's range
            "largest_drop": float(drops[k]),
            "largest_drop_at": [REGIME_FRACTIONS[k], REGIME_FRACTIONS[k + 1]],
            "drop_share": float(drops[k] / max(rates[0] - rates[-1], 1e-9))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--null", default="depth_stratified")
    # Both defaults come from the frozen config; fraction = 1 / gain on the normalised operator.
    p.add_argument("--fraction", type=float, default=None,
                   help="coincidence fraction; default: the frozen gain of "
                        "configs/base.yaml, converted for the configured operator "
                        "(1 / (gain x median row mass); = 1/gain when normalised)")
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--warmup", type=int,
                   default=int(config.get("dynamics.warmup_ticks")))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--null-seeds", type=int, nargs="+", default=[0],
                   help="one null realisation per seed; more is better")
    p.add_argument("--regime-seeds", type=int, nargs="+", default=[1, 2],
                   help="noise seeds per regime curve: tells a cliff that belongs "
                        "to the wiring from one that belongs to a noise draw")
    p.add_argument("--bank", default="calibration")
    p.add_argument("--transforms", nargs="+", default=[v[0] for v in VARIANTS],
                   choices=[v[0] for v in VARIANTS],
                   help="restrict the measured variants (smoke tests)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel processes; the LIF step is single-threaded, so "
                        "throughput scales with this until memory runs out")
    p.add_argument("--backend", choices=["cpu", "gpu"],
                   default=config.get("dynamics.backend", "cpu"),
                   help="gpu: all (scene, seed) brains of a measurement in one batch "
                        "on the GPU. Part of the run's identity -- never mixed.")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="use seeds offset+1 .. offset+N: a replication on seeds the "
                        "first run never saw (the null-realisation floor does not "
                        "contain seed-set variability)")
    p.add_argument("--comparisons", nargs="+", default=list(COMPARISONS),
                   choices=list(COMPARISONS))
    p.add_argument("--skip-regime", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.backend == "gpu":
        # One device: several processes would only time-slice it. Workers are
        # forked before any CUDA initialisation in this process.
        args.workers = 1

    t0 = time.perf_counter()
    g = datasets.load(args.dataset)
    if args.fraction is None:
        wt = config.get("dynamics.weight_transform")
        _, _, mass = build_operator(g, bool(wt["log1p"]), bool(wt["normalize_in_degree"]), 1.0)
        args.fraction = round(1.0 / (float(config.get("dynamics.gain")) * mass), 6)
    # This process's brain stays on the CPU so CUDA is first touched inside a worker.
    cfg = AgentConfig(backend="cpu")
    agent = FlyAgent(g, cfg)
    reg = regions(g, args.dataset)
    tpd = cfg.ticks_per_decision
    seeds = np.arange(1 + args.seed_offset, args.seeds + 1 + args.seed_offset)
    bank = stimulus_bank.build(which=args.bank, length=args.length)
    target = float(config.get("dynamics.target_rate"))

    print(f"{g.source} · null {args.null} · fraction {args.fraction} "
          f"· {tpd} ticks/decision · bank {args.bank} · {args.workers} workers")
    print("no reward, score, completion or distance enters anything below\n")

    arms = {"connectome": g}
    for sd in args.null_seeds:
        t = time.perf_counter()
        arms[f"{args.null}_s{sd}"], how = load_or_build_null(args.null, g, sd,
                                                             args.dataset)
        print(f"null realisation seed {sd}: {how} ({time.perf_counter() - t:.0f} s)",
              flush=True)

    # Everything that decides what a measurement means; a partial file is resumed only if
    # these match exactly.
    params = {"dataset": args.dataset, "null": args.null,
              "fraction": args.fraction, "length": args.length,
              "warmup_ticks": args.warmup, "seeds": [int(x) for x in seeds],
              "null_seeds": list(args.null_seeds), "bank": args.bank,
              "ticks_per_decision": tpd, "noise": NOISE_FRACTION,
              "weight_transform_in_config": config.get("dynamics.weight_transform"),
              "comparisons": list(COMPARISONS), "transforms": list(args.transforms),
              "target_rate": target, "regime_seeds": list(args.regime_seeds),
              "regime_fractions": list(REGIME_FRACTIONS), "backend": args.backend,
              "run_comparisons": list(args.comparisons), "skip_regime": args.skip_regime}
    rep = {"params": params, "bank": bank.describe(), "complete": False,
           "regime_curves": {}, "results": []}

    # Flush after every result: a run of this size outlives the session that
    # started it. `complete` stays False until the end, so a partial file is
    # never mistaken for a finished one.
    RUNS.mkdir(parents=True, exist_ok=True)
    out = (Path(args.output) if args.output
           else RUNS / f"transform_exposure_{args.dataset}.json")

    def flush() -> None:
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=2))
        tmp.replace(out)  # atomic: a reader never sees a half-written file

    # Resume, or set an incompatible old file aside -- never silently overwrite.
    resumed = False
    if out.exists():
        old_rep = json.loads(out.read_text())
        if old_rep.get("params") == params and not old_rep.get("complete"):
            rep, resumed = old_rep, True
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            aside = out.with_name(f"{out.stem}.superseded_{stamp}.json")
            out.replace(aside)
            print(f"previous output has different parameters -> kept as {aside.name}")

    # --- the work, as independent tasks -------------------------------------
    arm_names = list(arms)
    tasks = [] if args.skip_regime else [("regime", t, a, sd) for t in args.transforms
                                         for a in arm_names for sd in args.regime_seeds]
    tasks += [("measure", c, t, a) for c in args.comparisons for t in args.transforms
              for a in arm_names]
    finished = set(rep["regime_curves"]) | {
        task_key(("measure", r["comparison"], r["transform"], r["arm"]))
        for r in rep["results"]}
    todo = [t for t in tasks if task_key(t) not in finished]
    if resumed:
        print(f"resuming: {len(tasks) - len(todo)} of {len(tasks)} tasks already on disk")

    # --- live control and curves (make follow) -----------------------------
    from flymario.runctl import RunControl, StopRequested
    ctl = RunControl(
        out.stem, total=len(tasks), checkpoint=flush, unit="task",
        already_done=len(tasks) - len(todo), fresh=not resumed,
        meta={
            "question": ("Does the difference between the real fly wiring and a "
                         "rewired copy of it survive the way synapse counts are "
                         "turned into connection strengths?"),
            "why": ("Four reasonable ways exist (compress strong synapses or not; "
                    "give every neuron the same total input or not). The project "
                    "used one without ever justifying it. If the real-vs-rewired "
                    "result flips between them, every structural conclusion depends "
                    "on that unjustified choice — which is itself a result "
                    "(protocol H5)."),
            "method": ("First, for each variant, how the firing rate responds to "
                       "the coupling strength: a gentle slope can be tuned, a cliff "
                       "cannot. Then how much of the visual input a decoder can "
                       "read from the motor neurons, compared twice (§9.5): with "
                       "identical settings, and with each network tuned to the "
                       "same 7% firing rate. No game score enters anything."),
            "params": params})
    ctl.parallel = max(1, args.workers)
    # Charts are declarations, not data: declared on every launch, including resumes.
    if True:
        for t in args.transforms:
            ctl.chart(f"regime_{t}", f"Operating regime · {t}",
                      "coupling (log scale) → weaker",
                      "global firing rate", ["connectome", "null"], logx=True,
                      yfmt="pct",
                      refs=[{"axis": "y", "value": target, "label": "target 7%"},
                            {"axis": "x", "value": args.fraction, "label": "frozen"}],
                      note=("One line per noise seed (and per rewired copy). A "
                            "gentle slope can be tuned to any rate; a cliff means "
                            "the network is either on or off."))
        for c, word in (("shared_parameter", "same settings"),
                        ("matched_regime", "same 7% firing rate")):
            ctl.chart(f"decode_{c}", f"Visual information at the motor neurons · {word}",
                      "weight transform", "decoder accuracy", ["connectome", "null"],
                      kind="points",
                      refs=[{"axis": "y", "value": 1.0 / len(bank.sequences),
                             "label": "chance"}],
                      note=("How often a decoder names which of the 10 scenes was "
                            "shown, from motor activity alone. Dots above the "
                            "chance line mean the picture got through."))

    if True:
        ctl.chart("regions", "Where the activity sits · same settings",
                  "brain region", "firing rate", ["connectome", "null"],
                  kind="points", yfmt="pct",
                  note=("Each dot is one measurement (one transform, one copy of "
                        "the wiring). Optic = the eye's first layers; motor = the "
                        "neurons the controller reads."))
    # The ten scenes, as the fly's eye receives them.
    for i, name in enumerate(sorted(bank.sequences)):
        fr = bank.sequences[name]
        ctl.image(f"scene_{i:02d}.png",
                  scene_png(fr[len(fr) // 2], agent.vision, f"scene {i + 1}: {name}"),
                  caption=f"scene {i + 1}", group="What the fly sees")
    for t in tasks:
        ctl.task(task_key(t), "done" if task_key(t) in finished else "todo")

    def publish(task, res) -> None:
        if task[0] == "regime":
            _, t, a, sd = task
            series = "connectome" if a == "connectome" else "null"
            for f, r in zip(res["fractions"], res["rates"]):
                ctl.point(f"regime_{t}", series, f, r, line=f"{a} · seed {sd}")
        elif res.get("decode") is not None:
            series = "connectome" if res["arm"] == "connectome" else "null"
            ctl.point(f"decode_{res['comparison']}", series, res["transform"],
                      res["decode"], line=res["arm"], arm=res["arm"],
                      rate=round(res["global_rate"], 4),
                      fraction=round(res["fraction_used"], 4))
            if res["comparison"] == "shared_parameter":
                for region, r in res["region_rates"].items():
                    if r == r:   # skip NaN (a region with no cells)
                        ctl.point("regions", series, region, r, line=res["arm"],
                                  transform=res["transform"], arm=res["arm"])

    # --- run ----------------------------------------------------------------
    import multiprocessing as mp
    started = mp.get_context("fork").Queue()
    _W.update(arms=arms, vision=agent.vision, bank=bank, motor_idx=agent.motor_idx,
              reg=reg, warmup=args.warmup, tpd=tpd, seeds=seeds, target=target,
              fraction=args.fraction, started=started, backend=args.backend)
    pool = mp.get_context("fork").Pool(max(1, args.workers), initializer=_worker_init)
    inflight = {task_key(t): (t, pool.apply_async(run_task, (t,))) for t in todo}
    try:
        while inflight:
            ctl.poll()              # checkpoint / stop requests land here
            while not started.empty():
                ctl.task(started.get_nowait(), "running")
            ctl.beat(f"{len(inflight)} task(s) queued or in flight "
                     f"on {args.workers} workers")
            ready = [k for k, (_, ar) in inflight.items() if ar.ready()]
            for k in ready:
                task, res, secs = inflight.pop(k)[1].get()
                if task[0] == "regime":
                    rep["regime_curves"][k] = res
                    summ = curve_summary(res["rates"], target)
                    line = (f"regime  {task[1]:<9} {task[2]:<22} seed {task[3]}  "
                            f"in-band run {summ['consecutive_in_band']}  largest "
                            f"drop {summ['largest_drop'] * 100:.1f} pts")
                    brief = {"in_band": summ["consecutive_in_band"],
                             "largest_drop_pts": round(summ["largest_drop"] * 100, 1)}
                else:
                    rep["results"].append(res)
                    if res.get("decode") is None:
                        line = (f"{task[1][:7]} {task[2]:<9} {task[3]:<22} "
                                f"INFEASIBLE: {res['calibration'].get('note') or 'no state at the target rate'}")
                        brief = {"decode": None, "infeasible": True}
                    else:
                        line = (f"{task[1][:7]} {task[2]:<9} {task[3]:<22} decode "
                                f"{res['decode']:.3f}  rate {res['global_rate'] * 100:.2f}%"
                                f"  fraction {res['fraction_used']:.4f}")
                        brief = {"decode": round(res["decode"], 3),
                                 "rate_pct": round(res["global_rate"] * 100, 2)}
                flush()
                publish(task, res)
                ctl.task(k, "infeasible" if (task[0] == "measure"
                                             and res.get("decode") is None) else "done")
                ctl.done(brief, seconds=secs, label=" · ".join(str(x) for x in task),
                         kind=task[0])
                print(line, flush=True)
            time.sleep(0.5)
    except StopRequested:
        pool.terminate()
        pool.join()
        flush()
        ctl.close("stopped")
        print(f"\nstopped cleanly: {ctl.completed}/{len(tasks)} tasks on disk. "
              "Relaunch the same command to resume.")
        return
    pool.close()
    pool.join()

    # --- the regime, summarised ---------------------------------------------
    print("\noperating regime: usable = >=3 consecutive grid points within x2 of "
          f"{target:.0%}; largest drop between neighbouring points")
    rep["regime_summary"] = {}
    for t in ([] if args.skip_regime else args.transforms):
        for a in arm_names:
            curves = [rep["regime_curves"][task_key(("regime", t, a, sd))]["rates"]
                      for sd in args.regime_seeds]
            summ = [curve_summary(c, target) for c in curves]
            rep["regime_summary"][f"{t}|{a}"] = {
                "usable_every_seed": all(x["usable_regime"] for x in summ),
                "largest_drop_pts": [round(x["largest_drop"] * 100, 2) for x in summ],
                "drop_share": [round(x["drop_share"], 3) for x in summ],
                "largest_drop_at": [x["largest_drop_at"] for x in summ]}
            rs = rep["regime_summary"][f"{t}|{a}"]
            print(f"  {t:<9} {a:<22} usable {'yes' if rs['usable_every_seed'] else 'NO '}"
                  f"  largest drop {rs['largest_drop_pts']} pts"
                  f"  ({[f'{x:.0%}' for x in rs['drop_share']]} of the curve's range)")

    # --- the contrast, per comparison and transform ------------------------
    by_cmp: dict[str, dict[str, dict]] = {c: {} for c in COMPARISONS}
    for r in rep["results"]:
        by_cmp[r["comparison"]].setdefault(r["transform"], {})[r["arm"]] = r
    rep["contrasts"], rep["summary"] = {}, {}
    for comparison in args.comparisons:
        by_transform = by_cmp[comparison]
        print(f"\n[{comparison}]")
        print(f"{'transform':<11}{'contrast':>10}{'  range over nulls':>20}"
              f"{'null sd':>8}   null realisations")
        contrasts = {}
        for tname in args.transforms:
            arms_r = by_transform.get(tname, {})
            infeasible = [k for k, v in arms_r.items() if v.get("decode") is None]
            if infeasible:
                # §9.5: a matched-regime contrast needs every arm at the target.
                contrasts[tname] = {"infeasible_arms": infeasible, "resolved": False}
                print(f"{tname:<11}{'—':>10}   infeasible: {len(infeasible)} arm(s) "
                      f"have no state at {target:.0%}")
                continue
            c = arms_r["connectome"]["decode"]
            nulls = [v["decode"] for k, v in arms_r.items() if k != "connectome"]
            nm = float(np.mean(nulls))
            d = c - nm
            # Spread between realisations of the same null: the floor any contrast has to
            # clear (rough with few realisations; reported, not trusted).
            null_spread = float(np.ptp(nulls)) if len(nulls) > 1 else float("nan")
            null_sd = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else float("nan")
            # 2*SD is a readability threshold for the table, not a test (exploratory).
            exceeds_2sd = bool(len(nulls) > 1 and abs(d) > 2 * null_sd)
            alt = [c - x for x in nulls]
            lo, hi = ((float(np.min(alt)), float(np.max(alt))) if len(nulls) > 1
                      else (d, d))
            rate_gap = (arms_r["connectome"]["global_rate"]
                        - float(np.mean([v["global_rate"] for k, v in arms_r.items()
                                         if k != "connectome"])))
            contrasts[tname] = {"connectome": c, "null_mean": nm, "difference": d,
                                "null_values": [float(x) for x in nulls],
                                "null_spread": null_spread, "null_sd": null_sd,
                                "contrast_range": [lo, hi],
                                "exceeds_2sd": exceeds_2sd,
                                # under shared parameters the rates differ, and
                                # that difference travels with the contrast
                                "rate_gap_connectome_minus_null": rate_gap,
                                "resolved": exceeds_2sd}
            nulls_txt = " ".join(f"{x:.2f}" for x in sorted(nulls))
            print(f"{tname:<11}{d:>+10.3f}  [{lo:+.3f},{hi:+.3f}]{null_sd:>8.3f}"
                  f"   {nulls_txt}   (rate gap {rate_gap * 100:+.1f} pts)")

        feasible = {t: v for t, v in contrasts.items() if "difference" in v}
        diffs = np.array([v["difference"] for v in feasible.values()]) \
            if feasible else np.array([np.nan])
        resolved = {t: v for t, v in feasible.items() if v["resolved"]}
        signs = {np.sign(v["difference"]) for v in resolved.values()}
        sign_stable = bool(len(signs - {0.0}) <= 1) if resolved else None
        rep["contrasts"][comparison] = contrasts
        rep["summary"][comparison] = {
            "n_feasible": len(feasible), "n_resolved": len(resolved),
            "sign_stable": sign_stable,
            "difference_range": [float(np.nanmin(diffs)), float(np.nanmax(diffs))]}
        print("-" * 70)
        print(f"{comparison}: EXPLORATORY DIAGNOSTIC. Nothing here is a test.")
        print(f"  feasible {len(feasible)}/{len(contrasts)} · resolved "
              f"{len(resolved)}/{len(feasible)} · sign stable among resolved: {sign_stable}")

    print("=" * 70)
    print("Read the intervals, not the point estimates. A sign only means")
    print("something where the whole interval shares it, across every transform,")
    print("and that still needs confirming separately.")

    graph_path = datasets.cache_path(args.dataset)
    rep["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": git_sha(), "python": platform.python_version(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "wall_seconds": round(time.perf_counter() - t0, 1), "argv": vars(args),
    }
    rep["complete"] = True
    flush()
    ctl.close("complete")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
