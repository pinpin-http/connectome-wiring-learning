#!/usr/bin/env python
"""Gate E, re-certified for the weight transform the config now declares.

    python scripts/gates/gate_e_recert.py            # GPU, live: make follow

Couplings are swept as a coincidence fraction f: gain = 1 / (f x median absolute input per neuron).
Criteria, declared before running: RATE within +-0.5 pt of dynamics.target_rate on real frames, motor
> 0.2 %, no region > 30 %; INFORMATION within 2 binomial SE of the sweep's best and above both controls
by > 2 SE; CHOICE the qualifying coupling closest to the target rate, which must meet RATE, else FAIL.
Progress and curves under artifacts/runs/live/gate_e_recert/ (make follow / make stop).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from flymario import config
from flymario.connectome import datasets
from flymario.dynamics.calibrate import interface_drive, regions
from flymario.env import stimulus_bank
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.models.torch_lif import TorchLIF
from flymario.paths import ROOT, RUNS
from flymario.runctl import RunControl, StopRequested

FRACTIONS = (0.006, 0.008, 0.010, 0.012, 0.014, 0.0158, 0.018, 0.021, 0.025, 0.030, 0.040)
RATE_TOL = 0.005           # +- 0.5 point around the target
MEASURE_TICKS = 60         # as calibrate.measure
NOISE = 0.02


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def decode(X, y, groups, folds: int = 4) -> float:
    """decode_interface.decode, unchanged: the instrument Gate E was cleared with."""
    n = min(folds, len(np.unique(groups)))
    accs = []
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--seeds", type=int, default=12, help="as the plan-B certification")
    p.add_argument("--rate-seeds", type=int, default=6)
    p.add_argument("--bank", default="calibration")
    # A confirmation on fresh seeds: the peak of 11 noisy points is biased up
    # (winner's curse), so the chosen point and its neighbours are re-measured
    # on seeds the choice never saw.
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--name", default="gate_e_recert")
    args = p.parse_args()

    t0 = time.perf_counter()
    g = datasets.load(args.dataset)
    wt = config.get("dynamics.weight_transform")
    W = g.to_sparse()                       # the configured transform
    norm = bool(wt["normalize_in_degree"])
    rows = np.abs(W).sum(axis=1).A.ravel()
    mass = 1.0 if norm else float(np.median(rows[rows > 0]))
    target = float(config.get("dynamics.target_rate"))
    warm = int(config.get("dynamics.warmup_ticks"))
    tpd = int(config.get("timing.ticks_per_decision"))
    reg = regions(g, args.dataset)
    read = datasets.motor_index(g, args.dataset)
    bank = stimulus_bank.build(which=args.bank)
    vision = RetinotopicVision(g)
    names = sorted(bank.sequences)
    seeds = list(range(1 + args.seed_offset, args.seeds + 1 + args.seed_offset))
    rate_seeds = list(range(1 + args.seed_offset, args.rate_seeds + 1 + args.seed_offset))
    fractions = tuple(args.fractions)
    gains = {f: 1.0 / (f * mass) for f in fractions}

    print(f"{g.source} · transform {wt} · median input {mass:.1f} · target {target:.1%} · "
          f"{len(names)} scenes x {len(seeds)} seeds per coupling · GPU")

    out = RUNS / f"{args.name}.json"
    rep = {"params": {"fractions": list(fractions), "mass": mass, "transform": wt,
                      "seeds": seeds, "rate_seeds": rate_seeds, "target": target,
                      "warmup": warm, "tpd": tpd, "backend": "gpu", "bank": args.bank},
           "rate": {}, "info": {}, "complete": False}
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("params") == rep["params"] and not old.get("complete"):
            rep = old                       # resume

    def flush():
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=2))
        tmp.replace(out)

    tasks = [f"rate|{f}" for f in fractions] + [f"info|{f}" for f in fractions]
    done = {f"rate|{k}" for k in rep["rate"]} | {f"info|{k}" for k in rep["info"]}
    ctl = RunControl(args.name, total=len(tasks), checkpoint=flush, unit="task",
                     already_done=len(done), fresh=not done, meta={
        "question": ("Where should the brain run, now that synapse counts are used as "
                     "strengths directly (lin_raw)?"),
        "why": ("The old operating point (gain 20) belonged to the old conversion. Gate E "
                "freezes the point where two things agree: the brain fires at its target "
                "rate under the real game pictures, AND a decoder can best tell the scenes "
                "apart from its motor neurons."),
        "method": ("Sweep the coupling; at each, measure the firing rate under real frames "
                   "and the scene decodability (10 scenes x 12 seeds, held-out seeds, two "
                   "controls). Choice rule written in the script before running.")})
    for t in tasks:
        ctl.task(t, "done" if t in done else "todo")
    ctl.chart("rate", "Firing rate under real game frames", "coupling (fraction, log)",
              "global firing rate", ["whole brain", "motor"], logx=True, yfmt="pct",
              refs=[{"axis": "y", "value": target, "label": "target"}],
              note="Weaker coupling to the right. The target band is ±0.5 point.")
    ctl.chart("info", "Can a decoder tell the scenes apart?", "coupling (fraction, log)",
              "decoder accuracy", ["scenes", "shuffled labels", "no input"], logx=True,
              refs=[{"axis": "y", "value": 1 / len(names), "label": "chance"}],
              note="The two controls must stay at chance, or the number measures something else.")

    # the interface drive, as calibrate.measure uses it: one sequence of frames
    at = interface_drive(g, args.dataset, bank=bank)
    drive_seq = [at(t) for t in range(warm + MEASURE_TICKS)]
    # the decodability stimuli: each scene's first frame, encoded from a reset
    enc = []
    for nm in names:
        vision.reset()
        enc.append(vision.encode(bank.sequences[nm][0]))
    import torch
    reg_t = None

    try:
        for f in fractions:
            key = f"rate|{f}"
            if key in done:
                continue
            ctl.begin(f"rate at fraction {f}")
            ctl.task(key, "running")
            b = TorchLIF(W, LIFParams(gain=gains[f], noise_std=NOISE), batch=len(rate_seeds),
                         normalize=norm, seeds=rate_seeds, noise="numpy", cache_key=("gate_e", 0))
            if reg_t is None:
                reg_t = {k: torch.as_tensor(v, device=b.device, dtype=torch.long) for k, v in reg.items()}
            for t in range(warm):
                b.step(drive_seq[t])
                if t % 25 == 0:
                    ctl.sub(t, warm + MEASURE_TICKS, "tick")
            glob, per = [], {k: [] for k in reg}
            for t in range(warm, warm + MEASURE_TICKS):
                s = b.step(drive_seq[t]).float()
                glob.append(s.mean(0).cpu().numpy())
                for k, idx in reg_t.items():
                    per[k].append(s[idx].mean(0).cpu().numpy() if idx.numel() else np.full(len(rate_seeds), np.nan))
            gseed = np.mean(glob, 0)                       # per seed
            rr = {k: float(np.nanmean(v)) for k, v in per.items()}
            live = [v for v in rr.values() if np.isfinite(v) and v > 0]
            rep["rate"][str(f)] = {"gain": gains[f], "global": float(gseed.mean()),
                                   "global_sd": float(gseed.std(ddof=1)), "regions": rr,
                                   "alive": rr.get("motor", 0) > 0.002,
                                   "saturated": any(v > 0.30 for v in live),
                                   "spread": float(max(live) / min(live)) if len(live) > 1 else None}
            flush()
            ctl.point("rate", "whole brain", f, gseed.mean())
            ctl.point("rate", "motor", f, rr.get("motor", 0.0))
            ctl.task(key, "done")
            ctl.done({"rate_pct": round(gseed.mean() * 100, 2)}, label=key, kind="rate")
            print(f"rate  f={f:<7} gain {gains[f]:.4f}  global {gseed.mean() * 100:6.2f}% "
                  f"± {gseed.std(ddof=1) * 100:.2f}  motor {rr.get('motor', 0) * 100:5.2f}%", flush=True)

        C, S = len(names), len(seeds)
        brain_seeds = [sd for _ in range(C) for sd in seeds]
        y = np.repeat(np.arange(C), S)
        grp = np.tile(np.asarray(seeds), C)
        cur = torch.from_numpy(np.stack(enc, 1))           # (n, C)
        for f in fractions:
            key = f"info|{f}"
            if key in done:
                continue
            ctl.begin(f"decodability at fraction {f}")
            ctl.task(key, "running")
            b = TorchLIF(W, LIFParams(gain=gains[f], noise_std=NOISE), batch=C * S,
                         normalize=norm, seeds=brain_seeds, noise="numpy", cache_key=("gate_e", 0))
            drive = cur.to(b.device).repeat_interleave(S, dim=1)
            read_t = torch.as_tensor(read, device=b.device, dtype=torch.long)

            def collect(on: bool):
                b.reset(seeds=brain_seeds)
                for t in range(warm):
                    b.step(drive if on else None)
                    if t % 25 == 0:
                        ctl.sub(t, 2 * (warm + tpd), "tick")
                c = torch.zeros((len(read), C * S), device=b.device)
                for _ in range(tpd):
                    c += b.step(drive if on else None).index_select(0, read_t).float()
                return c.t().cpu().numpy()

            X = collect(True)
            rate_last = float(b.spikes.float().mean())
            Xn = collect(False)
            real = decode(X, y, grp)
            shuf = decode(X, np.random.default_rng(1).permutation(y), grp)
            noin = decode(Xn, y, grp)
            rep["info"][str(f)] = {"gain": gains[f], "decode": real, "shuffled": shuf,
                                   "no_input": noin, "active_units": int((X > 0).any(0).sum()),
                                   "rate_at_readout": rate_last}
            flush()
            ctl.point("info", "scenes", f, real)
            ctl.point("info", "shuffled labels", f, shuf)
            ctl.point("info", "no input", f, noin)
            ctl.task(key, "done")
            ctl.done({"decode": round(real, 3)}, label=key, kind="info")
            print(f"info  f={f:<7} decode {real:.3f}  shuffled {shuf:.3f}  no input {noin:.3f}",
                  flush=True)
    except StopRequested:
        flush()
        ctl.close("stopped")
        print("stopped cleanly; relaunch to resume")
        return

    # ---- the declared choice ------------------------------------------------
    n = C * S
    best = max(v["decode"] for v in rep["info"].values())
    se_best = np.sqrt(best * (1 - best) / n)
    near = [float(f) for f, v in rep["info"].items() if v["decode"] >= best - 2 * se_best]
    rate_ok = lambda f: (abs(rep["rate"][str(f)]["global"] - target) <= RATE_TOL
                         and rep["rate"][str(f)]["alive"] and not rep["rate"][str(f)]["saturated"])
    near.sort(key=lambda f: abs(rep["rate"][str(f)]["global"] - target))
    chosen = next((f for f in near if rate_ok(f)), None)
    verdict = {"best_decode": best, "two_se": 2 * se_best, "within_2se_of_best": near,
               "chosen_fraction": chosen, "pass": chosen is not None}
    if chosen is not None:
        i = rep["info"][str(chosen)]
        se = np.sqrt(i["decode"] * (1 - i["decode"]) / n)
        ctrl_ok = i["decode"] - max(i["shuffled"], i["no_input"]) > 2 * se
        verdict.update(chosen_gain=gains[chosen], rate=rep["rate"][str(chosen)],
                       decode=i, controls_ok=bool(ctrl_ok), **{"pass": bool(ctrl_ok)})
    rep["verdict"] = verdict
    rep["complete"] = True
    rep["provenance"] = {"utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
                         "commit": git_sha(), "python": platform.python_version(),
                         "graph_sha256": hashlib.sha256(datasets.cache_path(args.dataset).read_bytes()).hexdigest(),
                         "wall_seconds": round(time.perf_counter() - t0, 1)}
    flush()
    ctl.close("complete")
    print("\n" + "=" * 70)
    print(f"best decodability {best:.3f} (2 SE = {2 * se_best:.3f}); within 2 SE of it: {sorted(near)}")
    if verdict["pass"]:
        r = verdict["rate"]
        print(f"GATE E PASS for {wt}: fraction {chosen} -> gain {gains[chosen]:.5f}; "
              f"rate {r['global'] * 100:.2f}% (target {target:.0%}); decode {verdict['decode']['decode']:.3f}")
    else:
        print("GATE E FAIL: no coupling meets both criteria -- the owner decides (see JSON)")
    print("=" * 70)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
