#!/usr/bin/env python
"""Which operating point (gain) lets a decoder tell which scene the network was shown?

Under the interface-derived drive, gain 12 is balanced but barely active and gain 20 hits the
target rate with central-brain ignition. Controls: shuffled labels and no-input must fall to chance.

    python scripts/diagnostics/decode_interface.py --gains 12 20 30
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from flymario.config import get
from flymario.connectome import datasets
from flymario.env import stimulus_bank
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFBrain, LIFParams
from flymario.paths import ROOT, RUNS


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def decode(X, y, groups, folds: int = 4) -> float:
    n = min(folds, len(np.unique(groups)))
    if n < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="malecns")
    p.add_argument("--gains", type=float, nargs="+", default=[12.0, 20.0, 30.0])
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--bank", default="calibration")
    args = p.parse_args()

    g = datasets.load(args.dataset)
    W = g.to_sparse()
    read = datasets.motor_index(g, args.dataset)
    bank = stimulus_bank.build(which=args.bank)
    vision = RetinotopicVision(g)
    tpd, warm = get("timing.ticks_per_decision"), get("dynamics.warmup_ticks")
    names = sorted(bank.sequences)
    seeds = list(range(1, args.seeds + 1))

    # Encode once: the interface is deterministic, so both arms see identical inputs.
    encoded = {}
    for nm in names:
        vision.reset()
        encoded[nm] = [vision.encode(f) for f in bank.sequences[nm]]

    print(f"{g.source} · {len(names)} classes · chance {1 / len(names):.3f} · "
          f"{len(seeds)} seeds · real interface, bank {args.bank}")
    print(f"{'gain':>6}{'decode':>9}{'shuffled':>10}{'no input':>10}"
          f"{'active':>8}{'rate':>8}")

    results, t0 = [], time.perf_counter()
    for gain in args.gains:
        def collect(drive: bool):
            X, y, grp, rates = [], [], [], []
            for ci, nm in enumerate(names):
                enc = encoded[nm]
                for sd in seeds:
                    b = LIFBrain(W, LIFParams(gain=gain, noise_std=0.02, seed=sd))
                    zero = np.zeros(g.n_nodes, np.float32)
                    for t in range(warm):
                        b.step(enc[0] if drive else zero)
                    c = np.zeros(len(read), np.float32)
                    for t in range(tpd):
                        c += b.step(enc[(t // tpd) % len(enc)] if drive else zero)[read]
                    X.append(c)
                    y.append(ci)
                    grp.append(sd)
                    rates.append(b.firing_rate)
            return (np.array(X, np.float32), np.array(y), np.array(grp),
                    float(np.mean(rates)))

        X, y, grp, rate = collect(True)
        Xn, yn, gn, _ = collect(False)
        real = decode(X, y, grp)
        shuf = decode(X, np.random.default_rng(1).permutation(y), grp)
        noin = decode(Xn, yn, gn)
        active = int((X > 0).any(axis=0).sum())
        print(f"{gain:6.0f}{real:9.3f}{shuf:10.3f}{noin:10.3f}"
              f"{active:8d}{rate * 100:7.2f}%")
        results.append({"gain": gain, "decode": real, "shuffled": shuf,
                        "no_input": noin, "active_units": active,
                        "firing_rate": rate})

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "decode_interface_gains.json"
    out.write_text(json.dumps({
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "machine": platform.node(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "args": vars(args), "chance": 1.0 / len(names),
        "classes": names, "seeds": seeds,
        "drive": "interface_drive (real frames through RetinotopicVision)",
        "results": results,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
