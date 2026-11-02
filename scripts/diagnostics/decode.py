#!/usr/bin/env python
"""Decode the readout: can it tell which stimulus was shown, on held-out noise seeds?

Replaces the distance ratio behind the retracted Gate D verdict. Controls, run by default:
shuffled (labels permuted) and no_input (drive removed); both must recover chance.
Everything needed to reproduce a run is written next to its result.

    python scripts/diagnostics/decode.py
    python scripts/diagnostics/decode.py --noise 0.0 --ticks 8 16 32
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

from flymario.connectome import datasets
from flymario.models.spiking_reservoir import LIFBrain, LIFParams
from flymario.paths import ROOT, RUNS


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_stimuli(lam: np.ndarray, n_classes: int, rng) -> list[np.ndarray]:
    """One fixed lamina subset per class. Classes differ only in which cells
    are driven, so nothing but the connectome can tell them apart."""
    return [rng.choice(lam, len(lam) // 3, replace=False) for _ in range(n_classes)]


def collect(graph, stimuli, seeds, ticks, gain, noise, warmup, amp, drive=True):
    """Readout traces. Returns (X, y, groups) with groups = the noise seed, so
    the split can hold entire seeds out rather than shuffling within them."""
    W = graph.to_sparse()
    read = datasets.motor_index(graph, "malecns" if "malecns" in graph.source else "flywire")
    n = graph.n_nodes
    base = np.zeros(n, np.float32)

    X, y, grp = [], [], []
    for ci, stim in enumerate(stimuli):
        cur = np.zeros(n, np.float32)
        if drive:
            cur[stim] = amp
        for sd in seeds:
            b = LIFBrain(W, LIFParams(gain=gain, noise_std=noise, seed=int(sd)))
            for _ in range(warmup):
                b.step(base)
            counts = np.zeros(len(read), np.float32)
            for _ in range(ticks):
                counts += b.step(cur)[read]
            X.append(counts)
            y.append(ci)
            grp.append(sd)
    return np.array(X, np.float32), np.array(y), np.array(grp)


def decode(X, y, groups, seed: int = 0) -> dict:
    """Leave-seeds-out cross-validated accuracy of a regularised linear model."""
    n_splits = min(len(np.unique(groups)), 5)
    if n_splits < 2:
        return {"accuracy": float("nan"), "note": "needs >= 2 seeds"}
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(float(clf.score(sc.transform(X[te]), y[te])))
    return {"accuracy": float(np.mean(accs)), "std": float(np.std(accs)),
            "folds": accs, "chance": 1.0 / len(np.unique(y))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="malecns")
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--ticks", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--gain", type=float, default=12.0)
    p.add_argument("--noise", type=float, nargs="+", default=[0.02, 0.0])
    p.add_argument("--amp", type=float, default=2.5)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--stim-seed", type=int, default=0)
    args = p.parse_args()

    g = datasets.load(args.dataset)
    lam = datasets.by_types(g, args.dataset, "lamina_types")
    rng = np.random.default_rng(args.stim_seed)
    stimuli = make_stimuli(lam, args.classes, rng)
    seeds = np.arange(1, args.seeds + 1)

    print(f"dataset {g.source} · {args.classes} classes · {args.seeds} seeds · "
          f"readout {len(datasets.motor_index(g, args.dataset)):,} units")
    print(f"chance = {1 / args.classes:.3f}\n")
    print(f"{'noise':>7}{'ticks':>7}{'decode':>9}{'shuffled':>10}{'no input':>10}"
          f"{'active':>8}{'verdict':>12}")

    results, t0 = [], time.perf_counter()
    for noise in args.noise:
        for ticks in args.ticks:
            X, y, grp = collect(g, stimuli, seeds, ticks, args.gain, noise,
                                args.warmup, args.amp)
            real = decode(X, y, grp)
            shuf = decode(X, np.random.default_rng(1).permutation(y), grp)
            Xn, yn, gn = collect(g, stimuli, seeds, ticks, args.gain, noise,
                                 args.warmup, args.amp, drive=False)
            noin = decode(Xn, yn, gn)
            active = int((X > 0).any(axis=0).sum())
            ok = (real["accuracy"] > 0.6 and shuf["accuracy"] < 0.45
                  and noin["accuracy"] < 0.45)
            print(f"{noise:>7.3f}{ticks:>7}{real['accuracy']:>9.3f}"
                  f"{shuf['accuracy']:>10.3f}{noin['accuracy']:>10.3f}"
                  f"{active:>8}{'decodable' if ok else '-':>12}")
            results.append({"noise": noise, "ticks": ticks, "real": real,
                            "shuffled": shuf, "no_input": noin,
                            "active_units": active})

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"decode_{args.dataset}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "machine": platform.node(),
        "python": platform.python_version(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "args": vars(args), "graph": g.stats(),
        "stimulus_sizes": [int(len(s)) for s in stimuli],
        "seeds": seeds.tolist(), "results": results,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
