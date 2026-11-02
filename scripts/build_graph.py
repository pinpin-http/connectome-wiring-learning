#!/usr/bin/env python
"""Raw connectome tables -> validated, cached ConnectomeGraph files.

Protocol rev 2 keeps every neuron: no aggregation, no top-K, no
largest-component restriction. The primary graph is unthresholded
(`min_syn=1`); `min_syn=5` is built alongside purely as the declared
sensitivity analysis, never as the default.

    python scripts/build_graph.py                # primary dataset only
    python scripts/build_graph.py --all          # both, for the replication axis
"""
from __future__ import annotations

import argparse
import json

from flymario.connectome import datasets
from flymario.paths import CONNECTOME, ensure_dirs

#: (suffix, min_syn). The first is primary; the second is the sensitivity.
VARIANTS = ((1, ""), (5, "_min5"))


def build_one(name: str) -> None:
    print(f"=== {name}: {datasets.spec(name)['about']} ===")
    for min_syn, suffix in VARIANTS:
        try:
            g = datasets.build(name, min_syn=min_syn)
        except FileNotFoundError as e:
            print(f"  raw tables missing ({e.filename}); run `make download`")
            return
        path = CONNECTOME / f"{name}{suffix}.npz"
        g.save(path)
        led = g.meta["ledger"]
        pct = led["edges_with_fast_current"] / max(1, led["pairs_after_threshold"]) * 100
        print(f"  min_syn={min_syn}: {led['nodes_retained']:,} nodes / "
              f"{led['pairs_after_threshold']:,} pairs / "
              f"{led['synapses_retained']:,} synapses / {pct:.1f}% carrying current")
        (CONNECTOME / f"{name}{suffix}_stats.json").write_text(
            json.dumps(g.stats(), indent=2))
        print(f"  -> {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=datasets.DEFAULT, choices=list(datasets.REGISTRY))
    p.add_argument("--all", action="store_true", help="build every registered dataset")
    args = p.parse_args()
    ensure_dirs()
    for name in (list(datasets.REGISTRY) if args.all else [args.dataset]):
        build_one(name)


if __name__ == "__main__":
    main()
