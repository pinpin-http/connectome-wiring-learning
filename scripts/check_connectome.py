#!/usr/bin/env python
"""Print every cached graph with its retention ledger, and re-validate it."""
from __future__ import annotations

import json

import numpy as np

from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import CONNECTOME

MOTOR = ("descending", "motor")


def main() -> None:
    files = sorted(CONNECTOME.glob("*.npz"))
    if not files:
        print("no cached graphs; run `make build-graphs`")
        return
    print(f"{'file':<22}{'nodes':>9}{'edges':>12}{'exc':>7}{'inh':>7}{'none':>7}"
          f"{'motor':>8}  valid")
    for f in files:
        g = ConnectomeGraph.load(f)
        try:
            g.validate()
            ok = "yes"
        except ValueError as e:
            ok = f"NO: {e}"
        motor = int(np.isin(g.node_class, MOTOR).sum()) if g.node_class is not None else 0
        print(f"{f.stem:<22}{g.n_nodes:>9,}{g.n_edges:>12,}"
              f"{(g.sign > 0).mean() * 100:>6.1f}%{(g.sign < 0).mean() * 100:>6.1f}%"
              f"{(g.sign == 0).mean() * 100:>6.1f}%{motor:>8,}  {ok}")

    for led in sorted(CONNECTOME.glob("*_ledger.json")):
        print(f"\n--- {led.stem} ---")
        print(json.dumps(json.loads(led.read_text()), indent=2))


if __name__ == "__main__":
    main()
