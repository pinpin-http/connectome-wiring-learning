"""Sensorimotor depth strata, and the null that preserves them.

Each neuron is labelled by its shortest-path distance from the sensory input and to the motor readout
(computed on the real graph); edges are swapped only within the same strata and sign. The layered
architecture is preserved by construction, so the null asks: given this architecture, does the specific
wiring matter? (Plain degree-preserving rewiring turns the graph into a shallow expander.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: Depths beyond this are pooled. Past 4 hops the populations are tiny
#: (687 neurons at 5, 19 at 6 on MaleCNS) and would fragment the strata.
MAX_DEPTH = 4

#: Sentinel for neurons the BFS never reaches. They form their own stratum
#: rather than being dropped: they are part of the anatomy.
UNREACHED = -1


@dataclass
class Strata:
    """Per-neuron depth labels, computed once on the anatomical graph."""

    from_input: np.ndarray   # hops from the sensory stratum
    to_output: np.ndarray    # hops to the motor readout
    key: str

    def edge_class(self, src: np.ndarray, dst: np.ndarray,
                   sign: np.ndarray) -> np.ndarray:
        """Class id for each edge. Swaps are confined within a class.

        Sign is part of the key, so Dale's principle survives.
        """
        f, t = self.from_input, self.to_output
        parts = {
            "from": (f[src], f[dst]),
            "to": (t[src], t[dst]),
            "both": (f[src], t[src], f[dst], t[dst]),
        }[self.key]
        code = np.zeros(len(src), dtype=np.int64)
        for p in parts:
            code = code * (MAX_DEPTH + 3) + (np.asarray(p) + 1)
        return code * 3 + (sign.astype(np.int64) + 1)


def _bfs(adj: sp.csr_matrix, seeds: np.ndarray, n: int, max_steps: int = 12) -> np.ndarray:
    d = np.full(n, UNREACHED, dtype=np.int16)
    d[seeds] = 0
    front = np.zeros(n, bool)
    front[seeds] = True
    for step in range(1, max_steps):
        nxt = (adj.T @ front.astype(np.int8)) > 0
        nxt &= d < 0
        if not nxt.any():
            break
        d[nxt] = step
        front = nxt
    return np.clip(d, UNREACHED, MAX_DEPTH)


def compute(graph, sensory: np.ndarray, readout: np.ndarray,
            key: str = "both") -> Strata:
    """Label every neuron by its position in the sensorimotor chain.

    Computed on the **anatomical** graph and then frozen; never recompute it on a rewired graph.
    """
    n = graph.n_nodes
    adj = sp.csr_matrix((np.ones(graph.n_edges, np.int8), (graph.src, graph.dst)),
                        shape=(n, n))
    return Strata(from_input=_bfs(adj, sensory, n),
                  to_output=_bfs(adj.T.tocsr(), readout, n),
                  key=key)


def report(strata: Strata, graph, sign: np.ndarray | None = None) -> dict:
    """How finely the strata cut the edge set, and therefore how much room the
    rewiring has to mix. A class of size 1 cannot be shuffled with anything."""
    sign = graph.sign if sign is None else sign
    cls = strata.edge_class(graph.src, graph.dst, sign)
    _, counts = np.unique(cls, return_counts=True)
    return {
        "key": strata.key,
        "n_classes": int(len(counts)),
        "largest_class": int(counts.max()),
        "singleton_classes": int((counts == 1).sum()),
        "edges_unswappable": int(counts[counts == 1].sum()),
        "edges_swappable_frac": float(counts[counts > 1].sum() / counts.sum()),
    }
