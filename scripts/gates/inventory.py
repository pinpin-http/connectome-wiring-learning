#!/usr/bin/env python
"""Structural inventory (read-only) of a graph: the numbers the protocol's design depends on.

Plasticity budget `k` = |KC -> MBON|; synaptic depth from the lamina and from the motor readout;
the `P_depth` control's candidate pool; the monosynaptic sensory -> readout shortcut count, real
and expected under a degree-preserving rewiring. Regenerate quoted numbers here, never copy them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy.sparse as sp

from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import CONNECTOME, RUNS

#: Population names are resolved from the dataset registry, never spelled here.
MAX_DEPTH = 12


def _bfs(adj: sp.csr_matrix, seeds: np.ndarray, n: int) -> np.ndarray:
    depth = np.full(n, -1, np.int16)
    depth[seeds] = 0
    frontier = seeds
    for d in range(1, MAX_DEPTH + 1):
        nxt = np.unique(adj[frontier].indices)
        nxt = nxt[depth[nxt] < 0]
        if not len(nxt):
            break
        depth[nxt] = d
        frontier = nxt
    return depth


def _depth_view(g, edge_mask, lam, mot, kc, mbon) -> dict:
    """Structural distances for an explicitly defined edge set and interface."""
    src, dst = g.src[edge_mask], g.dst[edge_mask]
    adj = sp.csr_matrix((np.ones(len(src), np.int8), (src, dst)),
                        shape=(g.n_nodes, g.n_nodes))
    fwd = _bfs(adj, lam, g.n_nodes)
    bwd = _bfs(adj.T.tocsr(), mot, g.n_nodes)
    live = fwd[mot][fwd[mot] >= 0]
    mb_nodes = np.zeros(g.n_nodes, bool)
    mb_nodes[kc] = True
    mb_nodes[mbon] = True
    is_kc = np.isin(src, kc)
    is_mbon = np.isin(dst, mbon)
    plastic = is_kc & is_mbon
    elsewhere = ~mb_nodes[src] & ~mb_nodes[dst]
    # Count the full joint depth strata, not just a pair of population medians.
    strata = {}
    for depths in np.unique(np.column_stack((bwd[src[plastic]], bwd[dst[plastic]])), axis=0):
        ds, dd = (int(x) for x in depths)
        matches = (bwd[src] == ds) & (bwd[dst] == dd)
        strata[f"{ds}->{dd}"] = {
            "mb_pairs": int((plastic & matches).sum()),
            "elsewhere_candidates": int((elsewhere & matches).sum()),
            "reachable": ds >= 0 and dd >= 0,
        }
    direct = np.isin(src, lam) & np.isin(dst, mot)
    return {
        "n_edges": len(src), "n_input_neurons": len(lam),
        "depth_histogram": {str(int(d)): int(c) for d, c in zip(*np.unique(fwd, return_counts=True))},
        "motor_forward_depth": {
            "min": int(live.min()) if len(live) else None,
            "median": float(np.median(live)) if len(live) else None,
            "max": int(live.max()) if len(live) else None,
            "unreached": int((fwd[mot] < 0).sum()),
        },
        "direct_input_motor_pairs": int(direct.sum()),
        "direct_motor_neurons_hit": int(len(np.unique(dst[direct]))),
        "plastic_depth_strata": strata,
        "note": "Unweighted shortest paths; not a measurement of spiking latency. -1 means not reached within the BFS limit.",
    }


def busiest_side(g: ConnectomeGraph, lam: np.ndarray) -> str:
    """The side carrying the most lamina cells (MaleCNS writes "R", FlyWire "right")."""
    side = np.asarray(g.node_side)[lam]
    vals, counts = np.unique(side, return_counts=True)
    return str(vals[counts.argmax()])


def inventory(g: ConnectomeGraph, eye: str | None = None,
              dataset: str | None = None) -> dict:
    n = g.n_nodes
    t = np.asarray(g.node_type).astype(str)
    c = np.asarray(g.node_class)
    dataset = dataset or datasets.of(g)
    spec = datasets.spec(dataset)

    kc = np.flatnonzero(np.char.startswith(t, "KC"))
    mbon = np.flatnonzero(np.char.startswith(t, "MBON"))
    lam = np.flatnonzero(np.isin(t, spec["lamina_types"]))
    mot = datasets.motor_index(g, dataset)
    eye = eye or busiest_side(g, lam)

    mask = np.zeros((4, n), bool)
    for i, idx in enumerate((kc, mbon, lam, mot)):
        mask[i, idx] = True
    is_kc, is_mbon, is_lam, is_mot = mask

    # --- k, the plasticity budget ------------------------------------
    kmask = is_kc[g.src] & is_mbon[g.dst]
    k_pairs = int(kmask.sum())
    k_syn = float(g.weight[kmask].sum())

    # --- depth --------------------------------------------------------
    adj = sp.csr_matrix((np.ones(g.n_edges, np.int8), (g.src, g.dst)), shape=(n, n))
    fwd = _bfs(adj, lam, n)
    bwd = _bfs(adj.T.tocsr(), mot, n)

    live = lambda a: a[a >= 0]
    kc_bd = int(np.median(live(bwd[kc]))) if len(live(bwd[kc])) else -1
    mb_bd = int(np.median(live(bwd[mbon]))) if len(live(bwd[mbon])) else -1

    # --- P_depth candidate pool --------------------------------------
    mb_nodes = is_kc | is_mbon
    pool = int(((bwd[g.src] == kc_bd) & (bwd[g.dst] == mb_bd)
                & ~mb_nodes[g.src] & ~mb_nodes[g.dst]).sum())

    # --- shortcuts ----------------------------------------------------
    smask = is_lam[g.src] & is_mot[g.dst]
    real_short = int(smask.sum())
    real_hit = int(len(np.unique(g.dst[smask])))
    indeg = np.bincount(g.dst, minlength=n).astype(np.float64)
    E = float(g.n_edges)
    lam_out = int(is_lam[g.src].sum())
    exp_short = lam_out * (indeg[mot].sum() / E) if E else 0.0
    exp_hit = float((1.0 - np.exp(-lam_out * (indeg[mot] / E))).sum()) if E else 0.0

    selected_lam = lam[np.asarray(g.node_side)[lam] == eye]
    effective = (g.sign != 0) & (g.weight != 0)
    all_edges = np.ones(g.n_edges, bool)
    views = {
        "anatomical_both_eyes": _depth_view(g, all_edges, lam, mot, kc, mbon),
        "anatomical_selected_eye": _depth_view(g, all_edges, selected_lam, mot, kc, mbon),
        "effective_both_eyes": _depth_view(g, effective, lam, mot, kc, mbon),
        "effective_selected_eye": _depth_view(g, effective, selected_lam, mot, kc, mbon),
    }

    return {
        "graph": g.source, "dataset": dataset,
        "n_nodes": n, "n_edges": g.n_edges, "eye": eye, "operator_views": views,
        "legacy_summary_scope": "anatomical edges, both eyes; use operator_views for model interface",
        "populations": {"KC": len(kc), "MBON": len(mbon),
                        "lamina": len(lam), "motor_readout": len(mot)},
        "plasticity_budget": {"k_pairs": k_pairs, "raw_synapses": k_syn},
        "depth_forward_from_lamina": {
            str(d): int((fwd == d).sum()) for d in range(0, MAX_DEPTH + 1)
            if (fwd == d).sum()},
        "depth_backward_from_readout": {
            str(d): int((bwd == d).sum()) for d in range(0, MAX_DEPTH + 1)
            if (bwd == d).sum()},
        "unreached_forward": int((fwd < 0).sum()),
        "motor_forward_depth": {
            "min": int(live(fwd[mot]).min()) if len(live(fwd[mot])) else -1,
            "median": int(np.median(live(fwd[mot]))) if len(live(fwd[mot])) else -1,
            "max": int(live(fwd[mot]).max()) if len(live(fwd[mot])) else -1},
        "mb_stratum": {"kc_backward_depth": kc_bd, "mbon_backward_depth": mb_bd,
                       "p_depth_candidate_pool": pool,
                       "pool_over_k": round(pool / max(k_pairs, 1), 1)},
        "shortcuts": {
            "expectation_note": "Unconstrained configuration-model approximation, not the exact sign-constrained simple-graph null.",
            "real_edges": real_short, "real_readout_neurons_hit": real_hit,
            "expected_edges_under_rewiring": round(exp_short),
            "expected_readout_neurons_hit": round(exp_hit),
            "inflation_factor": round(exp_short / max(real_short, 1), 1)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default=datasets.default_cache_name())
    p.add_argument("--eye", default=None,
                   help="dataset spelling; resolved from the graph if omitted")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    g = ConnectomeGraph.load(CONNECTOME / args.graph)
    inv = inventory(g, eye=args.eye)
    graph_path = CONNECTOME / args.graph
    inv["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    }

    P = inv["populations"]; K = inv["plasticity_budget"]
    M = inv["mb_stratum"]; S = inv["shortcuts"]
    print(f"{inv['graph']}   {inv['n_nodes']:,} nodes / {inv['n_edges']:,} pairs\n")
    print(f"  KC {P['KC']:,}   MBON {P['MBON']:,}   "
          f"lamina {P['lamina']:,}   motor readout {P['motor_readout']:,}\n")
    print(f"  k = KC->MBON pairs              {K['k_pairs']:>12,}")
    print(f"      raw synapses                {K['raw_synapses']:>12,.0f}\n")
    print("  forward depth from lamina")
    for d, m in inv["depth_forward_from_lamina"].items():
        print(f"      depth {d}: {m:>10,}")
    print(f"      unreached: {inv['unreached_forward']:,}")
    mf = inv["motor_forward_depth"]
    print(f"      motor readout depth  min {mf['min']}  "
          f"median {mf['median']}  max {mf['max']}\n")
    print(f"  MB stratum  KC bdepth {M['kc_backward_depth']} -> "
          f"MBON bdepth {M['mbon_backward_depth']}")
    print(f"      P_depth candidate pool      {M['p_depth_candidate_pool']:>12,}"
          f"   ({M['pool_over_k']}x k)\n")
    print("  monosynaptic lamina -> readout shortcuts")
    print(f"      real                        {S['real_edges']:>12,}"
          f"   hitting {S['real_readout_neurons_hit']:,} readout neurons")
    print(f"      expected under rewiring     {S['expected_edges_under_rewiring']:>12,}"
          f"   hitting {S['expected_readout_neurons_hit']:,}")
    print(f"      inflation                   {S['inflation_factor']:>12}x")
    print("\n  Operator/interface views (shortest-path distance, not response latency):")
    for name, view in inv["operator_views"].items():
        print(f"    {name}: input={view['n_input_neurons']:,}, edges={view['n_edges']:,}, "
              f"motor={view['motor_forward_depth']}, shortcuts={view['direct_input_motor_pairs']}")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / (args.output or "inventory.json")
    out.write_text(json.dumps(inv, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
