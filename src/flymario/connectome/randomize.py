"""Null-model graphs.

Directed double-edge swaps (`u->v, x->y` becomes `u->y, x->v`) preserve every node's in- and
out-degree exactly. Swaps are confined to edges whose source shares a transmitter class, so no
neuron's sign changes, and each source keeps its own outgoing weight multiset.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

#: Swaps per edge needed before the chain is considered mixed. Reported, not
#: assumed: `mixing_report` measures how much of the original wiring survives.
SWAP_MULTIPLIER = 10


def _rewire_class(src: np.ndarray, dst: np.ndarray, n_nodes: int,
                  seed: int, multiplier: int, loops: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Degree-preserving double-edge swaps on one transmitter class, via igraph.

    `loops=True` (historical nulls) lets a swap create a self-loop (never a multi-edge); `loops=False`
    (revision48, autapse-preserving nulls) forbids both."""
    import random

    import igraph as ig

    if len(src) < 2:
        return src, dst
    g = ig.Graph(n=n_nodes, edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    # igraph wants a stdlib Random, not a numpy Generator.
    ig.set_random_number_generator(random.Random(seed))
    g.rewire(n=multiplier * len(src), allowed_edge_types="loops" if loops else "simple")
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e[:, 0], e[:, 1]


def degree_preserving(graph, seed: int = 0, multiplier: int = SWAP_MULTIPLIER):
    """Primary null: exact in/out degrees, signs and per-source weight multisets.

    Returns a new ConnectomeGraph, not a matrix: the sign vector is rebuilt alongside the edges.
    """
    from flymario.connectome.schema import ConnectomeGraph

    rng = np.random.default_rng(seed)
    n = graph.n_nodes
    out_src, out_dst, out_w, out_sign = [], [], [], []

    # Rewire one sign class at a time, so the excitatory/inhibitory balance is unchanged.

    # Partition by the presynaptic sign class, so Dale's principle survives.
    for cls in np.unique(graph.sign):
        m = graph.sign == cls
        # Derive a non-negative per-class seed: sign classes include -1.
        cls_seed = (seed * 977 + int(cls) * 7919) % (2**31 - 1)
        s, d = _rewire_class(graph.src[m], graph.dst[m], n, cls_seed, multiplier)
        w = graph.weight[m]
        # Weights follow their source: sort both edge lists by source and hand each neuron's
        # original outgoing weights to its new outgoing edges (same count, so the multiset
        # transfers exactly; only which partner receives which strength changes).
        order = np.lexsort((rng.random(len(s)), s))
        src_order = np.lexsort((rng.random(len(graph.src[m])), graph.src[m]))
        w_new = np.empty_like(w)
        w_new[order] = w[src_order]
        out_src.append(s)
        out_dst.append(d)
        out_w.append(w_new)
        out_sign.append(np.full(len(s), cls, dtype=np.float32))

    g = ConnectomeGraph(
        node_ids=graph.node_ids.copy(),
        src=np.concatenate(out_src), dst=np.concatenate(out_dst),
        weight=np.concatenate(out_w), sign=np.concatenate(out_sign),
        node_type=graph.node_type, node_class=graph.node_class,
        node_side=graph.node_side, node_pos=graph.node_pos,
        source=f"{graph.source}-degree_preserving-s{seed}",
        meta={**graph.meta, "null": "degree_preserving", "seed": seed,
              "swap_multiplier": multiplier},
    )
    g.validate()
    return g


def shuffled_weights(graph, seed: int = 0, **_):
    """Secondary: edge set, signs and degrees untouched; magnitudes permuted
    within each source's own outgoing edges."""
    from flymario.connectome.schema import ConnectomeGraph

    rng = np.random.default_rng(seed)
    w = graph.weight.copy()
    order = np.argsort(graph.src, kind="stable")
    bounds = np.flatnonzero(np.diff(graph.src[order])) + 1
    for chunk in np.split(order, bounds):
        w[chunk] = rng.permutation(w[chunk])
    g = ConnectomeGraph(
        node_ids=graph.node_ids.copy(), src=graph.src.copy(), dst=graph.dst.copy(),
        weight=w, sign=graph.sign.copy(), node_type=graph.node_type,
        node_class=graph.node_class, node_side=graph.node_side, node_pos=graph.node_pos,
        source=f"{graph.source}-shuffled_weights-s{seed}",
        meta={**graph.meta, "null": "shuffled_weights", "seed": seed})
    g.validate()
    return g


def shuffled_edges(graph, seed: int = 0, **_):
    """Secondary, weakest null: same node and edge counts per transmitter class,
    degrees free. Beating this only shows that degree structure matters."""
    from flymario.connectome.schema import ConnectomeGraph

    rng = np.random.default_rng(seed)
    n = graph.n_nodes
    src = np.empty_like(graph.src)
    dst = np.empty_like(graph.dst)
    for cls in np.unique(graph.sign):
        m = graph.sign == cls
        src[m] = rng.integers(0, n, m.sum())
        dst[m] = rng.integers(0, n, m.sum())
    g = ConnectomeGraph(
        node_ids=graph.node_ids.copy(), src=src, dst=dst,
        weight=graph.weight.copy(), sign=graph.sign.copy(),
        node_type=graph.node_type, node_class=graph.node_class,
        node_side=graph.node_side, node_pos=graph.node_pos,
        source=f"{graph.source}-shuffled_edges-s{seed}",
        meta={**graph.meta, "null": "shuffled_edges", "seed": seed})
    g.validate()
    return g


def depth_stratified(graph, seed: int = 0, multiplier: int = SWAP_MULTIPLIER,
                     key: str = "both", sensory=None, readout=None, lateral: bool = False,
                     weights_by: str = "src", keep_loops: bool = False):
    """Primary null: rewire only within sensorimotor depth strata (and sign, so Dale's principle holds).

    An edge may only be swapped with one whose endpoints share the same frozen distance from the sensory
    input and to the motor readout, so the layered routing survives (a degree-matched graph is an expander).
    `lateral` adds both endpoints' side (C / L / M / R) to the key; `weights_by` "src" keeps out-strength,
    "dst" in-strength; `keep_loops` (revision48) keeps the measured autapses and creates no new loop.
    """
    from flymario.connectome import datasets, strata
    from flymario.connectome.schema import ConnectomeGraph

    if sensory is None or readout is None:
        name = "malecns" if "malecns" in graph.source else "flywire"
        sensory = datasets.by_types(graph, name, "lamina_types")
        readout = datasets.motor_index(graph, name)

    st = strata.compute(graph, sensory, readout, key)
    cls = st.edge_class(graph.src, graph.dst, graph.sign)
    if lateral:
        _u, side = np.unique(np.asarray(graph.node_side), return_inverse=True)
        k = int(side.max()) + 1
        cls = (np.asarray(cls, dtype=np.int64) * k + side[graph.src]) * k + side[graph.dst]

    rng = np.random.default_rng(seed)
    n = graph.n_nodes
    out_src, out_dst, out_w, out_sign = [], [], [], []
    if keep_loops:
        loop = graph.src == graph.dst
        out_src.append(graph.src[loop])
        out_dst.append(graph.dst[loop])
        out_w.append(graph.weight[loop])
        out_sign.append(graph.sign[loop])
        cls = np.where(loop, -1, cls)               # loops leave the rewiring classes
    # One stable sort instead of a boolean mask per class; stable order keeps each class's edges
    # in their original order, so the realisations are unchanged.
    by_cls = np.argsort(cls, kind="stable")
    uniq, starts = np.unique(cls[by_cls], return_index=True)
    for c, a, b in zip(uniq, starts, np.append(starts[1:], len(cls)), strict=True):
        if keep_loops and c == -1:
            continue
        m = by_cls[a:b]
        s_in, d_in = graph.src[m], graph.dst[m]
        if len(m) < 4:
            # Too small to shuffle meaningfully; left intact rather than
            # forced, and counted in the mixing report.
            s_out, d_out = s_in, d_in
        else:
            cls_seed = (seed * 7919 + int(c)) % (2**31 - 1)
            s_out, d_out = _rewire_class(s_in, d_in, n, cls_seed, multiplier, loops=not keep_loops)
        w = graph.weight[m]
        key_out, key_in = (s_out, s_in) if weights_by == "src" else (d_out, d_in)
        order = np.lexsort((rng.random(len(s_out)), key_out))
        src_order = np.lexsort((rng.random(len(s_in)), key_in))
        w_new = np.empty_like(w)
        w_new[order] = w[src_order]
        out_src.append(s_out)
        out_dst.append(d_out)
        out_w.append(w_new)
        out_sign.append(graph.sign[m])

    g = ConnectomeGraph(
        node_ids=graph.node_ids.copy(),
        src=np.concatenate(out_src), dst=np.concatenate(out_dst),
        weight=np.concatenate(out_w), sign=np.concatenate(out_sign),
        node_type=graph.node_type, node_class=graph.node_class,
        node_side=graph.node_side, node_pos=graph.node_pos,
        source=f"{graph.source}-depth_stratified{'_lateral' if lateral else ''}-{key}-s{seed}"
               + ("-keeploops" if keep_loops else ""),
        meta={**graph.meta, "null": "depth_stratified_lateral" if lateral else "depth_stratified", "seed": seed,
              "strata_key": key, "swap_multiplier": multiplier,
              "n_strata": int(len(np.unique(cls))), **({"keep_loops": True} if keep_loops else {})})
    g.validate()
    return g


REGISTRY = {"connectome": lambda g, seed=0, **k: g,
            "depth_stratified": depth_stratified,
            "depth_stratified_lateral": lambda g, seed=0, **k: depth_stratified(g, seed, lateral=True, **k),
            # RL_STUDY §26: rung c, in-strength of every neuron kept as well
            "depth_stratified_lateral_instrength": lambda g, seed=0, **k: depth_stratified(g, seed, lateral=True,
                                                                                         weights_by="dst", **k),
            # revision48 (R3): the same two families with the measured autapses kept and no new one
            "depth_stratified_lateral_keeploops": lambda g, seed=0, **k: depth_stratified(g, seed, lateral=True,
                                                                                        keep_loops=True, **k),
            "depth_stratified_lateral_instrength_keeploops": lambda g, seed=0, **k: depth_stratified(
                g, seed, lateral=True, weights_by="dst", keep_loops=True, **k),
            "degree_preserving": degree_preserving,
            "shuffled_weights": shuffled_weights,
            "shuffled_edges": shuffled_edges}


def make(kind: str, graph, seed: int = 0, **kwargs):
    if kind not in REGISTRY:
        raise KeyError(f"unknown null {kind!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[kind](graph, seed=seed, **kwargs)


# -- diagnostics ----------------------------------------------------------
def degrees(graph) -> tuple[np.ndarray, np.ndarray]:
    n = graph.n_nodes
    return (np.bincount(graph.src, minlength=n), np.bincount(graph.dst, minlength=n))


def shortcut_report(graph, sensory: np.ndarray, readout: np.ndarray) -> dict:
    """Count direct sensory->readout edges (96 on FlyWire, 5,994 after degree-preserving rewiring)."""
    m = np.isin(graph.src, sensory) & np.isin(graph.dst, readout)
    return {"direct_edges": int(m.sum()),
            "readout_reached": int(len(np.unique(graph.dst[m]))),
            "readout_total": int(len(readout))}


def mixing_report(original, null, sample_size: int = 400_000, seed: int = 0) -> dict:
    """What survived the rewiring. A swap count is not proof of mixing."""
    o_out, o_in = degrees(original)
    n_out, n_in = degrees(null)
    if original.n_nodes != null.n_nodes:
        raise ValueError("overlap requires the same node inventory")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    # Canonical ordering makes the sample independent of file/edge order.
    # Check sampled original pairs against the COMPLETE null: sampling both
    # sides falsely reports rewiring when only their edge order changed.
    a = np.unique(original.src.astype(np.int64) * original.n_nodes + original.dst)
    b = np.unique(null.src.astype(np.int64) * null.n_nodes + null.dst)
    total = len(a)
    if total > sample_size:
        a = a[np.random.default_rng(seed).choice(total, sample_size, replace=False)]
    overlap = float(np.isin(a, b, assume_unique=True).mean()) if len(a) else 1.0
    return {
        "out_degree_exact": bool(np.array_equal(o_out, n_out)),
        "in_degree_exact": bool(np.array_equal(o_in, n_in)),
        "edge_count_equal": bool(original.n_edges == null.n_edges),
        "sign_multiset_equal": bool(np.array_equal(np.sort(original.sign),
                                                   np.sort(null.sign))),
        "weight_multiset_equal": bool(np.allclose(np.sort(original.weight),
                                                  np.sort(null.weight))),
        "edge_overlap_sampled": round(overlap, 4),
        "edge_overlap_sample_size": len(a),
        "edge_overlap_exact": len(a) == total,
        "edge_overlap_note": "Edge turnover alone does not establish MCMC mixing.",
    }
