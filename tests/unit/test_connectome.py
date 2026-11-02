"""Acceptance checks for the graph representation — protocol rev 2 §2.4.

These are not style tests. Each one corresponds to a bug that was actually
shipped and silently changed the data, so each is written to fail loudly if
that bug returns.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flymario.connectome import datasets, randomize
from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import CONNECTOME

#: Every registered dataset that has been built. Until 2026-09-22 this file
#: was pinned to `flywire.npz`, so Gate A had never run against MaleCNS -- the
#: primary dataset since 2026-09-21. Four of these tests failed on it.
DATASETS = [n for n in datasets.REGISTRY if datasets.cache_path(n).exists()]
pytestmark = pytest.mark.skipif(not DATASETS, reason="run `make build-graphs-all`")

#: Audited counts. Not decoration: a change here means the upstream release or
#: the loader changed, and §7.1 requires that explained before the number moves.
AUDITED = {
    "flywire": {"nodes": 139_262, "pairs": 15_091_983, "synapses": 54_492_922,
                "photoreceptors": 11_118},
    "malecns": {"nodes": 169_984, "pairs": 25_563_197, "synapses": 124_025_046,
                "photoreceptors": 4_109},
}

#: The two loaders name their transmitter sources differently. The invariant is
#: the same either way: measured evidence must outweigh prediction.
NT_SOURCES = {
    "flywire": ("from_known_nt", "from_prediction"),
    "malecns": ("from_ground_truth", "from_consensus"),
}


@pytest.fixture(scope="module", params=DATASETS)
def g(request) -> ConnectomeGraph:
    return ConnectomeGraph.load(datasets.cache_path(request.param))


@pytest.fixture(scope="module")
def name(g) -> str:
    return datasets.of(g)


# --- §7.5: node identity ---------------------------------------------------
def test_node_set_is_the_full_union(g, name):
    """Annotations union connection endpoints, fixed before any filtering.

    The earlier loader derived the node set *after* thresholding, so changing
    min_syn changed which neurons existed and two thresholds produced two
    incomparable networks.
    """
    assert g.n_nodes == AUDITED[name]["nodes"]
    assert len(np.unique(g.node_ids)) == g.n_nodes


@pytest.mark.parametrize("name", DATASETS)
def test_threshold_does_not_change_the_node_set(name):
    sens = datasets.cache_path(name, min_syn=5)
    if not sens.exists():
        pytest.skip("sensitivity graph not built")
    a = ConnectomeGraph.load(datasets.cache_path(name))
    b = ConnectomeGraph.load(sens)
    assert np.array_equal(a.node_ids, b.node_ids), "thresholding moved the node set"


def test_isolated_neurons_are_retained(g):
    """Silent nodes stay. Dropping them is a filtering decision in disguise."""
    deg = np.bincount(g.src, minlength=g.n_nodes) + np.bincount(g.dst, minlength=g.n_nodes)
    assert (deg == 0).sum() > 0, "no isolated nodes at all is suspicious"
    assert g.meta["ledger"]["isolated_nodes"] == int((deg == 0).sum())


# --- §7.5: pair aggregation and count conservation -------------------------
def test_pairs_are_unique(g):
    """Rows must already be summed per pair, not left for the matrix to merge."""
    key = g.src.astype(np.int64) * g.n_nodes + g.dst.astype(np.int64)
    assert len(np.unique(key)) == g.n_edges, "duplicate directed pairs remain"


def test_aggregation_conserves_synapses(g, name):
    """Not one synapse may go missing between the raw table and the ledger.

    MaleCNS reported 124,025,040 retained against 124,025,046 raw and it looked
    like six dropped edges. It was not: the ledger summed the float32 weight
    array, whose representable grid at 1.24e8 is spaced 8 apart. Nothing was
    ever lost -- the counter was imprecise. Counts are int64 now, and this test
    fails if anyone reintroduces a float sum.
    """
    led = g.meta["ledger"]
    audited = AUDITED[name]
    assert led["raw_synapses"] == audited["synapses"]
    assert led["synapses_retained"] == led["raw_synapses"]
    assert led["pairs_before_threshold"] == audited["pairs"]
    # And the ledger must agree with the array it describes.
    assert int(g.weight.astype(np.int64).sum()) == led["synapses_retained"]


def test_weights_are_raw_counts_not_transformed(g):
    """log1p belongs in the operator, not the anatomy. Storing it here is how
    log1p(7)+log1p(8) crept in ahead of the summation."""
    w = g.weight
    assert w.min() >= 1.0
    assert np.allclose(w, np.round(w)), "weights are not integer synapse counts"


# --- §7.5: transmitters ----------------------------------------------------
def test_unresolved_transmitters_carry_no_current(g):
    """Never silently excitatory. 'We do not know' must mean zero."""
    assert (g.sign == 0).sum() > 0
    assert set(np.unique(g.sign)) <= {-1.0, 0.0, 1.0}


def test_photoreceptors_are_inhibitory(g, name):
    """The histamine bug, in both spellings.

    FlyWire's classifier has no histamine class, so R1-6 came back 63%
    excitatory; they are histaminergic and inhibit L1/L2. MaleCNS writes the
    same cells `R1-R6`, which is why the type name comes from the registry and
    never from a literal here.
    """
    pr = datasets.by_types(g, name, "photoreceptor_types")
    # The count is audited per dataset rather than thresholded: MaleCNS
    # annotates 4,109 photoreceptors against FlyWire's 11,118, so any single
    # threshold either misses the spelling bug on one dataset or fails the
    # other for being itself.
    assert len(pr) == AUDITED[name]["photoreceptors"], (
        f"{name}: selected {len(pr)} photoreceptors; check the registry spelling")
    out = g.sign[np.isin(g.src, pr)]
    assert (out < 0).mean() > 0.95, f"{name}: only {(out < 0).mean():.1%} inhibitory"


def test_measured_transmitters_are_preferred(g, name):
    prov = g.meta["ledger"]["transmitter_provenance"]
    measured, predicted = NT_SOURCES[name]
    assert prov[measured] > prov[predicted], (
        f"{name}: {measured}={prov[measured]} is not above {predicted}={prov[predicted]}")


# --- §7.5: edge direction convention ---------------------------------------
@pytest.mark.parametrize("log_weights", [True, False])
def test_edge_direction_convention(g, log_weights):
    """`src[k] -> dst[k]` lands at `W[dst[k], src[k]]`, per schema.py.

    §7.5 requires this checked and nothing did. A silent transpose would leave
    every count, degree and acceptance test intact while reversing every
    pathway in the model. Checked for both weight transforms: which one the
    config selects is a separate question from which way the edges point.
    """
    W = g.to_sparse(log_weights=log_weights).tocsr()
    k = np.flatnonzero(g.sign != 0)[:200]
    for i in k[::20]:
        s, d = int(g.src[i]), int(g.dst[i])
        w = np.log1p(g.weight[i]) if log_weights else g.weight[i]
        expected = w * g.sign[i]
        assert W[d, s] == pytest.approx(expected, rel=1e-5), (
            f"edge {s}->{d} is not at W[{d},{s}]; direction convention broken")


# --- §7.5: sparse equivalence ----------------------------------------------
def test_sparse_matches_dense_on_a_fixture():
    small = ConnectomeGraph(
        node_ids=np.arange(5, dtype=np.int64),
        src=np.array([0, 1, 1, 3]), dst=np.array([1, 2, 3, 4]),
        weight=np.array([4.0, 9.0, 1.0, 2.0], np.float32),
        sign=np.array([1.0, -1.0, 1.0, 0.0], np.float32), source="fixture")
    small.validate()
    W = small.to_sparse(log_weights=True).toarray()
    assert W[1, 0] == pytest.approx(np.log1p(4.0))
    assert W[2, 1] == pytest.approx(-np.log1p(9.0))
    assert W[4, 3] == 0.0, "zero-sign edge must carry no current"
    assert W.shape == (5, 5)
    raw = small.to_sparse(log_weights=False).toarray()
    assert raw[1, 0] == pytest.approx(4.0) and raw[2, 1] == pytest.approx(-9.0)


def test_to_sparse_default_follows_the_config():
    """No argument = the declared transform, not a hardcoded one (§7.4)."""
    from flymario import config
    small = ConnectomeGraph(
        node_ids=np.arange(2, dtype=np.int64), src=np.array([0]), dst=np.array([1]),
        weight=np.array([4.0], np.float32), sign=np.array([1.0], np.float32),
        source="fixture")
    want_log = bool(config.get("dynamics.weight_transform")["log1p"])
    expected = np.log1p(4.0) if want_log else 4.0
    assert small.to_sparse().toarray()[1, 0] == pytest.approx(expected)


def test_sparse_preserves_matrix_dimensions(g):
    W = g.to_sparse()
    assert W.shape == (g.n_nodes, g.n_nodes), "sparse conversion dropped nodes"


# --- §7.5: deterministic derivation ----------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("name", DATASETS)
def test_graph_derivation_is_deterministic(name):
    """Same raw tables and same configuration must give the same graph.

    §7.5 asks for this and nothing checked it: only null reproducibility was
    tested, which covers the seeded rewiring and not the loader underneath it.
    Marked slow because it rebuilds from the raw feathers twice --
    `pytest -m slow` to run it, and it belongs in any pre-freeze run (§42).
    """
    import hashlib

    def digest(g: ConnectomeGraph) -> str:
        h = hashlib.sha256()
        for arr in (g.node_ids, g.src, g.dst, g.weight, g.sign):
            h.update(np.ascontiguousarray(arr).tobytes())
        for meta in (g.node_type, g.node_class, g.node_side):
            h.update(np.asarray(meta).astype(str).tobytes())
        for opt in (g.node_pos, g.node_retino):
            h.update(b"none" if opt is None else np.ascontiguousarray(opt).tobytes())
        return h.hexdigest()

    a = datasets.build(name, min_syn=1)
    b = datasets.build(name, min_syn=1)
    assert digest(a) == digest(b), f"{name}: rebuilding the graph changed it"


# --- §3: the null models --------------------------------------------------
@pytest.fixture(scope="module")
def small() -> ConnectomeGraph:
    """A graph big enough to rewire meaningfully, small enough for a test run."""
    rng = np.random.default_rng(0)
    n, m = 3_000, 40_000
    src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    key = src.astype(np.int64) * n + dst
    _, uniq = np.unique(key, return_index=True)
    src, dst = src[uniq], dst[uniq]
    sign = rng.choice([1.0, -1.0], len(src)).astype(np.float32)
    return ConnectomeGraph(
        node_ids=np.arange(n, dtype=np.int64), src=src, dst=dst,
        weight=rng.integers(1, 50, len(src)).astype(np.float32),
        sign=sign, node_class=np.full(n, "central"), source="fixture")


def test_degree_preserving_preserves_degrees_exactly(small):
    """The study's primary control. The previous implementation permuted
    endpoints independently: 2,658 nodes changed out-degree, one by 692."""
    null = randomize.degree_preserving(small, seed=0, multiplier=5)
    r = randomize.mixing_report(small, null)
    assert r["out_degree_exact"], "out-degrees changed"
    assert r["in_degree_exact"], "in-degrees changed"
    assert r["edge_count_equal"]


def test_degree_preserving_keeps_signs_and_weights(small):
    null = randomize.degree_preserving(small, seed=0, multiplier=5)
    r = randomize.mixing_report(small, null)
    assert r["sign_multiset_equal"], "rewiring changed the E/I balance"
    assert r["weight_multiset_equal"]


def test_degree_preserving_actually_mixes(small):
    """Preserving degrees is worthless if the wiring barely moved."""
    null = randomize.degree_preserving(small, seed=0, multiplier=10)
    assert randomize.mixing_report(small, null)["edge_overlap_sampled"] < 0.2


def test_nulls_keep_every_node(small):
    for kind in ("degree_preserving", "shuffled_weights", "shuffled_edges"):
        null = randomize.make(kind, small, seed=0, multiplier=3)
        assert null.n_nodes == small.n_nodes, f"{kind} dropped nodes"
        assert np.array_equal(null.node_ids, small.node_ids)


def test_shuffled_weights_keeps_the_edge_set(small):
    null = randomize.shuffled_weights(small, seed=0)
    assert randomize.mixing_report(small, null)["edge_overlap_sampled"] == 1.0


def test_nulls_are_seed_reproducible(small):
    a = randomize.degree_preserving(small, seed=7, multiplier=3)
    b = randomize.degree_preserving(small, seed=7, multiplier=3)
    assert np.array_equal(a.src, b.src) and np.array_equal(a.dst, b.dst)


# --- §8: the depth-stratified null, which is what clears Gate B ------------
@pytest.fixture(scope="module")
def layered() -> ConnectomeGraph:
    """A graph with a real sensory→motor chain, so depth strata mean something."""
    rng = np.random.default_rng(0)
    sizes = [300, 700, 700, 400, 200]          # input .. output
    offs = np.cumsum([0] + sizes)
    n = int(offs[-1])
    src, dst = [], []
    for layer in range(len(sizes) - 1):
        a = rng.integers(offs[layer], offs[layer + 1], 6_000)
        b = rng.integers(offs[layer + 1], offs[layer + 2], 6_000)
        src.append(a)
        dst.append(b)
    # A handful of genuine long-range shortcuts, as the real graph has.
    src.append(rng.integers(offs[0], offs[1], 12))
    dst.append(rng.integers(offs[4], offs[5], 12))
    src, dst = np.concatenate(src), np.concatenate(dst)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    key = src.astype(np.int64) * n + dst
    _, uniq = np.unique(key, return_index=True)
    src, dst = src[uniq], dst[uniq]
    return ConnectomeGraph(
        node_ids=np.arange(n, dtype=np.int64), src=src, dst=dst,
        weight=rng.integers(1, 40, len(src)).astype(np.float32),
        sign=rng.choice([1.0, -1.0], len(src)).astype(np.float32),
        node_class=np.full(n, "central"), source="layered-fixture",
        meta={"sensory": (0, int(offs[1])), "readout": (int(offs[4]), n)})


def _ends(g):
    a, b = g.meta["sensory"], g.meta["readout"]
    return np.arange(*a, dtype=np.int32), np.arange(*b, dtype=np.int32)


def test_strata_partition_is_swappable(layered):
    """A stratum of one edge cannot be shuffled with anything."""
    from flymario.connectome import strata

    sens, read = _ends(layered)
    st = strata.compute(layered, sens, read, key="both")
    r = strata.report(st, layered)
    assert r["n_classes"] > 4
    assert r["edges_swappable_frac"] > 0.95


def test_depth_stratified_preserves_shortcut_count(layered):
    """The whole reason this null exists: degree-preserving inflated direct
    sensory→readout edges 165× on the real graph. This must not."""
    sens, read = _ends(layered)
    before = randomize.shortcut_report(layered, sens, read)
    null = randomize.depth_stratified(layered, seed=0, multiplier=6,
                                      sensory=sens, readout=read)
    after = randomize.shortcut_report(null, sens, read)
    assert after["direct_edges"] == before["direct_edges"]


def test_depth_stratified_still_preserves_degrees(layered):
    sens, read = _ends(layered)
    null = randomize.depth_stratified(layered, seed=0, multiplier=6,
                                      sensory=sens, readout=read)
    r = randomize.mixing_report(layered, null)
    assert r["out_degree_exact"] and r["in_degree_exact"]
    assert r["sign_multiset_equal"] and r["weight_multiset_equal"]


def test_depth_stratified_actually_mixes(layered):
    """Preserving structure is worthless if nothing moved."""
    sens, read = _ends(layered)
    null = randomize.depth_stratified(layered, seed=0, multiplier=10,
                                      sensory=sens, readout=read)
    assert randomize.mixing_report(layered, null)["edge_overlap_sampled"] < 0.25


def _sided(g):
    """The layered fixture with sides, mostly ipsilateral wiring (as the real graph)."""
    import dataclasses
    side = np.where(np.arange(g.n_nodes) % 2 == 0, "L", "R")
    keep = (side[g.src] == side[g.dst]) | (np.arange(g.n_edges) % 5 == 0)
    return dataclasses.replace(g, src=g.src[keep], dst=g.dst[keep], weight=g.weight[keep], sign=g.sign[keep],
                               node_side=side)


def _side_pairs(g):
    s = np.asarray(g.node_side)
    return {(a, b): int(((s[g.src] == a) & (s[g.dst] == b)).sum()) for a in "LR" for b in "LR"}


def test_lateral_null_keeps_sides_and_degrees(layered):
    """2026-10-06: the primary null swaps targets between the two eyes; the lateral
    variant must keep every side pair's edge count, and still degrees and mixing."""
    g = _sided(layered)
    sens, read = _ends(g)
    lat = randomize.depth_stratified(g, seed=0, multiplier=10, sensory=sens, readout=read, lateral=True)
    plain = randomize.depth_stratified(g, seed=0, multiplier=10, sensory=sens, readout=read)
    assert _side_pairs(lat) == _side_pairs(g)
    assert _side_pairs(plain) != _side_pairs(g)            # guards the finding itself
    r = randomize.mixing_report(g, lat)
    assert r["out_degree_exact"] and r["in_degree_exact"] and r["weight_multiset_equal"]
    assert r["edge_overlap_sampled"] < 0.5


def test_weights_by_dst_keeps_in_strength(layered):
    sens, read = _ends(layered)
    by_dst = randomize.depth_stratified(layered, seed=0, multiplier=6, sensory=sens, readout=read, weights_by="dst")
    by_src = randomize.depth_stratified(layered, seed=0, multiplier=6, sensory=sens, readout=read)
    ins = lambda h: np.bincount(h.dst, weights=h.weight, minlength=h.n_nodes)
    outs = lambda h: np.bincount(h.src, weights=h.weight, minlength=h.n_nodes)
    assert np.allclose(ins(by_dst), ins(layered))
    assert np.allclose(outs(by_src), outs(layered))


def test_degree_preserving_does_inflate_shortcuts(layered):
    """Guards the finding itself. If this ever stops failing, either the
    fixture stopped being layered or the null stopped being naive."""
    sens, read = _ends(layered)
    before = randomize.shortcut_report(layered, sens, read)
    null = randomize.degree_preserving(layered, seed=0, multiplier=6)
    after = randomize.shortcut_report(null, sens, read)
    assert after["direct_edges"] > before["direct_edges"] * 3


# --- §11.2: the declared primary readout ----------------------------------
def test_fixed_readout_is_untrainable():
    """`fixed` means the projection cannot be written to, not merely that we
    promise not to. A readout that could be optimised is a different category."""
    from flymario.policy.readout import FixedRandomProjection

    r = FixedRandomProjection(n_features=64, n_actions=8, seed=0)
    assert r.CATEGORY == "fixed"
    with pytest.raises(ValueError):
        r.W[0, 0] = 1.0


def test_fixed_readout_is_seed_reproducible():
    from flymario.policy.readout import FixedRandomProjection

    a = FixedRandomProjection(64, 8, seed=7)
    b = FixedRandomProjection(64, 8, seed=7)
    c = FixedRandomProjection(64, 8, seed=8)
    assert np.array_equal(a.W, b.W)
    assert not np.array_equal(a.W, c.W)
    assert a.provenance()["checksum"] == b.provenance()["checksum"]


def test_fixed_readout_favours_no_action_by_construction():
    """The owner's stated constraint. Motor activity has a large common-mode
    component; against an uncentred projection that component alone picks the
    argmax and the readout emits one action forever."""
    from flymario.policy.readout import FixedRandomProjection, action_balance

    rng = np.random.default_rng(0)
    states = np.abs(rng.standard_normal((400, 256)) * 0.3
                    + rng.random((400, 1)) * 2).astype(np.float32)
    r = FixedRandomProjection(256, 8, seed=3)
    bal = action_balance(r, states)
    assert bal["structurally_unbiased"]
    assert bal["empirical_top_share"] < 0.35, bal


def test_uncentred_projection_would_have_been_biased():
    """Guards the reason centring is in the spec. If this stops failing, the
    common-mode component has gone, and the justification needs revisiting."""
    from flymario.policy.readout import (FixedRandomProjection, ProjectionSpec,
                                         action_balance)

    rng = np.random.default_rng(0)
    common = np.abs(rng.standard_normal((300, 256)) * 0.05
                    + rng.random((300, 1)) * 3).astype(np.float32)
    raw = FixedRandomProjection(256, 8, seed=3,
                                spec=ProjectionSpec(centre_rows=False))
    assert action_balance(raw, common)["empirical_top_share"] > 0.9


def test_block_seeds_are_distinct_and_reproducible():
    from flymario.policy.readout import block_seeds

    a, b = block_seeds(6, master_seed=0), block_seeds(6, master_seed=0)
    assert a == b
    assert len(set(a)) == 6
    assert a != block_seeds(6, master_seed=1)


def test_every_readout_declares_a_category():
    from flymario.agent import MotorPopulationReadout, RandomReadout
    from flymario.policy.readout import FixedRandomProjection

    cats = {FixedRandomProjection: "fixed", RandomReadout: "floor_control",
            MotorPopulationReadout: "demonstration"}
    for cls, expected in cats.items():
        assert cls.CATEGORY == expected, cls.__name__
    assert len(set(cats.values())) == len(cats), "categories must not collide"
