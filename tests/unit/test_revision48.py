"""REVISION48 pieces on small synthetic inputs (no network simulation, no GPU): the analysis on curves with a known
effect, the autapse-preserving rewiring on a small graph, the log conversion, the fixed-order frozen product and the
tuning selection rule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/revision48"))

from flymario.revision48 import analysis as A

TASKS = ("where", "loom")
CK = [0, 250, 500, 1000, 2000, 4000, 6000]


def synth(effect: float, spread: float, seed: int, specs=None, n=64, R=6) -> dict:
    """Correctness arrays whose expected learning area differs by `effect` for the connectome; nulls spread by
    `spread` around their family mean."""
    rng = np.random.default_rng(seed)
    specs = specs or (["connectome"] + [f"nulllat:{i}" for i in range(10)] + [f"nullin:{i}" for i in range(10)])
    G, T, C = len(specs), len(TASKS), len(CK)
    cls_p = np.tile(np.repeat(np.arange(8), n // 8), (R, T, 1))
    cls_f = np.tile(np.repeat(np.arange(8), 32), (R, T, 1))
    base = np.linspace(0.125, 0.6, C)
    X = np.zeros((G, T, R, C, n), np.uint8)
    F = np.zeros((G, T, R, 256), np.uint8)
    for g in range(G):
        shift = effect if g == 0 else spread * rng.standard_normal()
        p = np.clip(base + shift * (np.arange(C) > 0), 0, 1)
        X[g] = rng.random((T, R, C, n)) < p[None, None, :, None]
        F[g] = rng.random((T, R, 256)) < np.clip(0.6 + shift, 0, 1)
    return {"X": X, "F": F, "cls_p": cls_p, "cls_f": cls_f, "roles": A.roles_of(specs), "specs": specs}


def test_trapezoid_is_a_mean_accuracy():
    w = A.trapezoid_weights(CK)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(np.full(len(CK), 0.4) @ w, 0.4)


def test_known_advantage_is_recovered_and_read_as_an_advantage():
    c = synth(0.15, 0.01, 1)
    rep = A.report({"tuned": c}, TASKS, CK, n_rep=300, seed=3, margin=0.03, primary="tuned")
    for r in rep["contrasts"]:
        assert r["G"] > 0.10 and r["lo"] > 0.03
        assert r["reading"] == "practical advantage in this design"


def test_identical_graphs_give_zero_and_equivalence():
    c = synth(0.0, 0.0, 2)
    c["X"][:] = c["X"][0:1]
    c["F"][:] = c["F"][0:1]
    rep = A.report({"tuned": c}, TASKS, CK, n_rep=200, seed=4, margin=0.03)
    for r in rep["contrasts"]:                                          # zero up to float rounding of the means
        assert abs(r["G"]) < 1e-9 and abs(r["lo"]) < 1e-9 and abs(r["hi"]) < 1e-9
        assert r["reading"] == "practical equivalence at this margin"


def test_change_after_tuning_is_paired():
    ref = synth(0.10, 0.02, 5)
    same = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ref.items()}
    rep = A.report({"reference": ref, "tuned": same}, TASKS, CK, n_rep=200, seed=6, margin=0.03)
    for d in rep["change_after_tuning"]:
        assert d["D"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0     # identical conditions: zero in every replicate


def test_graph_spread_widens_the_interval():
    # 512 progress images per block, as in the real banks (with 64, image noise dominates the width)
    narrow = A.report({"tuned": synth(0.05, 0.002, 7, n=512)}, TASKS, CK, n_rep=200, seed=8, margin=0.03)
    wide = A.report({"tuned": synth(0.05, 0.08, 7, n=512)}, TASKS, CK, n_rep=200, seed=8, margin=0.03)
    wn = np.mean([r["hi"] - r["lo"] for r in narrow["contrasts"]])
    ww = np.mean([r["hi"] - r["lo"] for r in wide["contrasts"]])
    assert ww > 2 * wn


def test_reading_table():
    assert A.reading(0.04, 0.1, 0.03) == "practical advantage in this design"
    assert A.reading(-0.02, 0.02, 0.03) == "practical equivalence at this margin"
    assert A.reading(-0.1, -0.04, 0.03) == "practical disadvantage in this design"
    assert A.reading(0.0, 0.1, 0.03) == "not settled at this precision"


def _toy_graph(seed=0, n=60):
    from flymario.connectome.schema import ConnectomeGraph
    rng = np.random.default_rng(seed)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, n, (900, 2))}
    pairs |= {(i, i) for i in range(0, n, 7)}                               # a few autapses
    src, dst = map(np.array, zip(*sorted(pairs)))
    sign = np.where(rng.random(n) < 0.7, 1.0, -1.0)[src]
    return ConnectomeGraph(node_ids=np.arange(n), src=src, dst=dst, weight=rng.integers(1, 9, len(src)).astype(np.float32),
                           sign=sign.astype(np.float32), node_type=np.array(["x"] * n), node_class=np.array(["c"] * n),
                           node_side=np.array(["L", "R"] * (n // 2)), node_pos=rng.random((n, 3)).astype(np.float32),
                           source="toy", meta={})


@pytest.mark.parametrize("weights_by", ["src", "dst"])
def test_keep_loops_rewiring_keeps_autapses_and_creates_none(weights_by):
    from flymario.connectome import randomize
    g = _toy_graph()
    sens, ro = np.arange(0, 6), np.arange(50, 60)
    hist = randomize.depth_stratified(g, seed=3, lateral=True, weights_by=weights_by, sensory=sens, readout=ro)
    new = randomize.depth_stratified(g, seed=3, lateral=True, weights_by=weights_by, sensory=sens, readout=ro,
                                     keep_loops=True)
    loops = lambda x: sorted(zip(x.src[x.src == x.dst], x.weight[x.src == x.dst], x.sign[x.src == x.dst]))
    assert loops(new) == loops(g)                                            # identity, weight, sign
    od, idg = randomize.degrees(g)
    od2, id2 = randomize.degrees(new)
    assert np.array_equal(od, od2) and np.array_equal(idg, id2)
    assert np.array_equal(np.sort(g.weight), np.sort(new.weight))
    pairs = new.src * g.n_nodes + new.dst
    assert len(np.unique(pairs)) == len(pairs)                               # no multi-edge
    sw, sw2 = g.weight * g.sign, new.weight * new.sign
    key = (g.src, new.src) if weights_by == "src" else (g.dst, new.dst)
    assert np.allclose(np.bincount(key[0], sw, g.n_nodes), np.bincount(key[1], sw2, g.n_nodes))
    assert (hist.src == hist.dst).sum() >= 0                                 # historical builder still runs


def test_lognorm_operator_logs_after_summing_and_normalises_rows():
    from flymario.experience_commune.learn import lognorm_operator
    W = sp.csr_matrix((np.array([3.0, 4.0, -7.0, 2.0], np.float32), (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 2]))),
                      shape=(3, 3))                                          # row 0 has a duplicate (3 + 4)
    L = lognorm_operator(W).toarray()
    assert np.isclose(L[0, 1], 1.0)                                         # log1p(7) normalised alone in its row
    assert np.isclose(np.abs(L[1]).sum(), 1.0)
    assert L[1, 0] < 0 and np.isclose(-L[1, 0] / L[1, 2], np.log1p(7) / np.log1p(2))


def test_segment_operator_equals_the_csr_product():
    import torch

    from flymario.experience_commune.learn import SegmentOperator
    rng = np.random.default_rng(0)
    W = sp.random(40, 40, density=0.2, random_state=1, format="csr", dtype=np.float32)
    T = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int64)), torch.from_numpy(W.indices.astype(np.int64)),
                                torch.from_numpy(W.data), size=W.shape)
    x = torch.from_numpy((rng.random((40, 37)) < 0.3).astype(np.float32))
    op = SegmentOperator(T, chunk_cols=8)
    assert torch.allclose(op @ x, T @ x, atol=1e-6)
    assert torch.equal(op @ x, op @ x)


def test_exact64_operator_is_the_exact_sum_whatever_the_order():
    import torch

    from flymario.experience_commune.learn import Exact64Operator, lognorm_operator
    rng = np.random.default_rng(2)
    raw = sp.random(60, 60, density=0.3, random_state=3, format="csr", dtype=np.float32)
    raw.data = np.ceil(raw.data * 20) * np.where(rng.random(raw.nnz) < 0.3, -1, 1).astype(np.float32)
    W = lognorm_operator(raw)
    T = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int64)), torch.from_numpy(W.indices.astype(np.int64)),
                                torch.from_numpy(W.data), size=W.shape)
    x = (rng.random((60, 12)) < 0.4).astype(np.float32)
    op = Exact64Operator(T)
    assert op.max_grid_change == 0.0                  # float32 log weights are already on the grid
    exact = (W.astype(np.float64).toarray() @ x.astype(np.float64)).astype(np.float32)
    y = op @ torch.from_numpy(x)
    assert np.array_equal(y.numpy(), exact)
    # reversed column order inside every row: the same bits
    Wr = sp.csr_matrix(W[:, ::-1])
    Tr = torch.sparse_csr_tensor(torch.from_numpy(Wr.indptr.astype(np.int64)), torch.from_numpy(Wr.indices.astype(np.int64)),
                                 torch.from_numpy(Wr.data), size=Wr.shape)
    assert torch.equal(Exact64Operator(Tr) @ torch.from_numpy(x[::-1].copy()), y)


def test_tuning_selection_takes_the_best_mean_and_the_lowest_trial_on_ties(tmp_path, monkeypatch):
    import r48
    c = {"tuning": {"checkpoints": [0, 750, 1500], "trials": [
        {"trial": t, "gamma": g, "eta": e, "tau": u} for t, g, e, u in
        [(1, 1.0, 0.001, 2.0), (2, 0.5, 0.001, 2.0), (3, 1.0, 0.003, 1.0), (4, 0.5, 0.003, 1.0)]]}}
    acc = {1: 0.5, 2: 0.7, 3: 0.7, 4: float("nan")}                         # 2 and 3 tie; 4 is invalid
    for gamma, trials in ((1.0, (1, 3)), (0.5, (2, 4))):
        d = tmp_path / "runs/tuning" / f"g{gamma}" / "connectome"
        d.mkdir(parents=True)
        lt = [t for t in trials for _ in range(4)]
        lab = np.zeros((len(lt), 8), np.int8)
        preds = {f"progress_{k}": np.where(np.array([acc[t] for t in lt])[:, None] > np.linspace(0, 1, 8)[None, :], 0, 1)
                 .astype(np.int8) for k in (0, 750, 1500)}
        np.savez(d / "predictions.npz", labels_progress=lab, **preds)
        np.savez(d / "final_weights.npz", m=np.where(np.isnan([acc[t] for t in lt])[:, None], np.nan, 1.0) * np.ones((len(lt), 3)))
        (d / "run.json").write_text(json.dumps({"complete": True}))
        (d / "trials.json").write_text(json.dumps({"learner_trial": lt}))
    sc = r48.tuning_scores(tmp_path, c, "connectome")
    assert sc[4] is None and sc[2] == sc[3]
    valid = {t: v for t, v in sc.items() if v is not None}
    assert max(valid, key=lambda t: (valid[t], -t)) == 2


def test_loader_reads_block_then_task_order_and_follows_reuse(tmp_path):
    """Fake run directories: learner j = block r * T + task t; a tuned directory with reuse.json reads reference."""
    R, T, n, nf, ck = 2, 2, 16, 8, [0, 6000]
    lab_p = np.tile(np.arange(n) % 8, (R * T, 1)).astype(np.int8)
    lab_f = np.tile(np.arange(nf) % 8, (R * T, 1)).astype(np.int8)
    def write(d, correct_rt):                                     # correct_rt[r][t]: fraction of correct predictions
        d.mkdir(parents=True)
        preds = {}
        for k in ck:
            p = lab_p.copy()
            for r in range(R):
                for t in range(T):
                    wrong = round((1 - correct_rt[r][t]) * n)
                    p[r * T + t, :wrong] = (p[r * T + t, :wrong] + 1) % 8
            preds[f"progress_{k}"] = p
        np.savez(d / "predictions.npz", labels_progress=lab_p, labels_final=lab_f, final=lab_f, **preds)
    write(tmp_path / "reference/connectome", [[1.0, 0.5], [0.25, 0.75]])
    (tmp_path / "tuned/connectome").mkdir(parents=True)
    (tmp_path / "tuned/connectome/reuse.json").write_text("{}")
    d = A.load_condition([tmp_path / "tuned/connectome"], ["connectome"], ("where", "loom"), ["r0", "r1"], ck)
    acc = d["X"][0].mean(-1)[..., 0]                             # (T, R)
    assert np.allclose(acc, [[1.0, 0.25], [0.5, 0.75]])
    assert d["cls_p"].shape == (R, T, n) and d["F"].shape == (1, T, R, nf)
