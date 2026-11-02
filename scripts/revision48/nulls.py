#!/usr/bin/env python
"""Autapse-preserving nulls (reviewer R3) and the regression check of the historical builder.

    nulls.py build --specs nulllatL:100,nullinL:100 [--workers 4]   build into artifacts/cache/nulls (CPU)
    nulls.py validate --specs ...                                     invariants -> artifacts/revision48/nulls/
    nulls.py regression --spec nullin:100                             rebuild a historical null in memory with the
                                                                      current code; its file must equal the archived one

Specs: nulllatL:<s> = family S (out-strength kept), nullinL:<s> = family E (in-strength kept), both keeping
the connectome's autapses and adding none. Validation checks degrees, sign and weight multisets, stratum x side
edge classes, the family's strength, autapses, duplicates and edge overlap; it does not show uniform sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts/revision48/nulls"
KIND = {"nulllat": "depth_stratified_lateral", "nullin": "depth_stratified_lateral_instrength",
        "nulllatL": "depth_stratified_lateral_keeploops", "nullinL": "depth_stratified_lateral_instrength_keeploops"}
HIST = {"nulllatL": "nulllat", "nullinL": "nullin"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def _build(spec: str) -> dict:
    from flymario.connectome import datasets, nullcache
    fam, seed = spec.split(":")
    t = time.time()
    g = datasets.load(datasets.DEFAULT)
    _n, status = nullcache.load_or_build(KIND[fam], g, int(seed), datasets.DEFAULT)
    p = nullcache.null_path(KIND[fam], datasets.DEFAULT, int(seed))
    return {"spec": spec, "status": status, "seconds": round(time.time() - t, 1), "file": p.name,
            "sha256": sha(p), "bytes": p.stat().st_size}


def cmd_build(a) -> int:
    rec_p = OUT / "build.json"
    rec = json.loads(rec_p.read_text()) if rec_p.exists() else {"graphs": {}}
    specs = [s for s in a.specs.split(",") if s]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_build, s) for s in specs]):
            r = f.result()
            rec["graphs"][r["spec"]] = r
            write_json(rec_p, rec)
            print(json.dumps(r), flush=True)
    return 0


def _strata_classes(g, sensory, readout):
    from flymario.connectome import strata
    st = strata.compute(g, sensory, readout, "both")
    cls = np.asarray(st.edge_class(g.src, g.dst, g.sign), np.int64)
    _u, side = np.unique(np.asarray(g.node_side), return_inverse=True)
    k = int(side.max()) + 1
    return (cls * k + side[g.src]) * k + side[g.dst]


def validate(spec: str) -> dict:
    from flymario.connectome import datasets, nullcache, randomize
    from flymario.connectome.schema import ConnectomeGraph
    fam, seed = spec.split(":")
    g = datasets.load(datasets.DEFAULT)
    p = nullcache.null_path(KIND[fam], datasets.DEFAULT, int(seed))
    n = ConnectomeGraph.load(p)
    sensory = datasets.by_types(g, "malecns", "lamina_types")
    readout = datasets.motor_index(g, "malecns")
    N = g.n_nodes
    r: dict = {"spec": spec, "file": p.name, "sha256": sha(p)}
    r["node_ids_identical"] = bool(np.array_equal(g.node_ids, n.node_ids))
    r["edge_count_equal"] = int(g.n_edges) == int(n.n_edges)
    od, idg = randomize.degrees(g)
    od2, id2 = randomize.degrees(n)
    r["out_degree_exact"], r["in_degree_exact"] = bool(np.array_equal(od, od2)), bool(np.array_equal(idg, id2))
    r["sign_multiset_equal"] = bool(np.array_equal(np.sort(g.sign), np.sort(n.sign)))
    r["weight_multiset_equal"] = bool(np.array_equal(np.sort(g.weight), np.sort(n.weight)))
    c0, c1 = _strata_classes(g, sensory, readout), _strata_classes(n, sensory, readout)
    r["stratum_side_classes_equal"] = bool(np.array_equal(np.sort(c0), np.sort(c1)))
    sw0, sw1 = g.weight * g.sign, n.weight * n.sign
    out0, out1 = np.bincount(g.src, sw0, N), np.bincount(n.src, sw1, N)
    in0, in1 = np.bincount(g.dst, sw0, N), np.bincount(n.dst, sw1, N)
    r["signed_out_strength_exact"] = bool(np.array_equal(out0, out1))
    r["signed_in_strength_exact"] = bool(np.array_equal(in0, in1))
    r["family_strength_kept"] = r["signed_in_strength_exact"] if fam.startswith("nullin") else r["signed_out_strength_exact"]
    loops0 = sorted(zip(g.src[g.src == g.dst].tolist(), g.weight[g.src == g.dst].tolist(), g.sign[g.src == g.dst].tolist()))
    loops1 = sorted(zip(n.src[n.src == n.dst].tolist(), n.weight[n.src == n.dst].tolist(), n.sign[n.src == n.dst].tolist()))
    r["autapses_connectome"], r["autapses_null"] = len(loops0), len(loops1)
    r["autapses_identical"] = loops0 == loops1
    pairs = n.src.astype(np.int64) * N + n.dst
    r["duplicate_pairs"] = int(len(pairs) - len(np.unique(pairs)))
    r["edge_overlap_connectome"] = randomize.mixing_report(g, n, sample_size=10**9)["edge_overlap_sampled"]
    hist = nullcache.null_path(KIND[HIST[fam]], datasets.DEFAULT, int(seed)) if fam in HIST else None
    if hist is not None and hist.exists():
        h = ConnectomeGraph.load(hist)
        r["historical_null_autapses"] = int((h.src == h.dst).sum())
        r["edge_overlap_historical_same_seed"] = randomize.mixing_report(h, n, sample_size=10**9)["edge_overlap_sampled"]
    must = ["node_ids_identical", "edge_count_equal", "out_degree_exact", "in_degree_exact", "sign_multiset_equal",
            "weight_multiset_equal", "stratum_side_classes_equal", "family_strength_kept", "autapses_identical"]
    r["pass"] = all(r[k] for k in must) and r["duplicate_pairs"] == 0
    r["note"] = "invariants hold; this does not show that the swap chain sampled uniformly"
    return r


def cmd_validate(a) -> int:
    specs = [s for s in a.specs.split(",") if s]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(validate, s) for s in specs]):
            r = f.result()
            res[r["spec"]] = r
            print(json.dumps({k: r[k] for k in ("spec", "pass", "autapses_null", "duplicate_pairs",
                                                "edge_overlap_connectome")}), flush=True)
    rec_p = OUT / "validate.json"
    rec = json.loads(rec_p.read_text()) if rec_p.exists() else {}
    rec.update(res)
    write_json(rec_p, rec)
    return 0 if all(r["pass"] for r in res.values()) else 1


def cmd_regression(a) -> int:
    """The modified builder must still produce the historical nulls bit for bit."""
    from flymario.connectome import datasets, nullcache, randomize
    fam, seed = a.spec.split(":")
    g = datasets.load(datasets.DEFAULT)
    ref = nullcache.null_path(KIND[fam], datasets.DEFAULT, int(seed))
    t = time.time()
    n = randomize.make(KIND[fam], g, seed=int(seed))
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as d:
        p = Path(d) / ref.name
        n.save(p)
        same = sha(p) == sha(ref)
    out = {"spec": a.spec, "archived_file": ref.name, "rebuilt_identical": same, "seconds": round(time.time() - t, 1)}
    write_json(OUT / f"regression_{a.spec.replace(':', '')}.json", out)
    print(json.dumps(out))
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "validate", "regression"])
    ap.add_argument("--specs", default="")
    ap.add_argument("--spec", default="nullin:100")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return {"build": cmd_build, "validate": cmd_validate, "regression": cmd_regression}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
