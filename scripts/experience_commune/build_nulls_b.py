#!/usr/bin/env python
"""Build and validate the family-b nulls (nulllat:<s>, depth_stratified_lateral) EXPERIENCE_COMMUNE_V1 needs.

No family-b file existed on this server (2026-10-20); the earlier ensemble was not archived, so these are a
reconstruction from the recorded seeds with this machine's pinned stack, not a bit-identical replay. Each file's
sha256 is recorded; each graph is checked against the connectome for the invariants the construction promises:
same nodes and edge count, exact in- and out-degrees, sign and weight multisets, exact per-source signed and
unsigned out-strength, per-source sign counts, and identical depth-stratum x side edge classes.

    .venv/bin/python scripts/experience_commune/build_nulls_b.py --seeds 100,39,21 --workers 2 \
        --out artifacts/experience_commune_v1/manifests/nulls_b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
KIND = "depth_stratified_lateral"


def seeds_of(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def validate(g0, g) -> dict:
    from flymario.connectome import datasets, strata
    n = len(g0.node_ids)
    r = {"node_ids_identical": bool(np.array_equal(g0.node_ids, g.node_ids)),
         "n_edges_equal": len(g.src) == len(g0.src)}
    for nm, a0, a1 in (("out_degree", g0.src, g.src), ("in_degree", g0.dst, g.dst)):
        r[f"{nm}_exact"] = bool(np.array_equal(np.bincount(a0, minlength=n), np.bincount(a1, minlength=n)))
    r["sign_multiset_equal"] = bool(np.array_equal(np.sort(g0.sign), np.sort(g.sign)))
    r["weight_multiset_equal"] = bool(np.array_equal(np.sort(g0.weight), np.sort(g.weight)))
    for nm, w0, w1 in (("signed", g0.weight * g0.sign, g.weight * g.sign), ("unsigned", g0.weight, g.weight)):
        r[f"outstrength_{nm}_exact"] = bool(np.array_equal(np.bincount(g0.src, w0.astype(np.float64), n),
                                                           np.bincount(g.src, w1.astype(np.float64), n)))
    r["instrength_unsigned_max_abs_change"] = float(np.abs(np.bincount(g0.dst, g0.weight.astype(np.float64), n)
                                                           - np.bincount(g.dst, g.weight.astype(np.float64), n)).max())
    for s in (-1, 1):
        r[f"per_source_sign{s:+d}_counts_exact"] = bool(np.array_equal(
            np.bincount(g0.src[g0.sign == s], minlength=n), np.bincount(g.src[g.sign == s], minlength=n)))
    sens = datasets.by_types(g0, "malecns", "lamina_types")
    ro = datasets.motor_index(g0, "malecns")
    st = strata.compute(g0, sens, ro, "both")
    _u, side = np.unique(np.asarray(g0.node_side), return_inverse=True)
    k = int(side.max()) + 1

    def classes(gg):
        c = (np.asarray(st.edge_class(gg.src, gg.dst, gg.sign), np.int64) * k + side[gg.src]) * k + side[gg.dst]
        return np.unique(c, return_counts=True)
    c0, c1 = classes(g0), classes(g)
    r["strata_side_edge_classes_equal"] = bool(np.array_equal(c0[0], c1[0]) and np.array_equal(c0[1], c1[1]))
    key0 = g0.src.astype(np.int64) * n + g0.dst
    key1 = g.src.astype(np.int64) * n + g.dst
    r["duplicate_pairs"] = int(len(key1) - len(np.unique(key1)))
    r["edge_overlap"] = float(np.isin(key1, key0).mean())
    r["pass"] = all(v for kk, v in r.items() if kk.endswith(("_exact", "_equal", "_identical"))) and r["duplicate_pairs"] == 0
    return r


def build(seed: int) -> dict:
    from flymario.connectome import datasets, nullcache
    t = time.time()
    g0 = datasets.load(datasets.DEFAULT)
    null, status = nullcache.load_or_build(KIND, g0, seed, datasets.DEFAULT)
    p = nullcache.null_path(KIND, datasets.DEFAULT, seed)
    rec = {"seed": seed, "spec": f"nulllat:{seed}", "status": status, "build_s": round(time.time() - t, 1),
           "file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    t = time.time()
    rec["validation"] = validate(g0, null)
    rec["validate_s"] = round(time.time() - t, 1)
    rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import scipy
    rec = {"kind": KIND, "numpy": np.__version__, "scipy": scipy.__version__,
           "started_utc": datetime.now(UTC).isoformat(timespec="seconds"), "graphs": {}}
    if a.out.exists():
        rec["graphs"] = json.loads(a.out.read_text()).get("graphs", {})
    todo = [s for s in seeds_of(a.seeds) if f"nulllat:{s}" not in rec["graphs"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = a.out.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, a.out)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(build, s): s for s in todo}
        for f in as_completed(futs):
            r = f.result()
            rec["graphs"][r["spec"]] = r
            rec["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            save()
            print(r["spec"], r["status"], r["build_s"], "s", "pass" if r["validation"]["pass"] else "FAIL", flush=True)
    save()
    return 0 if all(g["validation"]["pass"] for g in rec["graphs"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
