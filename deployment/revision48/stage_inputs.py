#!/usr/bin/env python
"""Graph inputs of the revision campaign: one manifest, verified on the PC and again on the server.

    stage_inputs.py manifest   write configs/revision48/graph_manifest.json from the X3 selection
                               (artifacts/revision48_prep/inputs.json) plus the development graphs; every
                               sha256 is read from the archived files themselves
    stage_inputs.py link       PC only: verify each archived file and symlink it where graphsel looks
                               (data/connectome, artifacts/cache/nulls); nothing is copied or modified
    stage_inputs.py rebuild    server: rebuild the historical nulls from their seeds (the builder is
                               reproducible across machines) with --workers processes, then verify
    stage_inputs.py verify     every graph of the manifest is present with its sha256 (exit 1 otherwise)

The connectome itself travels (deployment `core` set); nulls never need to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MANIFEST = ROOT / "configs/revision48/graph_manifest.json"
PREP = ROOT / "artifacts/revision48_prep/inputs.json"
ARCHIVE = Path("/home/soheib/Downloads/campagne")
GRAPH = "6befb67a0278568316c26d32c3f2cadf76fbb8cc35d93137819c86f804a1df9f"
KIND = {"nulllat": "depth_stratified_lateral", "nullin": "depth_stratified_lateral_instrength"}
FAMILY = {"nulllat": "S", "nullin": "E"}          # the manuscript's names: S out-strength kept, E in-strength kept
DEV = ["nulllat:100", "nullin:100", "nullin:101"]  # development / pilot only, never analysed


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_name(spec: str) -> str:
    fam, seed = spec.split(":")
    return f"malecns_{GRAPH[:12]}_{KIND[fam]}_s{seed}.npz"


def dest(spec: str) -> Path:
    return ROOT / ("data/connectome/malecns.npz" if spec == "connectome" else f"artifacts/cache/nulls/{cache_name(spec)}")


def cmd_manifest(_a) -> int:
    prep = json.loads(PREP.read_text())
    rows = []
    for r in prep["inputs"]:
        spec = r["graph"]
        role = "connectome" if spec == "connectome" else FAMILY[spec.split(":")[0]]
        rows.append({"spec": spec, "role": role, "file": str(dest(spec).relative_to(ROOT)), "sha256": r["sha256"],
                     "bytes": r["bytes"], "archived_source": r["path"]})
    found = list(ARCHIVE.rglob("artifacts/cache/nulls/*.npz"))
    for spec in DEV:
        m = [p for p in found if p.name == cache_name(spec)]
        if len(m) != 1:
            raise SystemExit(f"{spec}: expected one archived file, found {len(m)}")
        rows.append({"spec": spec, "role": "dev", "file": str(dest(spec).relative_to(ROOT)), "sha256": sha(m[0]),
                     "bytes": m[0].stat().st_size, "archived_source": str(m[0])})
    man = {"what": "graphs of the revision campaign: the connectome, the X3 selection (10 S = nulllat, 10 E = nullin; "
                   "selected before any new result, historical subset, not a new random sample) and the development "
                   "graphs (pilot only)",
           "selection_source": prep["selection_source"], "selection_member": prep["selection_member"],
           "graph_sha256": GRAPH, "graphs": rows,
           "order": [r["spec"] for r in rows if r["role"] != "dev"]}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    print(f"{len(rows)} graphs -> {MANIFEST.relative_to(ROOT)}")
    return 0


def cmd_link(_a) -> int:
    man = json.loads(MANIFEST.read_text())
    for r in man["graphs"]:
        src, d = Path(r["archived_source"]), ROOT / r["file"]
        if sha(src) != r["sha256"]:
            raise SystemExit(f"{r['spec']}: archived file does not match the manifest")
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.is_symlink() or d.exists():
            if d.resolve() == src.resolve():
                continue
            raise SystemExit(f"{d} exists and is not the archived file: not replaced")
        d.symlink_to(src)
        print("linked", r["spec"])
    return 0


def _build(spec: str) -> dict:
    from flymario.connectome import datasets, nullcache
    fam, seed = spec.split(":")
    g = datasets.load(datasets.DEFAULT)
    _n, status = nullcache.load_or_build(KIND[fam], g, int(seed), datasets.DEFAULT)
    return {"spec": spec, "status": status}


def cmd_rebuild(a) -> int:
    man = json.loads(MANIFEST.read_text())
    todo = [r["spec"] for r in man["graphs"] if r["spec"] != "connectome" and not (ROOT / r["file"]).exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_build, s) for s in todo]):
            print(json.dumps(f.result()), flush=True)
    return cmd_verify(a)


def cmd_verify(_a) -> int:
    man = json.loads(MANIFEST.read_text())
    bad = []
    for r in man["graphs"]:
        p = ROOT / r["file"]
        ok = p.exists() and sha(p) == r["sha256"]
        print(f"{'ok ' if ok else 'BAD'} {r['spec']:12s} {r['role']:10s} {r['file']}")
        if not ok:
            bad.append(r["spec"])
    if bad:
        print("not identical to the archived graphs:", ", ".join(bad))
        return 1
    print(f"all {len(man['graphs'])} graphs identical to the archived ones (sha256)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["manifest", "link", "rebuild", "verify"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return {"manifest": cmd_manifest, "link": cmd_link, "rebuild": cmd_rebuild, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
