"""Build family-c null graphs (nullin:<s>) and record their sha256 (RETINO_ALIGN_V1 preparation).

Run on the owner's PC before the rental (2026-10-19) so that the server's rebuild can be compared file by
file: identical hashes mean the ensemble is reproduced bit for bit across machines; any difference is
recorded, not hidden (deployment/HANDOFF_RETINO.md).

    .venv/bin/python scripts/build_nullin_hashes.py --seeds 0-39,100,101 --workers 4 \
        --out artifacts/retino_align_prep/nullin_hashes_local.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
KIND = "depth_stratified_lateral_instrength"


def seeds_of(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def build(seed: int) -> dict:
    from flymario.connectome import datasets, nullcache
    t = time.time()
    g = datasets.load(datasets.DEFAULT)
    _null, status = nullcache.load_or_build(KIND, g, seed, datasets.DEFAULT)
    p = nullcache.null_path(KIND, datasets.DEFAULT, seed)
    return {"seed": seed, "spec": f"nullin:{seed}", "status": status, "seconds": round(time.time() - t, 1),
            "file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-39,100,101")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import numpy
    import scipy
    try:
        import igraph
        igv = igraph.__version__
    except ImportError:
        igv = None
    rec = {"what": "sha256 of family-c null graphs (nullin), RETINO_ALIGN_V1 preparation", "kind": KIND,
           "host": platform.node(), "python": platform.python_version(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "igraph": igv, "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "graphs": {}}
    if a.out.exists():
        rec["graphs"] = json.loads(a.out.read_text()).get("graphs", {})
    todo = [s for s in seeds_of(a.seeds) if f"nullin:{s}" not in rec["graphs"]]
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
            print(json.dumps(r), flush=True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
