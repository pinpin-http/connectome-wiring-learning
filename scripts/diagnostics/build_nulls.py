#!/usr/bin/env python
"""Build (and cache) laterality-preserving null realisations nulllat:<s> for the given seeds
(RL_STUDY §24). Each is fully determined by its seed (randomize.depth_stratified_lateral), saved
under artifacts/cache/nulls and read back for a round-trip check by nullcache.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/diagnostics/build_nulls.py 3 4 5
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from flymario.connectome import datasets, nullcache  # noqa: E402

g = datasets.load(datasets.DEFAULT)
kind = "depth_stratified_lateral"
args = sys.argv[1:]
if args and args[0] == "--instrength":         # RL_STUDY §26
    kind, args = "depth_stratified_lateral_instrength", args[1:]
for s in map(int, args):
    t0 = time.time()
    _null, status = nullcache.load_or_build(kind, g, s, datasets.DEFAULT, build=True)
    print(f"{kind}:{s} {status} {time.time() - t0:.0f}s", flush=True)
