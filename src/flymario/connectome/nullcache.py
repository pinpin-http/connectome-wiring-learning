"""Null realisations cached on disk, keyed on the graph they were drawn from.

A `depth_stratified` null on MaleCNS takes ~4 minutes to build and is fully
determined by its seed, so it is built once and reused -- by the measurement
campaigns and by the desktop lab, which can swap the real wiring for its
rewired copy while the fly plays. One module, so both read the same files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from flymario.connectome import datasets, randomize
from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import ROOT

NULL_CACHE = ROOT / "artifacts" / "cache" / "nulls"


def null_path(kind: str, dataset: str, seed: int) -> Path:
    """Where the realisation lives. The name carries the hash of the source
    graph, so a rebuilt graph can never be paired with a stale null."""
    base = hashlib.sha256(datasets.cache_path(dataset).read_bytes()).hexdigest()[:12]
    return NULL_CACHE / f"{dataset}_{base}_{kind}_s{seed}.npz"


def load_or_build(kind: str, g: ConnectomeGraph, seed: int, dataset: str,
                  build: bool = True) -> tuple[ConnectomeGraph | None, str]:
    """(graph, "cache" | "built" | "missing"). With build=False a missing
    realisation is reported, not built -- the lab must not freeze for minutes."""
    path = null_path(kind, dataset, seed)
    if path.exists():
        return ConnectomeGraph.load(path), "cache"
    if not build:
        return None, "missing"
    null = randomize.make(kind, g, seed=seed)
    null.save(path)
    # A cache nobody checked is a new way to be wrong: read it straight back
    # and require the wiring to match what was just built.
    back = ConnectomeGraph.load(path)
    for name in ("src", "dst", "weight", "sign"):
        if not np.array_equal(getattr(back, name), getattr(null, name)):
            path.unlink()
            raise RuntimeError(f"null cache round-trip changed `{name}`")
    return null, "built"
