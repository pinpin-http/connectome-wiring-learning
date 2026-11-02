"""Pick the wiring a run trains and is evaluated on: the measured connectome or one
of its cached structural nulls (PROTOCOL §8). Training and every evaluation of a
run must use the same spec, or learnt weights would be read through another graph.

    spec "connectome"   the MaleCNS graph
    spec "null:<s>"     the depth-stratified null with seed s (artifacts/cache/nulls)
    spec "nulllat:<s>"  the same with the side of each endpoint kept (2026-10-06)
    spec "nullin:<s>"   nulllat plus every neuron's in-strength kept (2026-10-10, RL_STUDY §26)
"""
from __future__ import annotations

from flymario.connectome import datasets, nullcache


def load(spec: str = "connectome"):
    """(graph for the eye and node identities, sparse W to simulate, motor index)."""
    g = datasets.load(datasets.DEFAULT)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    if spec == "connectome":
        return g, g.to_sparse(), motor
    kind, seed = spec.split(":")
    kind = {"null": "depth_stratified", "nulllat": "depth_stratified_lateral",
            "nullin": "depth_stratified_lateral_instrength",
            # revision48 (R3): autapse-preserving versions of the two families
            "nulllatL": "depth_stratified_lateral_keeploops",
            "nullinL": "depth_stratified_lateral_instrength_keeploops"}.get(kind, kind)
    null, status = nullcache.load_or_build(kind, g, int(seed), datasets.DEFAULT, build=False)
    if null is None:
        raise FileNotFoundError(f"{spec}: null not cached ({status}); build it first")
    W, W0 = null.to_sparse(), g.to_sparse()
    if W.shape != W0.shape:
        raise ValueError(f"{spec}: {W.shape} nodes vs {W0.shape} for the connectome")
    return g, W, motor
