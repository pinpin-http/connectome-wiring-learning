"""Dataset registry: which connectome the study runs on is a config change, never an edit.

Everything downstream consumes `ConnectomeGraph`; the adapters are the only place a
dataset's quirks are allowed to exist.

    from flymario.connectome import datasets
    g = datasets.load()                   # the primary dataset, whatever it is
    g = datasets.load("flywire")          # the replication axis, explicitly
"""
from __future__ import annotations

from typing import Callable

from flymario.connectome import flywire, malecns
from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import CONNECTOME

#: name -> (loader, human description, motor super-class vocabulary)
REGISTRY: dict[str, dict] = {
    "flywire": {
        "load": flywire.load,
        "about": "FlyWire 783, adult female brain. Replication axis since 2026-09-21.",
        "motor_classes": ("descending", "motor"),
        "sensory_classes": ("sensory", "visual_projection", "optic"),
        "lamina_types": ("L1", "L2", "L3", "L5"),
        # Cell-type spellings differ between datasets; anything that selects
        # neurons by name must come through here, or it silently returns none.
        "photoreceptor_types": ("R1-6", "R7", "R8"),
    },
    "malecns": {
        "load": malecns.load,
        "about": "MaleCNS v1.0, adult male brain + ventral nerve cord. Primary dataset.",
        "motor_classes": ("descending_neuron", "vnc_motor", "cb_motor",
                          "efferent_descending", "vnc_efferent"),
        "sensory_classes": ("ol_sensory", "cb_sensory", "vnc_sensory",
                            "visual_projection", "ol_intrinsic"),
        "lamina_types": ("L1", "L2", "L3", "L5"),
        # MaleCNS writes "R1-R6" where FlyWire writes "R1-6", and splits R7/R8
        # into pale/yellow/dorsal subtypes.
        "photoreceptor_types": ("R1-R6", "R7y", "R8y", "R7p", "R8p", "R7d", "R8d",
                                "R7_unclear", "R8_unclear", "R7R8_unclear"),
    },
}

#: MaleCNS covers brain and ventral nerve cord, so its 2,227 motor units drive muscles
#: (FlyWire's 1,413 descending neurons are the command bus leaving the brain), and its
#: readout stays active at a global rate where FlyWire's was silent.
DEFAULT = "malecns"


def spec(name: str = DEFAULT) -> dict:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def ledger_path(name: str = DEFAULT, min_syn: int = 1) -> "object":
    """Provenance file for one built variant; the suffix keeps the min_syn=5 sensitivity
    from overwriting the primary graph's provenance."""
    return CONNECTOME / cache_path(name, min_syn).name.replace(".npz", "_ledger.json")


def build(name: str = DEFAULT, min_syn: int = 1) -> ConnectomeGraph:
    """Build from the raw tables. Slow; prefer `load` for the cached version."""
    return spec(name)["load"](min_syn=min_syn,
                              ledger_path=ledger_path(name, min_syn))


def cache_path(name: str = DEFAULT, min_syn: int = 1) -> "object":
    suffix = "" if min_syn == 1 else f"_min{min_syn}"
    return CONNECTOME / f"{name}{suffix}.npz"


def load(name: str = DEFAULT, min_syn: int = 1) -> ConnectomeGraph:
    """Load the cached graph, building it if the cache is missing."""
    p = cache_path(name, min_syn)
    if p.exists():
        return ConnectomeGraph.load(p)
    g = build(name, min_syn)
    g.save(p)
    return g


def of(g: ConnectomeGraph) -> str:
    """Which registry entry a built graph came from, by substring of its own `source`
    (null models carry `"<base>-<null>-s<seed>"` and keep their base's vocabularies)."""
    for name in REGISTRY:
        if name in (g.source or ""):
            return name
    raise ValueError(f"cannot map graph source {g.source!r} to a dataset")


def default_cache_name(min_syn: int = 1) -> str:
    """Filename of the primary dataset's cache, for scripts that take `--graph`."""
    return cache_path(DEFAULT, min_syn).name


def by_types(g: ConnectomeGraph, name: str, key: str):
    """Select neurons by a named type group, using this dataset's spelling."""
    import numpy as np

    return np.flatnonzero(np.isin(g.node_type, spec(name)[key])).astype(np.int32)


def motor_index(g: ConnectomeGraph, name: str = DEFAULT):
    """Readout population, using this dataset's super-class names ('descending' vs
    'descending_neuron'); a mismatch silently yields an empty readout."""
    import numpy as np

    return np.flatnonzero(np.isin(g.node_class, spec(name)["motor_classes"])).astype(np.int32)
