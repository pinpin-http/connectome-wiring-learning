"""Dataset-independent connectome container: every graph (FlyWire, MaleCNS, rewired nulls) uses this type.

Convention: `src[k] -> dst[k]` with strength `weight[k]`, node ids are
contiguous integers `0..n_nodes-1`, and `sign[k]` in {-1, 0, +1} carries the
predicted neurotransmitter polarity. Raw dataset ids live in `node_ids`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

#: Drosophila polarity. Glutamate is predominantly INHIBITORY in the fly
#: (GluCl-alpha) -- the opposite of the vertebrate default, and an easy bug.
NT_SIGN: dict[str, float] = {
    "acetylcholine": +1.0, "ach": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0, "glut": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0, "da": 0.0,
    "octopamine": 0.0, "oct": 0.0,
    "serotonin": 0.0, "ser": 0.0,
    # Deliberately absent: "unclear" and "unknown". An unresolved transmitter carries
    # NO fast current (docs/PROTOCOL.md §7.4). Callers must use .get(nt, 0.0), never [nt].
}


@dataclass
class ConnectomeGraph:
    """A directed, weighted, signed graph plus whatever node metadata exists."""

    node_ids: np.ndarray                 # (n,) original dataset ids (int64)
    src: np.ndarray                      # (m,) contiguous node index
    dst: np.ndarray                      # (m,)
    weight: np.ndarray                   # (m,) float32, >= 0 before signing
    sign: np.ndarray                     # (m,) float32 in {-1, 0, +1}
    node_type: np.ndarray | None = None  # (n,) cell type label
    node_class: np.ndarray | None = None # (n,) super class label
    node_side: np.ndarray | None = None  # (n,) 'L' / 'R' / 'C'
    #: (n,2) retinotopic coordinates in [0,1]^2 for neurons that have one, NaN
    #: otherwise. Supplied by the adapter (MaleCNS: ommatidial columns), not
    #: derived from soma positions.
    node_retino: np.ndarray | None = None
    node_pos: np.ndarray | None = None   # (n,3) nanometre coordinates in the EM
                                         # volume; lets the graph be drawn in the
                                         # real anatomy of the brain
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    # -- basics ------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def validate(self) -> None:
        """Acceptance criteria from the spec. Raises on any violation."""
        n, m = self.n_nodes, self.n_edges
        if len(np.unique(self.node_ids)) != n:
            raise ValueError("duplicate node ids")
        for name, arr in (("src", self.src), ("dst", self.dst)):
            if arr.min() < 0 or arr.max() >= n:
                raise ValueError(f"{name} out of range [0,{n})")
            if len(arr) != m:
                raise ValueError(f"{name} length != n_edges")
        if not np.isfinite(self.weight).all():
            raise ValueError("non-finite weights")
        if (self.weight < 0).any():
            raise ValueError("negative weights before signing; sign lives in .sign")
        if not np.isin(self.sign, (-1.0, 0.0, 1.0)).all():
            raise ValueError("sign must be in {-1,0,+1}")

    # -- conversions -------------------------------------------------------
    def to_sparse(self, signed: bool = True, log_weights: bool | None = None) -> sp.csr_matrix:
        """Adjacency as W[dst, src], so `W @ activity` propagates forward.

        `log_weights=None` follows `dynamics.weight_transform.log1p` in
        configs/base.yaml.
        """
        if log_weights is None:
            from flymario import config
            log_weights = bool(config.get("dynamics.weight_transform")["log1p"])
        w = np.log1p(self.weight) if log_weights else self.weight.copy()
        if signed:
            w = w * self.sign
        keep = w != 0.0
        W = sp.csr_matrix(
            (w[keep].astype(np.float32), (self.dst[keep], self.src[keep])),
            shape=(self.n_nodes, self.n_nodes), dtype=np.float32,
        )
        W.sum_duplicates()
        return W

    def stats(self) -> dict:
        W = self.to_sparse()
        outdeg = np.bincount(self.src, minlength=self.n_nodes)
        indeg = np.bincount(self.dst, minlength=self.n_nodes)
        return {
            "source": self.source,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "nnz_after_signing": int(W.nnz),
            "density": float(self.n_edges / max(1, self.n_nodes**2)),
            "excitatory_frac": float((self.sign > 0).mean()),
            "inhibitory_frac": float((self.sign < 0).mean()),
            "mean_out_degree": float(outdeg.mean()),
            "max_out_degree": int(outdeg.max()),
            "mean_in_degree": float(indeg.mean()),
            "max_in_degree": int(indeg.max()),
            "isolated_nodes": int(((outdeg + indeg) == 0).sum()),
            **self.meta,
        }

    # -- io ----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            node_ids=self.node_ids, src=self.src, dst=self.dst,
            weight=self.weight, sign=self.sign,
        )
        for name in ("node_type", "node_class", "node_side"):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = np.asarray(v, dtype=object).astype("U64")
        for name in ("node_pos", "node_retino"):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(path, source=np.array(self.source),
                            meta=np.array(json.dumps(self.meta)), **arrays)

    @staticmethod
    def load(path: str | Path) -> "ConnectomeGraph":
        z = np.load(path, allow_pickle=False)
        return ConnectomeGraph(
            node_ids=z["node_ids"], src=z["src"], dst=z["dst"],
            weight=z["weight"], sign=z["sign"],
            node_type=z["node_type"] if "node_type" in z else None,
            node_class=z["node_class"] if "node_class" in z else None,
            node_side=z["node_side"] if "node_side" in z else None,
            node_pos=z["node_pos"] if "node_pos" in z else None,
            node_retino=z["node_retino"] if "node_retino" in z else None,
            source=str(z["source"]), meta=json.loads(str(z["meta"])),
        )
