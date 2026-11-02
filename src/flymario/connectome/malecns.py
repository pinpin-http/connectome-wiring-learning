"""MaleCNS v1.0 adapter -> ConnectomeGraph; the primary dataset (docs/PROTOCOL.md §7.1).

Covers brain and ventral nerve cord (2,227 efferent units), and its consensus transmitter call includes
histamine, so photoreceptors come out inhibitory without an override. The weights table already has one
row per directed pair; this is asserted rather than assumed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.feather as ft

from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import MALECNS_FILES, RAW

#: MaleCNS EM is 8 nm isotropic, unlike FlyWire's anisotropic 4x4x40.
VOXEL_NM = np.array([8.0, 8.0, 8.0], dtype=np.float32)

#: Same fast-current policy as FlyWire: monoamines and unresolved labels carry
#: no fast current, and "we don't know" is never quietly turned into "+1".
FAST_SIGN = {"acetylcholine": +1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}

#: statusLabel values that are not usable neurons.
DROP_STATUS = frozenset({"Glia", "Orphan", "Orphan-artifact", "Out of scope", "Unimportant"})


def _positions(ann: pd.DataFrame, node_ids: np.ndarray) -> np.ndarray:
    """somaLocation is a list column; missing somata get the centroid."""
    loc = ann.set_index("bodyId").reindex(node_ids)["somaLocation"]
    pos = np.full((len(node_ids), 3), np.nan, dtype=np.float32)
    for i, v in enumerate(loc.to_numpy()):
        if isinstance(v, (list, np.ndarray)) and len(v) == 3:
            pos[i] = v
    missing = ~np.isfinite(pos).all(axis=1)
    if missing.any():
        pos[missing] = np.nanmedian(pos[~missing], axis=0)
    return pos * VOXEL_NM


def _retinotopy(ann: pd.DataFrame, node_ids: np.ndarray) -> np.ndarray:
    """Ommatidial column indices -> normalised [0,1]^2, NaN where unknown.

    Uses `assignedOlHex1/2` (the measured eye column), not soma positions: only 15% of lamina
    cells carry a soma location.
    """
    a = ann.set_index("bodyId").reindex(node_ids)
    h1 = a["assignedOlHex1"].to_numpy(dtype=float)
    h2 = a["assignedOlHex2"].to_numpy(dtype=float)
    ok = np.isfinite(h1) & np.isfinite(h2)
    out = np.full((len(node_ids), 2), np.nan, dtype=np.float32)
    if ok.any():
        # Axial hex -> cartesian, then to the unit square.
        x = h1[ok] - 0.5 * h2[ok]
        y = (np.sqrt(3.0) / 2.0) * h2[ok]
        xy = np.stack([x, y], axis=1)
        lo, hi = xy.min(0), xy.max(0)
        out[ok] = ((xy - lo) / np.where(hi - lo == 0, 1.0, hi - lo)).astype(np.float32)
    return out


def load(min_syn: int = 1, ledger_path=None) -> ConnectomeGraph:
    """Build the anatomical graph and its signed operator.

    min_syn  Pair-level threshold. 1 is the primary condition; 5 is the
             declared sensitivity, exactly as for FlyWire.
    """
    ann = ft.read_table(
        RAW / MALECNS_FILES["annotations"][0],
        columns=["bodyId", "type", "class", "superclass", "somaSide",
                 "statusLabel", "somaLocation", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()
    ann["statusLabel"] = ann["statusLabel"].astype(str)
    ann = ann[~ann["statusLabel"].isin(DROP_STATUS)].drop_duplicates("bodyId")

    edges = ft.read_table(
        RAW / MALECNS_FILES["weights"][0],
        columns=["body_pre", "body_post", "weight"],
    ).to_pandas()
    raw_rows, raw_syn = len(edges), int(edges["weight"].sum())

    # Assert rather than assume: FlyWire needs this aggregation, MaleCNS does not;
    # a future release with duplicate pairs must not be silently mis-loaded.
    pairs = (edges.groupby(["body_pre", "body_post"], sort=False)["weight"]
             .sum().reset_index())
    assert len(pairs) == raw_rows, "MaleCNS gained duplicate pairs; check the release"

    n_pairs_all = len(pairs)
    if min_syn > 1:
        pairs = pairs[pairs["weight"] >= min_syn]

    # Node set fixed before any filtering, exactly as for FlyWire.
    node_ids = np.union1d(
        ann["bodyId"].to_numpy(),
        np.union1d(edges["body_pre"].to_numpy(), edges["body_post"].to_numpy()),
    )
    # Bodies excluded as glia/orphans are not neurons at all, so they leave.
    node_ids = np.intersect1d(node_ids, np.union1d(
        ann["bodyId"].to_numpy(),
        np.setdiff1d(np.union1d(edges["body_pre"], edges["body_post"]),
                     np.array([], dtype=np.int64))))
    index = pd.Series(np.arange(len(node_ids), dtype=np.int64), index=node_ids)

    # --- transmitters: measured first, consensus second ---
    nt = ft.read_table(
        RAW / MALECNS_FILES["neurotransmitters"][0],
        columns=["body", "consensus_nt", "ground_truth"],
    ).to_pandas().drop_duplicates("body").set_index("body").reindex(node_ids)
    known = nt["ground_truth"].fillna("unresolved").astype(str).str.lower()
    consensus = nt["consensus_nt"].fillna("unclear").astype(str).str.lower()
    final = np.where(known.to_numpy() != "unresolved", known.to_numpy(),
                     consensus.to_numpy())
    final = np.where(pd.isna(final), "unresolved", final)
    nt_prov = {
        "from_ground_truth": int((known.to_numpy() != "unresolved").sum()),
        "from_consensus": int(((known.to_numpy() == "unresolved")
                               & (consensus.to_numpy() != "unclear")).sum()),
        "unresolved": int(((known.to_numpy() == "unresolved")
                           & (consensus.to_numpy() == "unclear")).sum()),
    }

    src = index.reindex(pairs["body_pre"].to_numpy()).to_numpy()
    dst = index.reindex(pairs["body_post"].to_numpy()).to_numpy()
    ok = ~(np.isnan(src) | np.isnan(dst))
    src, dst = src[ok].astype(np.int64), dst[ok].astype(np.int64)
    weight = pairs["weight"].to_numpy(np.float32)[ok]
    # Counted in int64, never from `weight`: a float32 sum near 1.24e8 is only
    # exact to a grid of 8. The graph stores float32; the ledger must not.
    syn_retained = int(pairs["weight"].to_numpy(np.int64)[ok].sum())
    sign = pd.Series(final).map(FAST_SIGN).fillna(0.0).to_numpy(np.float32)[src]

    a = ann.set_index("bodyId").reindex(node_ids)
    g = ConnectomeGraph(
        node_ids=node_ids, src=src, dst=dst, weight=weight, sign=sign,
        node_type=a["type"].fillna("unknown").to_numpy(),
        node_class=a["superclass"].fillna("unknown").to_numpy(),
        node_side=a["somaSide"].fillna("C").to_numpy(),
        node_pos=_positions(ann, node_ids),
        node_retino=_retinotopy(ann, node_ids),
        source="malecns-v1.0",
        meta={"min_syn": min_syn, "nt": nt_prov},
    )
    g.validate()

    ledger = {
        "source": "malecns-v1.0", "min_syn": min_syn,
        "nodes_retained": int(len(node_ids)),
        "raw_rows": raw_rows, "raw_synapses": raw_syn,
        "pairs_before_threshold": n_pairs_all,
        "pairs_after_threshold": int(len(src)),
        "pairs_excluded_by_threshold": int(n_pairs_all - len(src)),
        "synapses_retained": syn_retained,
        "edges_with_fast_current": int((sign != 0).sum()),
        "edges_without_fast_current": int((sign == 0).sum()),
        "excitatory_edges": int((sign > 0).sum()),
        "inhibitory_edges": int((sign < 0).sum()),
        "transmitter_provenance": nt_prov,
        "isolated_nodes": int(len(node_ids)
                              - len(np.union1d(np.unique(src), np.unique(dst)))),
    }
    g.meta["ledger"] = ledger
    if ledger_path:
        ledger_path.write_text(json.dumps(ledger, indent=2))
    return g
