"""FlyWire 783 adapter -> ConnectomeGraph. Protocol revision 2 compliant.

Sources (both CC-BY, no account required):
  connectivity  Zenodo 10676866, proofread_connections_783.feather
  annotations   github.com/flyconnectome/flywire_annotations

Load-bearing choices: the node set (139,262 IDs) is fixed before any threshold; rows are summed per
(pre, post) pair before thresholding; log1p is applied after summing; measured `known_nt` overrides
the predicted `top_nt`, which has no histamine class (it would make photoreceptors excitatory).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.feather as ft

from flymario.connectome.schema import ConnectomeGraph
from flymario.paths import RAW

CONNECTIONS = "flywire_proofread_connections_783.feather"
ANNOTATIONS = "flywire_neuron_annotations_783.tsv"

#: FAFB voxels are anisotropic: 4 nm in x and y, 40 nm in z.
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float32)

#: Per-edge transmitter probability columns -> canonical names.
NT_COLS = {"gaba_avg": "gaba", "ach_avg": "acetylcholine", "glut_avg": "glutamate",
           "oct_avg": "octopamine", "ser_avg": "serotonin", "da_avg": "dopamine"}

#: Fast ionotropic sign. Absent keys carry NO fast current: monoamines are
#: neuromodulators rather than fast transmitters, and an unresolved label must
#: not be silently treated as excitatory.
FAST_SIGN = {"acetylcholine": +1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}

#: Below this, `top_nt` is treated as unresolved rather than believed.
#: 17% of neurons fall under it; without a floor the "unknown" branch never fires.
NT_CONF_FLOOR = 0.5


def _canonical_nt(raw: str | float) -> str:
    """Normalise a `known_nt` cell. The column is free text with entries like
    'acetylcholine; sNPF; acetylcholine, sNPF' and 'gaba-negative'."""
    if not isinstance(raw, str):
        return "unresolved"
    s = raw.lower()
    if "negative" in s:
        return "unresolved"
    for nt in ("histamine", "acetylcholine", "glutamate", "gaba",
               "dopamine", "octopamine", "serotonin"):
        if nt in s:
            return nt
    return "unresolved"


def _transmitter_per_neuron(edges: pd.DataFrame, ann: pd.DataFrame,
                            node_ids: np.ndarray) -> tuple[pd.Series, dict]:
    """One transmitter per presynaptic neuron (Dale), measured where possible.

    Prediction is per *synapse*, but a real neuron releases the same transmitter
    at all its outputs, so the per-edge argmax is reduced by a synapse-count
    weighted vote. Ground truth overrides the vote wherever it exists.
    """
    # Step 1 — read the per-synapse prediction. FlyWire scores six transmitters
    # for every connection row; the winner is whichever column scores highest.
    probs = edges[list(NT_COLS)].to_numpy()
    edge_nt = np.array(list(NT_COLS.values()))[probs.argmax(axis=1)]
    # Step 2 — collapse those per-synapse guesses into one per neuron, weighting
    # each guess by how many synapses back it. A neuron with 900 synapses voting
    # "GABA" and 3 voting "glutamate" is a GABA neuron.
    votes = (pd.DataFrame({"src": edges["pre_pt_root_id"].to_numpy(),
                           "nt": edge_nt, "w": edges["syn_count"].to_numpy()})
             .groupby(["src", "nt"])["w"].sum().reset_index()
             .sort_values(["w", "nt"]).drop_duplicates("src", keep="last")
             .set_index("src")["nt"])

    # Step 3 — throw away low-confidence guesses. Below the floor we say
    # "unresolved" rather than believe the classifier; an unresolved neuron
    # simply carries no fast current instead of being assumed excitatory.
    a = ann.drop_duplicates("root_id").set_index("root_id").reindex(node_ids)
    known = a["known_nt"].map(_canonical_nt)
    conf = a["top_nt_conf"].to_numpy(dtype=float)
    predicted = votes.reindex(node_ids).to_numpy(dtype=object)
    predicted = np.where(np.isfinite(conf) & (conf >= NT_CONF_FLOOR),
                         predicted, "unresolved")
    predicted = np.where(pd.isna(predicted), "unresolved", predicted)

    # Step 4 — measurement beats prediction. Where a neuron was experimentally
    # characterised, that label wins outright. This is what restores histamine
    # to the photoreceptors and flips the visual input back to inhibitory.
    final = np.where(known.to_numpy() != "unresolved", known.to_numpy(), predicted)
    provenance = {
        "from_known_nt": int((known.to_numpy() != "unresolved").sum()),
        "from_prediction": int(((known.to_numpy() == "unresolved")
                                & (predicted != "unresolved")).sum()),
        "unresolved": int((final == "unresolved").sum()),
        "conf_floor": NT_CONF_FLOOR,
    }
    return pd.Series(final, index=node_ids), provenance


def load(min_syn: int = 1, ledger_path=None) -> ConnectomeGraph:
    """Build the anatomical graph and its signed operator.

    min_syn  Pair-level threshold, applied AFTER summing rows per pair. 1 is the
             primary condition (no filtering); 5 is a declared sensitivity.
    """
    edges = ft.read_table(
        RAW / CONNECTIONS,
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", *NT_COLS],
    ).to_pandas()
    ann = pd.read_csv(
        RAW / ANNOTATIONS, sep="\t",
        usecols=["root_id", "super_class", "cell_class", "cell_type", "side",
                 "top_nt", "top_nt_conf", "known_nt", "pos_x", "pos_y", "pos_z"],
        dtype={"root_id": np.int64},
    )

    # --- node set, fixed before any filtering ---
    # Annotated or at either end of a connection: `min_syn` cannot change which neurons exist.
    node_ids = np.union1d(
        ann["root_id"].to_numpy(),
        np.union1d(edges["pre_pt_root_id"].to_numpy(), edges["post_pt_root_id"].to_numpy()),
    )
    index = pd.Series(np.arange(len(node_ids), dtype=np.int64), index=node_ids)

    nt_per_neuron, nt_prov = _transmitter_per_neuron(edges, ann, node_ids)

    # --- aggregate rows into directed pairs, then threshold, then transform ---
    raw_rows, raw_syn = len(edges), int(edges["syn_count"].sum())
    # One neuron pair can appear on several rows, one per brain region the
    # connection passes through. Add those back together FIRST, so a pair with
    # 3+3+3 synapses counts as 9 rather than being judged three times as 3.
    pairs = (edges.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)["syn_count"]
             .sum().reset_index())
    n_pairs_all = len(pairs)
    if min_syn > 1:
        pairs = pairs[pairs["syn_count"] >= min_syn]

    # Translate the 18-digit FlyWire IDs into 0..N-1 matrix positions. The
    # original IDs stay in `node_ids` so results can be traced back.
    src = index.reindex(pairs["pre_pt_root_id"].to_numpy()).to_numpy(np.int64)
    dst = index.reindex(pairs["post_pt_root_id"].to_numpy()).to_numpy(np.int64)
    weight = pairs["syn_count"].to_numpy(np.float32)      # raw counts, retained
    # Each edge inherits the sign of the neuron it leaves from (Dale's
    # principle). fillna(0) is the "unresolved or neuromodulator" case: the
    # connection exists anatomically but carries no fast current in this model.
    sign = (nt_per_neuron.reindex(node_ids).map(FAST_SIGN).fillna(0.0)
            .to_numpy(np.float32))[src]

    # 3D coordinates, for the anatomical views. A handful of nodes appear in
    # the connections but not the annotations and have none; parking them at the
    # centroid keeps every bounding box finite instead of poisoning it with NaN.
    a = ann.drop_duplicates("root_id").set_index("root_id").reindex(node_ids)
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float32)
    missing = ~np.isfinite(pos).all(axis=1)
    if missing.any():
        pos[missing] = np.nanmedian(pos[~missing], axis=0)

    # No ommatidial column labels in FlyWire: the retinotopic sheet is recovered by
    # projecting the lamina somata onto their two principal axes (100% coverage).
    retino = np.full((len(node_ids), 2), np.nan, dtype=np.float32)
    lam = np.isin(a["cell_type"].to_numpy(), ("L1", "L2", "L3", "L5"))
    if lam.any():
        c = pos[lam] - pos[lam].mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        uv = c @ vt[:2].T
        lo, hi = uv.min(0), uv.max(0)
        retino[lam] = ((uv - lo) / np.where(hi - lo == 0, 1.0, hi - lo)).astype(np.float32)

    g = ConnectomeGraph(
        node_ids=node_ids, src=src, dst=dst, weight=weight, sign=sign,
        node_type=a["cell_type"].fillna("unknown").to_numpy(),
        node_class=a["super_class"].fillna("unknown").to_numpy(),
        node_side=a["side"].fillna("C").to_numpy(),
        node_pos=pos * VOXEL_NM,
        node_retino=retino,
        source="flywire-783",
        meta={"min_syn": min_syn, "nt": nt_prov},
    )
    g.validate()

    ledger = {
        "source": "flywire-783", "min_syn": min_syn,
        "annotated_ids": int(ann["root_id"].nunique()),
        "ids_in_connections": int(len(np.union1d(edges["pre_pt_root_id"],
                                                 edges["post_pt_root_id"]))),
        "nodes_retained": int(len(node_ids)),
        "raw_rows": raw_rows, "raw_synapses": raw_syn,
        "pairs_before_threshold": n_pairs_all,
        "pairs_after_threshold": int(len(pairs)),
        "pairs_excluded_by_threshold": int(n_pairs_all - len(pairs)),
        "synapses_retained": int(pairs["syn_count"].sum()),
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
