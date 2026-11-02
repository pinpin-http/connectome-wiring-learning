"""Retinal columns of the injected lamina cells and the column permutations.

Input vector: the encoder's current on the injected cells, in ascending node order. A cell's column is its
measured ommatidial coordinate (`node_retino`, MaleCNS `assignedOlHex1/2`). A permutation moves whole columns
between columns of the same type signature, keeping each cell's type and slot, so ON and OFF move together;
columns in groups smaller than `min_group` stay fixed. Convention: the permuted input is `x[..., sigma]` (cell i
receives cell sigma[i]'s signal); at column level, column c receives column pi[c]'s signals.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

LAMINA = ("L1", "L2", "L3", "L5")
ON_TYPES, OFF_TYPES = ("L1", "L5"), ("L2", "L3")
MIN_GROUP = 10


@dataclass
class Retina:
    inj: np.ndarray        # (N,) node index of every injected cell, ascending: the input vector's order
    types: np.ndarray      # (N,) cell type
    col: np.ndarray        # (N,) column id, 0..C-1
    col_uv: np.ndarray     # (C, 2) normalised retinotopic coordinate (what the encoder samples)
    col_xy: np.ndarray     # (C, 2) hex-cartesian coordinate, in units of the column spacing
    col_sig: np.ndarray    # (C,) signature string, e.g. "L1:1|L2:1|L3:1|L5:1"
    side: str

    @property
    def n_cells(self) -> int:
        return len(self.inj)

    @property
    def n_cols(self) -> int:
        return len(self.col_uv)

    def channel(self) -> np.ndarray:
        return np.where(np.isin(self.types, ON_TYPES), "ON", "OFF")

    def groups(self, min_group: int = MIN_GROUP) -> dict[str, np.ndarray]:
        """Signature -> columns; only groups with at least `min_group` columns are permutable."""
        out = {}
        for s in np.unique(self.col_sig):
            cols = np.flatnonzero(self.col_sig == s)
            out[str(s)] = cols
        return {s: c for s, c in out.items() if len(c) >= min_group}

    def neighbours(self) -> np.ndarray:
        """(E, 2) unordered pairs of columns one lattice step apart (hex-cartesian distance 1)."""
        d = np.linalg.norm(self.col_xy[:, None, :] - self.col_xy[None, :, :], axis=2)
        i, j = np.nonzero(np.triu(np.abs(d - 1.0) < 0.02, 1))
        return np.stack([i, j], 1)

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.inj.astype(np.int64), self.col.astype(np.int64), self.col_uv.astype(np.float32)):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def _hex_xy(uv: np.ndarray) -> np.ndarray:
    """Undo the per-axis normalisation of `malecns._retinotopy`: x = h1 - h2/2 moves in half steps, y = (√3/2) h2
    in steps of √3/2, so the smallest non-zero gaps along each axis give the scale. Checked by `build`."""
    out = np.empty_like(uv, dtype=np.float64)
    for k, unit in ((0, 0.5), (1, np.sqrt(3.0) / 2.0)):
        u = np.unique(np.round(uv[:, k].astype(np.float64), 7))
        step = np.diff(u)
        step = step[step > 1e-6].min()
        out[:, k] = (uv[:, k] - uv[:, k].min()) / step * unit
    return out


def build(graph, vis) -> Retina:
    """Columns of the cells that `RetinotopicVision` injects, from the graph's measured coordinates."""
    inj = np.unique(np.r_[vis.on_idx, vis.off_idx]).astype(np.int64)
    types = np.asarray(graph.node_type)[inj].astype(str)
    side = np.asarray(graph.node_side)[inj].astype(str)
    if not np.isin(types, LAMINA).all():
        raise ValueError("an injected cell is not L1/L2/L3/L5")
    if len(np.unique(side)) != 1:
        raise ValueError(f"injected cells span several sides: {np.unique(side)}")
    uv = np.asarray(graph.node_retino, dtype=np.float32)[inj]
    key = np.round(uv.astype(np.float64), 6)
    col_uv_key, col = np.unique(key, axis=0, return_inverse=True)
    col = col.ravel()
    col_uv = np.stack([uv[col == c][0] for c in range(len(col_uv_key))])
    sig = []
    for c in range(len(col_uv)):
        t = types[col == c]
        sig.append("|".join(f"{x}:{int((t == x).sum())}" for x in LAMINA))
    xy = _hex_xy(col_uv)
    # lattice check: rounded hex coordinates must sit on the grid
    gx = xy[:, 0] / 0.5
    gy = xy[:, 1] / (np.sqrt(3.0) / 2.0)
    if max(np.abs(gx - np.round(gx)).max(), np.abs(gy - np.round(gy)).max()) > 1e-3:
        raise ValueError("column coordinates do not sit on a hexagonal lattice")
    return Retina(inj=inj, types=types, col=col, col_uv=col_uv, col_xy=xy, col_sig=np.array(sig),
                  side=str(side[0]))


def column_permutation(ret: Retina, seed: int, min_group: int = MIN_GROUP) -> np.ndarray:
    """pi (C,): column c receives the signals of column pi[c]. Uniform within each permutable signature group,
    identity elsewhere. Groups are visited in sorted signature order, so pi depends on the seed only."""
    rng = np.random.default_rng(seed)
    pi = np.arange(ret.n_cols)
    for s, cols in sorted(ret.groups(min_group).items()):
        pi[cols] = cols[rng.permutation(len(cols))]
    return pi


def cell_permutation(ret: Retina, pi: np.ndarray) -> np.ndarray:
    """sigma (N,): cell (column c, type t, k-th of that type in c) reads cell (pi[c], t, k). Same signature
    guarantees that the slot exists; slots are ordered by node index."""
    slot = {}
    for i in np.lexsort((ret.inj, ret.types, ret.col)):
        c, t = int(ret.col[i]), str(ret.types[i])
        slot.setdefault((c, t), []).append(int(i))
    sigma = np.empty(ret.n_cells, np.int64)
    for (c, t), cells in slot.items():
        src = slot.get((int(pi[c]), t), [])
        if len(src) != len(cells):
            raise ValueError(f"column {c} and {pi[c]} differ in their {t} count")
        sigma[cells] = src
    if not np.array_equal(np.sort(sigma), np.arange(ret.n_cells)):
        raise ValueError("cell map is not a bijection")
    return sigma


def inverse(sigma: np.ndarray) -> np.ndarray:
    inv = np.empty_like(sigma)
    inv[sigma] = np.arange(len(sigma))
    return inv


def report(ret: Retina, pi: np.ndarray, sigma: np.ndarray, min_group: int = MIN_GROUP) -> dict:
    """Coverage, displacement, preserved neighbourhoods, per-type checks."""
    groups = ret.groups(min_group)
    perm_cols = np.concatenate(list(groups.values())) if groups else np.array([], int)
    in_group = np.isin(ret.col, perm_cols)
    moved_cols = pi != np.arange(ret.n_cols)
    disp = np.linalg.norm(ret.col_xy[pi] - ret.col_xy, axis=1)
    nb = ret.neighbours()
    nbset = {(int(a), int(b)) for a, b in nb} | {(int(b), int(a)) for a, b in nb}
    kept = np.mean([(int(pi[a]), int(pi[b])) in nbset for a, b in nb]) if len(nb) else float("nan")
    return {
        "n_cells": ret.n_cells, "n_columns": ret.n_cols, "min_group": min_group,
        "groups": {s: int(len(c)) for s, c in sorted(groups.items())},
        "fixed_signatures": {str(s): int((ret.col_sig == s).sum()) for s in np.unique(ret.col_sig)
                             if str(s) not in groups},
        "frac_cells_permutable": float(in_group.mean()),
        "frac_cells_moved": float((sigma != np.arange(ret.n_cells)).mean()),
        "frac_columns_moved": float(moved_cols.mean()),
        "fixed_points_in_groups": int((~moved_cols[perm_cols]).sum()) if len(perm_cols) else 0,
        "displacement_hex_units": {"mean_all_columns": float(disp.mean()),
                                   "mean_moved": float(disp[moved_cols].mean()) if moved_cols.any() else 0.0,
                                   "quantiles_moved": {f"q{q}": float(np.quantile(disp[moved_cols], q / 100))
                                                       for q in (5, 25, 50, 75, 95)} if moved_cols.any() else {},
                                   "max": float(disp.max())},
        "neighbour_pairs": int(len(nb)),
        "frac_neighbour_pairs_kept": float(kept),
        "types_preserved": bool(np.array_equal(ret.types[sigma], ret.types)),
        "sides_preserved": True,  # one eye: checked in build()
        "on_off_same_displacement": bool(_on_off_together(ret, sigma)),
    }


def _on_off_together(ret: Retina, sigma: np.ndarray) -> bool:
    """Every cell of a column reads from one and the same source column."""
    src_col = ret.col[sigma]
    for c in range(ret.n_cols):
        if len(np.unique(src_col[ret.col == c])) != 1:
            return False
    return True
