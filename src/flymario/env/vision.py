"""NES framebuffer -> injected current on retinotopically mapped neurons.

Zero learned parameters, so no front-end can do the work instead of the connectome. The retinotopic
map is recovered from anatomy: PCA over one optic lobe's lamina somata gives two in-sheet axes that
approximate azimuth and elevation. Channels follow the ON/OFF split of the Drosophila lamina:

    luminance increment (ON)  -> L1, L5  -> Mi1 -> T4
    luminance decrement (OFF) -> L2, L3  -> Tm1/Tm2/Tm9 -> T5
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from flymario.connectome.schema import ConnectomeGraph

ON_TYPES = ("L1", "L5")
OFF_TYPES = ("L2", "L3")

# The SMB HUD (score / coins / world / timer) occupies the top ~32 rows.
# Feeding it in would leak the score and the clock into "vision".
HUD_ROWS = 32
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class VisionParams:
    # `eye` is a dataset-specific spelling ("right" vs "R") and must not be hard-coded;
    # resolved from the graph when left as None.
    hud_rows: int = HUD_ROWS
    # Calibrated against threshold=1.0 and decay=0.846: a neuron needs a
    # sustained current above ~0.154 to fire at all, so these are far larger
    # than the values one would guess.
    on_gain: float = 14.0
    off_gain: float = 14.0
    static_gain: float = 2.0
    blur_sigma: float = 1.0    # crude stand-in for ommatidial acceptance angle
    eye: str | None = None


class RetinotopicVision:
    def __init__(self, graph: ConnectomeGraph, params: VisionParams | None = None):
        self.p = params or VisionParams()
        if graph.node_pos is None:
            raise ValueError("graph has no 3D positions; rebuild with `make build-graphs`")

        if graph.node_retino is None:
            raise ValueError("graph carries no retinotopic map; rebuild with "
                             "`make build-graphs`")
        side = np.asarray(graph.node_side)
        types = np.asarray(graph.node_type)
        retino = np.asarray(graph.node_retino, dtype=np.float32)
        mapped = np.isfinite(retino).all(axis=1)

        # Pick the eye with the most mapped lamina cells, unless one is named.
        if self.p.eye is None:
            cand = [s for s in np.unique(side)
                    if (np.isin(types, ON_TYPES + OFF_TYPES) & (side == s) & mapped).sum()]
            if not cand:
                raise ValueError("no lamina neuron carries a retinotopic position")
            self.eye = max(cand, key=lambda s: int(
                (np.isin(types, ON_TYPES + OFF_TYPES) & (side == s) & mapped).sum()))
        else:
            self.eye = self.p.eye

        on = np.flatnonzero(np.isin(types, ON_TYPES) & (side == self.eye) & mapped)
        off = np.flatnonzero(np.isin(types, OFF_TYPES) & (side == self.eye) & mapped)
        if len(on) == 0 or len(off) == 0:
            raise ValueError(f"no mapped lamina neurons for eye {self.eye!r}")

        # Cells without a measured position are dropped rather than imputed (imputing to the
        # centroid would stack them on one pixel).
        self.on_idx, self.off_idx = on.astype(np.int32), off.astype(np.int32)
        self.n = graph.n_nodes
        self.on_uv, self.off_uv = retino[on], retino[off]
        self._prev: np.ndarray | None = None

    @staticmethod
    def _sheet_coords(pos: np.ndarray) -> np.ndarray:
        """Project a curved sheet of somata onto its two principal axes, in [0,1]."""
        centred = pos - pos.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        uv = centred @ vt[:2].T
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        return ((uv - lo) / np.where(hi - lo == 0, 1.0, hi - lo)).astype(np.float32)

    def reset(self) -> None:
        self._prev = None

    def _luminance(self, frame: np.ndarray) -> np.ndarray:
        gray = frame[self.p.hud_rows :, :, :3].astype(np.float32) @ _LUMA
        if self.p.blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.p.blur_sigma)
        return gray / 255.0

    @staticmethod
    def _sample(gray: np.ndarray, uv: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        rows = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
        cols = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
        return gray[rows, cols]

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Return a length-n_nodes current vector; only lamina entries are non-zero."""
        gray = self._luminance(frame)
        on_now, off_now = self._sample(gray, self.on_uv), self._sample(gray, self.off_uv)
        if self._prev is None:
            self._prev = (on_now.copy(), off_now.copy())
        d_on = on_now - self._prev[0]
        d_off = off_now - self._prev[1]
        self._prev = (on_now.copy(), off_now.copy())

        current = np.zeros(self.n, dtype=np.float32)
        current[self.on_idx] = np.maximum(d_on, 0.0) * self.p.on_gain + on_now * self.p.static_gain
        current[self.off_idx] = (np.maximum(-d_off, 0.0) * self.p.off_gain
                                 + (1.0 - off_now) * self.p.static_gain)
        return current

    @property
    def n_injected(self) -> int:
        return len(self.on_idx) + len(self.off_idx)

    def coverage(self) -> dict:
        """How many distinct directions the interface can actually resolve (cells can share positions)."""
        def distinct(uv):
            return int(len(np.unique(np.round(uv, 6), axis=0)))
        return {"eye": str(self.eye),
                "on_cells": len(self.on_idx), "on_positions": distinct(self.on_uv),
                "off_cells": len(self.off_idx), "off_positions": distinct(self.off_uv)}
