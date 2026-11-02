"""Leaky integrate-and-fire dynamics on the frozen FlyWire 783 graph.

The connectome gives the wiring only: time constants, thresholds, conductances, delays and plasticity
are modelling choices, not biology. Each tick sums only the CSC columns of the neurons that spiked
(~1-5% of them), which is fast enough for 60 FPS on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from flymario import config


#: Frozen operating point, from configs/base.yaml. Defined here so the dataclass
#: defaults are inspectable, but the file is the authority.
_GAIN = config.get("dynamics.gain")
_NOISE_STD = config.get("dynamics.noise_std")
_BIAS = config.get("dynamics.bias")


@dataclass
class LIFParams:
    """Every field here is a modelling assumption, not a measurement."""

    tau_ms: float = 20.0          # membrane time constant
    dt_ms: float = 3.33           # 5 neural ticks per 60 Hz Mario frame
    threshold: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    refractory_ms: float = 2.0
    # Read from configs/base.yaml, never duplicated, so LIFParams() runs on the frozen
    # operating point (docs/PROTOCOL.md §9.4).
    gain: float = _GAIN
    noise_std: float = _NOISE_STD
    bias: float = _BIAS
    seed: int = 0

    @property
    def decay(self) -> float:
        return float(np.exp(-self.dt_ms / self.tau_ms))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.refractory_ms / self.dt_ms))


@dataclass
class BrainState:
    v: np.ndarray
    refrac: np.ndarray
    spikes: np.ndarray = field(repr=False)


class LIFBrain:
    """Spike-driven LIF simulator over a frozen signed CSC adjacency."""

    def __init__(
        self,
        W: sp.csc_matrix,
        params: LIFParams | None = None,
        normalize: bool | None = None,
    ):
        # None follows dynamics.weight_transform.normalize_in_degree, so every
        # caller that does not choose explicitly runs the declared operator.
        if normalize is None:
            normalize = bool(config.get("dynamics.weight_transform")["normalize_in_degree"])
        if normalize:
            W = normalize_in_degree(W)
        if not sp.isspmatrix_csc(W):
            W = sp.csc_matrix(W)
        self.W = W
        self.n = W.shape[0]
        self.p = params or LIFParams()
        # Cache the raw CSC arrays: touching the scipy object per tick is slow.
        self._indptr = W.indptr.astype(np.int64)
        self._indices = W.indices.astype(np.int32)
        self._data = W.data.astype(np.float32)
        self._rng = np.random.default_rng(self.p.seed)
        self._zero = np.zeros(self.n, dtype=np.float32)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Zero the state. With `seed`, also restart the noise stream from it.

        `reset(seed=s)` equals `LIFBrain(W, params with seed=s)` (same state, same noise)
        without re-copying the operator.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=bool)
        self._active = np.empty(0, dtype=np.int32)

    # -- core --------------------------------------------------------------
    def _recurrent_input(self) -> np.ndarray:
        """Sum the CSC columns of the neurons that spiked on the previous tick.

        scipy's column slice + row-sum (~5 ms at 5% activity) beats dense `W @ s` and `np.bincount`.
        """
        if self._active.size == 0:
            return self._zero
        sub = self.W[:, self._active]
        return np.asarray(sub.sum(axis=1), dtype=np.float32).ravel()

    def step(self, sensory_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one neural tick. Returns the boolean spike vector."""
        p = self.p
        i_rec = self._recurrent_input() * p.gain
        self.v *= p.decay
        self.v += i_rec + p.bias
        if sensory_current is not None:
            self.v += sensory_current
        if p.noise_std > 0:
            self.v += self._rng.standard_normal(self.n, dtype=np.float32) * p.noise_std

        free = self.refrac <= 0
        spikes = free & (self.v >= p.threshold)
        self.v[spikes] = p.v_reset
        # Count down to zero and stop there: an unfloored int16 counter would wrap
        # after 32,768 silent ticks (a full Mario episode is ~38,000).
        np.subtract(self.refrac, 1, out=self.refrac, where=self.refrac > 0)
        self.refrac[spikes] = p.refractory_steps

        self.spikes = spikes
        self._active = np.flatnonzero(spikes).astype(np.int32)
        return spikes

    def run(self, ticks: int, sensory_current: np.ndarray | None = None) -> np.ndarray:
        """Advance several ticks with a held sensory input; return spike counts."""
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(ticks):
            counts += self.step(sensory_current)
        return counts

    # -- diagnostics -------------------------------------------------------
    @property
    def firing_rate(self) -> float:
        return float(self.spikes.mean())


def normalize_in_degree(W: sp.spmatrix) -> sp.csc_matrix:
    """Scale each neuron's incoming weights so they sum to unit absolute mass.

    In-degree spans four orders of magnitude; without this, hubs make the network jump from silence
    to saturation. Normalising gives a broad stable regime but discards absolute connection strength.
    """
    Wr = W.tocsr()
    inflow = np.abs(Wr).sum(axis=1).A.ravel()
    scale = np.where(inflow > 0, 1.0 / np.maximum(inflow, 1e-6), 0.0).astype(np.float32)
    return (sp.diags(scale) @ Wr).tocsc()


def load_brain(path=None, params: LIFParams | None = None, normalize: bool | None = None) -> LIFBrain:
    from flymario.paths import BRAIN_NPZ

    W = sp.load_npz(path or BRAIN_NPZ).tocsc()
    return LIFBrain(W, params, normalize=normalize)
