"""Model B: a reference-style LIF, for comparison with the project's own (Model A, `spiking_reservoir.LIFBrain`).

After Shiu et al., Nature 634:210-219 (2024): synapse counts -> x sign -> x w_syn (mV) -> LIF with synaptic
filtering and a delay; no log, no normalisation. Constants from the reference: rest/reset -52 mV, threshold
-45 mV, tau_m 20 ms, tau_syn 5 ms, refractory 2.2 ms, delay 1.8 ms; `w_syn` is free and swept. Departures:
dt = 3.33 ms as in Model A (delay and refractory round to one tick), and sensory input scaled by the 7 mV gap
so both models receive the same fraction of threshold. A comparator only; nothing here selects a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: Shiu et al. 2024, methods. Everything except `w_syn` is inherited.
V_REST_MV = -52.0
V_THRESHOLD_MV = -45.0
V_RESET_MV = -52.0
TAU_MEMBRANE_MS = 20.0
TAU_SYNAPSE_MS = 5.0
REFRACTORY_MS = 2.2
DELAY_MS = 1.8
#: The reference's fitted value, on a different connectome. Recorded for
#: provenance; not a default, because it does not transfer.
W_SYN_REFERENCE_MV = 0.275


@dataclass
class ReferenceParams:
    """Reference-style LIF. Only `w_syn_mV` is free."""

    w_syn_mV: float = W_SYN_REFERENCE_MV
    dt_ms: float = 3.33
    tau_membrane_ms: float = TAU_MEMBRANE_MS
    tau_synapse_ms: float = TAU_SYNAPSE_MS
    v_rest_mV: float = V_REST_MV
    v_threshold_mV: float = V_THRESHOLD_MV
    v_reset_mV: float = V_RESET_MV
    refractory_ms: float = REFRACTORY_MS
    delay_ms: float = DELAY_MS
    noise_mV: float = 0.0
    seed: int = 0

    @property
    def threshold_gap_mV(self) -> float:
        return self.v_threshold_mV - self.v_rest_mV

    @property
    def membrane_decay(self) -> float:
        return float(np.exp(-self.dt_ms / self.tau_membrane_ms))

    @property
    def synapse_decay(self) -> float:
        return float(np.exp(-self.dt_ms / self.tau_synapse_ms))

    @property
    def refractory_steps(self) -> int:
        return max(1, int(round(self.refractory_ms / self.dt_ms)))

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.delay_ms / self.dt_ms)))

    @property
    def synapses_to_fire(self) -> float:
        """Simultaneous synapses needed to cross threshold from rest."""
        return self.threshold_gap_mV / self.w_syn_mV


def operator(graph, w_syn_mV: float) -> sp.csc_matrix:
    """`W[post, pre]` in millivolts per presynaptic spike: linear in synapse count, signed,
    and not normalised (unlike `spiking_reservoir.normalize_in_degree`)."""
    n = graph.n_nodes
    data = (graph.weight.astype(np.float64) * graph.sign.astype(np.float64)
            * float(w_syn_mV))
    W = sp.csr_matrix((data, (graph.dst, graph.src)), shape=(n, n),
                      dtype=np.float32)
    W.sum_duplicates()
    return W.tocsc()


class ReferenceBrain:
    """LIF with current-based exponential synapses and a transmission delay.

        g[t+1] = g[t] * exp(-dt/tau_syn) + W @ spikes[t - delay]
        v[t+1] = v_rest + (v[t] - v_rest) * exp(-dt/tau_m) + g[t+1] * (dt/tau_m)

    `step` takes the interface's output in threshold units, as Model A does, and scales it.
    """

    def __init__(self, W: sp.spmatrix, params: ReferenceParams | None = None):
        self.p = params or ReferenceParams()
        self.W = W.tocsc() if not sp.isspmatrix_csc(W) else W
        self.n = self.W.shape[0]
        self._rng = np.random.default_rng(self.p.seed)
        self._zero = np.zeros(self.n, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.v = np.full(self.n, self.p.v_rest_mV, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=bool)
        # Ring of spike index arrays, so a spike arrives `delay_steps` later.
        self._pipeline: list[np.ndarray] = [
            np.empty(0, dtype=np.int32) for _ in range(self.p.delay_steps)
        ]

    def _arriving(self) -> np.ndarray:
        idx = self._pipeline.pop(0)
        if idx.size == 0:
            return self._zero
        sub = self.W[:, idx]
        return np.asarray(sub.sum(axis=1), dtype=np.float32).ravel()

    def step(self, sensory_threshold_units: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        self.g *= p.synapse_decay
        self.g += self._arriving()

        drive = self.g.copy()
        if sensory_threshold_units is not None:
            # 1.0 from the interface means "one threshold", in both models.
            drive = drive + sensory_threshold_units * p.threshold_gap_mV
        if p.noise_mV > 0:
            drive = drive + self._rng.standard_normal(self.n, dtype=np.float32) * p.noise_mV

        self.v = p.v_rest_mV + (self.v - p.v_rest_mV) * p.membrane_decay \
            + drive * (p.dt_ms / p.tau_membrane_ms)

        free = self.refrac <= 0
        spikes = free & (self.v >= p.v_threshold_mV)
        self.v[spikes] = p.v_reset_mV
        self.refrac -= 1
        self.refrac[spikes] = p.refractory_steps

        self.spikes = spikes
        self._pipeline.append(np.flatnonzero(spikes).astype(np.int32))
        return spikes

    @property
    def firing_rate(self) -> float:
        return float(self.spikes.mean())

    def describe(self) -> dict:
        return {
            "model": "reference_lif",
            "w_syn_mV": self.p.w_syn_mV,
            "synapses_to_fire": round(self.p.synapses_to_fire, 2),
            "threshold_gap_mV": self.p.threshold_gap_mV,
            "membrane_decay": round(self.p.membrane_decay, 4),
            "synapse_decay": round(self.p.synapse_decay, 4),
            "refractory_steps": self.p.refractory_steps,
            "delay_steps": self.p.delay_steps,
            "normalised": False,
            "log_weights": False,
            "nnz": int(self.W.nnz),
        }
