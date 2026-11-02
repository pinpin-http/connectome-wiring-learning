"""The frozen network: one sweep from rest per image, readout spike counts before any action projection (mission §8).

The equation is `TorchLIF.step` (models/torch_lif.py), operation for operation, on the plain operator:

    v = v * decay ; v += (W @ s_prev) * gain ; v += current ; v += noise
    s = (refrac <= 0) & (v >= threshold) ; v[s] = v_reset ; refrac = max(refrac - 1, 0) ; refrac[s] = steps

No plastic split, no eligibility, no learning signal: nothing here can change a weight. Before every image the
state (v, refractory, last spikes) is zeroed, so nothing carries over between images.

Why not `TorchLIF` itself: it keeps the activity column-major, measured fastest on the Radeon. On this RTX 3090
(cuSPARSE) the same CSR product against a row-major activity is ~30x faster (B = 64: 4.8 vs 146 ms per tick,
2026-10-19) and gives identical numbers. The equivalence with `TorchLIF` is tested on the full connectome.

Noise key (mission §8.3): (dynamic seed, example uid) -> one CUDA generator per image, which draws the image's
whole (ticks, neurons) block at once; tick t uses row t. It does not depend on the graph, the retina condition, the
label or the batch layout, so matched examples get matched noise in every graph and condition, and a batch of any
size gives the same counts (tested).

Determinism: lin_raw weights are integer synapse counts times a sign (max |w| 2,591, max |row sum| 114,020) and
spikes are 0/1, so W @ s is a sum of integers, exact in float32 whatever the summation order (< 2^24); the gain
multiplies after. Checked by running the same images twice and in different batch sizes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFParams

STUDY_SEED = 20261019          # namespace of every RETINO_ALIGN_V1 noise key


def noise_key(dyn_seed: int, uid: int) -> int:
    """63-bit generator seed for one image under one dynamic seed."""
    s = np.random.SeedSequence([STUDY_SEED, int(dyn_seed), int(uid)]).generate_state(2, dtype=np.uint32)
    return int((int(s[0]) << 31) ^ int(s[1]))


def operator_digest(W: sp.spmatrix) -> str:
    W = sp.csr_matrix(W, dtype=np.float32)
    W.sort_indices()
    h = hashlib.sha256()
    for a in (W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float32)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass
class SweepOut:
    counts: np.ndarray          # (B, M) uint8 readout spike counts over the window
    rate_all: np.ndarray        # (B,) fraction of all neurons x ticks that spiked
    rate_input: np.ndarray      # (B,) same, injected cells
    v_readout_mean: np.ndarray  # (B,) mean readout membrane potential at the last tick
    v_readout_sd: np.ndarray    # (B,) its SD over readout neurons


class FrozenNet:
    """B brains on one graph's operator; `run` simulates any number of images in batches of at most B."""

    def __init__(self, W: sp.spmatrix, readout: np.ndarray, inj: np.ndarray, params: LIFParams,
                 ticks: int, batch: int = 128, device: str = "cuda"):
        import torch
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("no GPU visible to torch; never fall back silently")
        self.torch = torch
        Wc = sp.csr_matrix(W, dtype=np.float32)
        Wc.sort_indices()
        if not np.array_equal(Wc.data, np.round(Wc.data)):
            raise ValueError("operator is not integer-valued: exact summation (determinism) would not hold")
        self.digest = operator_digest(Wc)
        self.dev = torch.device(device)
        self.W = torch.sparse_csr_tensor(torch.from_numpy(Wc.indptr.astype(np.int32)),
                                         torch.from_numpy(Wc.indices.astype(np.int32)),
                                         torch.from_numpy(Wc.data), size=Wc.shape).to(self.dev)
        self.n = Wc.shape[0]
        self.readout = np.asarray(readout, np.int64)
        self.inj = np.asarray(inj, np.int64)
        self.p, self.ticks, self.B = params, int(ticks), int(batch)
        self.decay, self.steps = float(params.decay), int(params.refractory_steps)
        self._ro = torch.as_tensor(self.readout, device=self.dev)
        self._inj = torch.as_tensor(self.inj, device=self.dev)

    def device_operator_digest(self) -> str:
        """sha256 of the operator as it sits on the GPU, in `operator_digest`'s layout: equal before and after."""
        T = self.W
        h = hashlib.sha256()
        for a, dt in ((T.crow_indices(), np.int64), (T.col_indices(), np.int64), (T.values(), np.float32)):
            h.update(np.ascontiguousarray(a.detach().cpu().numpy().astype(dt)).tobytes())
        return h.hexdigest()

    def _noise(self, keys: list[int]):
        torch = self.torch
        if self.p.noise_std <= 0:
            return None
        z = torch.empty((self.ticks, self.n, len(keys)), dtype=torch.float32, device=self.dev)
        for j, k in enumerate(keys):
            g = torch.Generator(device=self.dev)
            g.manual_seed(int(k))
            z[:, :, j] = torch.randn((self.ticks, self.n), generator=g, device=self.dev)
        return z

    def _sweep(self, X: np.ndarray, keys: list[int]) -> SweepOut:
        torch, p = self.torch, self.p
        b = len(keys)
        n = self.n
        v = torch.zeros((n, b), dtype=torch.float32, device=self.dev)
        refrac = torch.zeros((n, b), dtype=torch.int32, device=self.dev)
        s_f = torch.zeros((n, b), dtype=torch.float32, device=self.dev)       # row-major activity
        cur = torch.zeros((n, b), dtype=torch.float32, device=self.dev)
        cur[self._inj] = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.dev).t()
        z = self._noise(keys)
        counts = torch.zeros((len(self.readout), b), dtype=torch.int16, device=self.dev)
        n_all = torch.zeros(b, dtype=torch.float64, device=self.dev)
        n_inp = torch.zeros(b, dtype=torch.float64, device=self.dev)
        any_spike = False
        for t in range(self.ticks):
            v.mul_(self.decay)
            if any_spike:
                v.add_((self.W @ s_f) * p.gain)
            if p.bias:
                v.add_(p.bias)
            v.add_(cur)
            if z is not None:
                v.add_(z[t] * p.noise_std)
            s = (refrac <= 0) & (v >= p.threshold)
            v.masked_fill_(s, p.v_reset)
            refrac.sub_(1).clamp_(min=0)
            refrac.masked_fill_(s, self.steps)
            s_f.copy_(s)
            any_spike = True          # one sync saved per tick; W @ 0 = 0 exactly, so this changes no number
            counts += s.index_select(0, self._ro).to(torch.int16)
            n_all += s.sum(0, dtype=torch.float64)
            n_inp += s.index_select(0, self._inj).sum(0, dtype=torch.float64)
        vr = v.index_select(0, self._ro)
        cmax = int(counts.max())
        if cmax > 255:
            raise RuntimeError("a count exceeds uint8")
        return SweepOut(counts=counts.t().cpu().numpy().astype(np.uint8),
                        rate_all=(n_all / (n * self.ticks)).cpu().numpy(),
                        rate_input=(n_inp / (len(self.inj) * self.ticks)).cpu().numpy(),
                        v_readout_mean=vr.mean(0).cpu().numpy(),
                        v_readout_sd=vr.std(0).cpu().numpy())

    def run(self, X: np.ndarray, uids: np.ndarray, dyn_seed: int) -> SweepOut:
        """X (n_images, N) input currents on `inj`, uids (n_images,) example identifiers."""
        if X.shape[1] != len(self.inj):
            raise ValueError("X must have one column per injected cell")
        parts = []
        for a in range(0, len(X), self.B):
            keys = [noise_key(dyn_seed, u) for u in uids[a:a + self.B]]
            parts.append(self._sweep(X[a:a + self.B], keys))
        return SweepOut(**{f: np.concatenate([getattr(q, f) for q in parts])
                           for f in SweepOut.__dataclass_fields__})
