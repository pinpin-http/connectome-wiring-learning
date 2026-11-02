"""Many brains on the GPU whose connections onto the motor neurons can learn.

Plastic set (docs/GATE_G_PLAN.md §3): every connection onto the readout's motor neurons; sign fixed (Dale),
magnitude in [0, 2 x initial count]. Frozen part: one shared deterministic SpMM, as in `TorchLIF`. Plastic part:
one block-diagonal sparse matrix (a block per brain) on ROCm; on CUDA, where that SpMM is not bit-repeatable
with fractional weights, a segment reduction. Runs are reproducible per platform, not bit-identical across.
FLYMARIO_PLASTIC_PRODUCT=spmm|segment overrides the choice (tests only). `observe=True` exposes what the
learning rules need (training/rules.py).
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFParams, normalize_in_degree
from flymario.models.torch_lif import gpu_available, upload_operator


class PlasticLIF:
    """B LIF brains: shared frozen operator, per-brain plastic motor inputs."""

    def __init__(self, W: sp.spmatrix, motor: np.ndarray, params: LIFParams,
                 seeds: list[int], normalize: bool | None = None, device: str = "cuda",
                 cache_key=None, bound: float = 2.0):
        import torch
        if torch.device(device).type == "cuda" and not gpu_available():
            raise RuntimeError("PlasticLIF needs the requested GPU; never fall back silently.")
        self.torch, self.p = torch, params
        if normalize is None:
            from flymario import config
            normalize = bool(config.get("dynamics.weight_transform")["normalize_in_degree"])
        W = normalize_in_degree(W) if normalize else W
        W = sp.csr_matrix(W, dtype=np.float32)
        self.n, self.B = W.shape[0], len(seeds)
        self.dev = torch.device(device)
        motor = np.asarray(motor, dtype=np.int64)
        self.motor_np, self.M = motor, len(motor)
        keep = np.ones(self.n, np.float32)
        keep[motor] = 0.0
        frozen = sp.csr_matrix(sp.diags(keep) @ W)
        frozen.eliminate_zeros()
        self.W, self.nnz_frozen = upload_operator(frozen, False, device,
                                                  key=None if cache_key is None else (cache_key, "frozen"))
        Wm = W[motor]                                   # rows in the readout's order
        Wm.sort_indices()
        self.P = int(Wm.nnz)
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.dev)  # noqa: E731
        self.pre = t(Wm.indices.astype(np.int64), torch.long)
        self.post = t(np.repeat(np.arange(self.M), np.diff(Wm.indptr)), torch.long)
        self.sign = t(np.sign(Wm.data), torch.float32)
        self.m0 = t(np.abs(Wm.data), torch.float32)
        self.hi = self.m0 * float(bound)
        self.motor = t(motor, torch.long)
        # block-diagonal layout: brain b's block reads columns b*n .. (b+1)*n
        B, n, P = self.B, self.n, self.P
        indeg = np.diff(Wm.indptr)
        self._crow = t(np.concatenate([[0], np.cumsum(np.tile(indeg, B))]).astype(np.int32), torch.int32)
        self._col = t((np.tile(Wm.indices.astype(np.int64), B) +
                       np.repeat(np.arange(B, dtype=np.int64) * n, P)).astype(np.int32), torch.int32)
        self.m = self.m0[None, :].repeat(B, 1).contiguous()          # (B, P) synapse counts
        self._indeg = t(indeg, torch.long)                             # connections per motor row
        mode = os.environ.get("FLYMARIO_PLASTIC_PRODUCT", "auto")
        if mode not in ("auto", "spmm", "segment"):
            raise ValueError(f"FLYMARIO_PLASTIC_PRODUCT={mode!r}: auto, spmm or segment")
        self.segment = (mode == "segment" or
                        (mode == "auto" and self.dev.type == "cuda" and torch.version.hip is None))
        self._build()
        self.decay, self.steps = float(params.decay), int(params.refractory_steps)
        self.seeds = list(seeds)
        self.reset()

    # -- weights -------------------------------------------------------------
    def _build(self) -> None:
        """The block matrix from the current magnitudes (called after updates)."""
        vals = (self.m * self.sign[None, :]).reshape(-1)
        self.A = self.torch.sparse_csr_tensor(self._crow, self._col, vals,
                                              size=(self.M * self.B, self.n * self.B))
        if self.segment:                                               # (P, B) signed magnitudes
            self._wsig = vals.view(self.B, self.P).t().contiguous()

    def update(self, delta) -> dict:
        """m += delta (B, P), then bounded to [0, 2 m0]. Returns clip counts."""
        torch = self.torch
        self.m.add_(delta)
        lo = int((self.m < 0).sum())
        hi = int((self.m > self.hi[None, :]).sum())
        self.m.clamp_(min=0.0)
        torch.minimum(self.m, self.hi[None, :], out=self.m)
        self._build()
        return {"clipped_low": lo, "clipped_high": hi}

    def set_weights(self, m) -> None:
        self.m.copy_(m)
        self._build()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seeds: list[int] | None = None) -> None:
        """Zero the state; restart every noise stream from its seed."""
        torch = self.torch
        if seeds is not None:
            if len(seeds) != self.B:
                raise ValueError("one seed per brain")
            self.seeds = list(seeds)
        n, B, dev = self.n, self.B, self.dev
        self.v = torch.zeros((n, B), dtype=torch.float32, device=dev)
        self.refrac = torch.zeros((n, B), dtype=torch.int32, device=dev)
        self.spikes_f = torch.zeros((B, n), dtype=torch.float32, device=dev).t()
        self.spikes = torch.zeros((n, B), dtype=torch.bool, device=dev)
        self._useeds = sorted(set(self.seeds))
        self._pick = torch.tensor([self._useeds.index(s) for s in self.seeds], device=dev,
                                  dtype=torch.long)
        self._gens = []
        for s in self._useeds:
            g = torch.Generator(device=dev)
            g.manual_seed(int(s))
            self._gens.append(g)
        self.obs = None

    # -- core --------------------------------------------------------------
    def step(self, current=None, observe: bool = False, noise=None):
        """One tick for every brain. `current`: (n,) or (n, B) torch tensor.

        Optional `noise` is an already-scaled (n, B) tensor on this brain's
        device. Explicit samples leave the internal random generators untouched,
        allowing different simulator implementations to replay identical input.
        """
        torch, p = self.torch, self.p
        if noise is not None:
            if not isinstance(noise, torch.Tensor) or noise.shape != (self.n, self.B):
                raise ValueError(f"noise must be a torch tensor with shape {(self.n, self.B)}")
            if noise.device != self.v.device or noise.dtype != self.v.dtype:
                raise ValueError("noise must share the membrane's device and dtype")
        if observe:
            pre_spk = self.spikes_f.index_select(0, self.pre)          # s_j(t-1), (P, B)
        self.v.mul_(self.decay)
        if bool(self.spikes.any()):
            self.v.add_((self.W @ self.spikes_f) * p.gain)
            if self.segment:                                           # CUDA: bit-repeatable
                pre = pre_spk if observe else self.spikes_f.index_select(0, self.pre)
                motor_in = torch.segment_reduce(self._wsig * pre, "sum", lengths=self._indeg, axis=0)
            else:
                flat = self.spikes_f.t().reshape(-1, 1)                # brain-major
                motor_in = (self.A @ flat).view(self.B, self.M).t()    # (M, B)
            # one addition per motor row: deterministic despite index_add_
            self.v.index_add_(0, self.motor, motor_in * p.gain)
        if p.bias:
            self.v.add_(p.bias)
        if current is not None:
            self.v.add_(current[:, None] if current.dim() == 1 else current)
        if observe:
            vhat = self.v.index_select(0, self.motor)
        z = noise
        if z is None and p.noise_std > 0:
            z = torch.stack([torch.randn(self.n, generator=g, device=self.dev)
                             for g in self._gens], dim=1).index_select(1, self._pick) * p.noise_std
        if z is not None:
            self.v.add_(z)
        free = self.refrac <= 0
        spikes = free & (self.v >= p.threshold)
        self.v.masked_fill_(spikes, p.v_reset)
        self.refrac.sub_(1).clamp_(min=0)
        self.refrac.masked_fill_(spikes, self.steps)
        self.spikes = spikes
        self.spikes_f.copy_(spikes)
        if observe:
            zm = (z.index_select(0, self.motor) if z is not None
                  else torch.zeros((self.M, self.B), device=self.dev))
            self.obs = (pre_spk, zm, vhat, spikes.index_select(0, self.motor),
                        free.index_select(0, self.motor))
        return spikes

    # -- checkpoint ----------------------------------------------------------
    def state_dict(self) -> dict:
        return {"m": self.m.clone(), "v": self.v.clone(), "refrac": self.refrac.clone(),
                "spikes": self.spikes.clone(), "seeds": list(self.seeds),
                "gens": [g.get_state() for g in self._gens]}

    def load_state_dict(self, s: dict) -> None:
        self.reset(seeds=s["seeds"])
        self.set_weights(s["m"].to(self.dev))
        self.v.copy_(s["v"])
        self.refrac.copy_(s["refrac"])
        self.spikes.copy_(s["spikes"])
        self.spikes_f.copy_(s["spikes"])
        for g, st in zip(self._gens, s["gens"]):
            g.set_state(st.cpu())       # a generator state is a CPU ByteTensor, even
                                        # for a GPU generator; map_location moves it
