"""The LIF brain on the GPU: many brains at once, one operator, same equation.

The equation is `LIFBrain.step` (spiking_reservoir.py), with a deterministic SpMM kernel (int32 CSR,
column-major activity). Summation order differs from the CPU (rounding ~1e-7), so runs are
statistically equivalent, not identical: a campaign uses one backend, recorded with every result.
`noise="numpy"` reproduces the CPU's noise stream exactly; `noise="torch"` draws on the GPU.
`gains` may give every brain its own coupling, so a whole regime sweep is one batch.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFParams, normalize_in_degree


def _torch():
    import torch
    return torch


_OPERATORS: dict = {}


def upload_operator(W: sp.spmatrix, normalize: bool, device: str = "cuda", key=None):
    """The effective operator as a GPU CSR tensor (int32 indices), cached by `key`
    (rescaling and uploading takes ~2 s)."""
    torch = _torch()
    if key is not None and (key, normalize, device) in _OPERATORS:
        return _OPERATORS[(key, normalize, device)]
    if normalize:
        W = normalize_in_degree(W)
    W = sp.csr_matrix(W, dtype=np.float32)
    T = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int32)),
                                torch.from_numpy(W.indices.astype(np.int32)),
                                torch.from_numpy(W.data), size=W.shape).to(device)
    if key is not None:
        _OPERATORS[(key, normalize, device)] = (T, int(W.nnz))
    return T, int(W.nnz)


def gpu_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class TorchLIF:
    """B independent LIF brains sharing one effective operator, on the GPU."""

    def __init__(self, W: sp.spmatrix, params: LIFParams | None = None, batch: int = 1,
                 normalize: bool | None = None, device: str = "cuda",
                 seeds: list[int] | None = None, noise: str = "torch",
                 gains: np.ndarray | None = None, cache_key=None):
        torch = _torch()
        if not gpu_available():
            raise RuntimeError("dynamics.backend is 'gpu' but no GPU is visible to torch. "
                               "Set dynamics.backend: cpu -- never fall back silently: "
                               "the backend is part of what a run is.")
        self.torch = torch
        self.p = params or LIFParams()
        if normalize is None:
            from flymario import config
            normalize = bool(config.get("dynamics.weight_transform")["normalize_in_degree"])
        self.n, self.B = W.shape[0], int(batch)
        self.device = torch.device(device)
        # int32 indices: less memory traffic, measured ~10-25 % faster than int64
        self.W, self.nnz = upload_operator(W, normalize, device, key=cache_key)
        g = np.full(self.B, self.p.gain, np.float32) if gains is None else np.asarray(gains, np.float32)
        if g.shape != (self.B,):
            raise ValueError(f"gains must have shape ({self.B},)")
        self.gains = torch.from_numpy(g).to(self.device)[None, :]
        self.decay = float(self.p.decay)
        self.steps = int(self.p.refractory_steps)
        if noise not in ("torch", "numpy"):
            raise ValueError("noise must be 'torch' or 'numpy'")
        self.noise_mode = noise
        self.seeds = list(seeds) if seeds is not None else [self.p.seed + b for b in range(self.B)]
        if len(self.seeds) != self.B:
            raise ValueError("one seed per brain")
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seeds: list[int] | None = None) -> None:
        torch = self.torch
        if seeds is not None:
            if len(seeds) != self.B:
                raise ValueError("one seed per brain")
            self.seeds = list(seeds)
        n, B, dev = self.n, self.B, self.device
        self.v = torch.zeros((n, B), dtype=torch.float32, device=dev)
        self.refrac = torch.zeros((n, B), dtype=torch.int32, device=dev)
        # Column-major activity: measured fastest for CSR x dense on this card.
        self.spikes_f = torch.zeros((B, n), dtype=torch.float32, device=dev).t()
        self.spikes = torch.zeros((n, B), dtype=torch.bool, device=dev)
        self._active = None                                  # brains that spiked last tick
        # Noise is drawn once per distinct seed and shared by every brain carrying it (as on CPU);
        # a brain's noise depends on its seed only, never on the batch layout.
        self._useeds = sorted(set(self.seeds))
        self._pick = torch.tensor([self._useeds.index(s) for s in self.seeds],
                                  device=dev, dtype=torch.long)
        if self.noise_mode == "numpy":
            self._rngs = [np.random.default_rng(s) for s in self._useeds]
        else:
            self._gens = []
            for s in self._useeds:
                g = torch.Generator(device=dev)
                g.manual_seed(int(s))
                self._gens.append(g)

    # -- core --------------------------------------------------------------
    def _noise(self):
        torch = self.torch
        p = self.p
        if p.noise_std <= 0:
            return None
        if self.noise_mode == "numpy":
            z = np.stack([r.standard_normal(self.n, dtype=np.float32) for r in self._rngs], axis=1)
            z = torch.from_numpy(z).to(self.device)
        else:
            z = torch.stack([torch.randn(self.n, generator=g, device=self.device)
                             for g in self._gens], dim=1)
        return z.index_select(1, self._pick) * p.noise_std

    def step(self, current=None):
        """Advance every brain one tick. `current`: (n,) shared or (n, B), numpy
        or torch. Returns the (n, B) boolean spike tensor, on the GPU."""
        torch = self.torch
        p = self.p
        self.v.mul_(self.decay)
        # Brains with no spike last tick get no recurrent input (as on CPU), so SpMM runs only on
        # the active columns: same kernel on a column subset, still deterministic.
        if self._active is None:
            pass                                             # all silent
        elif self._active.numel() == self.B:
            self.v.add_((self.W @ self.spikes_f) * self.gains)
        else:
            a = self._active
            sub = self.W @ self.spikes_f.index_select(1, a)
            self.v.index_add_(1, a, sub * self.gains.index_select(1, a))
        if p.bias:
            self.v.add_(p.bias)
        if current is not None:
            c = current if isinstance(current, torch.Tensor) else torch.from_numpy(
                np.asarray(current, dtype=np.float32))
            c = c.to(self.device, non_blocking=True)
            self.v.add_(c[:, None] if c.dim() == 1 else c)
        z = self._noise()
        if z is not None:
            self.v.add_(z)
        spikes = (self.refrac <= 0) & (self.v >= p.threshold)
        self.v.masked_fill_(spikes, p.v_reset)
        self.refrac.sub_(1).clamp_(min=0)                 # floored, as on CPU
        self.refrac.masked_fill_(spikes, self.steps)
        self.spikes = spikes
        self.spikes_f.copy_(spikes)
        live = spikes.any(0)
        n_live = int(live.sum())                             # one sync per tick
        self._active = (None if n_live == 0 else
                        torch.arange(self.B, device=self.device) if n_live == self.B else
                        live.nonzero().squeeze(1))
        return spikes

    @property
    def firing_rate(self) -> np.ndarray:
        return self.spikes.float().mean(0).cpu().numpy()


class TorchBrain:
    """One GPU brain behind `LIFBrain`'s interface, so FlyAgent can use either.

    Noise is the CPU's own stream; only summation order differs. Spikes come back as numpy.
    """

    def __init__(self, W: sp.spmatrix, params: LIFParams | None = None,
                 normalize: bool | None = None, device: str = "cuda", noise: str = "numpy"):
        self.p = params or LIFParams()
        self._b = TorchLIF(W, self.p, batch=1, normalize=normalize, device=device,
                           seeds=[self.p.seed], noise=noise)
        self.n = self._b.n
        self.W = _NnzView(self._b.nnz)
        self.spikes = np.zeros(self.n, bool)

    def reset(self, seed: int | None = None) -> None:
        self._b.reset(seeds=[seed] if seed is not None else None)
        self.spikes = np.zeros(self.n, bool)

    def step(self, sensory_current: np.ndarray | None = None) -> np.ndarray:
        s = self._b.step(sensory_current)
        self.spikes = s[:, 0].cpu().numpy()
        return self.spikes

    def run(self, ticks: int, sensory_current: np.ndarray | None = None) -> np.ndarray:
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(ticks):
            counts += self.step(sensory_current)
        return counts

    @property
    def firing_rate(self) -> float:
        return float(self.spikes.mean())


class _NnzView:
    """Enough of a scipy matrix for code that only asks `brain.W.nnz`."""

    def __init__(self, nnz: int):
        self.nnz = nnz


def make_brain(W: sp.spmatrix, params: LIFParams | None = None, normalize: bool | None = None,
               backend: str | None = None, noise: str = "numpy"):
    """The one place a single brain is built: `cpu` (reference) or `gpu`.
    `backend=None` follows dynamics.backend in configs/base.yaml."""
    if backend is None:
        from flymario import config
        backend = config.get("dynamics.backend", "cpu")
    if backend == "gpu":
        return TorchBrain(W, params, normalize=normalize, noise=noise)
    from flymario.models.spiking_reservoir import LIFBrain
    return LIFBrain(W, params, normalize=normalize)
