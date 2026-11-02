"""Projected surrogate-gradient BPTT control for Gate G.

The hard-spike LIF dynamics are used in the forward pass. Autograd replaces
only the threshold derivative with a fast-sigmoid surrogate. The fixed readout
is never optimized; gradients can change magnitudes only on the selected
incoming motor edges, with signs, support and [0, 2*m0] bounds preserved.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import scipy.sparse as sp

from flymario import config
from flymario.agent import AgentConfig
from flymario.models.spiking_reservoir import LIFParams, normalize_in_degree


class _SurrogateSpike:
    """Hard threshold forward; fast-sigmoid derivative in the backward pass."""

    @staticmethod
    def function(slope: float = 1.0):
        import torch

        slope = float(slope)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("surrogate slope must be finite and positive")

        class HardStep(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return (x >= 0).to(x.dtype)

            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return grad / (1.0 + slope * x.abs()).square()

        return HardStep


class SurrogateBPTT:
    """BPTT over short decision blocks for one fixed-readout learner.

    `cue_currents` has shape (2, neurons), `readout` has shape (2, motors) in
    RIGHT, LEFT order, and each target is 0 (RIGHT) or 1 (LEFT). State is
    carried between blocks but detached at update boundaries (truncated BPTT).
    """

    def __init__(self, W: sp.spmatrix, motor: np.ndarray, cue_currents: np.ndarray,
                 readout: np.ndarray, params: LIFParams, seed: int,
                 device: str = "cuda", block_len: int = 10, temperature: float = 1.0,
                 surrogate_slope: float = 1.0, detach_reset: bool = False):
        import torch

        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("surrogate BPTT requires the configured GPU; no silent fallback")
        self.torch, self.dev = torch, torch.device(device)
        self.n, self.motor_np = W.shape[0], np.asarray(motor, dtype=np.int64)
        self.M, self.block_len = len(self.motor_np), int(block_len)
        self.params, self.temperature = params, float(temperature)
        if self.block_len <= 0 or self.temperature <= 0:
            raise ValueError("block_len and temperature must be positive")

        if bool(config.get("dynamics.weight_transform")['normalize_in_degree']):
            W = normalize_in_degree(W)
        W = sp.csr_matrix(W, dtype=np.float32)
        keep = np.ones(self.n, dtype=np.float32)
        keep[self.motor_np] = 0.0
        Wf = (sp.diags(keep) @ W).tocsr()
        Wf.eliminate_zeros()
        Wm_csr = W[self.motor_np].tocsr()
        Wm_csr.sum_duplicates()
        Wm_csr.sort_indices()
        Wm = Wm_csr.tocoo()

        def tensor(a, dtype=torch.float32):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev)

        crow = torch.as_tensor(Wf.indptr.astype(np.int32), dtype=torch.int32, device=self.dev)
        col = torch.as_tensor(Wf.indices.astype(np.int32), dtype=torch.int32, device=self.dev)
        val = tensor(Wf.data)
        self.Wf = torch.sparse_csr_tensor(crow, col, val, size=Wf.shape, device=self.dev)
        self.edge_index = torch.stack((
            torch.as_tensor(Wm.row, dtype=torch.long, device=self.dev),
            torch.as_tensor(Wm.col, dtype=torch.long, device=self.dev),
        ))
        self.sign = tensor(np.sign(Wm.data))
        self.m0 = tensor(np.abs(Wm.data))
        self.hi = 2.0 * self.m0
        self.m = torch.nn.Parameter(self.m0.clone())
        self.motor = torch.as_tensor(self.motor_np, dtype=torch.long, device=self.dev)
        self.cues = tensor(cue_currents.T)[:, :, None]                 # (neurons, 2, 1)
        self.readout = tensor(readout)                                 # (2, motors)
        self.decay = float(params.decay)
        self.refractory_steps = int(params.refractory_steps)
        self.beta = float(np.exp(-params.dt_ms / AgentConfig().trace_tau_ms))
        self.surrogate_slope = float(surrogate_slope)
        self.detach_reset = bool(detach_reset)
        self.spike = _SurrogateSpike.function(self.surrogate_slope).apply
        self.noise_rng = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.state = self._zero_state()
        self.warm_done = False
        self.audit: dict | None = None
        self.state_grad_clip_norm: float | None = None

    def _audit_tensor(self, value) -> dict:
        """Summarize a tensor without retaining it or building an autograd graph."""
        torch = self.torch
        with torch.no_grad():
            finite = torch.isfinite(value)
            safe = torch.where(finite, value, torch.zeros_like(value))
            maximum = safe.abs().max() if value.numel() else torch.zeros((), device=value.device)
            rms = (maximum * (safe / maximum.clamp_min(torch.finfo(value.dtype).tiny))
                   .square().mean().sqrt()) if value.numel() else maximum
            return {
                "nonfinite": int((~finite).sum().item()),
                "max_abs_finite": float(maximum.item()),
                "rms_finite": float(rms.item()),
            }

    def _state_gradient_hook(self, kind: str, tick: int, grad):
        """Optionally cap a recurrent-state adjoint before it travels farther back."""
        torch = self.torch
        scale = torch.ones((), dtype=grad.dtype, device=grad.device)
        clipped = grad
        if self.state_grad_clip_norm is not None:
            with torch.no_grad():
                finite = torch.isfinite(grad)
                safe = torch.where(finite, grad, torch.zeros_like(grad))
                max_abs = safe.abs().max()
                scaled = safe / max_abs.clamp_min(torch.finfo(grad.dtype).tiny)
                scaled_norm = torch.linalg.vector_norm(scaled)
                scale = torch.clamp(
                    (self.state_grad_clip_norm /
                     scaled_norm.clamp_min(torch.finfo(grad.dtype).tiny)) /
                    max_abs.clamp_min(torch.finfo(grad.dtype).tiny),
                    max=1.0,
                )
                clipped = torch.where(finite, safe * scale, grad)
        if self.audit is not None:
            self.audit["state_gradients"].append({
                "kind": kind, "tick": tick,
                "pre_clip": self._audit_tensor(grad),
                "post_clip": self._audit_tensor(clipped),
                "clip_scale": float(scale.item()),
            })
        return clipped

    def _motor_matrix(self, magnitudes):
        """Build the plastic sparse matrix without sorting its fixed support.

        `Wm_csr` is deduplicated and sorted before conversion to COO, so its
        coordinates are already in canonical row/column order. The weights
        change, but the sparse support does not. Marking this fact avoids a
        full COO coalesce (a sort over ~769k edges) on every forward/recompute.
        """
        torch = self.torch
        values = self.sign * magnitudes
        return torch.sparse_coo_tensor(
            self.edge_index, values, (self.M, self.n), device=self.dev,
            is_coalesced=True,
        )

    def _zero_state(self):
        torch = self.torch
        return (torch.zeros((self.n, 1), device=self.dev),
                torch.zeros((self.n, 1), dtype=torch.int32, device=self.dev),
                torch.zeros((self.n, 1), device=self.dev),
                torch.zeros((self.M, 1), device=self.dev))

    def _tick(self, v, refrac, spikes_prev, trace, current, xi, Wm):
        torch = self.torch
        vhat = self.decay * v + self.params.gain * torch.sparse.mm(self.Wf, spikes_prev)
        motor_in = torch.sparse.mm(Wm, spikes_prev)
        vhat = vhat.index_copy(
            0, self.motor,
            vhat.index_select(0, self.motor) + self.params.gain * motor_in,
        )
        if self.params.bias:
            vhat = vhat + self.params.bias
        vhat = vhat + current
        noisy = vhat + xi
        free = (refrac <= 0).to(noisy.dtype)
        spikes = self.spike(noisy - self.params.threshold) * free
        reset_spikes = spikes.detach() if self.detach_reset else spikes
        v = noisy * (1.0 - reset_spikes) + self.params.v_reset * reset_spikes
        refrac = (refrac - 1).clamp_min(0)
        refrac = torch.where(spikes.detach().bool(), self.refractory_steps, refrac)
        spikes_prev = spikes
        trace = self.beta * trace + spikes_prev.index_select(0, self.motor)
        return v, refrac, spikes_prev, trace

    def _decision(self, v, refrac, spikes_prev, trace, current, noise, Wm):
        for xi in noise:
            v, refrac, spikes_prev, trace = self._tick(
                v, refrac, spikes_prev, trace, current, xi, Wm
            )
        scores = self.readout @ trace
        return v, refrac, spikes_prev, trace, scores

    def _noise(self, ticks: int):
        return self.torch.randn((ticks, self.n, 1), generator=self.noise_rng,
                                device=self.dev) * self.params.noise_std

    def _pace_gpu(self, clock: list[float], duty: float) -> None:
        if self.dev.type != "cuda" or duty >= 1.0:
            return
        self.torch.cuda.synchronize(self.dev)
        elapsed = time.monotonic() - clock[0]
        idle = elapsed * (1.0 / duty - 1.0)
        if idle > 0.0:
            time.sleep(idle)
        clock[0] = time.monotonic()

    def warm_up(self, first_cue: int, ticks: int, gpu_duty: float = 0.6,
                pace_clock: list[float] | None = None) -> None:
        """Run unscored burn-in, then detach before the first scored block.

        Differentiating through this long, unlabelled burn-in made the recurrent
        gradient non-finite on the MaleCNS graph. The burn-in still sets the
        exact forward state; BPTT starts at the first scored decision.
        """
        if self.warm_done:
            raise RuntimeError("the BPTT warm-up has already been run")
        current = self.cues[:, int(first_cue)]
        noise = self._noise(ticks)
        Wm = self._motor_matrix(self.m)
        pace_clock = pace_clock or [time.monotonic()]
        with self.torch.no_grad():
            v, refrac, spikes_prev, trace = self.state
            for tick, xi in enumerate(noise):
                v, refrac, spikes_prev, trace = self._tick(
                    v, refrac, spikes_prev, trace, current, xi, Wm
                )
                if self.audit is not None:
                    self.audit["warmup_ticks"].append({
                        "tick": tick,
                        "membrane": self._audit_tensor(v),
                        "motor_trace": self._audit_tensor(trace),
                    })
                self._pace_gpu(pace_clock, gpu_duty)
            self.state = (v, refrac, spikes_prev, trace)
        self.warm_done = True

    def fit(self, cues: np.ndarray, targets: np.ndarray, n_trials: int, lr: float,
            ticks_per_decision: int, warmup_ticks: int, ctl=None,
            gpu_duty: float = 0.6, apply_update: bool = True,
            gradient_audit: bool = False, detect_anomaly: bool = False,
            state_grad_clip_norm: float | None = None) -> dict:
        """Train with projected Adam and return per-trial loss and hard accuracy."""
        from torch.nn import functional as F
        from torch.utils.checkpoint import checkpoint

        torch = self.torch
        cues = np.asarray(cues[:n_trials], dtype=np.int64)
        targets = np.asarray(targets[:n_trials], dtype=np.int64)
        if cues.shape != (n_trials,) or targets.shape != (n_trials,):
            raise ValueError("cues and targets must contain one class per training trial")
        if not 0.0 < gpu_duty <= 1.0:
            raise ValueError("gpu_duty must be in (0, 1]")
        if state_grad_clip_norm is not None and state_grad_clip_norm <= 0:
            raise ValueError("state_grad_clip_norm must be positive")
        self.state_grad_clip_norm = (None if state_grad_clip_norm is None
                                     else float(state_grad_clip_norm))
        pace_clock = [time.monotonic()]
        self.audit = ({"warmup_ticks": [], "forward_ticks": [],
                       "decision_outputs": [], "state_gradients": []}
                      if gradient_audit else None)
        if not self.warm_done:
            self.warm_up(int(cues[0]), warmup_ticks, gpu_duty, pace_clock)
        optimizer = torch.optim.Adam([self.m], lr=float(lr))
        losses = np.zeros(n_trials, dtype=np.float32)
        correct = np.zeros(n_trials, dtype=np.float32)
        for start in range(0, n_trials, self.block_len):
            stop = min(start + self.block_len, n_trials)
            optimizer.zero_grad(set_to_none=True)
            v, refrac, spikes_prev, trace = self.state
            block_losses = []
            # Weights remain frozen for this entire block. Checkpoint each tick,
            # not the whole 16-tick decision: backward then rematerializes one
            # recurrent step at a time and avoids the previous per-decision
            # activation peak. Reuse the sparse plastic matrix throughout.
            Wm = self._motor_matrix(self.m)
            tick_index = 0
            for k in range(start, stop):
                current = self.cues[:, int(cues[k])]
                noise = self._noise(ticks_per_decision)
                for xi in noise:
                    v, refrac, spikes_prev, trace = checkpoint(
                        self._tick, v, refrac, spikes_prev, trace, current, xi, Wm,
                        use_reentrant=False,
                    )
                    if self.audit is not None:
                        self.audit["forward_ticks"].append({
                            "tick": tick_index,
                            "membrane": self._audit_tensor(v),
                            "motor_trace": self._audit_tensor(trace),
                        })
                    if v.requires_grad and (self.audit is not None or
                                            self.state_grad_clip_norm is not None):
                        v.register_hook(
                            lambda grad, tick=tick_index:
                            self._state_gradient_hook("membrane", tick, grad)
                        )
                    if spikes_prev.requires_grad:
                        def spike_hook(grad, clock=pace_clock, duty=gpu_duty,
                                       tick=tick_index):
                            self._pace_gpu(clock, duty)
                            return self._state_gradient_hook("spikes", tick, grad)
                        spikes_prev.register_hook(spike_hook)
                    self._pace_gpu(pace_clock, gpu_duty)
                    tick_index += 1
                scores = self.readout @ trace
                target = torch.tensor([int(targets[k])], dtype=torch.long, device=self.dev)
                loss = F.cross_entropy(scores.T / self.temperature, target)
                block_losses.append(loss)
                if self.audit is not None:
                    self.audit["decision_outputs"].append({
                        "decision": k,
                        "scores": [float(x) for x in scores.detach().cpu().flatten()],
                        "loss": float(loss.detach()),
                    })
                action = int(scores[:, 0].argmax().item())
                losses[k] = float(loss.detach())
                correct[k] = float(action == int(targets[k]))
            try:
                anomaly_context = (torch.autograd.detect_anomaly(check_nan=True)
                                   if detect_anomaly else nullcontext())
                with anomaly_context:
                    torch.stack(block_losses).mean().backward()
            except BaseException as exc:
                if self.audit is not None:
                    self.audit["backward_exception"] = f"{type(exc).__name__}: {exc}"
                raise
            if self.audit is not None:
                self.audit["plastic_edge_gradient"] = (
                    None if self.m.grad is None else self._audit_tensor(self.m.grad)
                )
            if self.m.grad is None or not bool(torch.isfinite(self.m.grad).all()):
                bad = (self.m.numel() if self.m.grad is None else
                       int((~torch.isfinite(self.m.grad)).sum().item()))
                if self.audit is not None and self.m.grad is not None:
                    self.audit["plastic_edge_gradient"].update({
                        "nan": int(torch.isnan(self.m.grad).sum().item()),
                        "inf": int(torch.isinf(self.m.grad).sum().item()),
                    })
                raise FloatingPointError(
                    f"surrogate BPTT produced non-finite gradients on {bad} weights"
                )
            if apply_update:
                optimizer.step()
                with torch.no_grad():
                    self.m.clamp_(min=0.0)
                    torch.minimum(self.m, self.hi, out=self.m)
                    if not bool(torch.isfinite(self.m).all()):
                        raise FloatingPointError("surrogate BPTT produced non-finite weights")
            # A no-update diagnostic still advances continuous forward time.
            # Otherwise every block restarts at the post-warmup state.
            self.state = (v.detach(), refrac.detach(), spikes_prev.detach(), trace.detach())
            if self.dev.type == "cuda":
                self._pace_gpu(pace_clock, gpu_duty)
            if ctl is not None:
                ctl.sub(stop, n_trials, "surrogate BPTT training trial")
        return {"loss": losses, "correct": correct, "weights": self.m.detach().clone(),
                "block_len": self.block_len, "lr": float(lr),
                "checkpoint_granularity": "one simulation tick",
                "gpu_duty_target": gpu_duty,
                "optimizer_step_applied": bool(apply_update),
                "state_grad_clip_norm": self.state_grad_clip_norm,
                "surrogate": "hard threshold forward; fast-sigmoid backward",
                "surrogate_slope": self.surrogate_slope,
                "detach_reset": self.detach_reset,
                "burn_in_gradient": "detached before first scored decision",
                "gradient_audit": self.audit}
