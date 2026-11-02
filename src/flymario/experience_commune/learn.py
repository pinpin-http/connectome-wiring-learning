"""Batched plastic learners of one graph (docs/experience_commune/PROTOCOL_FROZEN.md §4-§5).

Per decision k, for learner j (task, mode, block): sweep from rest -> counts c, eligibility e;
z = R c ; u = softmax(z / tau) ; pi = (1 - eps) u + eps / 8 ; action: N inverse CDF of pi at
N_uniform[k], U the block's U_action[k] ; r = 1[action = class], b = class-indexed baseline ;
v = rho (r - b) (1 - eps) (u_a / pi_a) (onehot(a) - u), rho = pi_a / q_a ; L = R^T v / tau ;
m += eta U m0 bounded to [0, 2 m0] (centred, RMS-normalised eligibility, as study_s.py).
Evaluation uses a separate brain with copies of the plastic magnitudes, noise keyed per image.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from flymario.experience_commune import policy as P

KAPPA, EBAR, RD, TAU, EPS, BRATE, ETA, PSI = 0.2, 0.05, 0.95, 2.0, 0.1, 0.1, 0.001, 0.5
CHECKPOINTS = (0, 250, 500, 1000, 2000, 4000, 6000)
STUDY = 20261020
BANK_KEY = {"progress": 1, "final": 2, "ood": 3}


def fast_plastic_class():
    """PlasticLIF with the activity held row-major (n, B) for the frozen operator's product.

    Column-major activity makes the CSR product ~30x slower on an RTX 3090. Results are identical
    (integer weights, 0/1 spikes); checked bit for bit by scripts/experience_commune/sim_checks.py."""
    from flymario.training.plastic import PlasticLIF

    class FastPlasticLIF(PlasticLIF):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._col = self._crow = self.A = None            # block-diagonal layout unused on the segment path
            self.torch.cuda.empty_cache()

        def _build(self):
            if not self.segment:
                raise RuntimeError("row-major activity is only valid on the segment (CUDA) path")
            self._wsig = (self.m * self.sign[None, :]).t().contiguous()      # (P, B), as PlasticLIF._build

        def reset(self, seeds=None):
            super().reset(seeds)
            self.spikes_f = self.torch.zeros((self.n, self.B), dtype=self.torch.float32, device=self.dev)
    return FastPlasticLIF


@dataclass(frozen=True)
class Learner:
    task: str
    mode: str                 # "N" own policy | "U" common uniform experience
    block: str
    kind: str = "main"        # main | supervised | shuffled | frozen (development controls only)
    eta: float | None = None  # revision48: per-learner learning rate (None: ETA)
    tau: float | None = None  # revision48: per-learner temperature (None: TAU)

    @property
    def key(self) -> str:
        k = f"{self.task}-{self.mode}-{self.block}-{self.kind}"
        if self.eta is not None or self.tau is not None:
            k += f"-eta{self.eta if self.eta is not None else ETA}-tau{self.tau if self.tau is not None else TAU}"
        return k


def eval_key(seed: int, uid: int) -> int:
    s = np.random.SeedSequence([STUDY, int(seed), int(uid)]).generate_state(2, dtype=np.uint32)
    return int((int(s[0]) << 31) ^ int(s[1]))


def m_digest(m_row) -> str:
    return hashlib.sha256(m_row.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def lognorm_operator(W):
    """revision48 weight conversion: log1p of each summed signed connection count, then every
    neuron's incoming absolute mass normalised to 1. `W` is the raw signed CSR W[dst, src]."""
    import scipy.sparse as sp

    from flymario.models.spiking_reservoir import normalize_in_degree
    W = sp.csr_matrix(W, dtype=np.float32, copy=True)
    W.sum_duplicates()
    W.data = (np.sign(W.data) * np.log1p(np.abs(W.data))).astype(np.float32)
    return sp.csr_matrix(normalize_in_degree(W), dtype=np.float32)


class SegmentOperator:
    """The frozen operator's product as a fixed-order segment reduction (revision48 option).

    Bit-repeatable with fractional (log) weights, unlike the CUDA CSR product; computed in column
    chunks to bound memory. It replaces `brain.W` (anything with `@`)."""

    def __init__(self, T, chunk_cols: int = 16):
        import torch
        T = T.to_sparse_csr() if T.layout != torch.sparse_csr else T
        self.torch, self.chunk = torch, int(chunk_cols)
        self.col = T.col_indices().long()
        self.val = T.values()
        self.lengths = (T.crow_indices()[1:] - T.crow_indices()[:-1]).long()
        self.shape = T.shape

    def __matmul__(self, x):
        torch = self.torch
        outs = []
        for c0 in range(0, x.shape[1], self.chunk):
            xs = x[:, c0:c0 + self.chunk]
            outs.append(torch.segment_reduce(self.val[:, None] * xs.index_select(0, self.col), "sum",
                                             lengths=self.lengths, axis=0))
        return torch.cat(outs, 1) if len(outs) > 1 else outs[0]


class Exact64Operator:
    """The frozen operator's product made exact, hence independent of summation order (revision48 option).

    Weights are put on the grid of multiples of 2^-GRID_BITS and held in float64; with binary inputs every
    partial sum is then exact while a row's absolute mass is below 2^(53 - GRID_BITS), rounded once to float32."""
    GRID_BITS = 40

    def __init__(self, T):
        import torch
        T = T.to_sparse_csr() if T.layout != torch.sparse_csr else T
        self.torch = torch
        v = T.values().double()
        q = torch.round(v * 2.0 ** self.GRID_BITS) / 2.0 ** self.GRID_BITS
        crow = T.crow_indices()
        mass = torch.zeros(T.shape[0], dtype=torch.float64, device=v.device)
        rows = torch.repeat_interleave(torch.arange(T.shape[0], device=v.device), crow[1:] - crow[:-1])
        mass.index_add_(0, rows, q.abs())
        self.max_row_mass = float(mass.max()) if mass.numel() else 0.0
        if self.max_row_mass >= 2.0 ** (53 - self.GRID_BITS):
            raise ValueError(f"row absolute mass {self.max_row_mass} too large for an exact float64 sum")
        self.max_grid_change = float((q - v).abs().max()) if v.numel() else 0.0
        self.T = torch.sparse_csr_tensor(crow, T.col_indices(), q, size=T.shape, device=v.device)
        self.shape = T.shape

    def __matmul__(self, x):
        return (self.T @ x.double()).float()


class GraphRun:
    def __init__(self, spec: str, learners: list[Learner], data_dir: Path, eval_cols: int = 128,
                 device: str = "cuda", pair_u_to_n: bool = False, slow: bool = False, gamma: float = 1.0,
                 horizon: int = 6000, weights: str = "raw", frozen_product: str = "csr", base_gain: float | None = None):
        import torch

        from flymario import config
        from flymario.connectome import graphsel
        from flymario.models.spiking_reservoir import LIFParams
        from flymario.policy.readout import FixedRandomProjection
        from flymario.training.plastic import PlasticLIF as _Slow
        PlasticLIF = _Slow if slow else fast_plastic_class()
        self.torch, self.spec, self.learners = torch, spec, list(learners)
        self.dev = torch.device(device)
        self.B = len(learners)
        # gamma: global-gain multiplier (1 in the frozen campaign; ACTIVITY_MATCH's calibrated gains otherwise).
        # revision48: `base_gain` replaces dynamics.gain, `weights` "raw" (synapse counts) or "lognorm",
        # `horizon` the number of training decisions
        if weights not in ("raw", "lognorm"):
            raise ValueError(f"weights={weights!r}: raw or lognorm")
        self.weights, self.horizon = weights, int(horizon)
        base = float(config.get("dynamics.gain")) if base_gain is None else float(base_gain)
        self.params = LIFParams(gain=base * float(gamma), noise_std=float(config.get("dynamics.noise_std")))
        self.ticks = int(config.get("timing.ticks_per_decision"))
        t0 = time.time()
        _g, W, motor = graphsel.load(spec)
        if weights == "lognorm":
            W = lognorm_operator(W)
        self.load_s = time.time() - t0
        groups = sorted({(l.block, l.task) for l in learners})
        self.groups = groups
        self.gid = np.array([groups.index((l.block, l.task)) for l in learners])
        self.data, self.data_sha = {}, {}
        from flymario.experience_commune.data import sha256
        for b, t in groups:
            f = Path(data_dir) / b / f"{t}.npz"
            z = np.load(f)
            self.data[(b, t)] = {k: z[k] for k in z.files}
            self.data_sha[f"{b}/{t}"] = sha256(f)
        # test hook: U learners take the action of the N learner of the same block and task, with
        # q = pi_a (rho = 1); both must then follow the same computation exactly
        self.pair_u_to_n = bool(pair_u_to_n)
        self.idx = self.data[groups[0]]["idx"]
        for d in self.data.values():
            if not np.array_equal(d["idx"], self.idx):
                raise RuntimeError("blocks disagree on the injected cells")
        seeds = [int(self.data[(l.block, l.task)]["noise_train_seed"]) for l in learners]
        self.brain = PlasticLIF(W, motor, self.params, seeds=seeds, normalize=False)
        if frozen_product == "segment":
            self.brain.W = SegmentOperator(self.brain.W)
        elif frozen_product == "exact64":
            self.brain.W = Exact64Operator(self.brain.W)
        elif frozen_product != "csr":
            raise ValueError(f"frozen_product={frozen_product!r}: csr, segment or exact64")
        self.frozen_product = frozen_product
        self.M, self.P_ = self.brain.M, self.brain.P
        self.R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(
            self.M, 8, int(self.data[(l.block, l.task)]["readout_seed"])).W) for l in learners]), device=self.dev)
        self.eta = torch.tensor([0.0 if l.kind == "frozen" else (ETA if l.eta is None else float(l.eta))
                                 for l in learners], device=self.dev)
        self.tau = np.array([TAU if l.tau is None else float(l.tau) for l in learners], np.float64)
        self.tau_t = torch.as_tensor(self.tau, dtype=torch.float32, device=self.dev)
        self.Xtr = torch.as_tensor(np.stack([self.data[g_]["train_x"] for g_ in groups])).pin_memory()   # CPU
        self.cls = np.stack([self.data[g_]["train_cls"] for g_ in groups]).astype(np.int64)
        self.U_action = np.stack([self.data[g_]["U_action"] for g_ in groups]).astype(np.int64)
        self.N_uniform = np.stack([self.data[g_]["N_uniform"] for g_ in groups])
        self.shuf_u = np.stack([self.data[g_]["shuffled_reward_uniform"] for g_ in groups])
        self._idx_t = torch.as_tensor(self.idx, device=self.dev, dtype=torch.long)
        self.eval_cols = int(eval_cols)
        self.img_chunk = 64
        self.eval_brain = PlasticLIF(W, motor, self.params, seeds=[1] * self.eval_cols, normalize=False)
        if frozen_product == "segment":
            self.eval_brain.W = SegmentOperator(self.eval_brain.W)
        elif frozen_product == "exact64":
            self.eval_brain.W = Exact64Operator(self.eval_brain.W)
        del W
        self.k = 0
        self.base = np.zeros((self.B, 8))
        self.ebar = None
        self.ms = torch.zeros(self.B, device=self.dev)
        n = self.horizon
        self.diag = {k: np.zeros((n, self.B), np.float32) for k in
                     ("entropy_pi", "max_pi", "baseline", "rho", "mix_factor", "v_norm", "upd_rms_pre", "upd_rms_post",
                      "active_out_frac", "clipped_frac")}
        self.diag["action"] = np.zeros((n, self.B), np.int8)
        self.diag["reward"] = np.zeros((n, self.B), np.int8)
        self.step_s = np.zeros(n, np.float32)
        self.preds: dict = {}
        self.weight_digests: dict = {}

    # -- training ---------------------------------------------------------------------------------------------------
    def _current(self, xs):
        cur = self.torch.zeros((self.brain.n, xs.shape[0]), device=self.dev)
        cur[self._idx_t] = xs.t()
        return cur

    def step(self) -> None:
        from flymario.training.sweep import sweep_once
        torch, k, B = self.torch, self.k, self.B
        t0 = time.time()
        xs = self.Xtr[torch.as_tensor(self.gid), k].to(self.dev, non_blocking=True)
        c, e = sweep_once(self.brain, self.params, self._current(xs), self.ticks, PSI, True)
        z = torch.einsum("bam,mb->ba", self.R, c).double().cpu().numpy()
        u, pi = P.probs(z, self.tau[:, None], EPS)
        cls = self.cls[self.gid, k]
        a = np.empty(B, np.int64)
        for j, l in enumerate(self.learners):
            if l.mode == "N":
                cdf = np.cumsum(pi[j] / pi[j].sum())
                a[j] = min(int(np.searchsorted(cdf, self.N_uniform[self.gid[j], k], side="right")), 7)
            else:
                a[j] = self.U_action[self.gid[j], k]
        if self.pair_u_to_n:
            for j, l in enumerate(self.learners):
                if l.mode == "U":
                    jn = next(i for i, o in enumerate(self.learners)
                              if o.mode == "N" and (o.block, o.task, o.kind) == (l.block, l.task, l.kind))
                    a[j] = a[jn]
        r = (a == cls).astype(np.float64)
        for j, l in enumerate(self.learners):
            if l.kind == "shuffled":               # reward independent of action and label, at the U rate 1/8
                r[j] = float(self.shuf_u[self.gid[j], k] < 0.125)
        b = self.base[np.arange(B), cls]
        q = np.array([pi[j, a[j]] if (l.mode == "N" or self.pair_u_to_n) else 1.0 / 8
                      for j, l in enumerate(self.learners)])
        s = P.signal(z, a, r, b, q, self.tau[:, None], EPS)
        v = s["v"]
        for j, l in enumerate(self.learners):
            if l.kind == "supervised":            # development reference: the target class, no policy, no reward
                v[j] = np.eye(8)[cls[j]] - u[j]
        self.base[np.arange(B), cls] += BRATE * (r - b)
        if self.ebar is None:
            self.ebar = e.clone()
        ec = (e - self.ebar) + KAPPA * self.ebar
        self.ebar.mul_(1 - EBAR).add_(e, alpha=EBAR)
        Lsig = torch.einsum("ba,bam->mb", torch.as_tensor(v, device=self.dev, dtype=torch.float32), self.R) / self.tau_t
        U = ec * Lsig.index_select(0, self.brain.post) * self.brain.sign[:, None]
        pre = (U * U).mean(0)
        self.ms = RD * self.ms + (1 - RD) * pre
        U = U / torch.sqrt(self.ms / (1 - RD ** (k + 1)) + 1e-30)[None, :]
        post = (U * U).mean(0)
        clip = self.brain.update((U * self.eta[None, :]).t() * self.brain.m0[None, :])
        d = self.diag
        d["entropy_pi"][k], d["max_pi"][k], d["baseline"][k] = P.entropy(pi), pi.max(1), b
        d["rho"][k], d["mix_factor"][k] = s["rho"], s["mix_factor"]
        d["v_norm"][k] = np.linalg.norm(v, axis=1)
        d["upd_rms_pre"][k], d["upd_rms_post"][k] = pre.sqrt().cpu().numpy(), post.sqrt().cpu().numpy()
        d["active_out_frac"][k] = (c > 0).float().mean(0).cpu().numpy()
        d["clipped_frac"][k] = (clip["clipped_low"] + clip["clipped_high"]) / (B * self.P_)
        d["action"][k], d["reward"][k] = a, r
        self.step_s[k] = time.time() - t0
        self.k += 1

    # -- evaluation -------------------------------------------------------------------------------------------------
    def evaluate(self, bank: str) -> np.ndarray:
        """Greedy predictions (B, n_bank) of every learner on its own block and task's bank; only the
        plastic magnitudes are read, nothing is written. Noise per chunk of `img_chunk` images is keyed by
        (block evaluation seed, bank, chunk index), so an image always gets the same noise."""
        from flymario.training.sweep import reset_network
        torch, eb, C = self.torch, self.eval_brain, self.eval_cols
        nb = len(self.data[self.groups[0]][f"{bank}_cls"])
        preds = np.zeros((self.B, nb), np.int8)
        for gi, g_ in enumerate(self.groups):
            js_all = [j for j in range(self.B) if self.gid[j] == gi]
            per = max(1, C // len(js_all))                     # images per sweep for this group
            X = torch.as_tensor(self.data[g_][f"{bank}_x"], device=self.dev)
            seed = int(self.data[g_]["eval_noise_seed"])
            for c0 in range(0, nb, self.img_chunk):
                imgs = list(range(c0, min(c0 + self.img_chunk, nb)))
                gen = torch.Generator(device=self.dev)
                gen.manual_seed(eval_key(seed, BANK_KEY[bank] * 100_000 + c0 // self.img_chunk))
                noise = torch.randn((self.ticks, eb.n, self.img_chunk), generator=gen, device=self.dev)
                noise.mul_(self.params.noise_std)
                for s0 in range(0, len(imgs), per):
                    sub = imgs[s0:s0 + per]
                    cols = [(j, i) for j in js_all for i in sub]
                    pad = cols + [cols[-1]] * (C - len(cols))
                    js = torch.as_tensor([j for j, _ in pad], device=self.dev)
                    ii = torch.as_tensor([i for _, i in pad], device=self.dev)
                    eb.set_weights(self.brain.m.index_select(0, js))
                    cur = self._current(X.index_select(0, ii))
                    loc = ii - c0
                    reset_network(eb)
                    cnt = torch.zeros((eb.M, C), device=self.dev)
                    for t in range(self.ticks):
                        spk = eb.step(cur, observe=False, noise=noise[t].index_select(1, loc))
                        cnt.add_(spk.index_select(0, eb.motor).float())
                    sc = torch.einsum("bam,mb->ba", self.R.index_select(0, js), cnt).argmax(1).cpu().numpy()
                    for col, (j, i) in enumerate(cols):
                        preds[j, i] = sc[col]
        return preds

    def labels(self, bank: str) -> np.ndarray:
        return np.stack([self.data[(l.block, l.task)][f"{bank}_cls"] for l in self.learners]).astype(np.int8)

    # -- checkpoints ------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        return {"k": self.k, "brain": self.brain.state_dict(), "ebar": self.ebar, "ms": self.ms.clone(),
                "base": self.base.copy(), "diag": {k: v.copy() for k, v in self.diag.items()},
                "step_s": self.step_s.copy(), "preds": dict(self.preds), "weight_digests": dict(self.weight_digests),
                "learners": [asdict(l) for l in self.learners], "spec": self.spec,
                "config": self.config()}

    def config(self) -> dict:
        return {"gain": float(self.params.gain), "weights": self.weights, "horizon": self.horizon,
                "frozen_product": self.frozen_product}

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".part")
        self.torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        s = self.torch.load(path, map_location=self.dev, weights_only=False)
        if s["spec"] != self.spec or s["learners"] != [asdict(l) for l in self.learners]:
            raise RuntimeError("checkpoint belongs to another run")
        if s.get("config", self.config()) != self.config():
            raise RuntimeError(f"checkpoint configuration {s.get('config')} != {self.config()}")
        self.brain.load_state_dict(s["brain"])
        self.k, self.ebar, self.ms, self.base = s["k"], s["ebar"], s["ms"], s["base"]
        self.diag, self.step_s, self.preds = s["diag"], s["step_s"], s["preds"]
        self.weight_digests = s["weight_digests"]


def run(spec: str, learners: list[Learner], data_dir: Path, out_dir: Path, stop_at: int | None = None,
        eval_cols: int = 128, banks_final=("final", "ood"), log=print, log_accuracy: bool = False,
        pair_u_to_n: bool = False, gamma: float = 1.0, horizon: int = 6000, checkpoints=CHECKPOINTS,
        weights: str = "raw", frozen_product: str = "csr", base_gain: float | None = None, slow: bool = False,
        stop_file: Path | None = None) -> dict:
    """Train every learner of one graph to `horizon` decisions (6,000) with the checkpoint evaluations; resumable.

    revision48: `stop_file`, when it exists at a checkpoint, saves and returns early (status "stopped")."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    checkpoints = tuple(int(c) for c in checkpoints)
    if checkpoints[0] != 0 or checkpoints[-1] != int(horizon):
        raise ValueError("checkpoints must start at 0 and end at the horizon")
    gr = GraphRun(spec, learners, data_dir, eval_cols=eval_cols, pair_u_to_n=pair_u_to_n, gamma=gamma,
                  horizon=horizon, weights=weights, frozen_product=frozen_product, base_gain=base_gain, slow=slow)
    ck = out_dir / "resume.pt"
    resumed_from = None
    if ck.exists():
        gr.load(ck)
        resumed_from = gr.k
        log(f"resumed {spec} at k={gr.k}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    eval_s = {}
    end = gr.horizon if stop_at is None else int(stop_at)
    stopped = False
    while True:
        if gr.k in checkpoints and f"progress_{gr.k}" not in gr.preds:
            t0 = time.time()
            gr.preds[f"progress_{gr.k}"] = gr.evaluate("progress")
            gr.weight_digests[str(gr.k)] = [m_digest(gr.brain.m[j]) for j in range(gr.B)]
            eval_s[str(gr.k)] = round(time.time() - t0, 1)
            gr.save(ck)
            msg = f"{spec} k={gr.k} eval {eval_s[str(gr.k)]}s"
            if log_accuracy:                       # development runs only; never in the main campaign
                acc = (gr.preds[f"progress_{gr.k}"] == gr.labels("progress")).mean(1)
                msg += " accuracy " + " ".join(f"{l.key}:{x:.3f}" for l, x in zip(learners, acc))
            log(msg)
            if stop_file is not None and Path(stop_file).exists() and gr.k < end:
                stopped = True
                log(f"{spec} stop file seen at k={gr.k}: saved, stopping")
                break
        if gr.k >= end:
            break
        gr.step()
    out = {"spec": spec, "k": gr.k, "complete": gr.k >= gr.horizon, "stopped": stopped, "resumed_from": resumed_from,
           "learners": [asdict(l) for l in learners], "graph_load_s": round(gr.load_s, 1),
           "train_step_s_median": float(np.median(gr.step_s[:gr.k])) if gr.k else None,
           "train_s_total": float(gr.step_s[:gr.k].sum()), "eval_s": eval_s,
           "weight_digests": gr.weight_digests,
           "data_sha256": gr.data_sha, "pair_u_to_n": pair_u_to_n, "gamma": float(gamma),
           "gain": float(gr.params.gain), "config": gr.config(), "checkpoints": list(checkpoints),
           "eta": gr.eta.cpu().tolist(), "tau": gr.tau.tolist(),
           "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}
    if gr.k >= gr.horizon:
        for bank in banks_final:
            t0 = time.time()
            gr.preds[bank] = gr.evaluate(bank)
            eval_s[bank] = round(time.time() - t0, 1)
        lab = {bank: gr.labels(bank) for bank in ("progress",) + tuple(banks_final)}
        np.savez(out_dir / "predictions.npz",
                 **{k: v for k, v in gr.preds.items()}, **{f"labels_{b}": v for b, v in lab.items()},
                 **{f"uid_{b}": np.stack([gr.data[(l.block, l.task)][f"{b}_uid"] for l in learners]) for b in lab})
        np.savez(out_dir / "diagnostics.npz", step_s=gr.step_s, **gr.diag)
        np.savez(out_dir / "final_weights.npz", m=gr.brain.m.cpu().numpy())
        if ck.exists():
            ck.unlink()
    out["wall_s"] = round(time.time() - t_all, 1)
    tmp = out_dir / "run.json.part"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, out_dir / "run.json")
    return out
