"""Gate G in the sweep regime: one fresh look per decision, learning at the motor synapses.

Each decision is one sweep (docs/GATE_G_PLAN.md): reset the network state (noise generators keep running), hold
the encoded frame for K = timing.ticks_per_decision ticks, count motor spikes c, score with FixedRandomProjection(c).

Rule `eprop_readout` on motor synapses j -> i (sign fixed, m_ij in [0, 2 m0_ij]): e_ij = sum_t psi_i eps_ij,
L_i = sum_a g_a R_a,i / tau, G_ij = sign_ij L_i e_ij, m_ij <- clip(m_ij + eta m0_ij G_ij / RMS(G), 0, 2 m0_ij).
g per reward mode: 'true' (onehot(a) - pi)(r - b_cue); 'shuffled' r drawn independently; 'supervised'
onehot(target) - pi; 'anti' the 'true' rule fed -r. Options (off by default): kappa, tau_adapt, block, delay, frames.
Every random stream is a pure function of (seed set, purpose), so a learner and its controls share them.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF

RIGHT, LEFT = ACTION_NAMES.index("RIGHT"), ACTION_NAMES.index("LEFT")
MASTER = 20260930
PURPOSES = {"explore": 1, "reference": 2, "confirm": 3, "test": 4, "phase2": 5}
CUE_BLOCK = 10                                   # cues balanced 5/5 within every 10 trials
REWARDS = ("true", "shuffled", "supervised", "anti")


@dataclass(frozen=True)
class SweepLearner:
    name: str
    seed_set: int
    mapping: int                 # 0: cue 0 -> RIGHT, cue 1 -> LEFT; 1: the reverse
    readout_seed: int
    eta: float = 0.0             # 0 = the frozen control
    reward: str = "true"
    kappa: float = 1.0           # 1 = raw eligibility; < 1 = centred, kappa x common mode kept


@dataclass(frozen=True)
class RuleConfig:
    tau: float = 2.0             # softmax temperature on s_R - s_L (spike-count units)
    tau_adapt: float = 0.0       # 0 = fixed; else tau_b = max(tau, tau_adapt * RMS(d_b))
    psi_width: float = 0.5       # surrogate width, membrane units (threshold = 1)
    baseline_rate: float = 0.1   # per-cue running mean of reward
    rms_decay: float = 0.95      # RMS normalisation of the raw update
    ebar_rate: float = 0.05      # running mean of the eligibility (kappa < 1)
    block: int = 1               # trials accumulated per update
    delay: int = 0               # decisions between an action and its reward
    trace_gamma: float = 0.0     # decay per decision of the cross-decision trace (delay > 0)
    epsilon: float = 0.0         # exploration floor: this share of training actions is uniform


def frame_draws(seed_set: int, purpose: str, n_trials: int) -> np.ndarray:
    """Uniforms that pick each trial's frame from a pool (separate stream, so the
    cue, noise and action streams are those of `streams`)."""
    ss = np.random.SeedSequence([MASTER, int(seed_set), PURPOSES[purpose], 99])
    return np.random.default_rng(ss).random(n_trials).astype(np.float64)


def explore_draws(seed_set: int, purpose: str, n_trials: int) -> np.ndarray:
    """Uniforms for the exploration floor (a separate stream: the other streams are
    those of runs without a floor)."""
    ss = np.random.SeedSequence([MASTER, int(seed_set), PURPOSES[purpose], 98])
    return np.random.default_rng(ss).random(n_trials).astype(np.float32)


def streams(seed_set: int, purpose: str, n_trials: int) -> dict:
    """Cue sequence, noise seed, shuffled rewards and sampling uniforms."""
    ss = np.random.SeedSequence([MASTER, int(seed_set), PURPOSES[purpose]])
    noise_ss, cue_ss, shuf_ss, act_ss = ss.spawn(4)
    rng = np.random.default_rng(cue_ss)
    blocks = -(-n_trials // CUE_BLOCK)
    cues = np.concatenate([rng.permutation([0] * (CUE_BLOCK // 2) + [1] * (CUE_BLOCK // 2))
                           for _ in range(blocks)])[:n_trials]
    return {"noise_seed": int(noise_ss.generate_state(1)[0] % (2**31 - 1)),
            "cues": cues.astype(np.int64),
            "shuffled": np.random.default_rng(shuf_ss).choice([-1.0, 1.0], n_trials).astype(np.float32),
            "uniform": np.random.default_rng(act_ss).random(n_trials).astype(np.float32)}


def reset_network(brain) -> None:
    """Rest state before a sweep. The noise generators keep running."""
    brain.v.zero_()
    brain.refrac.zero_()
    brain.spikes.zero_()
    brain.spikes_f.zero_()


def sweep_once(brain, params, cur, ticks: int, psi_width: float, observe: bool,
               record: list | None = None, record_col: int = 0):
    """One sweep of a PlasticLIF batch from rest under the held current `cur` (n, B).

    Returns the motor spike counts (M, B) and, when `observe`, the eligibility
    e_ij per plastic connection (P, B). Used by the two-cue task and the game.
    With `record` (a list), every tick's spike vector of brain `record_col` is appended.
    """
    import torch
    b, p = brain, params
    reset_network(b)
    c = torch.zeros((b.M, b.B), device=b.dev)
    e = eps = None
    if observe:
        eps = torch.zeros((b.P, b.B), device=b.dev)
        e = torch.zeros_like(eps)
    w = float(psi_width)
    for _ in range(ticks):
        spk = b.step(cur, observe=observe)
        if record is not None:
            record.append(spk[:, record_col].clone())
        if observe:
            pre, zm, vhat, sm, free = b.obs
            eps.mul_(p.decay).add_(pre)
            psi = free.float() / (1.0 + (vhat + zm - p.threshold).abs() / w) ** 2
            e.addcmul_(psi.index_select(0, b.post), eps)
            smf = sm.float()
            eps.mul_(1.0 - smf.index_select(0, b.post))
            c.add_(smf)
        else:
            c.add_(spk.index_select(0, b.motor).float())
    return c, e


class SweepTask:
    """A batch of learners on the two-cue task in the sweep regime."""

    def __init__(self, W, motor, cue_currents: np.ndarray, learners: list[SweepLearner],
                 purpose: str, n_trials: int, rule: RuleConfig | None = None,
                 ticks: int | None = None, device: str = "cuda", cache_key=None,
                 gain: float | None = None, noise_std: float | None = None,
                 frames: np.ndarray | None = None, frame_pool=None):
        import torch
        self.torch, self.dev = torch, torch.device(device)
        for l in learners:
            if l.reward not in REWARDS:
                raise ValueError(f"unknown reward mode {l.reward!r}")
        self.learners, self.purpose, self.n_trials = list(learners), purpose, int(n_trials)
        self.rule = rule or RuleConfig()
        self.B = len(learners)
        self.params = LIFParams(gain=float(config.get("dynamics.gain") if gain is None else gain),
                                noise_std=float(config.get("dynamics.noise_std")
                                                if noise_std is None else noise_std))
        self.K = int(config.get("timing.ticks_per_decision") if ticks is None else ticks)
        sets = sorted({l.seed_set for l in learners})
        self.streams = {s: streams(s, purpose, n_trials) for s in sets}
        seeds = [self.streams[l.seed_set]["noise_seed"] for l in learners]
        self.brain = PlasticLIF(W, motor, self.params, seeds=seeds, device=device,
                                cache_key=cache_key)
        self.M, self.P = self.brain.M, self.brain.P
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        per = lambda key: np.stack([self.streams[l.seed_set][key] for l in learners], 1)
        if frames is None:
            self.cue_cur = t(np.asarray(cue_currents, np.float32).T)         # (n, 2)
            self.F = 1
            self.fidx = None
        else:
            frames = np.asarray(frames, np.float32)                          # (2, F, n)
            self.F = frames.shape[1]
            self.cue_cur = t(frames.reshape(2 * self.F, -1).T)               # (n, 2F)
            pool = np.asarray(sorted(frame_pool) if frame_pool is not None else range(self.F))
            u = np.stack([frame_draws(l.seed_set, purpose, n_trials) for l in learners], 1)
            self.fidx = t(pool[np.minimum((u * len(pool)).astype(int), len(pool) - 1)], torch.long)
        self.frame_pool = None if frames is None else [int(x) for x in pool]
        self.cues = t(per("cues"), torch.long)                               # (T, B)
        self.shuf = t(per("shuffled"))
        self.unif = t(per("uniform"))
        self.unif_eps = t(np.stack([explore_draws(l.seed_set, purpose, n_trials) for l in learners], 1))
        self.eta = t([l.eta for l in learners])
        self.kappa = t([l.kappa for l in learners])
        self.mapping = t([l.mapping for l in learners], torch.long)
        self.is_shuf = t([l.reward == "shuffled" for l in learners], torch.bool)
        self.is_sup = t([l.reward == "supervised" for l in learners], torch.bool)
        self.is_anti = t([l.reward == "anti" for l in learners], torch.bool)
        rows = np.zeros((self.B, 2, self.M), np.float32)
        for b, l in enumerate(learners):
            Wr = np.asarray(FixedRandomProjection(self.M, len(ACTION_NAMES), l.readout_seed).W)
            rows[b] = Wr[[RIGHT, LEFT]]
        self.Rd = t(rows[:, 0] - rows[:, 1]).t().contiguous()               # (M, B): R_R - R_L
        self.centred = bool((self.kappa != 1.0).any())
        self.learning = bool((self.eta != 0).any())
        self.reset_learning_state()

    # -- state ---------------------------------------------------------------
    def reset_network(self) -> None:
        reset_network(self.brain)

    def reset_learning_state(self) -> None:
        torch, B, n = self.torch, self.B, self.n_trials
        self.trial = 0
        self.base = torch.zeros((B, 2), device=self.dev)
        self.ms = torch.zeros(B, device=self.dev)
        self.n_updates = 0
        self.d2 = torch.zeros(B, device=self.dev)
        self.n_d = 0
        self.ebar = None
        self.acc = None
        self.gbase = torch.zeros(B, device=self.dev)
        self.trace = None
        self.buffer = []                      # (G, correct, shuffled reward) per pending decision
        self.log = {k: np.zeros((n, B), np.float32) for k in
                    ("reward", "correct", "correct_greedy", "action_left", "d", "p_right",
                     "update_rms", "rate_motor")}          # update_rms: raw G before RMS scaling
        self.log["clipped"] = np.zeros((n, 2), np.int64)

    # -- one sweep -----------------------------------------------------------
    def sweep(self, cues, observe: bool):
        """One decision for every learner. Returns motor counts (M, B) and the
        eligibility (P, B) when `observe`."""
        return sweep_once(self.brain, self.params, self.cue_cur.index_select(1, cues), self.K,
                          self.rule.psi_width, observe)

    def _tau(self):
        torch = self.torch
        if self.rule.tau_adapt > 0 and self.n_d > 0:
            rms = torch.sqrt(self.d2 / (1 - 0.98 ** self.n_d))
            return torch.clamp(self.rule.tau_adapt * rms, min=self.rule.tau)
        return torch.full((self.B,), float(self.rule.tau), device=self.dev)

    def step_trial(self, learn: bool = True) -> dict:
        """Advance one trial of this task's stream."""
        torch, k = self.torch, self.trial
        cues = self.cues[k]
        observe = learn and self.learning
        cols = cues if self.fidx is None else cues * self.F + self.fidx[k]
        c, e = self.sweep(cols, observe)
        d = (self.Rd * c).sum(0)                                            # (B,) s_R - s_L
        tau = self._tau()
        pR = torch.sigmoid(d / tau)
        target_left = (cues ^ self.mapping) == 1
        a_left = self.unif[k] >= pR if learn else d < 0
        if learn and self.rule.epsilon > 0:
            # floor: with probability epsilon the action is uniform (u < eps/2 -> LEFT)
            ue = self.unif_eps[k]
            a_left = torch.where(ue < self.rule.epsilon, ue < self.rule.epsilon / 2, a_left)
        correct = a_left == target_left
        correct_greedy = (d < 0) == target_left
        out = {"d": d, "correct": correct, "correct_greedy": correct_greedy, "a_left": a_left}
        rms_b = torch.zeros(self.B, device=self.dev)
        clip = {"clipped_low": 0, "clipped_high": 0}
        if learn:
            self.n_d += 1
            self.d2.mul_(0.98).add_(0.02 * d * d)
            if self.rule.delay > 0:
                r, rms_b, clip = self._delayed(k, e, d, tau, pR, a_left, target_left, correct,
                                               observe)
                self.log["reward"][k] = r.cpu().numpy()
                return self._finish(k, out, c, d, pR, a_left, correct, correct_greedy,
                                    rms_b, clip)
            r = torch.where(self.is_shuf, self.shuf[k], correct.float() * 2 - 1)
            r = torch.where(self.is_anti, -r, r)
            ar = torch.arange(self.B, device=self.dev)
            bl = self.base[ar, cues]
            adv = r - bl
            self.base[ar, cues] = bl + (r - bl) * self.rule.baseline_rate
            if observe:
                chose_r = (~a_left).float()
                want_r = (~target_left).float()
                gR = torch.where(self.is_sup, want_r - pR, (chose_r - pR) * adv)  # (B,)
                L = self.Rd * (gR / tau)[None, :]                                  # (M, B)
                e = self._centre(e)
                G = e * L.index_select(0, self.brain.post) * self.brain.sign[:, None]
                rms_b, clip = self._apply(k, G)
            lg = self.log
            lg["reward"][k] = r.cpu().numpy()
        return self._finish(k, out, c, d, pR, a_left, correct, correct_greedy, rms_b, clip)

    def _centre(self, e):
        torch = self.torch
        if not self.centred:
            return e
        if self.ebar is None:
            self.ebar = e.clone()
        ec = (e - self.ebar) + self.kappa[None, :] * self.ebar
        self.ebar.mul_(1 - self.rule.ebar_rate).add_(e, alpha=self.rule.ebar_rate)
        return torch.where((self.kappa != 1.0)[None, :], ec, e)

    def _apply(self, k, G):
        """Accumulate G; every `block` trials, RMS-normalise and update the weights."""
        torch = self.torch
        rms_b = torch.zeros(self.B, device=self.dev)
        clip = {"clipped_low": 0, "clipped_high": 0}
        self.acc = G if self.acc is None else self.acc + G
        if (k + 1) % self.rule.block == 0:
            G, self.acc = self.acc, None
            self.n_updates += 1
            rd = self.rule.rms_decay
            sq = (G * G).mean(0)
            rms_b = torch.sqrt(sq) * (self.eta != 0)       # raw size, logged
            self.ms.mul_(rd).add_((1 - rd) * sq)
            msb = self.ms / (1 - rd ** self.n_updates)
            G = G / torch.sqrt(msb + 1e-30)[None, :]
            delta = (G * self.eta[None, :]).t() * self.brain.m0[None, :]
            clip = self.brain.update(delta)
        return rms_b, clip

    def _delayed(self, k, e, d, tau, pR, a_left, target_left, correct, observe):
        """Delay > 0: decision k's reward is applied at decision k + D."""
        torch, D = self.torch, int(self.rule.delay)
        rms_b = torch.zeros(self.B, device=self.dev)
        clip = {"clipped_low": 0, "clipped_high": 0}
        Gk = None
        if observe:
            chose_r, want_r = (~a_left).float(), (~target_left).float()
            g = torch.where(self.is_sup, want_r - pR, chose_r - pR)          # no reward yet
            L = self.Rd * (g / tau)[None, :]
            Gk = self._centre(e) * L.index_select(0, self.brain.post) * self.brain.sign[:, None]
            self.trace = Gk.clone() if self.trace is None else \
                self.trace.mul_(self.rule.trace_gamma).add_(Gk)
        self.buffer.append((Gk, correct.clone(), self.shuf[k].clone()))
        r = torch.zeros(self.B, device=self.dev)
        if len(self.buffer) > D:
            G_old, corr_old, shuf_old = self.buffer.pop(0)
            r = torch.where(self.is_shuf, shuf_old, corr_old.float() * 2 - 1)
            r = torch.where(self.is_anti, -r, r)
            adv = r - self.gbase
            self.gbase.add_((r - self.gbase) * self.rule.baseline_rate)
            if observe:
                upd = torch.where(self.is_sup[None, :], G_old, self.trace * adv[None, :])
                rms_b, clip = self._apply(k, upd)
        return r, rms_b, clip

    def _finish(self, k, out, c, d, pR, a_left, correct, correct_greedy, rms_b, clip):
        lg = self.log
        lg["correct"][k] = correct.cpu().numpy()
        lg["correct_greedy"][k] = correct_greedy.cpu().numpy()
        lg["action_left"][k] = a_left.cpu().numpy()
        lg["d"][k] = d.cpu().numpy()
        lg["p_right"][k] = pR.cpu().numpy()
        lg["update_rms"][k] = rms_b.cpu().numpy()
        lg["rate_motor"][k] = (c.mean(0) / self.K).cpu().numpy()
        lg["clipped"][k] = [clip["clipped_low"], clip["clipped_high"]]
        self.trial += 1
        return out

    def run(self, n_more: int, learn: bool = True, ctl=None, every: int = 10,
            on_trial=None) -> None:
        end = min(self.trial + int(n_more), self.n_trials)
        while self.trial < end:
            self.step_trial(learn=learn)
            if on_trial is not None:
                on_trial(self)
            if ctl is not None and (self.trial % every == 0 or self.trial == end):
                ctl.sub(self.trial, self.n_trials, "trial")

    # -- weights and checkpoints ----------------------------------------------
    def weights(self):
        return self.brain.m.clone()

    def set_weights(self, m) -> None:
        self.brain.set_weights(m.to(self.dev))

    def relative_rms(self):
        """RMS of (m - m0) / m0 per learner."""
        b = self.brain
        return ((b.m - b.m0[None, :]) / b.m0[None, :]).pow(2).mean(1).sqrt().cpu().numpy()

    def state_dict(self) -> dict:
        return {"trial": self.trial, "base": self.base.clone(), "ms": self.ms.clone(),
                "n_updates": self.n_updates, "d2": self.d2.clone(), "n_d": self.n_d,
                "ebar": None if self.ebar is None else self.ebar.clone(),
                "acc": None if self.acc is None else self.acc.clone(),
                "gbase": self.gbase.clone(),
                "trace": None if self.trace is None else self.trace.clone(),
                "buffer": [(None if g is None else g.clone(), c.clone(), r.clone())
                           for g, c, r in self.buffer],
                "brain": self.brain.state_dict(),
                "log": {k: v.copy() for k, v in self.log.items()},
                "learners": [asdict(l) for l in self.learners], "purpose": self.purpose,
                "n_trials": self.n_trials, "rule": asdict(self.rule), "K": self.K}

    def load_state_dict(self, s: dict) -> None:
        if (s["learners"] != [asdict(l) for l in self.learners] or s["purpose"] != self.purpose
                or s["rule"] != asdict(self.rule) or s["K"] != self.K):
            raise ValueError("checkpoint belongs to a different run")
        self.trial, self.n_updates, self.n_d = int(s["trial"]), int(s["n_updates"]), int(s["n_d"])
        self.base.copy_(s["base"])
        self.ms.copy_(s["ms"])
        self.d2.copy_(s["d2"])
        self.ebar = None if s["ebar"] is None else s["ebar"].to(self.dev).clone()
        self.acc = None if s["acc"] is None else s["acc"].to(self.dev).clone()
        self.gbase.copy_(s["gbase"])
        self.trace = None if s["trace"] is None else s["trace"].to(self.dev).clone()
        self.buffer = [(None if g is None else g.to(self.dev).clone(), c.to(self.dev).clone(),
                        r.to(self.dev).clone()) for g, c, r in s["buffer"]]
        self.brain.load_state_dict(s["brain"])
        for k, v in s["log"].items():
            self.log[k][:len(v)] = v

    def copy_learning_state(self, other: SweepTask) -> None:
        """Continue another task's learner: weights, baselines, RMS and centring
        statistics (used when the mapping reverses between two tasks)."""
        self.set_weights(other.weights())
        self.base.copy_(other.base)
        self.gbase.copy_(other.gbase)
        self.ms.copy_(other.ms)
        self.d2.copy_(other.d2)
        self.n_updates, self.n_d = other.n_updates, other.n_d
        self.ebar = None if other.ebar is None else other.ebar.clone()

    def save(self, path) -> None:
        """Atomic: a crash mid-write leaves the previous checkpoint intact."""
        tmp = str(path) + ".part"
        self.torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> None:
        self.load_state_dict(self.torch.load(path, map_location=self.dev, weights_only=False))


def evaluate(W, motor, cue_currents, learners, weights, n_test: int, cache_key=None,
             ticks: int | None = None, frames=None, frame_pool=None) -> dict:
    """Frozen-weight, greedy evaluation on the 'test' purpose streams.

    Returns per-learner accuracy, per-cue fraction of LEFT choices and mean
    score difference. `weights` is the (B, P) magnitude tensor to evaluate.
    """
    task = SweepTask(W, motor, cue_currents, learners, "test", n_test, ticks=ticks,
                     cache_key=cache_key, frames=frames, frame_pool=frame_pool)
    task.set_weights(weights)
    task.run(n_test, learn=False)
    cues = task.cues[:n_test].cpu().numpy()
    lg = task.log
    acc = lg["correct_greedy"][:n_test].mean(0)
    left = {c: np.array([lg["action_left"][:n_test][cues[:, b] == c, b].mean()
                         for b in range(task.B)]) for c in (0, 1)}
    dmean = {c: np.array([lg["d"][:n_test][cues[:, b] == c, b].mean() for b in range(task.B)])
             for c in (0, 1)}
    return {"accuracy": acc, "left_given_cue0": left[0], "left_given_cue1": left[1],
            "d_cue0": dmean[0], "d_cue1": dmean[1], "n_test": n_test}
