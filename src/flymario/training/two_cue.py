"""Gate G stage 1: two cues, two actions, reward ±1 (docs/GATE_G_PLAN.md §2).

Many learners run as one batch of brains (training/plastic.py). Each learner is
a brain plus what it does with reward:

    rule      node_perturbation | spike_reinforce | None (never learns)
    eta       learning rate; 0 = the "η = 0" control
    reward    "true" (±1 for the right action) | "shuffled" (±1 at random,
              independent of the action: the shuffled-reward control)
    mapping   0: cue 0 -> RIGHT, cue 1 -> LEFT;  1: the reverse
    readout   the fixed readout seed whose RIGHT and LEFT rows choose the action

Time runs as in play: one warm-up at the start with the first cue, then one
decision every `ticks_per_decision` ticks with the current cue held; the cue
changes between trials; nothing is reset between trials. The eligibility is
accumulated over each decision's ticks, then `Δm = η (R − R̄_cue) e` with a
per-learner, per-cue running baseline (τ = 50 trials), bounds [0, 2 m0].

`node_perturbation_full_trajectory` keeps its score across decisions in a
10-decision rollout. Weights stay fixed during that rollout; its accumulated
`Σ_k (R_k − R̄_k) S_k` update is applied once at the boundary. The brain state
continues across rollouts, while the score and pending update are cleared.

Every stream is a pure function of (seed set, purpose, trial index): the cue
sequence, the shuffled rewards, the noise seed. So a checkpoint only has to
hold the network state, the weights, the baselines, the rules' state and the
noise generators' states -- and a resumed run is bit-identical to an
uninterrupted one (checked on a small network by tests/unit/test_two_cue.py,
and on the connectome by scripts/gates/gate_g_task.py resume-check).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from flymario import config
from flymario.agent import AgentConfig
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF
from flymario.training.rules import RULES

RIGHT, LEFT = ACTION_NAMES.index("RIGHT"), ACTION_NAMES.index("LEFT")
MASTER = 20260923
PURPOSES = {"train": 1, "test": 2}
BASELINE_TAU = 50.0
CUE_BLOCK = 10                       # cues balanced 5/5 within every 10 trials


@dataclass(frozen=True)
class Learner:
    name: str
    seed_set: int
    rule: str | None = None
    eta: float = 0.0
    reward: str = "true"
    mapping: int = 0
    readout_seed: int | None = None


def streams(seed_set: int, purpose: str, n_trials: int) -> dict:
    """The cue sequence, shuffled rewards and noise seed of one seed set."""
    ss = np.random.SeedSequence([MASTER, int(seed_set), PURPOSES[purpose]])
    noise_ss, cue_ss, shuf_ss = ss.spawn(3)
    rng = np.random.default_rng(cue_ss)
    blocks = -(-n_trials // CUE_BLOCK)
    cues = np.concatenate([rng.permutation([0] * (CUE_BLOCK // 2) + [1] * (CUE_BLOCK // 2))
                           for _ in range(blocks)])[:n_trials]
    shuffled = np.random.default_rng(shuf_ss).choice([-1.0, 1.0], n_trials)
    return {"noise_seed": int(noise_ss.generate_state(1)[0] % (2**31 - 1)),
            "cues": cues.astype(np.int64), "shuffled": shuffled.astype(np.float32)}


class TwoCue:
    """A batch of learners on the two-cue task."""

    def __init__(self, W, motor, cue_currents: np.ndarray, learners: list[Learner],
                 purpose: str, n_trials: int, record: bool | list[int] = False,
                 device: str = "cuda",
                 cache_key=None, noise_std: float | None = None,
                 motor_noise_std: float | None = None, gain: float | None = None):
        import torch
        self.torch, self.dev = torch, torch.device(device)
        self.learners, self.purpose, self.n_trials = list(learners), purpose, int(n_trials)
        self.B = len(learners)
        # gain: None = the frozen operating point; tests on small nets set their own
        self.params = LIFParams(gain=float(config.get("dynamics.gain") if gain is None else gain),
                                noise_std=float(config.get("dynamics.noise_std")
                                                if noise_std is None else noise_std))
        self.tpd = int(config.get("timing.ticks_per_decision"))
        self.warm = int(config.get("dynamics.warmup_ticks"))
        self.beta = float(np.exp(-self.params.dt_ms / AgentConfig().trace_tau_ms))
        sets = sorted({l.seed_set for l in learners})
        self.streams = {s: streams(s, purpose, n_trials) for s in sets}
        seeds = [self.streams[l.seed_set]["noise_seed"] for l in learners]
        self.brain = PlasticLIF(W, motor, self.params, seeds=seeds, device=device,
                                cache_key=cache_key)
        if motor_noise_std is not None:
            raise NotImplementedError("the §4 fallback is declared but not yet needed")
        self.M, self.P = self.brain.M, self.brain.P
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)  # noqa: E731
        self.cue_cur = t(np.asarray(cue_currents, np.float32).T)          # (n, 2)
        self.cues = t(np.stack([self.streams[l.seed_set]["cues"] for l in learners]), torch.long)
        self.shuf = t(np.stack([self.streams[l.seed_set]["shuffled"] for l in learners]))
        self.eta = t([l.eta for l in learners])
        self.is_shuf = t([l.reward == "shuffled" for l in learners], torch.bool)
        self.mapping = t([l.mapping for l in learners], torch.long)
        rows = np.zeros((self.B, 2, self.M), np.float32)
        for b, l in enumerate(learners):
            if l.readout_seed is not None:
                Wr = np.asarray(FixedRandomProjection(self.M, len(ACTION_NAMES), l.readout_seed).W)
                rows[b] = Wr[[RIGHT, LEFT]]
        self.Rd = t(rows)
        rargs = (self.brain.post, self.brain.sign, self.params.gain, self.params.noise_std,
                 self.params.decay, self.params.threshold, self.B, self.dev)
        self.rules = {}
        for name, cls in RULES.items():
            use = [l.rule == name for l in learners]
            if any(use):
                self.rules[name] = (cls(*rargs), t(use, torch.bool))
        self.observe = bool(self.rules)
        # which brains keep their decision-time motor trace (the features a
        # supervised decoder or another readout can be evaluated on later)
        self.rec = (list(range(self.B)) if record is True else list(record) if record else [])
        self.reset_state()

    # -- state ---------------------------------------------------------------
    def reset_state(self) -> None:
        torch = self.torch
        self.trial = 0
        self.warm_done = False
        self.trace = torch.zeros((self.M, self.B), device=self.dev)
        self.Rbar = torch.zeros((self.B, 2), device=self.dev)
        for r, _ in self.rules.values():
            r.reset_state()
        self.pending = {name: torch.zeros_like(r.e) for name, (r, _) in self.rules.items()
                        if getattr(r, "block_update", False)}
        n = self.n_trials
        self.log = {k: np.zeros((n, self.B), np.float32) for k in
                    ("reward", "correct", "action", "margin", "update_rms",
                     "block_update_rms", "rate_motor")}
        self.log["clipped"] = np.zeros((n, 2), np.int64)
        self.features = np.zeros((n, self.M, len(self.rec)), np.float32) if self.rec else None

    def _current(self, k: int):
        return self.cue_cur.index_select(1, self.cues[:, k])              # (n, B)

    def _tick(self, cur):
        spk = self.brain.step(cur, observe=self.observe)
        if self.observe:
            for r, _ in self.rules.values():
                r.tick(*self.brain.obs)
        m = spk.index_select(0, self.brain.motor).float()
        self.trace.mul_(self.beta).add_(m)
        return m

    def _pace_gpu(self, clock: list[float], duty: float | None) -> None:
        if duty is None or self.dev.type != "cuda" or duty >= 1.0:
            return
        self.torch.cuda.synchronize(self.dev)
        elapsed = time.monotonic() - clock[0]
        idle = elapsed * (1.0 / duty - 1.0)
        if idle > 0.0:
            time.sleep(idle)
        clock[0] = time.monotonic()

    def run(self, n_more: int, ctl=None, every: int = 10,
            gpu_duty: float | None = None) -> None:
        """Advance `n_more` trials (learning where η > 0)."""
        torch = self.torch
        if gpu_duty is not None and not 0.0 < gpu_duty <= 1.0:
            raise ValueError("gpu_duty must be in (0, 1]")
        pace_clock = [time.monotonic()]
        if not self.warm_done:
            cur = self._current(0)
            for t in range(self.warm):
                self._tick(cur)
                self._pace_gpu(pace_clock, gpu_duty)
                if ctl is not None and t % 50 == 0:
                    ctl.sub(t, self.warm, "warm-up")
            self.warm_done = True
        end = min(self.trial + n_more, self.n_trials)
        ar = torch.arange(self.B, device=self.dev)
        while self.trial < end:
            k = self.trial
            cur = self._current(k)
            for r, _ in self.rules.values():
                r.reset_window()
            rate = torch.zeros(self.B, device=self.dev)
            for _ in range(self.tpd):
                rate += self._tick(cur).mean(0)
                self._pace_gpu(pace_clock, gpu_duty)
            scores = torch.einsum("bam,mb->ba", self.Rd, self.trace)
            action = (scores[:, 1] > scores[:, 0]).long()                  # argmax, ties -> RIGHT
            cue = self.cues[:, k]
            correct = action == (cue ^ self.mapping)
            R = torch.where(self.is_shuf, self.shuf[:, k], correct.float() * 2 - 1)
            base = self.Rbar[ar, cue]
            adv = R - base
            delta, rms = None, torch.zeros(self.B, device=self.dev)
            for r, use in self.rules.values():
                e = r.eligibility()                                          # (P, B)
                step = e * (adv * use)[None, :]                              # unscaled update
                rms += torch.sqrt((step * step).mean(0))
                if getattr(r, "block_update", False):
                    name = r.name
                    self.pending[name].add_(step)
                elif bool((self.eta * use != 0).any()):
                    d = (step * self.eta[None, :]).t()
                    delta = d if delta is None else delta + d

            rollout_end = ((k + 1) % CUE_BLOCK == 0 or (k + 1) == self.n_trials)
            block_rms = torch.zeros(self.B, device=self.dev)
            if rollout_end:
                for name, (r, use) in self.rules.items():
                    if not getattr(r, "block_update", False):
                        continue
                    block_step = self.pending[name]
                    block_rms += torch.sqrt((block_step * block_step).mean(0))
                    if bool((self.eta * use != 0).any()):
                        d = (block_step * self.eta[None, :]).t()
                        delta = d if delta is None else delta + d
                    block_step.zero_()
                    r.finish_rollout()
            clip = {"clipped_low": 0, "clipped_high": 0}
            if delta is not None:
                clip = self.brain.update(delta)
            self.Rbar[ar, cue] = base + (R - base) / BASELINE_TAU
            lg = self.log
            lg["reward"][k] = R.cpu().numpy()
            lg["correct"][k] = correct.cpu().numpy()
            lg["action"][k] = action.cpu().numpy()
            lg["margin"][k] = (scores[:, 0] - scores[:, 1]).cpu().numpy()
            lg["update_rms"][k] = rms.cpu().numpy()
            lg["block_update_rms"][k] = block_rms.cpu().numpy()
            lg["rate_motor"][k] = (rate / self.tpd).cpu().numpy()
            lg["clipped"][k] = [clip["clipped_low"], clip["clipped_high"]]
            if self.rec:
                self.features[k] = self.trace[:, self.rec].cpu().numpy()
            self.trial += 1
            self._pace_gpu(pace_clock, gpu_duty)
            if ctl is not None and (self.trial % every == 0 or self.trial == end):
                ctl.sub(self.trial, self.n_trials, "trial")

    # -- checkpoint ----------------------------------------------------------
    def state_dict(self) -> dict:
        rules = {}
        for name, (r, _) in self.rules.items():
            rules[name] = {"e": r.e.clone()}
            if hasattr(r, "x"):
                rules[name].update(x=r.x.clone(), spk_prev=r.spk_prev.clone())
        return {"trial": self.trial, "warm_done": self.warm_done, "trace": self.trace.clone(),
                "Rbar": self.Rbar.clone(), "brain": self.brain.state_dict(), "rules": rules,
                "pending": {k: v.clone() for k, v in self.pending.items()},
                "log": {k: v.copy() for k, v in self.log.items()},
                "features": None if self.features is None else self.features[:self.trial].copy(),
                "learners": [asdict(l) for l in self.learners], "purpose": self.purpose,
                "n_trials": self.n_trials}

    def load_state_dict(self, s: dict) -> None:
        if s["learners"] != [asdict(l) for l in self.learners] or s["purpose"] != self.purpose:
            raise ValueError("checkpoint belongs to a different run")
        self.trial, self.warm_done = int(s["trial"]), bool(s["warm_done"])
        self.trace.copy_(s["trace"])
        self.Rbar.copy_(s["Rbar"])
        self.brain.load_state_dict(s["brain"])
        for name, st in s["rules"].items():
            r = self.rules[name][0]
            r.e.copy_(st["e"])
            if "x" in st:
                r.x.copy_(st["x"])
                r.spk_prev.copy_(st["spk_prev"])
        for name, pending in s.get("pending", {}).items():
            self.pending[name].copy_(pending)
        for k, v in s["log"].items():
            self.log[k][:len(v)] = v
        if self.features is not None and s["features"] is not None:
            self.features[:len(s["features"])] = s["features"]

    def save(self, path) -> None:
        """Atomic: a crash mid-write leaves the previous checkpoint intact."""
        import os
        tmp = str(path) + ".part"
        self.torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> None:
        self.load_state_dict(self.torch.load(path, map_location=self.dev, weights_only=False))


def cue_currents(g, pair: list[str]) -> np.ndarray:
    """Each cue scene's first frame, encoded from a reset of the eye: the Gate E
    instrument's stimulus, which chose the pair (scripts/gates/gate_g_pair.py)."""
    from flymario.env import stimulus_bank
    from flymario.env.vision import RetinotopicVision
    bank = stimulus_bank.build(which="calibration")
    vision = RetinotopicVision(g)
    out = []
    for nm in pair:
        vision.reset()
        out.append(vision.encode(bank.sequences[nm][0]))
    return np.stack(out).astype(np.float32)
