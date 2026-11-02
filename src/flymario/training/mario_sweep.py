"""Learning in the game, sweep regime: the Gate G learner (docs/GATE_G_PLAN.md) playing Super Mario Bros.

One decision: frame -> RetinotopicVision -> one 16-tick sweep from rest -> motor counts -> fixed readout's
8 action scores -> softmax(scores / tau), sampled -> emulator (4 frames). Learning: centred eligibility x
policy factor, accumulated in a trace E (decay gamma, reset per episode); m <- clip(m + eta m0 U / RMS(U),
0, 2 m0) with U = normalised advantage * E. Controls in the same batch: a frozen twin (eta 0) and a
shuffled-reward twin. Each learner has its own emulator subprocess (`mario.EnvPool`); RAM reaches only
reward, termination and metrics.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.policy.readout import FixedRandomProjection
from flymario.training.mario import EnvPool
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once


@dataclass(frozen=True)
class SweepMarioLearner:
    name: str
    readout_seed: int
    noise_seed: int
    eta: float = 0.0
    kappa: float = 0.2
    reward: str = "true"            # true | shuffled
    group: int = 0                  # twins share a group: same noise, same random draws
    # generalisation study: what each episode looks like for this learner
    levels: tuple = ()              # savestates drawn per episode; () = settings.state
    max_delay: int = 0              # NOOP decisions before control, drawn in [0, max_delay]
    jitter: bool = False            # per-episode palette: RGB channels permuted, gain 0.75-1.25
    # perception-or-credit study: "schematic" = the eye sees the RAM drawing (env/schematic.py)
    view: str = "pixels"


#: actions that press A (a jump when Mario stands on the ground)
JUMP_ACTIONS = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]

#: coin rate of the take-off teacher's twin (Stage C): the take-off label's rate in a
#: 300-decision smoke run before the declaration (0.463; PERCEPTION_CREDIT_STUDY §14)
TAKEOFF_P = 0.45
#: coin rate of the Stage D twin: take-off label v2's rate in a smoke run before the declaration
TAKEOFF2_P = 0.48


@dataclass(frozen=True)
class SweepMarioSettings:
    """Declared in docs/MARIO_PILOT_PLAN.md (amendment 2026-10-02) before the run."""
    state: str = "Level1-1"
    frame_skip: int = 4
    max_frames: int = 6000          # 100 s of game; 1-1 takes ~30 s at a run
    stall_decisions: int = 150      # 10 s of game without a new furthest point
    gamma: float = 0.8              # stage 1: learns a 2-decision delay
    reward_tau: float = 1000.0      # decisions, for r_bar and the RMS
    tau: float = 2.0                # softmax temperature floor (count units)
    tau_adapt: float = 0.0          # 0 = fixed tau; else max(tau, tau_adapt * spread)
    psi_width: float = 0.5
    rms_decay: float = 0.95
    ebar_rate: float = 0.05
    sample_seed: int = 20261002
    # --- v2 (MARIO_PILOT_PLAN amendment 2026-10-03); the defaults reproduce v1
    algo: str = "reinforce"         # reinforce (v1) | actor_critic (v2)
    epsilon: float = 0.0            # exploration floor: share of actions drawn uniformly
    gamma_rl: float = 0.98          # discount of future reward (actor_critic)
    lam: float = 0.9                # trace decay: actor and critic traces use gamma_rl * lam
    critic_lr: float = 0.05         # normalised LMS step of the critic
    reward_scale: float = 10.0      # learning signal = reward / scale (actor_critic)
    death_extra: float = 0.0        # added to the learning signal on death (not to the score)
    # --- v3 procedure (docs/RL_STUDY.md §3); the defaults reproduce v1/v2
    pg_factor: str = "pi"           # pi: onehot(a) - pi (v1/v2); behaviour: x (1 - eps) pi(a) / mu(a)
    bootstrap_truncation: bool = False   # time-limit / stall ends bootstrap on V(s_T) instead of 0
    # --- v4 (docs/RL_STUDY.md §8): cost per jump decision in the learning signal only
    jump_cost: float = 0.0
    # critic warm-up (docs/RL_STUDY.md §9): no actor update for this many
    # decisions of the run, so a warm-started policy is not moved by an untrained critic
    actor_warmup: int = 0
    # evaluation switch (docs/RL_STUDY.md §9.2): no weight update of any kind (actor and
    # label teaching); the critic still learns, which does not affect actions. Not part of a
    # run's identity: a checkpoint may be resumed frozen.
    frozen: bool = False
    # stable random streams (docs/RL_STUDY.md §11): episode draws and per-group uniforms come from
    # generators keyed by the learner's group, not its batch position, so twins share them and
    # adding a learner changes no one else's draws. False reproduces every earlier run.
    stable_streams: bool = False


def episode_rng(seed: int, position: int, group: int, stable: bool) -> np.random.Generator:
    """Episode-draw generator of one learner: keyed by batch position (earlier runs) or by
    group (`stable_streams`)."""
    return np.random.default_rng([int(seed), int(group) if stable else int(position), 77 if not stable else 78])


class GroupStreams:
    """One torch generator per learner group (`stable_streams`): `rand()` gives one uniform
    per group, in the order of `groups`, each from its own stream, so a group's draws do not
    depend on which other groups are in the batch."""

    def __init__(self, seed: int, groups, device):
        import torch
        self.groups = list(groups)
        self.gens = []
        for gr in self.groups:
            g = torch.Generator(device=device)
            g.manual_seed(int(np.random.SeedSequence([int(seed), int(gr), 91]).generate_state(1)[0]))
            self.gens.append(g)
        self.device = device

    def rand(self):
        import torch
        return torch.cat([torch.rand(1, generator=g, device=self.device) for g in self.gens])

    def get_state(self):
        return [g.get_state() for g in self.gens]

    def set_state(self, states) -> None:
        for g, st in zip(self.gens, states):
            g.set_state(st.cpu() if hasattr(st, "cpu") else st)


def pg_weight(pi, actions, epsilon: float, mode: str):
    """Per-learner weight of the actor's score factor onehot(a) - pi (docs/RL_STUDY.md §3).

    Actions are drawn from mu = (1 - eps) pi + eps / A. "pi" (v1/v2) returns 1 (no importance
    weight); "behaviour" returns (1 - eps) pi(a) / mu(a), making the factor the score of mu."""
    import torch
    if mode == "pi" or epsilon <= 0:
        return torch.ones(pi.shape[0], device=pi.device)
    if mode != "behaviour":
        raise ValueError(f"unknown pg_factor {mode!r}")
    pa = pi.gather(1, actions[:, None]).squeeze(1)
    return (1 - epsilon) * pa / ((1 - epsilon) * pa + epsilon / pi.shape[1])


def end_delta(r, V, truncated, gamma: float, bootstrap: bool):
    """TD error of a decision that ended its episode. Deaths and flags are terminal
    (delta = r - V(s)); a time-limit or stall end is not a terminal state, and with
    `bootstrap` its unseen next state's value is replaced by V(s) (docs/RL_STUDY.md §3)."""
    if not bootstrap:
        return r - V
    return r + gamma * V * truncated.float() - V


class MarioSweepTrainer:
    def __init__(self, g, W, motor, learners: list[SweepMarioLearner],
                 settings: SweepMarioSettings, device: str = "cuda", cache_key=None):
        import torch

        from flymario.env.vision import RetinotopicVision
        self.torch, self.dev = torch, torch.device(device)
        self.learners, self.s, self.B = list(learners), settings, len(learners)
        self.params = LIFParams(gain=float(config.get("dynamics.gain")),
                                noise_std=float(config.get("dynamics.noise_std")))
        self.K = int(config.get("timing.ticks_per_decision"))
        self.brain = PlasticLIF(W, motor, self.params, seeds=[l.noise_seed for l in learners],
                                device=device, cache_key=cache_key)
        self.M, self.P = self.brain.M, self.brain.P
        self.visions = [RetinotopicVision(g) for _ in learners]
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self.R = t(np.stack([np.asarray(FixedRandomProjection(self.M, len(ACTION_NAMES),
                                                              l.readout_seed).W) for l in learners]))
        self.eta = t([l.eta for l in learners])
        self.kappa = t([l.kappa for l in learners])
        self.is_shuf = t([l.reward == "shuffled" for l in learners], torch.bool)
        # reward "teacher": supervised by the RAM danger label (PERCEPTION_CREDIT_STUDY),
        # target = RIGHT+A+B if an on-screen enemy is 8-64 px ahead, else RIGHT+B
        self.is_teacher = t([l.reward in ("teacher", "teacher_shuffled", "teacher_full", "teacher_full_shuffled",
                                          "teacher_takeoff", "teacher_takeoff_shuffled",
                                          "teacher_takeoff2", "teacher_takeoff2_shuffled")
                             for l in learners], torch.bool)
        # "teacher_takeoff" (Stage C): jump only at take-off distance (schematic.takeoff_label);
        # "teacher_takeoff_shuffled": its coin twin at the label's rate (TAKEOFF_P)
        self.is_ttko = t([l.reward == "teacher_takeoff" for l in learners], torch.bool)
        # "teacher_takeoff2" (Stage D): the take-off label with ground threats only
        # (schematic.takeoff_label_v2); its twin's coin rate TAKEOFF2_P
        self.is_ttko2 = t([l.reward == "teacher_takeoff2" for l in learners], torch.bool)
        # "script_takeoff2" (docs/RL_STUDY.md §11.1): a scripted reference inside the same
        # harness -- presses RIGHT+A+B while the take-off label v2 is on, RIGHT+B otherwise;
        # never learns (its brain slot is computed and ignored)
        self.is_script = t([l.reward == "script_takeoff2" for l in learners], torch.bool)
        # "teacher_full" (Stage B): the label also covers walls and pits (schematic.jump_label);
        # "teacher_full_shuffled": its non-contingent twin, coin with p 0.35
        self.is_tfull = t([l.reward == "teacher_full" for l in learners], torch.bool)
        self.coin_p = t([{"teacher_full_shuffled": 0.35, "teacher_takeoff_shuffled": TAKEOFF_P,
                          "teacher_takeoff2_shuffled": TAKEOFF2_P}.get(l.reward, 0.25)
                         for l in learners])
        # "teacher_shuffled": same supervised rule, target jump drawn at random (p 0.25),
        # independent of the frame -- the negative reference of criterion D
        self.is_tshuf = t([l.reward in ("teacher_shuffled", "teacher_full_shuffled", "teacher_takeoff_shuffled",
                                        "teacher_takeoff2_shuffled")
                           for l in learners], torch.bool)
        groups = sorted({l.group for l in learners})
        self.gidx = t([groups.index(l.group) for l in learners], torch.long)
        self.n_groups = len(groups)
        self.learning = bool((self.eta != 0).any())
        #: brain whose sweep is recorded tick by tick for the live view (None = none)
        self.focus: int | None = None
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(settings.sample_seed))
        self.gstreams = GroupStreams(settings.sample_seed, groups, self.dev) if settings.stable_streams else None
        # per-learner episode draws (level, delay, palette): a pure function of the seed
        self.ep_rng = [episode_rng(settings.sample_seed, b, l.group, settings.stable_streams)
                       for b, l in enumerate(learners)]
        self.palette = [None] * self.B
        self.pool = EnvPool(self.B, {"state": settings.state, "frame_skip": settings.frame_skip,
                                     "max_frames": settings.max_frames})
        self.reset_state()

    # -- state ---------------------------------------------------------------
    def reset_state(self) -> None:
        torch, B = self.torch, self.B
        self.decisions = 0
        self.started = False
        self.E = torch.zeros((self.P, B), device=self.dev) if self.learning else None
        self.ebar = None
        self.ms = torch.zeros(B, device=self.dev)
        self.n_updates = 0
        self.rbar = torch.zeros(B, device=self.dev)
        self.rvar = torch.ones(B, device=self.dev)
        self.spread = torch.zeros(B, device=self.dev)
        # critic (actor_critic): V(s) = wv . phi(s) + bv, phi = motor counts / 4
        self.wv = torch.zeros((self.M, B), device=self.dev)
        self.bv = torch.zeros(B, device=self.dev)
        self.zv = torch.zeros((self.M, B), device=self.dev)
        self.zb = torch.zeros(B, device=self.dev)
        self.V_prev = torch.zeros(B, device=self.dev)
        self.r_prev = torch.zeros(B, device=self.dev)
        self.has_prev = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.dvar = torch.ones(B, device=self.dev)        # running mean square of delta
        self.phi2 = torch.ones(B, device=self.dev)        # running mean of |phi|^2
        self.nu = torch.zeros(B, device=self.dev)         # actor updates per learner
        self.ms_t = torch.zeros(B, device=self.dev)       # teacher: running mean square
        self.n_t = torch.zeros(B, device=self.dev)        # teacher: updates per learner
        self.last_V = torch.zeros(B, device=self.dev)
        self.last_delta = torch.zeros(B, device=self.dev)
        self.obs, self.info = [None] * B, [None] * B
        self.ep = [self._new_episode() for _ in range(B)]
        self.episodes: list[dict] = []
        self.stats = {"entropy": [], "update_rms": []}
        self.last: dict = {}

    @staticmethod
    def _new_episode() -> dict:
        return {"decisions": 0, "reward": 0.0, "best_x": 0.0, "best_progress": -1e18,
                "since_best": 0, "actions": [0] * len(ACTION_NAMES)}

    def _group_uniform(self):
        """One uniform per group, broadcast to its learners (common random numbers)."""
        torch = self.torch
        if self.gstreams is not None:
            return self.gstreams.rand().index_select(0, self.gidx)
        return torch.rand(self.n_groups, generator=self.gen, device=self.dev).index_select(0, self.gidx)

    def _uniform_b(self):
        """Per-learner uniforms for coin labels: per group under `stable_streams`."""
        if self.gstreams is not None:
            return self.gstreams.rand().index_select(0, self.gidx)
        return self.torch.rand(self.B, generator=self.gen, device=self.dev)

    def _restart(self, b: int) -> None:
        l, rng = self.learners[b], self.ep_rng[b]
        if l.levels or l.max_delay:
            state = str(rng.choice(list(l.levels))) if l.levels else self.s.state
            delay = int(rng.integers(0, l.max_delay + 1)) if l.max_delay else 0
            obs, info = self.pool.reset_to(b, state, delay)
            info = dict(info, level=state, delay=delay)
        else:
            obs, info = self.pool.reset(b)
            info = dict(info, level=self.s.state, delay=0)
        self.palette[b] = ((rng.permutation(3), float(rng.uniform(0.75, 1.25)))
                           if l.jitter else None)
        self.obs[b], self.info[b] = obs, info
        self.visions[b].reset()
        if self.E is not None:
            self.E[:, b] = 0.0
        self.zv[:, b] = 0.0
        self.zb[b] = 0.0
        self.has_prev[b] = False
        self.ep[b] = self._new_episode()
        self.ep[b]["best_progress"] = info.get("progress", 0.0)
        self.ep[b]["level"], self.ep[b]["delay"] = info["level"], info["delay"]
        self.ep[b]["best_x"] = info.get("mario_x", 0.0)

    def start(self) -> None:
        for b in range(self.B):
            self._restart(b)
        self.started = True

    def seen(self, b: int) -> np.ndarray:
        """The frame as this learner's eye receives it (palette jitter applied)."""
        if self.learners[b].view == "schematic":
            from flymario.env.schematic import schematic_frame
            return schematic_frame(self.info[b]["ram"], self.obs[b].shape)
        pal = self.palette[b]
        if pal is None:
            return self.obs[b]
        perm, gain = pal
        return np.clip(self.obs[b][..., perm].astype(np.float32) * gain, 0, 255).astype(np.uint8)

    # -- one decision ----------------------------------------------------------
    def _learn(self, delta, mask) -> None:
        """Actor-critic update for the learners in `mask` with TD error `delta`."""
        torch, s = self.torch, self.s
        m = mask.float()
        # critic: normalised LMS on the TD(lambda) trace
        step = s.critic_lr * delta * m / self.phi2
        self.wv.add_(self.zv * step[None, :])
        self.bv.add_(self.zb * step)
        if not self.learning or self.decisions < s.actor_warmup or s.frozen:
            return
        mask = mask & ~self.is_teacher                   # teachers learn from the label, not delta
        m = mask.float()
        # actor: the three-factor update, with delta normalised by its running RMS
        self.dvar = torch.where(mask, 0.99 * self.dvar + 0.01 * delta * delta, self.dvar)
        adv = delta / torch.sqrt(self.dvar + 1e-8) * m
        U = self.E * adv[None, :]
        sq = (U * U).mean(0)
        rd = s.rms_decay
        self.nu = self.nu + m
        self.ms = torch.where(mask, rd * self.ms + (1 - rd) * sq, self.ms)
        msb = self.ms / (1 - rd ** self.nu.clamp(min=1))
        U = U / torch.sqrt(msb + 1e-30)[None, :] * m[None, :]
        self.brain.update((U * self.eta[None, :]).t() * self.brain.m0[None, :])

    def _teach(self, ec, pi, tau) -> None:
        """Supervised three-factor update for the 'teacher' learners, the Gate G
        supervised form: G = e_centred * L(onehot(target) - pi), RMS-normalised,
        applied every decision. The target comes from RAM (training signal only;
        the learner's input stays its eye)."""
        torch = self.torch
        dx = torch.tensor([float(i.get("danger_dx", 999.0)) for i in self.info], device=self.dev)
        jump = (dx >= 8) & (dx <= 64)
        if bool(self.is_tfull.any()):
            from flymario.env.schematic import jump_label
            full = torch.tensor([bool(jump_label(i["ram"])) if "ram" in i else False for i in self.info],
                                device=self.dev)
            jump = torch.where(self.is_tfull, full, jump)
        if bool(self.is_ttko.any()):
            from flymario.env.schematic import takeoff_label
            tko = torch.tensor([bool(takeoff_label(i["ram"])) if "ram" in i else False for i in self.info],
                               device=self.dev)
            jump = torch.where(self.is_ttko, tko, jump)
        if bool(self.is_ttko2.any()):
            from flymario.env.schematic import takeoff_label_v2
            tko2 = torch.tensor([bool(takeoff_label_v2(i["ram"])) if "ram" in i else False for i in self.info],
                                device=self.dev)
            jump = torch.where(self.is_ttko2, tko2, jump)
        if bool(self.is_tshuf.any()):
            coin = self._uniform_b() < self.coin_p
            jump = torch.where(self.is_tshuf, coin, jump)
        target = torch.where(jump, ACTION_NAMES.index("RIGHT+A+B"), ACTION_NAMES.index("RIGHT+B"))
        oh = torch.nn.functional.one_hot(target, pi.shape[1]).float()
        L = torch.einsum("ba,bam->mb", oh - pi, self.R) / tau[None, :]
        U = ec * L.index_select(0, self.brain.post) * self.brain.sign[:, None] * self.is_teacher.float()[None, :]
        rd = self.s.rms_decay
        self.n_t += self.is_teacher.float()
        self.ms_t = torch.where(self.is_teacher, rd * self.ms_t + (1 - rd) * (U * U).mean(0), self.ms_t)
        msb = self.ms_t / (1 - rd ** self.n_t.clamp(min=1))
        U = U / torch.sqrt(msb + 1e-30)[None, :]
        self.brain.update((U * self.eta[None, :]).t() * self.brain.m0[None, :])

    def decide(self) -> list[dict]:
        """One decision for every learner; returns the episodes that ended."""
        torch, s = self.torch, self.s
        if not self.started:
            self.start()
        ac = s.algo == "actor_critic"
        cur = torch.as_tensor(np.stack([self.visions[b].encode(self.seen(b))
                                        for b in range(self.B)], axis=1), device=self.dev)
        rec = [] if self.focus is not None else None
        c, e = sweep_once(self.brain, self.params, cur, self.K, s.psi_width, self.learning,
                          record=rec, record_col=self.focus or 0)
        phi = c / 4.0
        V = (self.wv * phi).sum(0) + self.bv
        delta_shown = torch.zeros(self.B, device=self.dev)
        if ac and bool(self.has_prev.any()):
            # the previous decision's TD error, now that its next state is seen
            delta = (self.r_prev + s.gamma_rl * V - self.V_prev) * self.has_prev.float()
            self._learn(delta, self.has_prev)
            delta_shown = delta
            V = (self.wv * phi).sum(0) + self.bv
        scores = torch.einsum("bam,mb->ba", self.R, c)                      # (B, A)
        centred = scores - scores.mean(1, keepdim=True)
        self.spread.mul_(0.98).add_(0.02 * centred.pow(2).mean(1))
        tau = torch.full((self.B,), float(s.tau), device=self.dev)
        if s.tau_adapt > 0 and self.decisions > 0:
            rms = torch.sqrt(self.spread / (1 - 0.98 ** (self.decisions + 1)))
            tau = torch.clamp(s.tau_adapt * rms, min=s.tau)
        pi = torch.softmax(scores / tau[:, None], dim=1)
        nA = pi.shape[1]
        # common random numbers: twins draw the same uniforms, so they act alike
        # until their weights differ
        u = self._group_uniform()
        actions = torch.searchsorted(pi.cumsum(1), u[:, None]).squeeze(1).clamp(max=nA - 1)
        if s.epsilon > 0:
            ue = self._group_uniform()
            ur = self._group_uniform()
            actions = torch.where(ue < s.epsilon, (ur * nA).long().clamp(max=nA - 1), actions)
        if bool(self.is_script.any()):
            from flymario.env.schematic import takeoff_label_v2
            lab = torch.tensor([bool(takeoff_label_v2(i["ram"])) if "ram" in i else False for i in self.info],
                               device=self.dev)
            scripted = torch.where(lab, ACTION_NAMES.index("RIGHT+A+B"), ACTION_NAMES.index("RIGHT+B"))
            actions = torch.where(self.is_script, scripted, actions)
        out = self.pool.step(actions.cpu().numpy())
        rew = torch.tensor([o[1] for o in out], device=self.dev, dtype=torch.float32)
        died = torch.tensor([bool(o[4].get("died", False)) for o in out], device=self.dev)
        flip = torch.where(self._group_uniform() < 0.5, -1.0, 1.0)
        ent = -(pi * torch.log(pi + 1e-12)).sum(1)
        urms = torch.zeros(self.B, device=self.dev)
        G = None
        if self.learning:
            if self.ebar is None:
                self.ebar = e.clone()
            ec = torch.where((self.kappa != 1.0)[None, :],
                             (e - self.ebar) + self.kappa[None, :] * self.ebar, e)
            self.ebar.mul_(1 - s.ebar_rate).add_(e, alpha=s.ebar_rate)
            onehot = torch.nn.functional.one_hot(actions, nA).float()
            w = pg_weight(pi, actions, s.epsilon, s.pg_factor)
            L = torch.einsum("ba,bam->mb", (onehot - pi) * w[:, None], self.R) / tau[None, :]   # (M, B)
            G = ec * L.index_select(0, self.brain.post) * self.brain.sign[:, None]
            if bool(self.is_teacher.any()) and not s.frozen:
                self._teach(ec, pi, tau)
        # episode bookkeeping first: which learners just finished
        acts = actions.cpu().numpy()
        done = np.zeros(self.B, bool)
        trunc_np = np.zeros(self.B, bool)
        ends = [None] * self.B
        for b, (obs, rr, term, trunc, info) in enumerate(out):
            ep = self.ep[b]
            self.obs[b], self.info[b] = obs, info
            ep["decisions"] += 1
            ep["reward"] += rr
            ep["actions"][int(acts[b])] += 1
            if info.get("progress", 0.0) > ep["best_progress"]:
                ep["best_progress"], ep["since_best"] = info["progress"], 0
            else:
                ep["since_best"] += 1
            ep["best_x"] = max(ep["best_x"], info.get("mario_x", 0.0))
            stalled = ep["since_best"] >= s.stall_decisions
            if term or trunc or stalled:
                done[b] = True
                ends[b] = ("flag" if info.get("cleared") else "death" if info.get("died") else
                           "stall" if stalled else "time")
                trunc_np[b] = ends[b] in ("stall", "time")
        done_t = torch.as_tensor(done, device=self.dev)
        if ac:
            r_total = rew + s.death_extra * died.float()
            if s.jump_cost:
                jumped = torch.isin(actions, torch.tensor(JUMP_ACTIONS, device=self.dev)).float()
                r_total = r_total - s.jump_cost * jumped
            r_learn = torch.where(self.is_shuf, r_total * flip, r_total) / s.reward_scale
            if G is not None:
                self.E.mul_(s.gamma_rl * s.lam).add_(G)
            self.zv.mul_(s.gamma_rl * s.lam).add_(phi)
            self.zb.mul_(s.gamma_rl * s.lam).add_(1.0)
            self.phi2.mul_(0.99).add_(0.01 * (phi * phi).sum(0) + 0.01)
            if done.any():
                trunc_t = torch.as_tensor(trunc_np, device=self.dev)
                delta_T = end_delta(r_learn, V, trunc_t, s.gamma_rl, s.bootstrap_truncation) * done_t.float()
                self._learn(delta_T, done_t)
                delta_shown = torch.where(done_t, delta_T, delta_shown)
            self.V_prev = V.clone()
            self.r_prev = r_learn
            self.has_prev = ~done_t
        else:
            r = torch.where(self.is_shuf, rew * flip, rew)
            dev_ = r - self.rbar
            adv = dev_ / torch.sqrt(self.rvar + 1e-6) * (~self.is_teacher).float()
            self.rbar += dev_ / s.reward_tau
            self.rvar += (dev_ * dev_ - self.rvar) / s.reward_tau
            if G is not None:
                self.E.mul_(s.gamma).add_(G)
                U = self.E * adv[None, :]
                sq = (U * U).mean(0)
                urms = torch.sqrt(sq) * (self.eta != 0)
                self.n_updates += 1
                self.ms.mul_(s.rms_decay).add_((1 - s.rms_decay) * sq)
                msb = self.ms / (1 - s.rms_decay ** self.n_updates)
                U = U / torch.sqrt(msb + 1e-30)[None, :]
                self.brain.update((U * self.eta[None, :]).t() * self.brain.m0[None, :])
        if ac and G is not None:
            urms = torch.sqrt((self.E * self.E).mean(0)) * (self.eta != 0)
        self.last_V, self.last_delta = V, delta_shown
        self.last = {"pi": pi.cpu().numpy(), "scores": scores.cpu().numpy(),
                     "actions": actions.cpu().numpy(), "reward": rew.cpu().numpy(),
                     "tau": tau.cpu().numpy(), "entropy": ent.cpu().numpy(),
                     "value": (V * s.reward_scale).cpu().numpy(),
                     "td": (delta_shown * s.reward_scale).cpu().numpy()}
        if rec is not None:
            f = self.focus
            self.last.update(focus=f, spikes=torch.stack(rec).cpu().numpy(),
                             motor_counts=c[:, f].cpu().numpy(),
                             current=cur[:, f].cpu().numpy())
        self.stats["entropy"].append(ent.cpu().numpy().round(4).tolist())
        self.stats["update_rms"].append(urms.cpu().numpy().tolist())
        ended = []
        for b in np.flatnonzero(done):
            ep, info = self.ep[b], self.info[b]
            rec_ep = {"learner": self.learners[b].name, "b": int(b),
                      "episode": sum(1 for x in self.episodes if x["b"] == b),
                      "end": ends[b], "decisions": ep["decisions"], "reward": ep["reward"],
                      "best_x": ep["best_x"], "frames": info.get("frames", 0.0),
                      "actions": ep["actions"], "at_decision": self.decisions + 1,
                      "level": ep.get("level"), "delay": ep.get("delay")}
            self.episodes.append(rec_ep)
            ended.append(rec_ep)
            self._restart(int(b))
        self.decisions += 1
        return ended

    # -- weights and checkpoints ----------------------------------------------
    def relative_rms(self):
        b = self.brain
        return ((b.m - b.m0[None, :]) / b.m0[None, :]).pow(2).mean(1).sqrt().cpu().numpy()

    def state_dict(self) -> dict:
        return {"decisions": self.decisions, "started": self.started,
                "E": None if self.E is None else self.E.clone(),
                "ebar": None if self.ebar is None else self.ebar.clone(),
                "ms": self.ms.clone(), "n_updates": self.n_updates,
                "rbar": self.rbar.clone(), "rvar": self.rvar.clone(), "spread": self.spread.clone(),
                "critic": {k: getattr(self, k).clone() for k in
                           ("wv", "bv", "zv", "zb", "V_prev", "r_prev", "has_prev", "dvar", "phi2", "nu",
                            "ms_t", "n_t")},
                "gen": self.gen.get_state(), "brain": self.brain.state_dict(),
                "gstreams": self.gstreams.get_state() if self.gstreams is not None else None,
                "ep_rng": [r.bit_generator.state for r in self.ep_rng],
                "palette": [None if p is None else (p[0].tolist(), p[1]) for p in self.palette],
                "obs": [o.copy() for o in self.obs], "info": list(self.info),
                "vision_prev": [copy.deepcopy(v._prev) for v in self.visions],
                "ep": [dict(e, actions=list(e["actions"])) for e in self.ep],
                "episodes": list(self.episodes), "stats": copy.deepcopy(self.stats),
                "envs": self.pool.get_states(),
                "learners": [asdict(l) for l in self.learners], "settings": asdict(self.s)}

    @staticmethod
    def _same_learners(saved: list[dict], cur: list[dict]) -> bool:
        """Equal, allowing fields added since the checkpoint if they hold their default."""
        from dataclasses import MISSING, fields
        dflt = {f.name: f.default for f in fields(SweepMarioLearner) if f.default is not MISSING}
        return len(saved) == len(cur) and all(
            all(c.get(k) == v for k, v in s.items()) and all(c[k] == dflt.get(k) for k in set(c) - set(s))
            for s, c in zip(saved, cur))

    @staticmethod
    def _same_settings(saved: dict, cur: dict) -> bool:
        """Equal, allowing settings added since the checkpoint if they hold their default."""
        from dataclasses import MISSING, fields
        dflt = {f.name: f.default for f in fields(SweepMarioSettings) if f.default is not MISSING}
        saved = {k: v for k, v in saved.items() if k != "frozen"}
        cur = {k: v for k, v in cur.items() if k != "frozen"}
        return all(cur.get(k) == v for k, v in saved.items()) and \
            all(cur[k] == dflt.get(k) for k in set(cur) - set(saved))

    def load_state_dict(self, st: dict) -> None:
        if not self._same_learners(st["learners"], [asdict(l) for l in self.learners]) \
                or not self._same_settings(st["settings"], asdict(self.s)):
            raise ValueError("checkpoint belongs to a different run")
        self.decisions, self.started = int(st["decisions"]), bool(st["started"])
        if self.E is not None:
            self.E.copy_(st["E"])
        self.ebar = None if st["ebar"] is None else st["ebar"].to(self.dev).clone()
        self.ms.copy_(st["ms"])
        self.n_updates = int(st["n_updates"])
        self.rbar.copy_(st["rbar"])
        self.rvar.copy_(st["rvar"])
        self.spread.copy_(st["spread"])
        for k, v in st.get("critic", {}).items():
            getattr(self, k).copy_(v.to(self.dev))
        self.gen.set_state(st["gen"].cpu() if hasattr(st["gen"], "cpu") else st["gen"])
        if self.gstreams is not None and st.get("gstreams") is not None:
            self.gstreams.set_state(st["gstreams"])
        self.brain.load_state_dict(st["brain"])
        for r, rs in zip(self.ep_rng, st.get("ep_rng", [])):
            r.bit_generator.state = rs
        self.palette = [None if p is None else (np.asarray(p[0]), float(p[1]))
                        for p in st.get("palette", [None] * self.B)]
        self.obs, self.info = [o.copy() for o in st["obs"]], list(st["info"])
        for v, p in zip(self.visions, st["vision_prev"]):
            v._prev = copy.deepcopy(p)
        self.ep = [dict(e, actions=list(e["actions"])) for e in st["ep"]]
        self.episodes = list(st["episodes"])
        self.stats = copy.deepcopy(st["stats"])
        self.pool.set_states(st["envs"])

    def save(self, path) -> None:
        import os
        tmp = str(path) + ".part"
        self.torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> None:
        self.load_state_dict(self.torch.load(path, map_location=self.dev, weights_only=False))

    def close(self) -> None:
        self.pool.close()
