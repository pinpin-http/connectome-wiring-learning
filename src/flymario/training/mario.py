"""Learning in the game: the Gate G rule, playing Super Mario Bros.

Loop (agent.py, batched): pixels -> RetinotopicVision -> connectome (16 ticks, motor inputs plastic) -> the fixed
readout's argmax over 8 actions -> emulator. Reward: retro_env.py's. Advantage per learner: (r - r̄) / running RMS,
both exponential averages over `reward_tau` decisions. Credit: an eligibility trace across decisions,
E ← γ E + e_decision, Δm = η · advantage · E, bounds [0, 2 m0]. At episode end the eye, the readout trace and E
restart, the brain does not. One emulator per subprocess (stable-retro); checkpoints hold their save states.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
from dataclasses import asdict, dataclass

import numpy as np

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.policy.readout import FixedRandomProjection
from flymario.training.plastic import PlasticLIF
from flymario.training.rules import NodePerturbation, SpikeReinforce

_INFO_KEYS = ("mario_x", "progress", "died", "cleared", "frames", "world", "area",
              "lives", "score", "time", "player_state", "enemy_dx", "enemy_type", "player_y",
              "enemy_sx", "enemy_sy", "mario_sx", "mario_sy", "danger_dx")


def _small(info: dict) -> dict:
    out = {k: (float(info[k]) if not isinstance(info[k], bool) else info[k])
           for k in _INFO_KEYS if k in info}
    if "ram" in info:                       # 2 KB, for learners whose eye sees the schematic
        out["ram"] = info["ram"]
    return out


def _worker(conn, cfg: dict) -> None:
    from flymario.env.retro_env import EnvConfig, MarioEnv
    env = MarioEnv(EnvConfig(**cfg))
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            obs, info = env.reset()
            conn.send((obs, _small(info)))
        elif cmd == "reset_to":
            # another level and/or a delayed start: Mario stands still (NOOP) for
            # n frames-steps while the enemies keep walking; or (play harness v2,
            # RL_STUDY §13) a prefix: a list of action indices played in order
            state, n_noop = arg
            if state != env.cfg.state:
                env.set_level(state)
            obs, info = env.reset()
            prefix = [0] * int(n_noop) if isinstance(n_noop, int) else [int(a) for a in n_noop]
            for a in prefix:
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    obs, info = env.reset()
                    break
            conn.send((obs, _small(info)))
        elif cmd == "step":
            obs, r, term, trunc, info = env.step(int(arg))
            conn.send((obs, float(r), bool(term), bool(trunc), _small(info)))
        elif cmd == "get_state":
            conn.send((env.env.em.get_state(), env._frames, env._best_progress, env._prev_progress))
        elif cmd == "set_state":
            # stable-retro refuses to step an env that was never reset, so a fresh
            # worker resets first, then loads the saved emulator state over it
            env.env.reset()
            state, frames, best, prev = arg
            env.env.em.set_state(state)
            env.env.data.update_ram()
            env._frames, env._best_progress, env._prev_progress = frames, best, prev
            conn.send(True)
        elif cmd == "close":
            env.close()
            conn.send(True)
            return


class EnvPool:
    """One Mario emulator per subprocess; steps them all at once."""

    def __init__(self, n: int, cfg: dict):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self, i: int):
        self.conns[i].send(("reset", None))
        return self.conns[i].recv()

    def reset_to(self, i: int, state: str, n_noop=0):
        """`n_noop`: NOOP decisions before control (int), or a prefix of action indices."""
        self.conns[i].send(("reset_to", (state, int(n_noop) if isinstance(n_noop, (int, np.integer)) else list(n_noop))))
        return self.conns[i].recv()

    def step(self, actions) -> list:
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        return [c.recv() for c in self.conns]

    def get_states(self) -> list:
        for c in self.conns:
            c.send(("get_state", None))
        return [c.recv() for c in self.conns]

    def set_states(self, states) -> None:
        for c, s in zip(self.conns, states):
            c.send(("set_state", s))
        for c in self.conns:
            c.recv()

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


@dataclass(frozen=True)
class MarioLearner:
    name: str
    readout_seed: int
    noise_seed: int
    rule: str | None = None
    eta: float = 0.0


@dataclass(frozen=True)
class MarioSettings:
    """Declared in docs/MARIO_PILOT_PLAN.md before the run."""
    state: str = "Level1-1"
    frame_skip: int = 4
    max_frames: int = 6000            # 100 s of game time; 1-1 takes ~30 s at a run
    stall_decisions: int = 150        # no new best progress for 10 s of game: time-out
    gamma: float = 0.9                # credit reaches ~10 decisions (0.7 s) back
    reward_tau: float = 1000.0        # decisions, for r̄ and the RMS


class MarioTrainer:
    def __init__(self, g, W, motor, learners: list[MarioLearner], settings: MarioSettings,
                 device: str = "cuda", cache_key=None):
        import torch

        from flymario.agent import AgentConfig
        from flymario.env.vision import RetinotopicVision
        self.torch, self.dev = torch, torch.device(device)
        self.learners, self.s, self.B = list(learners), settings, len(learners)
        self.params = LIFParams(gain=float(config.get("dynamics.gain")),
                                noise_std=float(config.get("dynamics.noise_std")))
        self.tpd = int(config.get("timing.ticks_per_decision"))
        self.warm = int(config.get("dynamics.warmup_ticks"))
        self.beta = float(np.exp(-self.params.dt_ms / AgentConfig().trace_tau_ms))
        self.brain = PlasticLIF(W, motor, self.params, seeds=[l.noise_seed for l in learners],
                                device=device, cache_key=cache_key)
        self.M, self.P = self.brain.M, self.brain.P
        self.visions = [RetinotopicVision(g) for _ in learners]
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self.Rd = t(np.stack([np.asarray(FixedRandomProjection(self.M, len(ACTION_NAMES),
                                                               l.readout_seed).W) for l in learners]))
        self.eta = t([l.eta for l in learners])
        rargs = (self.brain.post, self.brain.sign, self.params.gain, self.params.noise_std,
                 self.params.decay, self.params.threshold, self.B, self.dev)
        self.rules = {}
        for name, cls in (("node_perturbation", NodePerturbation), ("spike_reinforce", SpikeReinforce)):
            use = [l.rule == name for l in learners]
            if any(use):
                self.rules[name] = (cls(*rargs), t(use, torch.bool))
        self.observe = bool(self.rules)
        self.pool = EnvPool(self.B, {"state": settings.state, "frame_skip": settings.frame_skip,
                                     "max_frames": settings.max_frames})
        self.reset_state()

    # -- state ---------------------------------------------------------------
    def reset_state(self) -> None:
        torch = self.torch
        self.decisions = 0
        self.warm_done = False
        self.trace = torch.zeros((self.M, self.B), device=self.dev)
        self.E = torch.zeros((self.P, self.B), device=self.dev) if self.observe else None
        self.rbar = torch.zeros(self.B, device=self.dev)
        self.rvar = torch.ones(self.B, device=self.dev)
        self.obs, self.info = [None] * self.B, [None] * self.B
        self.ep = [self._new_episode() for _ in range(self.B)]
        self.episodes: list[dict] = []           # finished episodes, all learners
        for r, _ in self.rules.values():
            r.reset_state()

    @staticmethod
    def _new_episode() -> dict:
        return {"decisions": 0, "reward": 0.0, "best_x": 0.0, "best_progress": -1e18,
                "since_best": 0, "actions": [0] * len(ACTION_NAMES)}

    def _restart(self, b: int) -> None:
        obs, info = self.pool.reset(b)
        self.obs[b], self.info[b] = obs, info
        self.visions[b].reset()
        self.trace[:, b] = 0.0
        if self.E is not None:
            self.E[:, b] = 0.0
        self.ep[b] = self._new_episode()
        self.ep[b]["best_progress"] = info.get("progress", 0.0)
        self.ep[b]["best_x"] = info.get("mario_x", 0.0)

    def _currents(self):
        cur = np.stack([self.visions[b].encode(self.obs[b]) for b in range(self.B)], axis=1)
        return self.torch.as_tensor(cur, device=self.dev)

    def _tick(self, cur):
        spk = self.brain.step(cur, observe=self.observe)
        if self.observe:
            for r, _ in self.rules.values():
                r.tick(*self.brain.obs)
        self.trace.mul_(self.beta).add_(spk.index_select(0, self.brain.motor).float())

    # -- the loop --------------------------------------------------------------
    def start(self) -> None:
        """First episode of every learner, and the brain's warm-up on its first frame."""
        for b in range(self.B):
            self._restart(b)
        cur = self._currents()
        for _ in range(self.warm):
            self._tick(cur)
        self.trace.zero_()
        self.warm_done = True

    def decide(self) -> list[dict]:
        """One decision for every learner; returns the episodes that ended."""
        torch = self.torch
        if not self.warm_done:
            self.start()
        cur = self._currents()
        for r, _ in self.rules.values():
            r.reset_window()
        for _ in range(self.tpd):
            self._tick(cur)
        scores = torch.einsum("bam,mb->ba", self.Rd, self.trace)
        actions = scores.argmax(1).cpu().numpy()
        out = self.pool.step(actions)
        rew = torch.tensor([o[1] for o in out], device=self.dev, dtype=torch.float32)
        # advantage: centred and scaled by this learner's own running statistics
        tau = self.s.reward_tau
        adv = (rew - self.rbar) / torch.sqrt(self.rvar + 1e-6)
        dev_ = rew - self.rbar
        self.rbar += dev_ / tau
        self.rvar += (dev_ * dev_ - self.rvar) / tau
        if self.observe:
            self.E.mul_(self.s.gamma)
            for r, use in self.rules.values():
                self.E.add_(r.eligibility() * use[None, :].float())
            if bool((self.eta != 0).any()):
                self.brain.update((self.E * (self.eta * adv)[None, :]).t())
        ended = []
        for b, (obs, r, term, trunc, info) in enumerate(out):
            e = self.ep[b]
            self.obs[b], self.info[b] = obs, info
            e["decisions"] += 1
            e["reward"] += r
            e["actions"][int(actions[b])] += 1
            if info.get("progress", 0.0) > e["best_progress"]:
                e["best_progress"], e["since_best"] = info["progress"], 0
            else:
                e["since_best"] += 1
            e["best_x"] = max(e["best_x"], info.get("mario_x", 0.0))
            stalled = e["since_best"] >= self.s.stall_decisions
            if term or trunc or stalled:
                how = ("flag" if info.get("cleared") else "death" if info.get("died") else
                       "stall" if stalled else "time")
                rec = {"learner": self.learners[b].name, "b": b,
                       "episode": sum(1 for x in self.episodes if x["b"] == b),
                       "end": how, "decisions": e["decisions"], "reward": e["reward"],
                       "best_x": e["best_x"], "frames": info.get("frames", 0.0),
                       "actions": e["actions"], "at_decision": self.decisions + 1}
                self.episodes.append(rec)
                ended.append(rec)
                self._restart(b)
        self.decisions += 1
        return ended

    # -- checkpoint ----------------------------------------------------------
    def state_dict(self) -> dict:
        rules = {}
        for name, (r, _) in self.rules.items():
            rules[name] = {"e": r.e.clone()}
            if hasattr(r, "x"):
                rules[name].update(x=r.x.clone(), spk_prev=r.spk_prev.clone())
        return {"decisions": self.decisions, "warm_done": self.warm_done,
                "trace": self.trace.clone(), "E": None if self.E is None else self.E.clone(),
                "rbar": self.rbar.clone(), "rvar": self.rvar.clone(),
                "brain": self.brain.state_dict(), "rules": rules,
                "obs": [o.copy() for o in self.obs], "info": list(self.info),
                "vision_prev": [copy.deepcopy(v._prev) for v in self.visions],   # the eye's memory
                "ep": [dict(e, actions=list(e["actions"])) for e in self.ep],
                "episodes": list(self.episodes), "envs": self.pool.get_states(),
                "learners": [asdict(l) for l in self.learners], "settings": asdict(self.s)}

    def load_state_dict(self, s: dict) -> None:
        if s["learners"] != [asdict(l) for l in self.learners] or s["settings"] != asdict(self.s):
            raise ValueError("checkpoint belongs to a different run")
        self.decisions, self.warm_done = int(s["decisions"]), bool(s["warm_done"])
        self.trace.copy_(s["trace"])
        if self.E is not None:
            self.E.copy_(s["E"])
        self.rbar.copy_(s["rbar"])
        self.rvar.copy_(s["rvar"])
        self.brain.load_state_dict(s["brain"])
        for name, st in s["rules"].items():
            r = self.rules[name][0]
            r.e.copy_(st["e"])
            if "x" in st:
                r.x.copy_(st["x"])
                r.spk_prev.copy_(st["spk_prev"])
        self.obs, self.info = [o.copy() for o in s["obs"]], list(s["info"])
        for v, p in zip(self.visions, s["vision_prev"]):
            v._prev = copy.deepcopy(p)
        self.ep = [dict(e, actions=list(e["actions"])) for e in s["ep"]]
        self.episodes = list(s["episodes"])
        self.pool.set_states(s["envs"])

    def save(self, path) -> None:
        import os
        tmp = str(path) + ".part"
        self.torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> None:
        self.load_state_dict(self.torch.load(path, map_location=self.dev, weights_only=False))

    def close(self) -> None:
        self.pool.close()
