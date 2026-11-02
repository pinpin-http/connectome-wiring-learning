"""The closed loop: pixels -> fly vision -> connectome -> descending neurons -> joypad.

Mario's controller is driven by simulated activity through the Drosophila connectome; with an
untrained readout it plays badly, on purpose. What it establishes is that the loop is closed:
d(motor activity) / d(visual input) != 0, which `flymario.telemetry.probe` measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flymario import config
from flymario.connectome import datasets, randomize
from flymario.policy.readout import FixedRandomProjection
from flymario.connectome.schema import ConnectomeGraph
from flymario.env.vision import RetinotopicVision, VisionParams
from flymario.models.spiking_reservoir import LIFBrain, LIFParams  # noqa: F401
from flymario.models.torch_lif import make_brain

#: Deliberately absent: the readout population is defined once, in the dataset
#: registry, and read from there.


#: From configs/base.yaml (`timing.ticks_per_decision`).
_TICKS_PER_DECISION = config.get("timing.ticks_per_decision")


@dataclass
class AgentConfig:
    topology: str = "connectome"       # connectome | degree_preserving
                                       # | shuffled_weights | shuffled_edges
    topology_seed: int = 0
    #: Neural ticks per policy decision, not per emulator frame: one decision spans
    #: `frame_skip` emulator frames and the network is stepped this many times for it.
    ticks_per_decision: int = _TICKS_PER_DECISION
    # Shorter than one decision, so a decision is not dominated by the previous stimulus (Gate D).
    trace_tau_ms: float = 5.0
    lif: LIFParams = field(default_factory=LIFParams)
    vision: VisionParams = field(default_factory=VisionParams)
    #: None = dynamics.backend from the config. Scripts that fork workers pass "cpu":
    #: initialising CUDA before a fork makes every GPU worker fail.
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.ticks_per_decision < 1:
            raise ValueError("ticks_per_decision must be positive")
        if self.trace_tau_ms <= 0:
            raise ValueError("trace_tau_ms must be positive")


class FlyAgent:
    """Frozen biological reservoir + a (optionally trained) readout."""

    def __init__(self, graph: ConnectomeGraph, cfg: AgentConfig | None = None, readout=None):
        self.cfg = cfg or AgentConfig()
        self.graph = graph

        # The null models return a whole ConnectomeGraph (signs and node metadata travel
        # with the edges), so the sensory and motor interfaces stay identical across conditions.
        self.wired = randomize.make(self.cfg.topology, graph, self.cfg.topology_seed)
        # Weight transform and backend are declared in configs/base.yaml
        # (dynamics.weight_transform, dynamics.backend; docs/PROTOCOL.md §7.4):
        # to_sparse() and make_brain() follow them when not told otherwise.
        self.brain = make_brain(self.wired.to_sparse(), self.cfg.lif,
                                backend=self.cfg.backend)

        self.vision = RetinotopicVision(graph, self.cfg.vision)
        name = "malecns" if "malecns" in graph.source else "flywire"
        self.dataset = name
        self.motor_idx = datasets.motor_index(graph, name)
        if len(self.motor_idx) == 0:
            raise ValueError("no motor neurons found; check node_class vocabulary")

        self.beta = float(np.exp(-self.cfg.lif.dt_ms / self.cfg.trace_tau_ms))
        self.trace = np.zeros(len(self.motor_idx), dtype=np.float32)
        self.readout = readout
        self.last_spikes: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return len(self.motor_idx)

    def reset(self) -> None:
        self.brain.reset()
        self.vision.reset()
        self.trace[:] = 0.0
        self.last_spikes = None

    def tick(self, current: np.ndarray) -> np.ndarray:
        """One neural tick of the loop: step the brain, integrate the readout trace.

        The single place the trace is updated: `features()` calls it `ticks_per_decision` times.
        """
        spikes = self.brain.step(current)
        # Integrate every neural tick. Sampling only the last spike vector
        # loses earlier responses and changes trace time units when the
        # number of ticks per policy decision is varied.
        self.trace *= self.beta
        self.trace += spikes[self.motor_idx]
        self.last_spikes = spikes
        return spikes

    def features(self, frame: np.ndarray) -> np.ndarray:
        """One frame of the sensorimotor loop, up to (not including) the decision."""
        current = self.vision.encode(frame)
        for _ in range(self.cfg.ticks_per_decision):
            self.tick(current)
        return self.trace

    def act(self, frame: np.ndarray) -> int:
        z = self.features(frame)
        if self.readout is None:
            raise RuntimeError("agent has no readout")
        return self.readout.predict(z)

    def fixed_readout(self, n_actions: int, seed: int) -> FixedRandomProjection:
        """The primary readout, §11.2 category `fixed`.

        One realisation per block, shared by every topology compared inside it;
        `policy.readout.block_seeds` supplies a fresh seed per block.
        """
        return FixedRandomProjection(self.feature_dim, n_actions, seed)

    def describe(self) -> dict:
        return {
            "source": self.graph.source,
            "topology": self.cfg.topology,
            "wired_as": self.wired.source,
            "neurons": self.brain.n,
            "edges": int(self.brain.W.nnz),
            "injected_neurons": self.vision.n_injected,
            "motor_neurons": len(self.motor_idx),
            "ticks_per_decision": self.cfg.ticks_per_decision,
            "readout": (self.readout.provenance()
                        if hasattr(self.readout, "provenance") else None),
        }


class RandomReadout:
    """**Floor control.** Ignores `z` entirely, so it measures what the
    environment alone yields. Never the primary condition: §11.2 declares
    `fixed`, and this reads nothing from the network."""

    CATEGORY = "floor_control"

    def __init__(self, n_actions: int, seed: int = 0, bias_right: bool = True):
        self.rng = np.random.default_rng(seed)
        self.n = n_actions
        p = np.ones(n_actions)
        if bias_right and n_actions > 2:
            p[2] = 4.0                      # RIGHT+B, so the baseline is not pure noise
        self.p = p / p.sum()

    def predict(self, z: np.ndarray) -> int:
        return int(self.rng.choice(self.n, p=self.p))

    def provenance(self) -> dict:
        return {"readout": "RandomReadout", "category": self.CATEGORY,
                "n_actions": self.n, "reads_network": False}


class MotorPopulationReadout:
    """**Demonstration only.** Untrained but connectome-driven; not a scientific condition.

    Motor neurons are split into one contiguous group per action and the most active group
    wins, scored against a slow running baseline (raw group means differ by a constant
    offset, so a plain argmax would lock onto one action).
    """

    CATEGORY = "demonstration"

    def __init__(self, n_features: int, n_actions: int, hold_frames: int = 4,
                 baseline_tau: float = 200.0):
        self.bounds = np.linspace(0, n_features, n_actions + 1).astype(int)
        self.n_actions = n_actions
        self.hold = hold_frames
        self.alpha = 1.0 / baseline_tau
        self.baseline = np.zeros(n_actions, dtype=np.float32)
        self._warm = 0
        self._last, self._count = 1, 0

    def _scores(self, z: np.ndarray) -> np.ndarray:
        return np.array([z[a:b].mean() if b > a else 0.0
                         for a, b in zip(self.bounds[:-1], self.bounds[1:])], dtype=np.float32)

    def predict(self, z: np.ndarray) -> int:
        s = self._scores(z)
        self.baseline += self.alpha * (s - self.baseline)
        self._warm += 1
        if self._count > 0:
            self._count -= 1
            return self._last
        if self._warm < 30:                 # let the baseline settle first
            self._last = 2                  # RIGHT+B while warming up
        else:
            dev = (s - self.baseline) / (np.abs(self.baseline) + 1e-3)
            self._last = int(dev.argmax())
        self._count = self.hold - 1
        return self._last
