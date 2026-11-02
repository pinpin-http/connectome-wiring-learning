"""A policy-step boundary must not change the neural trace's time units."""
from types import SimpleNamespace

import numpy as np
import pytest

from flymario.agent import AgentConfig, FlyAgent


def agent_with_spikes(ticks, spikes):
    agent = FlyAgent.__new__(FlyAgent)
    agent.cfg = SimpleNamespace(ticks_per_decision=ticks)
    sequence = iter(np.asarray(spikes, dtype=bool))
    agent.brain = SimpleNamespace(step=lambda current: next(sequence))
    agent.vision = SimpleNamespace(encode=lambda frame: np.zeros(2))
    agent.motor_idx = np.array([0, 1])
    agent.beta = 0.5
    agent.trace = np.zeros(2, dtype=np.float32)
    agent.last_spikes = None
    return agent


def test_trace_preserves_spikes_before_last_tick():
    agent = agent_with_spikes(3, [[1, 0], [0, 1], [0, 0]])
    np.testing.assert_allclose(agent.features(None), [0.25, 0.5])
    np.testing.assert_array_equal(agent.last_spikes, [False, False])


def test_trace_is_invariant_to_policy_step_grouping():
    spikes = [[1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0]]
    grouped = agent_with_spikes(3, spikes)
    individual = agent_with_spikes(1, spikes)
    for _ in range(2):
        grouped.features(None)
    for _ in range(6):
        individual.features(None)
    np.testing.assert_allclose(grouped.trace, individual.trace)


@pytest.mark.parametrize("kwargs", [{"ticks_per_decision": 0}, {"trace_tau_ms": 0}])
def test_invalid_clock_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AgentConfig(**kwargs)
