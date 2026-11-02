"""The refractory counter must not wrap: a long-silent neuron must still fire."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flymario.models.spiking_reservoir import LIFBrain, LIFParams


def test_long_silence_does_not_lock_a_neuron_out():
    # int16 wrapped after 32,768 decrements and blocked the neuron for ~32k
    # ticks: a full Mario episode is ~38,000 ticks, so this is reachable.
    W = sp.csc_matrix((2, 2), dtype=np.float32)
    b = LIFBrain(W, LIFParams(gain=1.0, noise_std=0.0, bias=0.0), normalize=False)
    silent = np.zeros(2, np.float32)
    for _ in range(33_000):
        b.step(silent)
    assert b.refrac.min() >= 0
    assert b.step(np.array([5.0, 5.0], np.float32)).all()


def test_refractory_period_still_holds_right_after_a_spike():
    W = sp.csc_matrix((1, 1), dtype=np.float32)
    p = LIFParams(gain=1.0, noise_std=0.0, bias=0.0)
    b = LIFBrain(W, p, normalize=False)
    big = np.array([5.0], np.float32)
    assert b.step(big)[0]                       # fires
    blocked = [b.step(big)[0] for _ in range(p.refractory_steps)]
    assert not any(blocked)                     # refractory, however strong the drive
    assert b.step(big)[0]                       # and free again afterwards
