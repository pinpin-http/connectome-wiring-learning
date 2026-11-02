"""Training/evaluation dynamics agree under explicit, identical random inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from flymario.models.spiking_reservoir import LIFParams
from flymario.training.bptt import SurrogateBPTT
from flymario.training.plastic import PlasticLIF

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/gates/gate_g_forward_parity.py"
_SPEC = importlib.util.spec_from_file_location("forward_parity", _SCRIPT)
parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(parity)


@pytest.fixture(autouse=True)
def one_torch_thread():
    """Thread launch overhead dominates these deliberately small networks."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _compare(weight_case="initial"):
    return parity.compare_forward(
        parity.small_setup(), params=LIFParams(gain=0.15, noise_std=0.08,
                                               bias=0.02, refractory_ms=6.66,
                                               v_reset=-0.1),
        seed_set=1, readout_seed=11, weight_case=weight_case, n_trials=8,
        warmup_ticks=30, ticks_per_decision=8, device="cpu", gpu_duty=1,
    )


@pytest.mark.parametrize("weight_case", ["initial", "bounded_modified"])
def test_hard_forward_matches_evaluation_on_every_tick(weight_case):
    result = _compare(weight_case)
    assert result["exact_forward_parity"]
    assert result["weights_unchanged"]
    assert result["decision_action_mismatches"] == 0
    assert result["max_membrane_abs_error"] < 1e-5
    assert sum(t["plastic_spikes"] for t in result["ticks"]) > 100
    assert {t["cue"] for t in result["ticks"]} == {0, 1}


def test_forward_diagnostic_detects_a_mismatched_simulator(monkeypatch):
    original = SurrogateBPTT._tick

    def wrong_drive(self, v, refrac, spikes, trace, current, xi, Wm):
        return original(self, v, refrac, spikes, trace, current + 2, xi, Wm)

    monkeypatch.setattr(SurrogateBPTT, "_tick", wrong_drive)
    result = _compare()
    assert not result["exact_forward_parity"]
    assert result["spike_mismatches"] > 0
    assert result["first_spike_divergence"]["tick"] == 0
    assert result["max_trace_abs_error"] > 0


def test_explicit_noise_matches_default_draws_and_does_not_advance_generator():
    S = parity.small_setup()
    params = LIFParams(gain=0.15, noise_std=0.08)
    automatic = PlasticLIF(S["W"], S["motor"], params, seeds=[23], device="cpu")
    explicit = PlasticLIF(S["W"], S["motor"], params, seeds=[23], device="cpu")
    generator = torch.Generator().manual_seed(23)
    initial_rng_state = explicit._gens[0].get_state().clone()
    current = torch.as_tensor(S["cues"][0])
    for _ in range(60):
        noise = torch.randn(64, generator=generator)[:, None] * params.noise_std
        assert torch.equal(automatic.step(current), explicit.step(current, noise=noise))
        assert torch.equal(automatic.v, explicit.v)
    assert torch.equal(explicit._gens[0].get_state(), initial_rng_state)


def test_no_update_bptt_carries_state_across_blocks():
    """Without an update, truncation boundaries must not reset forward time."""
    S = parity.small_setup()
    params = LIFParams(gain=0.015, noise_std=0.08)
    readout = np.random.default_rng(17).normal(0, 0.1, (2, len(S["motor"]))).astype(np.float32)
    split = SurrogateBPTT(S["W"], S["motor"], S["cues"], readout,
                          params, seed=29, device="cpu", block_len=1)
    whole = SurrogateBPTT(S["W"], S["motor"], S["cues"], readout,
                          params, seed=29, device="cpu", block_len=3)
    before = split.m.detach().clone()
    kwargs = {"cues": np.array([0, 1, 0]), "targets": np.array([0, 1, 0]),
              "n_trials": 3, "lr": 1e-4, "ticks_per_decision": 3, "warmup_ticks": 4,
              "gpu_duty": 1, "apply_update": False}
    split_result = split.fit(**kwargs)
    whole_result = whole.fit(**kwargs)
    assert np.array_equal(split_result["loss"], whole_result["loss"])
    assert np.array_equal(split_result["correct"], whole_result["correct"])
    assert all(torch.equal(a, b) for a, b in zip(split.state, whole.state))
    assert torch.equal(split.m, before)
    assert all(not s.requires_grad for s in split.state)
