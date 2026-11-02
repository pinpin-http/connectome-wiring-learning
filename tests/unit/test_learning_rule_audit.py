"""Check analytical identities behind the small learning-rule audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/gates/learning_rule_audit.py"
SPEC = importlib.util.spec_from_file_location("learning_rule_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("weight", [-4.0, 0.0, 4.0])
def test_action_independent_reward_has_zero_true_and_score_gradient(weight):
    result = audit.exact_counterexample(weight)
    assert result["expected_reward"] == pytest.approx(0.5)
    assert result["true_reward_gradient"] == 0.0
    assert result["expected_score_function_update_before_learning_rate"] == pytest.approx(0)
    assert result["expected_hebbian_update_before_learning_rate"] > 0


def test_claimed_hebbian_estimator_has_exact_nonzero_bias():
    assert audit.exact_counterexample()[
        "expected_hebbian_update_before_learning_rate"] == pytest.approx(1 / 8)


def test_bernoulli_score_derivative_matches_finite_difference():
    # Expected reward for y=x with phi=[1,x], enumerating contexts and actions.
    theta = np.array([-0.3, 0.8])

    def reward(parameters):
        return 0.5 * (1 - audit.sigmoid(parameters[0])
                      + audit.sigmoid(parameters.sum()))

    analytic = np.zeros(2)
    for x in (0, 1):
        phi = np.array([1.0, x])
        probability = float(audit.sigmoid(theta @ phi))
        for y in (0, 1):
            mass = .5 * (probability if y else 1 - probability)
            analytic += mass * phi * (y - probability) * ((y == x) - 0.37)
    epsilon = 1e-6
    numeric = np.array([(reward(theta + epsilon * np.eye(2)[j])
                         - reward(theta - epsilon * np.eye(2)[j])) / (2 * epsilon)
                        for j in range(2)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)


def test_long_noiseless_curve_recovers_parameters():
    n = np.linspace(20, 5000, 20)
    result = audit.curve_profile(n, audit.curve(n, .8, 1000), .01)
    assert result["A_hat"] == pytest.approx(.8, abs=.002)
    assert result["tau_hat"] == pytest.approx(1000, rel=.01)
    assert result["A_profile_95"][0] <= .8 <= result["A_profile_95"][1]


def test_short_curve_cannot_separate_two_asymptotes_at_this_noise_scale():
    short = np.linspace(0, 100, 100)
    difference = np.max(np.abs(audit.curve(short, .8, 1000) - audit.curve(short, .4, 500)))
    assert difference < .004
    assert audit.curve(np.array(5000), .8, 1000) - audit.curve(
        np.array(5000), .4, 500) > .39
