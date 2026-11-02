#!/usr/bin/env python
"""Small, reproducible audits of Protocol III's learning and curve-fit claims.

These are analytic examples and small Bernoulli policies, NOT simulations of the
full LIF network. They test the claimed estimator identity and whether short
learning curves determine their asymptote. No connectome or emulator is loaded.

Run: .venv/bin/python scripts/gates/learning_rule_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("hebbian", "score_function")


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def exact_counterexample(weight: float = 0.0) -> dict:
    """Enumerate x,y: action-independent R=x, with its exact mean baseline."""
    hebbian = score_function = expected_reward = 0.0
    terms = []
    for x in (0, 1):
        probability = float(sigmoid(weight * x))
        for y in (0, 1):
            mass = 0.5 * (probability if y else 1.0 - probability)
            advantage = x - 0.5
            h = x * y * advantage
            s = x * (y - probability) * advantage
            hebbian += mass * h
            score_function += mass * s
            expected_reward += mass * x
            terms.append({"x": x, "y": y, "probability": mass,
                          "hebbian_contribution": mass * h,
                          "score_function_contribution": mass * s})
    return {
        "definition": "x~Bernoulli(0.5), y~Bernoulli(sigmoid(w*x)), R=x, b=0.5",
        "weight": weight,
        "expected_reward": expected_reward,
        "true_reward_gradient": 0.0,
        "expected_hebbian_update_before_learning_rate": hebbian,
        "expected_score_function_update_before_learning_rate": score_function,
        "enumerated_terms": terms,
        "interpretation": "A nonzero expected update when the true gradient is zero "
                          "disproves the general policy-gradient-estimator claim. "
                          "It does not imply that a Hebbian rule can never learn.",
    }


def distribution(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95))}


def bandit_audit(*, seeds: int = 64, episodes: int = 4000, seed: int = 20260921,
                 learning_rate: float = 0.05, baseline_alpha: float = 0.02) -> dict:
    """Paired random streams; baseline uses only rewards from preceding episodes.

    One-step episodes make E=e=pre*post in the protocol's proposed rule, so no
    eligibility-decay convention is needed. The comparator uses the exact
    Bernoulli log-probability derivative pre*(post-probability).
    """
    if seeds < 2 or episodes < 1:
        raise ValueError("need at least two seeds and one episode")
    results = {}
    for task_id, task in enumerate(("action_independent_reward", "context_matching")):
        dimensions = 1 if task_id == 0 else 2
        weights = np.zeros((2, seeds, dimensions))
        baseline = np.full((2, seeds), 0.5)
        generators = [np.random.default_rng(np.random.SeedSequence([seed, task_id, s]))
                      for s in range(seeds)]
        # Each run has independent contexts and uniforms; methods share each run's draws.
        contexts = np.stack([rng.integers(0, 2, size=episodes) for rng in generators])
        uniforms = np.stack([rng.random(episodes) for rng in generators])
        checkpoints = set(np.linspace(0, episodes, 41, dtype=int).tolist())
        curves = {method: [] for method in METHODS}
        clipping_count = np.zeros(2, dtype=int)

        def expected_performance():
            if task_id == 0:
                return np.full((2, seeds), 0.5)
            p0 = sigmoid(weights[:, :, 0])
            p1 = sigmoid(weights[:, :, 0] + weights[:, :, 1])
            return 0.5 * ((1.0 - p0) + p1)

        def record(n):
            performance = expected_performance()
            for method_id, method in enumerate(METHODS):
                curves[method].append({"episodes": n,
                                       "expected_reward": distribution(performance[method_id]),
                                       "weight_mean": weights[method_id].mean(axis=0).tolist()})

        record(0)
        for episode in range(episodes):
            x = contexts[:, episode]
            features = x[:, None] if task_id == 0 else np.column_stack((np.ones(seeds), x))
            probability = sigmoid(np.sum(weights * features[None, :, :], axis=-1))
            action = uniforms[None, :, episode] < probability
            reward = (np.broadcast_to(x, action.shape) if task_id == 0
                      else (action == x[None, :])).astype(float)
            advantage = reward - baseline
            eligibility = np.stack((features * action[0, :, None],
                                    features * (action[1] - probability[1])[:, None]))
            unbounded = weights + learning_rate * advantage[:, :, None] * eligibility
            clipping_count += np.any(np.abs(unbounded) > 8.0, axis=-1).sum(axis=-1)
            weights = np.clip(unbounded, -8.0, 8.0)
            baseline += baseline_alpha * advantage
            if episode + 1 in checkpoints:
                record(episode + 1)
        final_performance = expected_performance()
        result = {"feature_definition": "[x]" if task_id == 0 else "[1, x]",
                  "reward_definition": "R=x independent of action" if task_id == 0
                  else "R=1 if action equals context x, else 0",
                  "initial_expected_reward": 0.5,
                  "methods": {}}
        for method_id, method in enumerate(METHODS):
            result["methods"][method] = {
                "final_expected_reward": distribution(final_performance[method_id]),
                "final_weights_by_run": weights[method_id].tolist(),
                "final_expected_reward_by_run": final_performance[method_id].tolist(),
                "final_weight_mean": weights[method_id].mean(axis=0).tolist(),
                "clipped_run_updates": int(clipping_count[method_id]),
                "curve": curves[method],
            }
        result["paired_final_reward_difference_score_minus_hebbian"] = distribution(
            final_performance[1] - final_performance[0])
        results[task] = result
    return {"settings": {"runs": seeds, "episodes": episodes, "master_seed": seed,
                         "learning_rate": learning_rate, "baseline_alpha": baseline_alpha,
                         "baseline_initial": 0.5, "weight_clip": [-8, 8],
                         "evaluation": "exact expected reward enumerating both contexts",
                         "run_pairing": "same contexts and action uniforms across methods",
                         "hyperparameter_search": "none; identical fixed parameters"},
            "tasks": results,
            "scope": "Toy stochastic policies only. Neither success nor failure validates "
                     "the rule on the full LIF graph, Mario, or arbitrary plastic subsets."}


def curve(n: np.ndarray, asymptote: float, tau: float) -> np.ndarray:
    return asymptote * -np.expm1(-n / tau)


def curve_profile(n: np.ndarray, observations: np.ndarray, sigma: float,
                  *, a_points: int = 601, tau_points: int = 1801) -> dict:
    """Known-Gaussian-noise likelihood profiles; intervals are grid approximations.

    The 95% single-parameter profile threshold is chi-square(1)=3.84146.
    This is NOT a confidence procedure proposed for the actual Mario study.
    """
    tau_grid = np.geomspace(20.0, 100000.0, tau_points)
    a_grid = np.linspace(0.01, 1.0, a_points)
    basis = -np.expm1(-n[None, :] / tau_grid[:, None])
    cross = basis @ observations
    squares = np.sum(basis**2, axis=1)
    y_square = float(observations @ observations)
    best_a_per_tau = np.clip(cross / squares, 0.01, 1.0)
    sse_tau = y_square - 2 * best_a_per_tau * cross + best_a_per_tau**2 * squares
    best_idx = int(np.argmin(sse_tau))
    best_sse = float(sse_tau[best_idx])
    sse_a = np.min(y_square - 2 * a_grid[:, None] * cross[None, :]
                   + a_grid[:, None]**2 * squares[None, :], axis=1)
    threshold = best_sse + 3.841458820694124 * sigma**2
    accepted_a = a_grid[sse_a <= threshold]
    accepted_tau = tau_grid[sse_tau <= threshold]
    return {"A_hat": float(best_a_per_tau[best_idx]), "tau_hat": float(tau_grid[best_idx]),
            "sse": best_sse,
            "A_profile_95": [float(accepted_a[0]), float(accepted_a[-1])],
            "tau_profile_95": [float(accepted_tau[0]), float(accepted_tau[-1])],
            "A_profile_touches_upper_bound": bool(accepted_a[-1] == a_grid[-1]),
            "tau_profile_touches_upper_bound": bool(accepted_tau[-1] == tau_grid[-1])}


def asymptote_audit(*, seeds: int = 64, seed: int = 20260921) -> dict:
    true_a, true_tau, sigma = 0.8, 1000.0, 0.01
    horizons = {"short": np.linspace(20, 100, 20), "long": np.linspace(20, 5000, 20)}
    summaries = {}
    # Both horizons have identical observation counts and matched Gaussian errors.
    errors = np.stack([np.random.default_rng(np.random.SeedSequence([seed, 2, s]))
                       .normal(0, sigma, 20) for s in range(seeds)])
    for name, n in horizons.items():
        observations = curve(n, true_a, true_tau)[None, :] + errors
        fits = [curve_profile(n, y, sigma) for y in observations]
        a_width = np.array([f["A_profile_95"][1] - f["A_profile_95"][0] for f in fits])
        summaries[name] = {
            "checkpoint_episodes": n.tolist(), "fits_by_run": fits,
            "A_hat": distribution(np.array([f["A_hat"] for f in fits])),
            "tau_hat": distribution(np.array([f["tau_hat"] for f in fits])),
            "A_profile_95_width": distribution(a_width),
            "fraction_A_profiles_including_true_A": float(np.mean([
                f["A_profile_95"][0] <= true_a <= f["A_profile_95"][1] for f in fits])),
            "fraction_A_profiles_touching_upper_bound": float(np.mean([
                f["A_profile_touches_upper_bound"] for f in fits])),
            "representative_run_0_observations": observations[0].tolist(),
        }
    short_n = np.linspace(0, 100, 1001)
    candidate_difference = np.abs(curve(short_n, 0.8, 1000) - curve(short_n, 0.4, 500))
    return {"settings": {"runs": seeds, "master_seed": seed, "true_A": true_a,
                         "true_tau": true_tau, "known_Gaussian_sigma": sigma,
                         "checkpoints_per_horizon": 20, "A_bounds": [0.01, 1.0],
                         "tau_grid_bounds": [20.0, 100000.0],
                         "observation_model": "unclipped additive Gaussian measurement noise; "
                         "an illustrative statistical model, not simulated episode outcomes",
                         "interval_method": "single-parameter likelihood profiles, known sigma; "
                         "95% chi-square approximation on a fixed grid"},
            "analytic_pair": {"parameters": [{"A": 0.8, "tau": 1000},
                                                {"A": 0.4, "tau": 500}],
                              "same_initial_slope": 0.0008,
                              "maximum_prediction_difference_through_episode_100":
                              float(candidate_difference.max()),
                              "prediction_difference_at_episode_5000": float(
                                  curve(np.array(5000), .8, 1000)
                                  - curve(np.array(5000), .4, 500))},
            "horizons": summaries,
            "interpretation": "When n/tau is small, A*(1-exp(-n/tau)) is approximately "
            "(A/tau)*n. A good short-horizon fit need not identify A separately from tau. "
            "Even a well-identified fitted A describes this curve, not architectural expressivity."}


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/runs/learning_rule_audit.json")
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "metadata": {"started_utc": datetime.now(timezone.utc).isoformat(),
                     "hostname": platform.node(), "platform": platform.platform(),
                     "python": sys.version, "numpy": np.__version__,
                     "git_commit": git_output("rev-parse", "HEAD"),
                     "git_status_at_start": git_output("status", "--short"),
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "command": sys.argv},
        "exact_counterexample": exact_counterexample(),
        "bandits": bandit_audit(seeds=args.seeds, episodes=args.episodes, seed=args.seed),
        "asymptote": asymptote_audit(seeds=args.seeds, seed=args.seed),
    }
    report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Exact action-independent-reward counterexample:")
    print("  true gradient = 0; expected Hebbian update = 0.125; score-function update = 0")
    for task, result in report["bandits"]["tasks"].items():
        print(f"Bandit: {task} ({args.seeds} independent runs, {args.episodes} episodes)")
        for method, values in result["methods"].items():
            mean = values["final_expected_reward"]["mean"]
            print(f"  {method:16} expected reward={mean:.4f} "
                  f"mean weights={values['final_weight_mean']}")
    for horizon, values in report["asymptote"]["horizons"].items():
        width = values["A_profile_95_width"]["median"]
        print(f"Curve {horizon:5}: median 95% profile width for A={width:.4f}")
    print(f"Elapsed {report['metadata']['elapsed_seconds']:.2f}s; report: {args.output}")
    print("Scope: toy audits only; no full-graph LIF learning claim.")


if __name__ == "__main__":
    main()
