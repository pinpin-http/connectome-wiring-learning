#!/usr/bin/env python
"""Gate G §6.1 — does each learning rule measure the slope it claims to?

Checks, on a small LIF network, that each rule's estimate d·ĝ matches the true slope along d, a central
finite difference (J(m + δd) − J(m − δd)) / 2δ on common noise; estimates use independent draws. 20
prespecified random directions calibrate each estimator; each rule's own direction is validated separately.
References: FD vs independent FD (must pass); random vector and pre x post x reward (must fail).
Settings frozen in docs/GATE_G_PLAN.md §9; the simulator is checked against LIFBrain first (`selftest`).

Run:  .venv/bin/python scripts/gates/gate_g_estimator_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.models.spiking_reservoir import LIFBrain, LIFParams  # noqa: E402
from flymario.policy.readout import FixedRandomProjection  # noqa: E402
from flymario.runctl import RunControl, StopRequested  # noqa: E402
from flymario.run_manifest import RunManifest  # noqa: E402
from flymario.training.rules import (Hebbian, NodePerturbation,
                                     NodePerturbationFullTrajectory,
                                     SpikeReinforce)  # noqa: E402

# Versioned paths keep the completed v2 verdict byte-for-byte intact.
OUT = ROOT / "artifacts" / "runs" / "gate_g_estimator_check_v3.json"
POWER = ROOT / "artifacts" / "runs" / "gate_g_estimator_power_v3.json"
VERDICT = ROOT / "artifacts" / "runs" / "gate_g_estimator_verdict_v3.json"
V2_OUT = ROOT / "artifacts" / "runs" / "gate_g_estimator_check_v2.json"


def estimator_manifest(phase: str, args, seeds: dict, inputs: dict,
                       outputs: dict) -> RunManifest:
    return RunManifest(
        run_id=f"gate_g_estimator_v3_{phase}", study="Gate G",
        phase=f"estimator {phase}", configuration={
            "arguments": vars(args),
            "execution_device": args.device,
            "fixture_topology": {"neurons": N, "sensory": N_SENS, "motor": N_MOTOR,
                                 "connection_probability": P_CONN, "excitatory_fraction": EXC,
                                 "plastic_subset": "all edges into motor neurons",
                                 "feedback_condition": "both retained and removed"},
            "task": {"warmup_ticks": WARM, "rewarded_decisions": K,
                     "ticks_per_decision": T, "target_rate": RATE_TARGET},
            "rules": ["node_perturbation", "node_perturbation_full_trajectory",
                      "spike_reinforce", "pre_post_reward"]},
        seeds=seeds, dataset="synthetic 300-neuron LIF fixture",
        input_files={"base_config": ROOT / "configs" / "base.yaml",
                     "gate_plan": ROOT / "docs" / "GATE_G_PLAN.md",
                     "estimator_script": Path(__file__),
                     "learning_rules": ROOT / "src/flymario/training/rules.py", **inputs},
        output_files=outputs,
    )


def estimator_seeds(args, power: bool = False) -> dict:
    offset = 10 if power else 0
    return {
        "network": args.net_seed,
        "readout": args.net_seed + 1000,
        "calibration": 11,
        "baseline_batches": [71, 72],
        "direction_rng": args.net_seed + 77,
        "random_vector_rng": args.net_seed + 78,
        "estimate_batch_seeds": [73 + offset, 74 + offset],
        "finite_difference_sets": {"set_1": "100+i", "set_2": "200+i"},
        "own_direction_fd_seed": 310 if power else 300,
        "independent_noise_ceiling_fd_seed": 410 if power else 400,
        "episode_seed_formula": "batch_seed * 1000 + chunk_index",
    }

# -- the small network, declared --------------------------------------------
N, N_SENS, N_MOTOR = 300, 40, 40
P_CONN, EXC = 0.10, 0.70          # connection probability; excitatory fraction
WARM, K, T = 30, 3, 16            # warm-up ticks, rewarded trials, ticks per decision
TRACE_TAU_MS = 5.0                # agent.py: the readout trace
RATE_TARGET = 0.07                # configs/base.yaml target_rate


class SmallNet:
    """Random signed network with integer synapse counts, as the connectome has.

    `feedback=False` removes every connection leaving a motor neuron (the easiest
    case for every rule); `feedback=True` keeps them, as in the connectome.
    """

    def __init__(self, seed: int, feedback: bool):
        rng = np.random.default_rng(seed)
        order = rng.permutation(N)
        self.sens, self.motor = np.sort(order[:N_SENS]), np.sort(order[N_SENS:N_SENS + N_MOTOR])
        A = rng.random((N, N)) < P_CONN
        np.fill_diagonal(A, False)
        counts = 1.0 + np.floor(rng.lognormal(0.0, 1.0, (N, N)))       # median 2, heavy tail
        sign = np.where(rng.random(N) < EXC, 1.0, -1.0)                # Dale: per presynaptic
        W = A * counts * sign[None, :]                                  # W[i, j]: j -> i
        if not feedback:
            W[:, self.motor] = 0.0
        self.feedback = feedback
        rows = np.zeros(N, bool)
        rows[self.motor] = True
        ii, jj = np.nonzero(W * rows[:, None])                          # plastic: onto motor
        self.post_global, self.pre = ii, jj
        local = -np.ones(N, int)
        local[self.motor] = np.arange(N_MOTOR)
        self.post = local[ii]
        self.sign = np.sign(W[ii, jj])
        self.m0 = np.abs(W[ii, jj])
        self.W_fixed = W.copy()
        self.W_fixed[ii, jj] = 0.0
        self.W = W
        # two cues: random positive currents onto the sensory neurons
        self.cues = np.zeros((2, N), np.float32)
        self.cues[:, self.sens] = rng.uniform(0.0, 0.3, (2, N_SENS))
        self.readout = FixedRandomProjection(N_MOTOR, 2, seed=seed + 1000)
        self.target = np.array([0, 1])                                  # cue c -> action c

    @property
    def n_plastic(self) -> int:
        return len(self.pre)

    def describe(self) -> dict:
        return {"neurons": N, "sensory": N_SENS, "motor": N_MOTOR,
                "connections": int((self.W != 0).sum()), "plastic": self.n_plastic,
                "feedback": self.feedback,
                "motor_outgoing": int((self.W[:, self.motor] != 0).sum())}


class Sim:
    """Batched simulation of the small net on the GPU; one column per noise draw."""

    def __init__(self, net: SmallNet, gain: float, sigma: float, device="cuda", gpu_duty: float = 1.0):
        import torch
        self.torch, self.net, self.dev = torch, net, torch.device(device)
        self.gpu_duty = float(gpu_duty)
        self.p = LIFParams(gain=gain, noise_std=sigma)
        self.gain, self.sigma = float(gain), float(sigma)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)  # noqa: E731
        self.W_fixed = t(net.W_fixed)
        self.rows, self.cols = t(net.post_global, torch.long), t(net.pre, torch.long)
        self.sign = t(net.sign)
        self.pre, self.post = self.cols, t(net.post, torch.long)
        self.motor = t(net.motor, torch.long)
        self.cues = t(net.cues).t().contiguous()                       # (N, 2)
        self.Rd = t(np.asarray(net.readout.W))                          # (2, M)
        self.target = t(net.target, torch.long)
        self.beta = float(np.exp(-self.p.dt_ms / TRACE_TAU_MS))

    def operator(self, m):
        W = self.W_fixed.clone()
        W[self.rows, self.cols] = self.sign * m
        return W

    def run(self, m, seed: int, B: int, rules=None, baseline=None, noise=None, record=False):
        """One batch of B episodes: warm-up, then K rewarded decisions.

        Returns rewards (K, B), cues (K, B), and per rule Σ_draws Σ_trials
        (R − b_cue) e -- the unnormalised gradient estimate.
        `noise` (ticks, N) overrides the draws: the selftest feeds LIFBrain's.
        """
        torch, p = self.torch, self.p
        started = time.perf_counter() if self.dev.type == "cuda" and self.gpu_duty < 1.0 else None
        gn = torch.Generator(device=self.dev).manual_seed(int(seed))
        gc = torch.Generator(device=self.dev).manual_seed(int(seed) + 7919)
        W = self.operator(torch.as_tensor(m, dtype=torch.float32, device=self.dev))
        cue = torch.randint(0, 2, (K + 1, B), generator=gc, device=self.dev)
        v = torch.zeros((N, B), device=self.dev)
        refrac = torch.zeros((N, B), dtype=torch.int32, device=self.dev)
        s_prev = torch.zeros((N, B), device=self.dev)
        trace = torch.zeros((N_MOTOR, B), device=self.dev)
        rewards = torch.zeros((K, B), device=self.dev)
        ghat = {r.name + getattr(r, "tag", ""): torch.zeros(r.P, device=self.dev) for r in rules or []}
        for r in rules or []:
            r.reset_state()
        spikes, tick = [], 0
        for phase in range(K + 1):
            cur = self.cues.index_select(1, cue[phase])
            for r in rules or []:
                if phase and not getattr(r, "full", False):
                    r.reset_window()
            for _ in range(T if phase else WARM):
                # LIFBrain.step, same order of operations
                v.mul_(p.decay)
                v.add_(self.gain * (W @ s_prev) + p.bias)
                v.add_(cur)
                vhat = v.clone()
                if noise is not None:
                    xi = torch.as_tensor(noise[tick], device=self.dev)[:, None].expand(N, B)
                else:
                    xi = torch.randn((N, B), generator=gn, device=self.dev) * self.sigma
                v.add_(xi)
                free = refrac <= 0
                spk = free & (v >= p.threshold)
                v.masked_fill_(spk, p.v_reset)
                refrac.sub_(1).clamp_(min=0)
                refrac.masked_fill_(spk, p.refractory_steps)
                trace.mul_(self.beta).add_(spk.index_select(0, self.motor).float())
                if rules:
                    pre_spk = s_prev.index_select(0, self.pre)
                    xm, vm = xi.index_select(0, self.motor), vhat.index_select(0, self.motor)
                    sm, fm = spk.index_select(0, self.motor), free.index_select(0, self.motor)
                    for r in rules:
                        r.tick(pre_spk, xm, vm, sm, fm)
                s_prev = spk.float()
                if record:
                    spikes.append(spk[:, 0].cpu().numpy())
                tick += 1
            if phase:
                scores = self.Rd @ trace                                 # (2, B)
                action = (scores[1] > scores[0]).long()                  # argmax, ties -> 0
                R = torch.where(action == self.target[cue[phase]], 1.0, -1.0)
                rewards[phase - 1] = R
                if rules:
                    adv = R - baseline.index_select(0, cue[phase])
                    for r in rules:
                        ghat[r.name + getattr(r, "tag", "")] += r.eligibility() @ adv
        out = {"rewards": rewards, "cues": cue[1:], "ghat": ghat}
        if record:
            out["spikes"] = np.array(spikes)
        if started is not None:
            # Bound average GPU duty by syncing one small batch at a time, then
            # idling for the complementary fraction. The computation and RNG
            # draws are unchanged; short chunks keep display work schedulable.
            torch.cuda.synchronize(self.dev)
            active_seconds = time.perf_counter() - started
            time.sleep(active_seconds * (1.0 / self.gpu_duty - 1.0))
        return out

    def rules(self, B: int):
        """Every rule under test, sharing one set of per-connection constants."""
        args = (self.post, self.sign, self.gain, self.sigma, self.p.decay, self.p.threshold, B, self.dev)
        a_full = NodePerturbationFullTrajectory(*args)
        return [NodePerturbation(*args), a_full, SpikeReinforce(*args), Hebbian(*args)]

    def rate(self, m, seed: int, B: int = 256) -> float:
        """Mean firing rate over the rewarded trials."""
        torch = self.torch
        gn = torch.Generator(device=self.dev).manual_seed(int(seed))
        W = self.operator(torch.as_tensor(m, dtype=torch.float32, device=self.dev))
        v = torch.zeros((N, B), device=self.dev)
        refrac = torch.zeros((N, B), dtype=torch.int32, device=self.dev)
        s_prev = torch.zeros((N, B), device=self.dev)
        cur = self.cues.index_select(1, torch.arange(B, device=self.dev) % 2)
        tot, n = 0.0, 0
        for t in range(WARM + K * T):
            v.mul_(self.p.decay).add_(self.gain * (W @ s_prev) + self.p.bias).add_(cur)
            v.add_(torch.randn((N, B), generator=gn, device=self.dev) * self.sigma)
            free = refrac <= 0
            spk = free & (v >= self.p.threshold)
            v.masked_fill_(spk, 0.0)
            refrac.sub_(1).clamp_(min=0)
            refrac.masked_fill_(spk, self.p.refractory_steps)
            s_prev = spk.float()
            if t >= WARM:
                tot += float(s_prev.mean())
                n += 1
        return tot / n


# -- measurements -------------------------------------------------------------
def calibrate_gain(net: SmallNet, sigma: float, seed: int, device: str = "cuda",
                   gpu_duty: float = 1.0) -> tuple[float, float]:
    """Bisection on log gain for the target rate, as Gate E does for the brain."""
    lo, hi = 1e-3, 2.0
    for _ in range(30):
        mid = float(np.sqrt(lo * hi))
        r = Sim(net, mid, sigma, device=device, gpu_duty=gpu_duty).rate(net.m0, seed)
        lo, hi = (mid, hi) if r < RATE_TARGET else (lo, mid)
    g = float(np.sqrt(lo * hi))
    return g, Sim(net, g, sigma, device=device, gpu_duty=gpu_duty).rate(net.m0, seed + 1)


def expected_reward(sim: Sim, m, seed: int, draws: int, chunk: int) -> tuple[float, float]:
    """J(m) and its standard error, on `draws` episodes from `seed`."""
    tot = []
    for c in range(0, draws, chunk):
        out = sim.run(m, seed * 1000 + c // chunk, min(chunk, draws - c))
        tot.append(out["rewards"].mean(0).cpu().numpy())
    x = np.concatenate(tot)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def finite_difference(sim: Sim, m0, d, delta: float, seed: int, draws: int, chunk: int,
                      ctl=None, label: str = "finite difference"):
    """(J(m+δd) − J(m−δd)) / 2δ on common noise draws, with its standard error."""
    diffs = []
    chunks = list(range(0, draws, chunk))
    for i, c in enumerate(chunks):
        if ctl is not None:
            ctl.sub(i + 1, len(chunks), label)
        s, b = seed * 1000 + c // chunk, min(chunk, draws - c)
        up = sim.run(m0 + delta * d, s, b)["rewards"].mean(0)
        dn = sim.run(m0 - delta * d, s, b)["rewards"].mean(0)
        diffs.append(((up - dn) / (2 * delta)).cpu().numpy())
    x = np.concatenate(diffs)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def estimates(sim: Sim, m0, seed: int, draws: int, chunk: int, baseline,
              ctl=None, label: str = "gradient estimates"):
    """Every rule's gradient estimate on the same draws, normalised to ∂J/∂m."""
    torch = sim.torch
    acc, n = None, 0
    per_chunk = []
    chunks = list(range(0, draws, chunk))
    for i, c in enumerate(chunks):
        if ctl is not None:
            ctl.sub(i + 1, len(chunks), label)
        b = min(chunk, draws - c)
        out = sim.run(m0, seed * 1000 + c // chunk, b, rules=sim.rules(b), baseline=baseline)
        g = {k: v.cpu().numpy().astype(np.float64) for k, v in out["ghat"].items()}
        per_chunk.append({k: v / (b * K) for k, v in g.items()})
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
        n += b
        del out
        torch.cuda.empty_cache()
    return {k: v / (n * K) for k, v in acc.items()}, per_chunk


def regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Descriptive OLS of y (true) on x (estimated), conditional on x."""
    from scipy import stats
    res = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    lo, hi = res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr
    return {"slope": float(res.slope), "slope_ci95": [float(lo), float(hi)],
            "intercept": float(res.intercept), "r": float(res.rvalue), "n": int(len(x))}


def mean_ci95(se: float, n_chunks: int) -> list[float]:
    """Student-t interval width for a mean estimated from independent MC chunks."""
    from scipy import stats
    tcrit = stats.t.ppf(0.975, max(n_chunks - 1, 1))
    return [-float(tcrit * se), float(tcrit * se)]


def verdict(reg: dict, r_min: float = 0.8) -> dict:
    lo, hi = reg["slope_ci95"]
    ok_slope, ok_r = lo <= 1.0 <= hi, reg["r"] >= r_min
    return {"slope_ci_contains_1": bool(ok_slope), "r_at_least_0.8": bool(ok_r),
            "pass": bool(ok_slope and ok_r)}


def selftest(net_seed: int = 1, streams: int = 4, device: str = "cuda",
             gpu_duty: float = 1.0) -> dict:
    """The simulator against LIFBrain itself: same net, same noise, same cues.

    Counts ticks with any spike disagreement over several numpy noise streams fed
    to both, for both network versions, at the calibrated coupling.
    """
    total = {"ticks": 0, "spikes": 0, "ticks_disagreeing": 0}
    for fb in (False, True):
        net = SmallNet(net_seed, feedback=fb)
        for k in range(streams):
            r = _selftest_one(net, gain=0.065, sigma=0.1, stream=123 + k,
                              device=device, gpu_duty=gpu_duty)
            for key in total:
                total[key] += r[key]
    total["note"] = ("LIFBrain adds cue and noise as one current, the batch simulator adds "
                     "them in turn: same sum, rounding may differ in the last bit")
    return total


def _selftest_one(net: SmallNet, gain: float, sigma: float, stream: int, device: str = "cuda",
                  gpu_duty: float = 1.0) -> dict:
    import scipy.sparse as sp
    sim = Sim(net, gain, sigma, device=device, gpu_duty=gpu_duty)
    ticks = WARM + K * T
    rng = np.random.default_rng(stream)
    noise = [rng.standard_normal(N, dtype=np.float32) * sigma for _ in range(ticks)]
    out = sim.run(net.m0, seed=1, B=1, noise=noise, record=True)
    cue = out["cues"].cpu().numpy()[:, 0]
    # LIFBrain cannot take a changing cue by itself: drive it with the same schedule
    gc = sim.torch.Generator(device=sim.dev).manual_seed(1 + 7919)
    cues_all = sim.torch.randint(0, 2, (K + 1, 1), generator=gc, device=sim.dev).cpu().numpy()[:, 0]
    assert (cues_all[1:] == cue).all()
    brain = LIFBrain(sp.csc_matrix(net.W.astype(np.float32)), LIFParams(gain=gain, noise_std=0.0),
                     normalize=False)
    ref, tick = [], 0
    for phase in range(K + 1):
        for _ in range(T if phase else WARM):
            s = brain.step(net.cues[cues_all[phase]] + noise[tick])
            ref.append(s.copy())
            tick += 1
    ref = np.array(ref)
    bad = np.flatnonzero((ref != out["spikes"]).any(1))
    return {"ticks": ticks, "spikes": int(ref.sum()), "ticks_disagreeing": int(len(bad))}


# -- the check ----------------------------------------------------------------
def judge(dirs, ghat, chunks, own_dirs, own_fd, own_fd2, fd1, fd2, lin,
          fd_n_chunks) -> dict:
    """Calibrate on prespecified random directions; report selected directions
    separately as independent held-out ascent diagnostics."""
    fd1a, fd2a = np.array([x[0] for x in fd1]), np.array([x[0] for x in fd2])
    se1 = np.array([x[1] for x in fd1])
    se2 = np.array([x[1] for x in fd2])
    spread = float(fd1a.std(ddof=1))
    fd_pair_se = np.sqrt(se1 * se1 + se2 * se2)
    fd_pair_se_median = float(np.median(fd_pair_se))
    fd_noise_ceiling_powered = fd_pair_se_median <= spread / 3
    refs = {}
    reg = regression(fd2a, fd1a)
    refs["finite_difference_vs_independent"] = {
        **reg, **verdict(reg), "expected": "pass",
        "directions": "prespecified_random_only",
        "own_direction_excluded": True,
    }
    rules_out = {}
    for k in ghat:
        est = dirs @ ghat[k]
        true = fd1a
        reg = regression(est, true)
        if k == "random_vector":
            est_se = 0.0
        else:
            proj = np.array([[c[k] @ d for d in dirs] for c in chunks])          # chunks x dirs
            est_se = float(np.median(proj.std(axis=0, ddof=1) / np.sqrt(len(chunks))))
        own_est = float(own_dirs[k] @ ghat[k])
        if k == "random_vector":
            own_est_se, own_est_ci = 0.0, [own_est, own_est]
        else:
            own_proj = np.array([c[k] @ own_dirs[k] for c in chunks], dtype=np.float64)
            own_est_se = float(own_proj.std(ddof=1) / np.sqrt(len(own_proj)))
            own_est_ci_delta = mean_ci95(own_est_se, len(own_proj))
            own_est_ci = [own_est + own_est_ci_delta[0], own_est + own_est_ci_delta[1]]
        own_true, own_true_se = own_fd[k]
        own_true_ci_delta = mean_ci95(float(own_true_se), fd_n_chunks)
        own_true_ci = [float(own_true + own_true_ci_delta[0]),
                       float(own_true + own_true_ci_delta[1])]
        v = verdict(reg)
        powered = est_se <= spread / 3
        v["powered"] = bool(powered)
        v["outcome"] = ("pass" if v["pass"] else "fail") if powered else "inconclusive"
        direction_pass = own_est_ci[0] > 0.0 and own_true_ci[0] > 0.0
        entry = {**reg, **v, "estimate_se_median": est_se,
                 "estimates": est.tolist(), "true": true.tolist(),
                 "own_direction_estimate_independent": own_est,
                 "own_direction_estimate_se": own_est_se,
                 "own_direction_estimate_ci95": own_est_ci,
                 "own_direction_true": float(own_true),
                 "own_direction_se": float(own_true_se),
                 "own_direction_true_ci95": own_true_ci,
                 "direction_screen_pass": bool(direction_pass),
                 "cosine_with_node_perturbation": float(
                     ghat[k] @ ghat["node_perturbation"] /
                     (np.linalg.norm(ghat[k]) * np.linalg.norm(ghat["node_perturbation"])))}
        if k in ("random_vector", "pre_post_reward"):
            entry["expected"] = "fail"
            refs[k] = entry
        else:
            rules_out[k] = entry
    fd_fidelity_ok = refs["finite_difference_vs_independent"]["pass"]
    refs["finite_difference_vs_independent"]["outcome"] = (
        "pass" if fd_fidelity_ok else "inconclusive")
    return {"finite_difference_se_median": float(np.median(se1)),
            "finite_difference_pair_se_median": fd_pair_se_median,
            "finite_difference_noise_ceiling_powered": bool(fd_noise_ceiling_powered),
            "true_slope_spread_sd": spread,
            "linearity_half_step": lin,
            "references": refs, "rules": rules_out,
            "fd_set1": [list(t) for t in fd1], "fd_set2": [list(t) for t in fd2],
            "own_fd": {k: list(v) for k, v in own_fd.items()},
            "own_fd_set2_node_perturbation": list(own_fd2),
            "calibration_scope": "random directions only; selected directions reported separately",
            "fd_mc_chunks": int(fd_n_chunks)}


def power_rerun(key: str, args, ctl=None) -> dict:
    """GATE_G_PLAN §9 (power rerun): new estimates with more draws on fresh
    seeds; the powered random-direction finite differences are reused; the
    own-direction finite differences are remeasured along the new directions."""
    import torch
    first = json.loads(OUT.read_text())[key]
    fb, sigma = key.startswith("feedback"), float(key.split("_s")[1])
    net = SmallNet(args.net_seed, fb)
    sim = Sim(net, first["gain"], sigma, device=args.device, gpu_duty=args.gpu_duty)
    m0 = net.m0.astype(np.float32)
    P = net.n_plastic
    baseline = torch.as_tensor(np.asarray(first["baseline_per_cue"], np.float32), device=sim.dev)
    rng = np.random.default_rng(args.net_seed + 77)
    dirs = rng.standard_normal((args.directions, P))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ghat, chunks = estimates(sim, m0, 83, args.estimate_draws, args.chunk, baseline,
                             ctl, f"{key}: estimator draws")
    own, _ = estimates(sim, m0, 84, args.estimate_draws, args.chunk, baseline,
                        ctl, f"{key}: own-direction draws")
    rand_vec = np.random.default_rng(args.net_seed + 78).standard_normal(P)
    rand_vec *= np.linalg.norm(ghat["node_perturbation"]) / np.linalg.norm(rand_vec)
    ghat["random_vector"] = own["random_vector"] = rand_vec
    own_dirs = {k: v / np.linalg.norm(v) for k, v in own.items()}
    own_fd = {}
    for i, (k, d) in enumerate(own_dirs.items()):
        if ctl is not None:
            ctl.sub(i, len(own_dirs), f"{key}: own-direction finite differences")
        own_fd[k] = finite_difference(sim, m0, d.astype(np.float32), args.delta, 310,
                                      args.draws, args.chunk, ctl, f"{key}: own FD {k}")
    own_fd2 = finite_difference(sim, m0, own_dirs["node_perturbation"].astype(np.float32),
                                args.delta, 410, args.draws, args.chunk, ctl,
                                f"{key}: independent node FD")
    res = judge(dirs, ghat, chunks, own_dirs, own_fd, own_fd2,
                [tuple(x) for x in first["fd_set1"]], [tuple(x) for x in first["fd_set2"]],
                first["linearity_half_step"], int(np.ceil(args.draws / args.chunk)))
    res.update({"estimate_draws": args.estimate_draws, "estimate_seeds": [83, 84],
                "own_fd_seeds": [310, 410], "reused": "fd_set1, fd_set2, linearity from the first pass",
                "first_pass_outcomes": {k: v["outcome"] for k, v in
                                        {**first["references"], **first["rules"]}.items()}})
    return res


def check_config(feedback: bool, sigma: float, args, ctl, rep_key: str, rep: dict, flush):
    torch = __import__("torch")
    net = SmallNet(args.net_seed, feedback)
    m0 = net.m0.astype(np.float32)
    P = net.n_plastic
    stored_v2 = None
    if args.reuse_v2_random_fd:
        if not V2_OUT.exists():
            raise FileNotFoundError(f"v2 finite-difference artifact is missing: {V2_OUT}")
        old = json.loads(V2_OUT.read_text())
        old_params = old.get("params", {})
        required = {"net_seed": args.net_seed, "draws": args.draws,
                    "directions": args.directions,
                    "delta": args.delta}
        mismatches = {name: (old_params.get(name), value)
                      for name, value in required.items()
                      if old_params.get(name) != value}
        if mismatches:
            raise ValueError(f"cannot reuse v2 finite differences with changed settings: {mismatches}")
        stored_v2 = old[rep_key]
        gain, rate = float(stored_v2["gain"]), float(stored_v2["rate"])
        sim = Sim(net, gain, sigma, device=args.device, gpu_duty=args.gpu_duty)
        J0, J0se = stored_v2["J"], stored_v2["J_se"]
        base = np.asarray(stored_v2["baseline_per_cue"], dtype=np.float32)
    else:
        gain, rate = calibrate_gain(net, sigma, seed=11, device=args.device,
                                    gpu_duty=args.gpu_duty)
        sim = Sim(net, gain, sigma, device=args.device, gpu_duty=args.gpu_duty)
        ctl.sub(0, 1, f"{rep_key}: baseline")
        J0, J0se = expected_reward(sim, m0, 71, args.draws // 4, args.chunk)
        # per-cue baseline from an independent batch
        out = sim.run(m0, 72, args.chunk)
        R, cue = out["rewards"].cpu().numpy(), out["cues"].cpu().numpy()
        base = np.array([R[cue == c].mean() for c in (0, 1)], np.float32)
    res = rep.setdefault(rep_key, {"net": net.describe(), "sigma": sigma, "gain": gain,
                                   "rate": rate})
    res.update({"J": J0, "J_se": J0se, "baseline_per_cue": base.tolist()})
    if stored_v2 is not None:
        res["reused_v2_random_fd_source_chunk"] = old_params["chunk"]
    baseline = torch.as_tensor(base, device=sim.dev)

    # the directions: 20 random unit vectors, then each rule's own direction
    rng = np.random.default_rng(args.net_seed + 77)
    dirs = rng.standard_normal((args.directions, P))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    ghat, chunks = estimates(sim, m0, 73, args.draws, args.chunk, baseline,
                             ctl, f"{rep_key}: estimates")
    own, _ = estimates(sim, m0, 74, args.draws, args.chunk, baseline,
                       ctl, f"{rep_key}: independent own-direction estimates")
    rand_vec = np.random.default_rng(args.net_seed + 78).standard_normal(P)
    rand_vec *= np.linalg.norm(ghat["node_perturbation"]) / np.linalg.norm(rand_vec)
    ghat["random_vector"] = rand_vec
    own["random_vector"] = rand_vec
    own_dirs = {k: v / np.linalg.norm(v) for k, v in own.items()}

    # Finite differences along these exact fixed random directions are
    # independent of the revised regression. Reusing v2 saves all repeated
    # random-direction simulations; selected-direction FDs below are fresh.
    if stored_v2 is not None:
        fd1 = [tuple(x) for x in stored_v2["fd_set1"]]
        fd2 = [tuple(x) for x in stored_v2["fd_set2"]]
        lin = stored_v2["linearity_half_step"]
        res["reused_v2_random_fd"] = True
        for i, row in enumerate(fd1):
            ctl.point("fd", rep_key, i, row[0], line=f"{rep_key} reused v2 set 1")
    else:
        fd1, fd2, fd_half = [], [], []
        for i, d in enumerate(dirs):
            ctl.sub(i, len(dirs), f"{rep_key}: finite differences")
            fd1.append(finite_difference(sim, m0, d.astype(np.float32), args.delta, 100 + i,
                                         args.draws, args.chunk))
            fd2.append(finite_difference(sim, m0, d.astype(np.float32), args.delta, 200 + i,
                                         args.draws, args.chunk))
            # linearity diagnostic: half the step, on set 1's draws
            fd_half.append(finite_difference(sim, m0, d.astype(np.float32), args.delta / 2, 100 + i,
                                             args.draws, args.chunk))
            ctl.point("fd", rep_key, i, fd1[-1][0], line=f"{rep_key} set 1")
        lin = regression(np.array([x[0] for x in fd1]), np.array([x[0] for x in fd_half]))
    own_fd = {}
    for k, d in own_dirs.items():
        ctl.sub(0, 1, f"{rep_key}: finite difference along {k}")
        own_fd[k] = finite_difference(sim, m0, d.astype(np.float32), args.delta, 300,
                                      args.draws, args.chunk)
    # the noise ceiling needs its own "own direction": node perturbation's, measured twice
    own_fd2 = finite_difference(sim, m0, own_dirs["node_perturbation"].astype(np.float32),
                                args.delta, 400, args.draws, args.chunk)

    res.update(judge(dirs, ghat, chunks, own_dirs, own_fd, own_fd2, fd1, fd2, lin,
                     int(np.ceil(args.draws / args.chunk))))
    flush()
    return res


def summarize(manifest_path: str | None = None) -> dict:
    """Merge fidelity diagnostics; keep the direction screen separate."""
    first = json.loads(OUT.read_text())
    power = json.loads(POWER.read_text()) if POWER.exists() else {}
    keys = [k for k in first if "_s" in k and isinstance(first[k], dict) and "rules" in first[k]]
    table, refs_ok = {}, True
    for k in keys:
        src = first[k]
        use = {}
        for name, v in src["rules"].items():
            if v["outcome"] == "inconclusive" and k in power:
                use[name] = {**power[k]["rules"][name], "source": "power rerun"}
            else:
                use[name] = {**v, "source": "first pass"}
        ref = power[k]["references"] if k in power else src["references"]
        refs_ok &= (ref["random_vector"]["outcome"] == "fail"
                    and ref["pre_post_reward"]["outcome"] != "pass")
        fdsrc = power[k] if k in power else first[k]
        table[k] = {"powered": bool(fdsrc["finite_difference_noise_ceiling_powered"]),
                    "finite_difference_pair_se_median": fdsrc["finite_difference_pair_se_median"],
                    "finite_difference_fidelity_outcome": ref["finite_difference_vs_independent"]["outcome"],
                    "rules": use, "references": {n: {"outcome": r["outcome"], "slope": r["slope"],
                                                       "slope_ci95": r["slope_ci95"], "r": r["r"]}
                                                   for n, r in ref.items()}}
    names = sorted({n for k in keys for n in table[k]["rules"]})
    overall = {}
    for n in names:
        outs = [table[k]["rules"][n]["outcome"] for k in keys]
        overall[n] = ("pass" if all(o == "pass" for o in outs) else
                      "fail" if any(o == "fail" for o in outs) else "inconclusive")
    rep = {"protocol_version": 3,
           "complete": bool(first.get("complete")) and len(keys) == 4,
           "configurations": table, "fidelity_overall": overall,
           "overall": overall, "references_behaved": bool(refs_ok),
           "controls_behaved": bool(refs_ok),
           "noise_ceiling_powered_all": all(v["powered"] for v in table.values())}
    if manifest_path:
        rep["manifest"] = manifest_path
    tmp = VERDICT.with_suffix(".json.part")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(VERDICT)
    print(json.dumps({"fidelity_overall": overall, "references_behaved": refs_ok,
                      "complete": rep["complete"]}, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-seed", type=int, default=1)
    ap.add_argument("--draws", type=int, default=400_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--directions", type=int, default=20)
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--sigmas", default="0.1,0.02")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="execution device; the simulator selftest must show zero spike disagreements")
    ap.add_argument("--gpu-duty", type=float, default=1.0,
                    help="target average GPU duty (0, 1]; uses synchronized idle gaps per batch")
    ap.add_argument("--reuse-v2-random-fd", action="store_true",
                    help="reuse the unchanged v2 finite differences on the fixed random directions")
    ap.add_argument("--name", default="gate_g_estimator_check_v3")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--power-rerun", nargs="*", default=None,
                    help="optional extra draws for inconclusive exact-scale fidelity diagnostics")
    ap.add_argument("--estimate-draws", type=int, default=1_600_000)
    ap.add_argument("--summarize", action="store_true",
                    help="merge the first pass and power reruns into the verdict file")
    args = ap.parse_args()
    if not 0.0 < args.gpu_duty <= 1.0:
        ap.error("--gpu-duty must be greater than 0 and at most 1")
    out = Path(args.out)
    if args.summarize:
        inputs = {"first_pass": out, "power_rerun": POWER}
        manifest = estimator_manifest("summarize", args, estimator_seeds(args), inputs,
                                      {"verdict": VERDICT})
        try:
            summarize(str(manifest.path))
            manifest.finish("complete")
        except BaseException as exc:
            manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        return
    if args.power_rerun is not None:
        keys = list(dict.fromkeys(args.power_rerun))
        run_name = ("gate_g_estimator_power_v3"
                    if args.name == "gate_g_estimator_check_v3" else args.name)
        live = ROOT / "artifacts" / "runs" / "live" / run_name
        manifest = estimator_manifest(
            "power", args, estimator_seeds(args, power=True), {"first_pass": out},
            {"power_report": POWER, "live_status": live / "status.json",
             "live_series": live / "series.jsonl"})
        pr = POWER.read_text() if POWER.exists() else "{}"
        rep = json.loads(pr)
        rep["complete"] = False
        rep["manifest"] = str(manifest.path)

        def flush_power():
            tmp = POWER.with_suffix(".json.part")
            tmp.write_text(json.dumps(rep, indent=2))
            tmp.replace(POWER)

        already_done = [key for key in keys if key in rep]
        ctl = RunControl(run_name, total=len(keys), checkpoint=flush_power,
                         unit="configuration", already_done=len(already_done),
                         fresh=not already_done, meta={
            "question": "Do inconclusive estimator-fidelity diagnostics resolve with more draws?",
            "why": "Exact-scale calibration is a fidelity result; the learning screen uses an independent held-out ascent check.",
            "method": "Rerun estimates at 1.6M draws; reuse powered random-direction finite differences and remeasure selected directions.",
        })
        try:
            for key in keys:
                if key in already_done:
                    continue
                ctl.begin(key)
                t0 = time.time()
                rep[key] = power_rerun(key, args, ctl)
                rep[key]["seconds"] = time.time() - t0
                flush_power()
                r = rep[key]
                summary = {k: v["outcome"] for k, v in {**r["references"], **r["rules"]}.items()}
                ctl.done(summary)
                print(key, {k: (round(v["slope"], 3), [round(c, 3) for c in v["slope_ci95"]],
                                round(v["r"], 3), v["outcome"])
                            for k, v in {**r["references"], **r["rules"]}.items()}, flush=True)
        except StopRequested:
            ctl.close("stopped")
            manifest.finish("stopped")
            return
        except BaseException as exc:
            ctl.close("failed")
            manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        rep["complete"] = True
        flush_power()
        ctl.close("complete")
        manifest.finish("complete")
        return

    live = ROOT / "artifacts" / "runs" / "live" / args.name
    first_pass_inputs = {"v2_random_direction_finite_differences": V2_OUT} \
        if args.reuse_v2_random_fd else {}
    manifest = estimator_manifest(
        "first_pass", args, estimator_seeds(args), first_pass_inputs,
        {"estimator_report": out, "live_status": live / "status.json",
         "live_series": live / "series.jsonl"})
    try:
        st = selftest(device=args.device, gpu_duty=args.gpu_duty)
    except BaseException as exc:
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    print("selftest:", st, flush=True)
    if st["ticks_disagreeing"] != 0:
        manifest.finish("failed", error=f"simulator selftest had {st['ticks_disagreeing']} spike-disagreeing ticks")
        raise RuntimeError(f"simulator selftest failed: {st['ticks_disagreeing']} spike-disagreeing ticks")
    if args.selftest_only:
        manifest.finish("complete")
        return
    params = {k: v for k, v in vars(args).items()
              if k not in ("name", "selftest_only", "out", "power_rerun", "estimate_draws", "summarize")}
    rep = {"protocol_version": 3, "what": "Gate G §6.1 estimator check v3",
           "params": params, "selftest": st,
           "declared": {"fidelity_diagnostic": "random-direction OLS slope 95% CI contains 1 AND r >= 0.8",
                        "learning_direction_screen": "independent-estimate and held-out finite-difference 95% CIs both above zero",
                        "own_direction_in_calibration": False,
                        "reused_v2_random_direction_finite_differences": bool(args.reuse_v2_random_fd),
                        "negative_controls_expected_to_fail": ["random_vector", "pre_post_reward"],
                        "finite_difference_vs_independent": "fidelity diagnostic; not required for learning-pilot eligibility",
                        "judged_sigmas": [0.1, 0.02],
                        "fidelity_inconclusive": "FD noise ceiling fails, or random-direction estimate SE > spread of true slopes / 3",
                        "estimate_draws": "independent of the finite-difference draws",
                        "plan": "docs/GATE_G_PLAN.md §6.1 and §9"},
           "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rep["manifest"] = str(manifest.path)

    # Resume: a configuration already judged is kept, not recomputed. Every
    # configuration draws from its own fixed seeds, so a resumed run gives the
    # numbers an uninterrupted one would have.
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("params") == params and not old.get("complete"):
            kept = {k: v for k, v in old.items() if "_s" in k and isinstance(v, dict) and "rules" in v}
            rep.update(kept)
            rep["started"] = old.get("started", rep["started"])
            rep["resumed_with"] = sorted(kept)
            print("resuming; already judged:", sorted(kept), flush=True)

    def flush():
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=2))
        tmp.replace(out)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    configs = [(fb, s) for s in sigmas for fb in (False, True)]
    done = [k for k in rep if "_s" in k and isinstance(rep[k], dict) and "rules" in rep[k]]
    ctl = RunControl(args.name, total=len(configs), checkpoint=flush, unit="configuration",
                     already_done=len(done), fresh=not done, meta={
        "question": "Does each candidate learning rule point the way reward actually increases?",
        "why": ("A rule 'Δw = η (R − R̄) e' only learns if e is, on average, the true slope of "
                "expected reward. On a small network that slope can be measured directly."),
        "method": ("True slope = finite difference of expected reward along 20 prespecified random directions; "
                   "each rule's estimate on independent noise, with its selected direction validated separately. References first: finite "
                   "difference vs independent finite difference must pass; random vector "
                   "and pre x post x reward must fail.")})
    ctl.chart("fd", "True slope along each direction", "direction", "d(expected reward)/dδ",
              [f"{'feedback' if fb else 'no_feedback'}_s{s}" for fb, s in configs], kind="points",
              note="Finite differences: the ground truth each rule is compared against.")
    try:
        for fb, s in configs:
            key = f"{'feedback' if fb else 'no_feedback'}_s{s}"
            if key in done:
                continue
            rep.pop(key, None)                 # a half-done configuration restarts clean
            ctl.begin(key)
            t0 = time.time()
            res = check_config(fb, s, args, ctl, key, rep, flush)
            res["seconds"] = time.time() - t0
            flush()
            ctl.done({k: v["outcome"] for k, v in {**res["references"], **res["rules"]}.items()})
    except StopRequested:
        ctl.close("stopped")
        manifest.finish("stopped")
        raise SystemExit(0)
    except BaseException as exc:
        ctl.close("failed")
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    keys = [k for k in rep if "_s" in k]
    names = rep[keys[0]]["rules"].keys()
    rep["fidelity_overall"] = {
        n: ("pass" if all(rep[k]["rules"][n]["outcome"] == "pass" for k in keys) else
            "inconclusive" if any(rep[k]["rules"][n]["outcome"] == "inconclusive"
                                  for k in keys) else "fail") for n in names}
    rep["overall"] = rep["fidelity_overall"]
    rep["controls_behaved"] = all(
        rep[k]["references"]["random_vector"]["outcome"] == "fail" and
        rep[k]["references"]["pre_post_reward"]["outcome"] != "pass" for k in keys)
    rep["references_behaved"] = rep["controls_behaved"]
    rep["noise_ceiling_powered_all"] = all(
        rep[k]["finite_difference_noise_ceiling_powered"] for k in keys)
    rep["complete"] = True
    flush()
    ctl.close("complete")
    manifest.finish("complete")
    print(json.dumps({k: {**{r: v["outcome"] for r, v in rep[k]["references"].items()},
                          **{r: v["outcome"] for r, v in rep[k]["rules"].items()}}
                      for k in keys}, indent=1))
    print("fidelity outcomes:", rep["fidelity_overall"],
          "reference controls behaved:", rep["controls_behaved"],
          "noise ceiling powered:", rep["noise_ceiling_powered_all"])


if __name__ == "__main__":
    main()
