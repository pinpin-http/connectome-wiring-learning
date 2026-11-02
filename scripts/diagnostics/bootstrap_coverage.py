#!/usr/bin/env python
"""Statistics check: does the paired, graph-block, image-stratified percentile bootstrap used by RETINO_ALIGN and
ACTIVITY_MATCH cover the true connectome-minus-null-mean contrast at its nominal 95 %?

Simulation with a known truth, shaped like the campaigns: one connectome and K nulls, n test images in 16 cells.
Each image has a difficulty shared by every graph (this is what makes the design paired); each null's accuracy is
drawn from a population of nulls; correctness ~ Bernoulli(sigmoid(logit(p_g) + d_i)). The target is
G = acc(connectome) - E_null[acc(null)], marginal over the image population and the null population. For each of R
simulated datasets the study's bootstrap (images resampled within cells identically for every graph, nulls resampled
as whole graphs) gives a 95 % percentile interval; coverage = share of intervals containing the true G.

    .venv/bin/python scripts/diagnostics/bootstrap_coverage.py   # CPU, a few minutes; fixed seed 20261020

Writes artifacts/complements/bootstrap_coverage.json, input of fig_validation.pdf (scripts/figures/make_figures_campagne.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "artifacts/complements"
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
logit = lambda p: np.log(p / (1 - p))  # noqa: E731


def true_acc(mu_logit: float, d_sd: float, rng) -> float:
    d = rng.normal(0, d_sd, 400_000)
    return float(sig(mu_logit + d).mean())


def one_dataset(rng, K, n, p_conn, p_null, null_sd, d_sd, n_rep, avg_fits):
    d = rng.normal(0, d_sd, n)                              # image difficulty, shared by every graph
    strata = np.repeat(np.arange(16), n // 16)
    g_logit = np.r_[logit(p_conn), rng.normal(logit(p_null), null_sd, K)]
    # cbar = mean over `avg_fits` decoders (seeds x draws x budgets) of per-example correctness
    prob = sig(g_logit[:, None] + d[None, :])
    cbar = (rng.random((avg_fits, K + 1, n)) < prob[None]).mean(0)
    point = cbar[0].mean() - cbar[1:].mean()
    reps = np.empty(n_rep)
    idx_by_s = [np.flatnonzero(strata == s) for s in range(16)]
    for b in range(n_rep):
        w = np.zeros(n)
        for ix in idx_by_s:
            np.add.at(w, rng.choice(ix, len(ix)), 1.0)
        M = cbar @ w / w.sum()
        nul = 1 + rng.integers(0, K, K)
        reps[b] = M[0] - M[nul].mean()
    return point, np.quantile(reps, [0.025, 0.975])


def main():
    rng = np.random.default_rng(20261020)
    scenarios = [
        {"name": "ACTIVITY_MATCH-like (K=40, n=2560)", "K": 40, "n": 2560},
        {"name": "fewer images (K=40, n=640)", "K": 40, "n": 640},
        {"name": "fewer nulls (K=20, n=2560)", "K": 20, "n": 2560},
    ]
    p_conn, p_null, null_sd, d_sd, avg_fits = 0.29, 0.35, 0.08, 1.0, 20
    R, n_rep = 300, 600
    out = {"settings": {"p_conn": p_conn, "p_null_median": p_null, "null_logit_sd": null_sd, "image_logit_sd": d_sd,
                        "decoders_averaged": avg_fits, "datasets": R, "bootstrap_replicates": n_rep}, "results": []}
    # true G: marginal accuracies over images and over the null population
    t_conn = true_acc(logit(p_conn), d_sd, rng)
    nl = rng.normal(logit(p_null), null_sd, 4000)
    dd = rng.normal(0, d_sd, 20000)
    t_null = float(sig(nl[:, None] + dd[None, :]).mean())
    G_true = t_conn - t_null
    out["G_true"] = G_true
    for sc in scenarios:
        cov, width, bias = 0, [], []
        for _ in range(R):
            pt, (lo, hi) = one_dataset(rng, sc["K"], sc["n"], p_conn, p_null, null_sd, d_sd, n_rep, avg_fits)
            cov += lo <= G_true <= hi
            width.append(hi - lo)
            bias.append(pt - G_true)
        res = {**sc, "coverage_95": cov / R, "mean_width_pts": 100 * float(np.mean(width)),
               "mean_bias_pts": 100 * float(np.mean(bias)), "mc_se_coverage": float(np.sqrt(0.95 * 0.05 / R))}
        out["results"].append(res)
        print(json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bootstrap_coverage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
