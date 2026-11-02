"""Multiclass ridge decoder with the normalised-loss convention (mission §9).

Objective, per task, training subset and lambda:

    (1/n) ||Z W + 1 b^T - Y||_F^2 + lambda ||W||_F^2,      Y one-hot (n, 4), b not penalised

with Z the features standardised on the training subset only (mean, SD with ddof 0; a column constant on the
subset gets SD 1, so it is zero on the subset and its weight is zero). Because Z is centred, b = mean(Y) and
W = (Z^T Z + n lambda I)^-1 Z^T (Y - mean(Y)). With the thin SVD Z = U S V^T this is
W = V diag(s / (s^2 + n lambda)) U^T Yc: one factorisation serves every lambda and both tasks (same subset,
same features). In a library that minimises the unnormalised sum, alpha = n * lambda (`alpha_of`). The decision is
argmax of Z W + b (first index on an exact tie, which is reported).

Selection: on the validation set, per task, budget and fit, the lambda with the highest accuracy; ties go to the
largest lambda. No refit on train + validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_CLASSES = 4


def alpha_of(lam: float, n: int) -> float:
    """The sum-of-squares penalty equivalent to the normalised-loss lambda."""
    return float(n) * float(lam)


def standardise_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


@dataclass
class Fit:
    W: np.ndarray            # (L, d, 4) one per lambda
    b: np.ndarray            # (4,)
    mu: np.ndarray           # (d,)
    sd: np.ndarray           # (d,)


def fit_all_lambdas(X: np.ndarray, y: np.ndarray, lambdas=LAMBDAS) -> Fit:
    """Ridge solutions for every lambda on one training subset (float64)."""
    return fit_tasks(X, [y], lambdas)[0]


def fit_tasks(X: np.ndarray, ys: list[np.ndarray], lambdas=LAMBDAS) -> list[Fit]:
    """One standardisation and one SVD of the subset, then the solutions of every task's targets. The labels of a
    task are only ever its targets, never a feature of another."""
    X = np.asarray(X, np.float64)
    n = len(X)
    mu, sd = standardise_fit(X)
    Z = (X - mu) / sd
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    out = []
    for y in ys:
        Y = np.eye(N_CLASSES)[np.asarray(y, int)]
        b = Y.mean(0)
        A = U.T @ (Y - b)                                    # (r, 4)
        W = np.stack([Vt.T @ ((s / (s ** 2 + n * lam))[:, None] * A) for lam in lambdas])
        out.append(Fit(W=W, b=b, mu=mu, sd=sd))
    return out


def scores(fit: Fit, X: np.ndarray) -> np.ndarray:
    """(L, m, 4) decision scores for every lambda."""
    Z = (np.asarray(X, np.float64) - fit.mu) / fit.sd
    return np.einsum("md,ldk->lmk", Z, fit.W) + fit.b


def select(val_acc: np.ndarray, lambdas=LAMBDAS) -> int:
    """Index of the chosen lambda: best validation accuracy, ties to the largest lambda."""
    val_acc = np.asarray(val_acc)
    best = val_acc.max()
    cand = np.flatnonzero(val_acc >= best - 1e-12)
    return int(cand[np.argmax(np.asarray(lambdas)[cand])])


def subsets(pool_combo: np.ndarray, draws_seeds: list[int], per_combo_budgets=(2, 4, 8, 16, 32)) -> np.ndarray:
    """Nested, jointly balanced training subsets from the training pool.

    pool_combo: (n_pool,) joint label 0..15 of each pool image. Returns an int array (draws, budgets, max) of pool
    positions, where budget k uses the first 16 * per_combo_budgets[k] entries of each row (padded with -1). For
    each draw and combination a single random order of that combination's pool images is taken, and budget k keeps
    its first per_combo_budgets[k]: smaller budgets are nested in larger ones, and each budget has exactly the same
    number of images per (orientation, brightness) pair, so both tasks are balanced on the same subset."""
    nb = len(per_combo_budgets)
    mx = 16 * per_combo_budgets[-1]
    out = np.full((len(draws_seeds), nb, mx), -1, np.int64)
    for r, seed in enumerate(draws_seeds):
        rng = np.random.default_rng(seed)
        order = {c: np.flatnonzero(pool_combo == c)[rng.permutation(int((pool_combo == c).sum()))]
                 for c in range(16)}
        for k, m in enumerate(per_combo_budgets):
            sel = np.concatenate([order[c][:m] for c in range(16)])
            out[r, k, :len(sel)] = np.sort(sel)
    return out
