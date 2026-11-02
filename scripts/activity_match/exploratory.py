#!/usr/bin/env python
"""ACTIVITY_MATCH_V1 exploratory analyses, decided after the main result; none replaces the frozen primary reading.

X1: refit the decoders with the lambda grid extended to 100 and 1,000; recompute every declared contrast.
X2: orientation and brightness accuracy of every graph and condition against its readout rate and gamma.

    .venv/bin/python scripts/activity_match/exploratory.py --run main_am --x1 --workers 3
    .venv/bin/python scripts/activity_match/exploratory.py --run main_am            # X2 only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/activity_match"))
LAMBDAS_X1 = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def x1_one(run: str, spec: str, cond: str) -> tuple:
    import run_am
    ctx = run_am.Ctx(run)
    inp = ctx.inputs()
    subs = np.load(ctx.dir / "inputs/subsets.npz")["subsets"]
    part = inp["part"]
    te = part == 2
    y = {"orient": inp["orient"].astype(int), "lum": inp["lum"].astype(int)}
    corr, lam = [], []
    for seed in ctx.manifest["noise_seeds"]:
        F = np.load(ctx.feature_path(spec, cond, seed))["counts"].astype(np.float64)
        o = run_am.fit_predict(F, y, part, subs, LAMBDAS_X1)
        corr.append(np.moveaxis(o["pred"] == np.stack([y["orient"][te], y["lum"][te]])[None, None], 2, 0))
        lam.append(np.moveaxis(o["lam_idx"], 2, 0))
    return spec, cond, np.stack(corr, 1), np.stack(lam, 1)                  # (T, S, R, K, n), (T, S, R, K)


def x1(run: str, workers: int, reps: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    import run_am

    from flymario.activity_match import analysis
    ctx = run_am.Ctx(run)
    graphs, conds = ctx.manifest["graphs"], ctx.manifest["conditions"]
    inp = ctx.inputs()
    te = inp["part"] == 2
    strata = (inp["orient"].astype(int) * 4 + inp["lum"])[te]
    cbar = np.zeros((len(graphs), len(conds), 2, int(te.sum())))
    lamc = np.zeros((2, len(LAMBDAS_X1)), np.int64)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for spec, cond, corr, lam in pool.map(x1_one, *zip(*[(run, g, c) for g in graphs for c in conds])):
            cbar[graphs.index(spec), conds.index(cond)] = corr.mean(axis=(1, 2, 3))
            for ti in range(2):
                lamc[ti] += np.bincount(lam[ti].ravel(), minlength=len(LAMBDAS_X1))
    M = cbar.mean(-1)
    pt = analysis.contrasts(M, conds)
    bs = analysis.bootstrap(cbar, strata, conds, reps, int(ctx.cfg["analysis"]["bootstrap_seed"]))
    iv = analysis.intervals(bs)
    delta = float(ctx.cfg["analysis"]["practical_margin"])
    frozen = json.loads((ctx.dir / "metrics/analysis.json").read_text())
    return {"label": "EXPLORATORY (decided after the main result)", "lambdas": LAMBDAS_X1, "replicates": reps,
            "lambda_counts": {"orient": lamc[0].tolist(), "lum": lamc[1].tolist()},
            "point": pt, "intervals_95": iv,
            "reading_G_CM_orient_same_table": analysis.reading_G(*iv["G_CM_orient"], delta),
            "frozen_point_for_comparison": {k: frozen["point"][k] for k in pt}}


def x2(run: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_am
    ctx = run_am.Ctx(run)
    res = json.loads((ctx.dir / "metrics/analysis.json").read_text())
    graphs, conds = res["graphs"], res["conditions"]
    out = {"label": "EXPLORATORY (decided after the main result)", "per_family": {}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ti, t in enumerate(("orient", "lum")):
        ax = axes[ti]
        for fam, members in (("connectome", ["connectome"]), ("nulls", [g for g in graphs if g != "connectome"])):
            r = np.array([[res["activity_test"][g][c]["rate_hz"] for c in conds] for g in members])
            m = np.array([[res["M"][g][c][t] for c in conds] for g in members])
            gm = np.array([[ctx.gamma(g, c) for c in conds] for g in members])
            out["per_family"].setdefault(fam, {})[t] = {
                "corr_M_rate": float(np.corrcoef(r.ravel(), m.ravel())[0, 1]),
                "corr_M_log_gamma": float(np.corrcoef(np.log(gm.ravel()), m.ravel())[0, 1])}
            for k in range(len(members)):
                order = np.argsort(r[k])
                is_c = fam == "connectome"
                ax.plot(r[k][order], 100 * m[k][order], "-o", color="#c2410c" if is_c else "#9ca3af",
                        lw=2 if is_c else 0.5, ms=5 if is_c else 2, alpha=1 if is_c else 0.6, zorder=3 if is_c else 1)
        for ci, c in enumerate(conds):
            ax.annotate(c, (res["activity_test"]["connectome"][c]["rate_hz"],
                            100 * res["M"]["connectome"][c][t]), textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color="#c2410c")
        ax.axhline(25, color="k", lw=0.5, ls=":")
        ax.set_xlabel("readout rate on test images (Hz)")
        ax.set_ylabel(f"M, {'orientation' if t == 'orient' else 'brightness'} (%)")
        ax.set_title(f"{'Orientation' if t == 'orient' else 'Brightness'}: each graph's four conditions")
    fig.suptitle("EXPLORATORY: accuracy against readout activity (connectome orange, 40 nulls grey)", fontsize=10)
    fig.savefig(ctx.dir / "figures/figX2_accuracy_vs_activity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main_am")
    ap.add_argument("--x1", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--replicates", type=int, default=5000)
    a = ap.parse_args()
    import run_am
    d = run_am.ARTS / a.run / "metrics"
    rec = {"X2_accuracy_vs_activity": x2(a.run)}
    if a.x1:
        rec["X1_lambda_extended"] = x1(a.run, a.workers, a.replicates)
    run_am.write_json(d / ("exploratory_x1.json" if a.x1 else "exploratory_x2.json"), rec)
    print(json.dumps({k: (v.get("point", v) if isinstance(v, dict) else v) for k, v in rec.items()}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
