#!/usr/bin/env python
"""RETINO_ALIGN_V1 exploratory analyses, decided after reading the main result; none replaces the primary
contrast D.

    .venv/bin/python scripts/retino_align/exploratory.py --run main_v1

E1 headroom: orientation room for a loss, H = M(ordered) - 0.25, the loss relative to it, and the largest D the
   connectome could have produced had its permuted score fallen to chance.
E2 the connectome's ordered and permuted orientation scores against the input-level arrangement-blind references.
E3 brightness: G for brightness and the connectome's rank.
E4 activity regime of the readout, connectome vs the 40 nulls (ordered condition, mean of the two seeds).
E5 the same decoders with the lambda grid extended to 100 and 1000: D, G, S and their bootstrap.

    .venv/bin/python scripts/retino_align/exploratory.py --run main_v1 --e5 --workers 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


LAMBDAS_E5 = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def e5_one(run: str, spec: str, cond: str) -> tuple:
    """Per-example test correctness (T, S, R, K, n) with the extended grid; nothing else is saved."""
    sys.path.insert(0, str(ROOT / "src"))
    from flymario.retino_align import decode
    d = ROOT / "artifacts/retino_align" / run
    man = json.loads((d / "manifest.json").read_text())
    inp = np.load(d / "inputs/inputs.npz")
    subs = np.load(d / "inputs/subsets.npz")["subsets"]
    part = inp["part"]
    tr, va, te = part == 0, part == 1, part == 2
    y = {"orient": inp["orient"].astype(int), "lum": inp["lum"].astype(int)}
    R, K = subs.shape[:2]
    S = len(man["dynamic_seeds"])
    corr = np.zeros((2, S, R, K, int(te.sum())), np.uint8)
    lam_idx = np.zeros((2, S, R, K), np.int8)
    for si, seed in enumerate(man["dynamic_seeds"]):
        F = np.load(d / "features" / spec.replace(":", "") / f"{cond}_s{seed}.npz")["counts"].astype(np.float64)
        Ftr, Fva, Fte = F[tr], F[va], F[te]
        for r in range(R):
            for k in range(K):
                sel = subs[r, k][subs[r, k] >= 0]
                fits = decode.fit_tasks(Ftr[sel], [y["orient"][tr][sel], y["lum"][tr][sel]], LAMBDAS_E5)
                Zva = (Fva - fits[0].mu) / fits[0].sd
                Zte = (Fte - fits[0].mu) / fits[0].sd
                for ti, t in enumerate(("orient", "lum")):
                    f = fits[ti]
                    va_a = ((np.einsum("md,ldk->lmk", Zva, f.W) + f.b).argmax(2) == y[t][va]).mean(1)
                    j = decode.select(va_a, LAMBDAS_E5)
                    corr[ti, si, r, k] = (Zte @ f.W[j] + f.b).argmax(1) == y[t][te]
                    lam_idx[ti, si, r, k] = j
    return spec, cond, corr, lam_idx


def e5(run: str, workers: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(ROOT / "src"))
    from flymario.retino_align import analysis
    d = ROOT / "artifacts/retino_align" / run
    man = json.loads((d / "manifest.json").read_text())
    graphs, conds = man["graphs"], man["conditions"]
    inp = np.load(d / "inputs/inputs.npz")
    te = inp["part"] == 2
    strata = (inp["orient"].astype(int) * 4 + inp["lum"])[te]
    cbar = np.zeros((len(graphs), len(conds), 2, int(te.sum())))
    lam = np.zeros((2, len(LAMBDAS_E5)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(e5_one, run, g, c) for g in graphs for c in conds]
        for f in futs:
            g, c, corr, li = f.result()
            cbar[graphs.index(g), conds.index(c)] = corr.mean(axis=(1, 2, 3))
            for ti in range(2):
                lam[ti] += np.bincount(li[ti].ravel(), minlength=len(LAMBDAS_E5))
    M = analysis.scores(cbar)
    c = analysis.contrasts(M)
    bs = analysis.bootstrap(cbar, strata, 5000, 4242)
    return {"lambdas": list(LAMBDAS_E5),
            "lambda_share": {t: (lam[ti] / lam[ti].sum()).tolist() for ti, t in enumerate(("orient", "lum"))},
            "point": {k: c[k] for k in ("D_orientation", "D_brightness", "G", "S")},
            "intervals_95": {k: analysis.interval(v) for k, v in bs.items()},
            "M_connectome_ordered": {"orient": float(M[0, 0, 0]), "lum": float(M[0, 0, 1])},
            "M_nulls_ordered_mean": {"orient": float(M[1:, 0, 0].mean()), "lum": float(M[1:, 0, 1].mean())},
            "L_connectome": c["L"][0].tolist(), "L_nulls_mean": c["L"][1:].mean(0).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_v1")
    ap.add_argument("--e5", action="store_true", help="only the extended-grid sensitivity (E5)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.e5:
        res = e5(a.run, a.workers)
        p = ROOT / "artifacts/retino_align" / a.run / "metrics/exploratory_e5.json"
        p.write_text(json.dumps({"label": "EXPLORATORY E5 (DECISIONS X1)", **res}, indent=1))
        print(json.dumps(res, indent=1))
        return 0
    d = ROOT / "artifacts/retino_align" / a.run
    r = json.loads((d / "metrics/analysis.json").read_text())
    v = json.loads((d / "validation/validation.json").read_text())
    M, L = r["M"], r["L"]
    g0 = "connectome"
    nulls = [g for g in r["graphs_complete"] if g != g0]
    out = {"label": "EXPLORATORY (decided after the main result)", "run": a.run}
    H = {g: M[g]["ordered"]["orient"] - 0.25 for g in [g0] + nulls}
    Lo = {g: L[g]["orient"] for g in [g0] + nulls}
    Ln = np.array([Lo[g] for g in nulls])
    out["E1_headroom"] = {
        "H_connectome": H[g0], "H_nulls_mean": float(np.mean([H[g] for g in nulls])),
        "H_nulls_min": float(min(H[g] for g in nulls)),
        "relative_loss_connectome": Lo[g0] / H[g0],
        "relative_loss_nulls_mean": float(np.mean([Lo[g] / H[g] for g in nulls])),
        "relative_loss_nulls_sd": float(np.std([Lo[g] / H[g] for g in nulls])),
        "max_D_if_connectome_fell_to_chance": float(H[g0] - Ln.mean()),
        "delta": 0.05,
        "note": "a D above +delta was out of reach for this connectome even with a complete loss to chance"}
    perm = [M[g0][c]["orient"] for c in ("perm1", "perm2", "perm3")]
    out["E2_orientation_vs_references"] = {
        "connectome_ordered_M": M[g0]["ordered"]["orient"], "connectome_permuted_M": perm,
        "nulls_ordered_M_mean": float(np.mean([M[g]["ordered"]["orient"] for g in nulls])),
        "nulls_ordered_M_min": float(min(M[g]["ordered"]["orient"] for g in nulls)),
        "n_nulls_below_connectome_ordered": int(sum(M[g]["ordered"]["orient"] < M[g0]["ordered"]["orient"] for g in nulls)),
        "input_direct_decoder_mean_over_budgets": float(np.mean(v["direct_decoder_accuracy"]["orient"]["mean_by_budget"])),
        "input_sorted_values_mean_over_budgets": float(np.mean(v["multiset_reference_decoder"]["orient"]["mean_by_budget"])),
        "input_global_descriptors_mean_over_budgets": float(np.mean(v["global_descriptor_decoder"]["orient"]["mean_by_budget"])),
        "curves_connectome_ordered_by_budget": r["curves"][0][0][0],
        "curves_nulls_ordered_mean_by_budget": np.mean([r["curves"][i][0][0] for i in range(1, len(r["curves"]))], 0).tolist()}
    bn = np.array([M[g]["ordered"]["lum"] for g in nulls])
    out["E3_brightness"] = {"connectome_ordered_M": M[g0]["ordered"]["lum"], "nulls_mean": float(bn.mean()),
                            "nulls_max": float(bn.max()), "G_brightness": float(M[g0]["ordered"]["lum"] - bn.mean()),
                            "n_nulls_above_connectome": int((bn > M[g0]["ordered"]["lum"]).sum())}
    act = {}
    for p in sorted((d / "features").glob("*/ordered_s*.json")):
        j = json.loads(p.read_text())
        act.setdefault(j["lot"][0], []).append(j["activity"])
    keys = ("mean_count", "sd_count", "frac_silent_entries", "frac_near_ceiling_entries_ge7",
            "frac_neurons_silent_on_all_images", "rate_all", "rate_input", "v_readout_mean")
    E4 = {}
    for k in keys:
        c = float(np.mean([x[k] for x in act[g0]]))
        n = np.array([np.mean([x[k] for x in act[g]]) for g in nulls])
        E4[k] = {"connectome": c, "nulls_mean": float(n.mean()), "nulls_sd": float(n.std()),
                 "nulls_min": float(n.min()), "nulls_max": float(n.max()),
                 "connectome_outside_null_range": bool(c < n.min() or c > n.max())}
    out["E4_activity_ordered"] = E4
    (d / "metrics/exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
