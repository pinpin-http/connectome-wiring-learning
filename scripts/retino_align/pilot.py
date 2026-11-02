#!/usr/bin/env python
"""RETINO_ALIGN_V1 pilot (docs/retino_align/PILOT.md): decodability on the reserved graphs for a grid of generator
settings. Never runs the connectome or an analysed null.

    .venv/bin/python scripts/retino_align/pilot.py grid --step P1 --settings A,B,C,D
    .venv/bin/python scripts/retino_align/pilot.py grid --step P1b --settings A,B,C,D --period 40,64

Writes artifacts/retino_align/pilot/<step>.json (atomically, after every setting x graph).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts/retino_align/pilot"
PILOT_GRAPHS = ("nullin:100", "nullin:101")
AMPS = {"A": (0.05, 0.10), "B": (0.10, 0.15), "C": (0.15, 0.20), "D": (0.20, 0.25)}
BANK_SEEDS = {"pilot_train": 2002, "pilot_val": 2003, "pilot_test": 2004}
DRAW_SEEDS = [9001, 9002, 9003]


def save(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)


def evaluate(F: dict, y: dict, draws: np.ndarray, budgets_idx: list[int], per_combo: tuple) -> dict:
    from flymario.retino_align import decode
    res = {}
    for task in ("orient", "lum"):
        for k in budgets_idx:
            accs, lams = [], []
            for r in range(draws.shape[0]):
                sel = draws[r, k][draws[r, k] >= 0]
                f = decode.fit_all_lambdas(F["pilot_train"][sel], y["pilot_train"][task][sel])
                va = (decode.scores(f, F["pilot_val"]).argmax(2) == y["pilot_val"][task]).mean(1)
                j = decode.select(va)
                accs.append(float((decode.scores(f, F["pilot_test"]).argmax(2)[j] == y["pilot_test"][task]).mean()))
                lams.append(decode.LAMBDAS[j])
            res[f"{task}_{4 * per_combo[k]}"] = {"test_acc": accs, "mean": float(np.mean(accs)), "lambda": lams}
    return res


def grid(a) -> int:
    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.retino_align import decode, retina, simulate, stimuli
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.step}.json"
    rec = json.loads(path.read_text()) if path.exists() else {
        "step": a.step, "started_utc": datetime.now(UTC).isoformat(timespec="seconds"), "results": {}}
    g0, _W, motor = graphsel.load("connectome")
    del _W
    vis = RetinotopicVision(g0)
    ret = retina.build(g0, vis)
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    per_combo = (2, 4, 8, 16, 32)
    budgets_idx = [0, 4]
    period = tuple(float(x) for x in a.period.split(","))
    for s in a.settings.split(","):
        gp = stimuli.GenParams(amp_range=AMPS[s], period_range=period)
        t = time.time()
        banks = {b: stimuli.draw_bank(b, {"pilot_train": 64, "pilot_val": 32, "pilot_test": 64}[b], sd, gp)
                 for b, sd in BANK_SEEDS.items()}
        X = {b: stimuli.encode_bank(bk, gp, vis, ret.inj, ret.col_uv)[0] for b, bk in banks.items()}
        y = {b: {"orient": bk.orient.astype(int), "lum": bk.lum.astype(int)} for b, bk in banks.items()}
        combo = banks["pilot_train"].orient.astype(int) * 4 + banks["pilot_train"].lum
        draws = decode.subsets(combo, DRAW_SEEDS, per_combo)
        t_enc = time.time() - t
        key = f"{s}"
        rec["results"].setdefault(key, {"gen": gp.as_dict(), "encode_s": round(t_enc, 1), "graphs": {}})
        rec["results"][key]["input"] = evaluate({b: X[b].astype(np.float64) for b in X}, y, draws, budgets_idx,
                                                per_combo)
        for gs in PILOT_GRAPHS:
            if gs in rec["results"][key]["graphs"]:
                continue
            _g, W, _m = graphsel.load(gs)
            net = simulate.FrozenNet(W, motor, ret.inj, p, int(config.get("timing.ticks_per_decision")), batch=128)
            t = time.time()
            F, act = {}, {}
            for b in banks:
                o = net.run(X[b], banks[b].uid, 0)
                F[b] = o.counts.astype(np.float64)
                act[b] = {"mean_count": float(o.counts.mean()), "frac_silent": float((o.counts == 0).mean()),
                          "frac_at_8": float((o.counts >= 8).mean()), "rate_all": float(o.rate_all.mean()),
                          "rate_input": float(o.rate_input.mean())}
            t_sim = time.time() - t
            res = evaluate(F, y, draws, budgets_idx, per_combo)
            rec["results"][key]["graphs"][gs] = {"activity": act["pilot_test"], "sim_s": round(t_sim, 1), **res}
            rec["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            save(path, rec)
            print(s, gs, json.dumps({k: round(v["mean"], 3) for k, v in res.items()}), flush=True)
        save(path, rec)
    rec["complete"] = True
    save(path, rec)
    return 0


def controls(a) -> int:
    """P2 (PILOT.md): negative label control with many shuffles (true / shuffled validation labels, fixed lambda),
    non-spatial input decoders (global descriptors, sorted values per type) against fictitious labels."""
    sys.path.insert(0, str(ROOT / "scripts/retino_align"))
    from run_study import decode_curve, global_descriptors

    from flymario.retino_align import decode
    d = ROOT / "artifacts/retino_align" / a.run
    inp = np.load(d / "inputs/inputs.npz")
    pz = np.load(d / "permutations/permutations.npz")
    subs = np.load(d / "inputs/subsets.npz")["subsets"]
    part = inp["part"]
    tr, va, te = part == 0, part == 1, part == 2
    per = (2, 4, 8, 16, 32)
    rec = {"run": a.run, "utc": datetime.now(UTC).isoformat(timespec="seconds"), "graph": a.graph, "cond": "ordered"}
    F = np.load(d / "features" / a.graph.replace(":", "") / "ordered_s1.npz")["counts"].astype(np.float64)
    shuf = {}
    for task in ("orient", "lum"):
        y = inp[task].astype(int)
        acc = {"selected_true_val": np.zeros((10, 5, a.reps)), "selected_shuffled_val": np.zeros((10, 5, a.reps)),
               "fixed_lambda": np.zeros((10, 5, a.reps, 6))}
        for r in range(10):
            for k in range(5):
                sel = subs[r, k][subs[r, k] >= 0]
                for rep in range(a.reps):
                    rng = np.random.default_rng([r, k, rep, 1 if task == "lum" else 0])
                    f = decode.fit_all_lambdas(F[tr][sel], y[tr][sel][rng.permutation(len(sel))])
                    te_acc = (decode.scores(f, F[te]).argmax(2) == y[te]).mean(1)
                    sva = decode.scores(f, F[va]).argmax(2)
                    acc["fixed_lambda"][r, k, rep] = te_acc
                    acc["selected_true_val"][r, k, rep] = te_acc[decode.select((sva == y[va]).mean(1))]
                    yv = y[va][rng.permutation(int(va.sum()))]
                    acc["selected_shuffled_val"][r, k, rep] = te_acc[decode.select((sva == yv).mean(1))]
        shuf[task] = {k_: {"mean_by_budget": v.mean(axis=tuple(i for i in range(v.ndim) if i != 1)).tolist(),
                           "sd_over_fits_by_budget": v.std(axis=tuple(i for i in range(v.ndim) if i != 1)).tolist(),
                           "n_fits_per_budget": int(v.size // 5)} for k_, v in acc.items()}
        shuf[task]["fixed_lambda_mean_by_lambda"] = acc["fixed_lambda"].mean(axis=(0, 1, 2)).tolist()
    rec["label_shuffle"] = shuf
    X = inp["X"].astype(np.float64)
    types = pz["types"]
    feats = {"global_descriptors": global_descriptors(X, types),
             "sorted_values_per_type": np.concatenate([np.sort(X[:, types == t], 1) for t in ("L1", "L2", "L3", "L5")], 1)}
    s1 = pz["perm1_sigma"]
    rec["sorted_values_identical_under_perm1"] = bool(np.array_equal(
        feats["sorted_values_per_type"],
        np.concatenate([np.sort(X[:, s1][:, types == t], 1) for t in ("L1", "L2", "L3", "L5")], 1)))
    ns = {}
    for name, G in feats.items():
        ns[name] = {}
        for task in ("orient", "lum"):
            y = inp[task].astype(int)
            ns[name][task] = decode_curve(G[tr], y[tr], G[va], y[va], G[te], y[te], subs, per)["mean_by_budget"]
            fict = []
            for rep in range(a.fict):
                rng = np.random.default_rng(100 + rep)
                yf = y.copy()
                for p in (0, 1, 2):
                    m = np.flatnonzero(part == p)
                    yf[m] = y[m][rng.permutation(len(m))]
                fict.append(decode_curve(G[tr], yf[tr], G[va], yf[va], G[te], yf[te], subs, per)["mean"])
            ns[name][f"{task}_fictitious_labels_mean_sd"] = [float(np.mean(fict)), float(np.std(fict)), len(fict)]
    rec["non_spatial"] = ns
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "P2_controls.json", rec)
    print(json.dumps({"shuffle_lum_selected_true_val": shuf["lum"]["selected_true_val"]["mean_by_budget"],
                      "shuffle_orient_selected_true_val": shuf["orient"]["selected_true_val"]["mean_by_budget"],
                      "non_spatial": ns}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid", help="decodability on the pilot graphs for generator settings")
    g.add_argument("--step", required=True)
    g.add_argument("--settings", default="A,B,C,D")
    g.add_argument("--period", default="24,40")
    c = sub.add_parser("controls", help="P2 negative and non-spatial controls on a pilot run's saved data")
    c.add_argument("--run", default="pilot_e2e")
    c.add_argument("--graph", default="nullin:100")
    c.add_argument("--reps", type=int, default=10)
    c.add_argument("--fict", type=int, default=8)
    a = ap.parse_args()
    return grid(a) if a.cmd == "grid" else controls(a)


if __name__ == "__main__":
    sys.exit(main())
