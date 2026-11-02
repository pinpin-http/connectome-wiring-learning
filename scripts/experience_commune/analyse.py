#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 frozen analysis (mission §11-§13). Refuses to run while any planned run is incomplete.

    .venv/bin/python scripts/experience_commune/analyse.py [--replicates 10000]

Writes artifacts/experience_commune_v1/metrics/analysis.json, runs_table.csv (graph, family, task, mode, block,
checkpoint, accuracy), bootstrap.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"
CK = (0, 250, 500, 1000, 2000, 4000, 6000)


def main() -> int:
    from flymario.experience_commune import analysis as AN
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--status", default="campaign_status.json")
    ap.add_argument("--out", default="metrics")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/experience_commune_v1.yaml").read_text())
    st = json.loads((ART / a.status).read_text())
    graphs, blocks = st["graphs"], st["blocks"]
    tasks, modes = AN.TASKS, AN.MODES
    roles = np.array(["connectome" if g == "connectome" else ("b" if g.startswith("nulllat") else "c") for g in graphs])
    missing = [g for g in graphs if not (ART / a.runs / g.replace(":", "") / "predictions.npz").exists()]
    if missing:
        print("refused: incomplete runs", missing)
        return 2
    G, T, M, R, C = len(graphs), 2, 2, len(blocks), len(CK)
    prog = final = ood = None
    cls_p = cls_f = None
    uid_ref = None
    diag = {k: np.zeros((G, T, M, R)) for k in ("reward_rate", "entropy_first500", "entropy_last500", "maxpi_last500",
                                                "rho_mean", "rho_p99", "v_norm_mean", "upd_pre_mean", "upd_post_mean",
                                                "clipped_final", "active_out_frac")}
    act_freq = np.zeros((G, T, M, R, 8))
    for gi, g in enumerate(graphs):
        d = ART / a.runs / g.replace(":", "")
        run = json.loads((d / "run.json").read_text())
        z = np.load(d / "predictions.npz")
        dg = np.load(d / "diagnostics.npz")
        keys = [(l["task"], l["mode"], l["block"]) if l["kind"] == "main" else None for l in run["learners"]]
        nb = z["labels_progress"].shape[1]
        nf = z["labels_final"].shape[1]
        no = z["labels_ood"].shape[1] if "labels_ood" in z.files else 0
        if prog is None:
            prog = np.zeros((G, T, M, R, C, nb), np.uint8)
            final = np.zeros((G, T, M, R, nf), np.uint8)
            ood = np.zeros((G, T, M, R, max(no, 1)), np.uint8)
            cls_p = np.zeros((R, T, nb), np.int64)
            cls_f = np.zeros((R, T, nf), np.int64)
            uid_ref = {}
        for j, key in enumerate(keys):
            if key is None:
                continue
            t, m, b = key
            ti, mi, ri = tasks.index(t), modes.index(m), blocks.index(b)
            for ci, c in enumerate(CK):
                prog[gi, ti, mi, ri, ci] = z[f"progress_{c}"][j] == z["labels_progress"][j]
            final[gi, ti, mi, ri] = z["final"][j] == z["labels_final"][j]
            if no:
                ood[gi, ti, mi, ri] = z["ood"][j] == z["labels_ood"][j]
            cls_p[ri, ti], cls_f[ri, ti] = z["labels_progress"][j], z["labels_final"][j]
            u = (b, t)
            ids = (z["uid_progress"][j].tobytes(), z["uid_final"][j].tobytes())
            if u in uid_ref and uid_ref[u] != ids:
                raise RuntimeError(f"pairing violated: {g} {u} evaluated on different images")
            uid_ref[u] = ids
            diag["reward_rate"][gi, ti, mi, ri] = dg["reward"][:, j].mean()
            diag["entropy_first500"][gi, ti, mi, ri] = dg["entropy_pi"][:500, j].mean()
            diag["entropy_last500"][gi, ti, mi, ri] = dg["entropy_pi"][-500:, j].mean()
            diag["maxpi_last500"][gi, ti, mi, ri] = dg["max_pi"][-500:, j].mean()
            diag["rho_mean"][gi, ti, mi, ri] = dg["rho"][:, j].mean()
            diag["rho_p99"][gi, ti, mi, ri] = np.quantile(dg["rho"][:, j], 0.99)
            diag["v_norm_mean"][gi, ti, mi, ri] = dg["v_norm"][:, j].mean()
            diag["upd_pre_mean"][gi, ti, mi, ri] = dg["upd_rms_pre"][:, j].mean()
            diag["upd_post_mean"][gi, ti, mi, ri] = dg["upd_rms_post"][:, j].mean()
            diag["clipped_final"][gi, ti, mi, ri] = dg["clipped_frac"][-1, j]
            diag["active_out_frac"][gi, ti, mi, ri] = dg["active_out_frac"][:, j].mean()
            act_freq[gi, ti, mi, ri] = np.bincount(dg["action"][:, j].astype(int), minlength=8) / len(dg["action"])
    Ap, Af = prog.mean(-1), final.mean(-1)
    point = AN.stats(Ap, Af, roles)
    reps = int(a.replicates or cfg["analysis"]["bootstrap_replicates"])
    t0 = time.time()
    bs = AN.bootstrap(prog, final, cls_p, cls_f, roles, reps, int(cfg["analysis"]["bootstrap_seed"]))
    lv1, lv2 = float(cfg["analysis"]["primary_interval_confidence"]), float(cfg["analysis"]["secondary_interval_confidence"])
    delta = float(cfg["analysis"]["practical_margin"])
    thr = float(cfg["analysis"]["informative_final_accuracy_threshold"])
    iv95 = {k: AN.interval(v, lv2) for k, v in bs.items()}
    primary = {}
    for t in tasks:
        for f in AN.FAMILIES:
            k = f"L_G_{f}_{t}_U"
            lo, hi = AN.interval(bs[k], lv1)
            primary[f"{t}/{f}"] = {"G_U": point[k], "interval_98.75": [lo, hi], "reading": AN.reading(lo, hi, delta),
                                   "G_N": point[f"L_G_{f}_{t}_N"], "G_N_interval_95": iv95[f"L_G_{f}_{t}_N"],
                                   "I": point[f"L_I_{f}_{t}"], "I_interval_95": iv95[f"L_I_{f}_{t}"]}
    informative = {}
    for t in tasks:
        lows = {"connectome": iv95[f"final_connectome_{t}_U"][0],
                **{f: iv95[f"final_{f}_{t}_U"][0] for f in AN.FAMILIES}}
        informative[t] = {"lower_bounds_final_U_95": lows, "informative": all(v > thr for v in lows.values())}
    readings = [v["reading"] for v in primary.values()]
    res = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "plan": st["plan"], "graphs": graphs,
           "blocks": blocks, "roles": roles.tolist(), "point": point, "intervals_95": iv95, "primary": primary,
           "informative_final_U": informative, "coherent_across_four": len(set(readings)) == 1,
           "bootstrap": {"replicates": reps, "seed": int(cfg["analysis"]["bootstrap_seed"]),
                         "seconds": round(time.time() - t0, 1),
                         "scope": "blocks resampled (same draw for all graphs, modes, tasks); nulls resampled within "
                                  "each family as whole graphs; images resampled within the 8 classes per selected "
                                  "block and task, the same for every checkpoint, graph and mode"},
           "per_block": {}, "diagnostics": {}, "ood": {}}
    w = AN.trapezoid_weights()
    L = Ap @ w                                                       # (G, T, M, R)
    for ti, t in enumerate(tasks):
        for mi, m in enumerate(modes):
            res["per_block"][f"{t}_{m}"] = {
                "connectome": L[roles == "connectome", ti, mi].mean(0).tolist(),
                **{f: L[roles == f, ti, mi].mean(0).tolist() for f in AN.FAMILIES}}
            res["ood"][f"{t}_{m}"] = {r_: float(ood[roles == r_, ti, mi].mean()) for r_ in ("connectome", "b", "c")}
            res["diagnostics"][f"{t}_{m}"] = {k: {r_: float(v[roles == r_, ti, mi].mean()) for r_ in ("connectome", "b", "c")}
                                              for k, v in diag.items()}
    res["curves"] = {f"{t}_{m}_{r_}": Ap[roles == r_, ti, mi].mean(axis=(0, 1)).tolist()
                     for ti, t in enumerate(tasks) for mi, m in enumerate(modes) for r_ in ("connectome", "b", "c")}
    res["curves_per_graph"] = Ap.mean(3).tolist()                    # (G, T, M, C)
    res["action_frequencies"] = {f"{t}_{m}": {r_: act_freq[roles == r_, ti, mi].mean(axis=(0, 1)).tolist()
                                              for r_ in ("connectome", "b", "c")}
                                 for ti, t in enumerate(tasks) for mi, m in enumerate(modes)}
    res["E1_N_descriptive"] = {t: {r_: float(diag["reward_rate"][roles == r_, ti, 0].mean())
                                   for r_ in ("connectome", "b", "c")} for ti, t in enumerate(tasks)}
    res["U_journal_reward_rate"] = {t: float(diag["reward_rate"][:, ti, 1].mean()) for ti, t in enumerate(tasks)}
    out = ART / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "bootstrap.npz", **bs)
    with open(out / "runs_table.csv", "w") as f:
        f.write("graph,family,task,mode,block,checkpoint,accuracy,final_accuracy\n")
        for gi, g in enumerate(graphs):
            for ti, t in enumerate(tasks):
                for mi, m in enumerate(modes):
                    for ri, b in enumerate(blocks):
                        for ci, c in enumerate(CK):
                            f.write(f"{g},{roles[gi]},{t},{m},{b},{c},{Ap[gi, ti, mi, ri, ci]:.6f},"
                                    f"{Af[gi, ti, mi, ri]:.6f}\n")
    print(json.dumps({"primary": primary, "informative": informative}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
