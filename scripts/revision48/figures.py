#!/usr/bin/env python
"""REVISION48 material for the owner's interpretation: figures and tables, descriptive only (owner's request,
2026-10-26: "produis-moi des graphiques et des stats ... pas besoin de faire l'analyse").

    figures.py   -> artifacts/revision48/analysis/figures/*.png, analysis/tables/*.csv

Reads the run directories, the sealed tuning selection and analysis/report.json (run analyse.py first). Nothing here
changes the declared reading; every number is a mean of the stored greedy predictions, or copied from report.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/revision48"
OUT = ART / "analysis"
FIG, TAB = OUT / "figures", OUT / "tables"
CONDS = ("reference", "tuned", "supervised", "lognorm", "autapse")
FR = {"reference": "référence", "tuned": "réglé (tuned)", "supervised": "supervisé", "lognorm": "poids log (lognorm)",
      "autapse": "nulls avec autapses"}
COL = {"connectome": "black", "S": "#2b6cb0", "E": "#dd6b20"}
CHANCE = 0.125


def run_dir(cond: str, spec: str) -> Path:
    d = ART / "runs/main" / ("reference" if (cond == "autapse" and spec == "connectome") else cond) / spec.replace(":", "")
    if (d / "reuse.json").exists():
        d = d.parents[1] / "reference" / d.name
    return d


def load(c: dict) -> dict:
    analysed = json.loads((ROOT / c["graphs"]["manifest"]).read_text())["order"]
    autapse = ["connectome"] + [s.replace("nulllat:", "nulllatL:").replace("nullin:", "nullinL:") for s in analysed[1:]]
    tasks, blocks, ks = c["tasks"], c["blocks"]["main"], c["checkpoints"]
    T, R = len(tasks), len(blocks)
    data = {}
    for cond in CONDS:
        specs = autapse if cond == "autapse" else analysed
        rows = []
        for s in specs:
            d = run_dir(cond, s)
            z = np.load(d / "predictions.npz")
            # learners in block-then-task order: index = r * T + t  ->  (T, R, ...)
            acc = np.stack([(z[f"progress_{k}"] == z["labels_progress"]).mean(-1) for k in ks], -1)  # (12, C)
            fin = (z["final"] == z["labels_final"]).mean(-1)
            ood = (z["ood"] == z["labels_ood"]).mean(-1)
            dg = np.load(d / "diagnostics.npz")
            rows.append({"spec": s, "dir": d,
                         "A": acc.reshape(R, T, -1).transpose(1, 0, 2), "final": fin.reshape(R, T).T,
                         "ood": ood.reshape(R, T).T, "diag": {k: dg[k] for k in dg.files if dg[k].ndim == 2},
                         "run": json.loads((d / "run.json").read_text()),
                         "reused": (ART / "runs/main" / cond / s.replace(":", "") / "reuse.json").exists()})
        data[cond] = rows
    return data


def family(spec: str) -> str:
    if spec == "connectome":
        return "connectome"
    return "S" if spec.split(":")[0] in ("nulllat", "nulllatL") else "E"


def wcsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    from flymario.revision48.analysis import trapezoid_weights
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    tasks, blocks, ks = c["tasks"], c["blocks"]["main"], c["checkpoints"]
    w = trapezoid_weights(ks)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data = load(c)
    rep = json.loads((OUT / "report.json").read_text()) if (OUT / "report.json").exists() else None
    for rows in data.values():
        for r in rows:
            r["L"] = r["A"] @ w                                    # (T, R)

    # ------------------------------------------------------------------ tables
    per_block, per_graph, ranks, meta = [], [], [], []
    for cond, rows in data.items():
        for r in rows:
            for ti, t in enumerate(tasks):
                for bi, b in enumerate(blocks):
                    per_block.append({"condition": cond, "graph": r["spec"], "family": family(r["spec"]), "task": t,
                                      "block": b, **{f"A_{k}": round(float(r["A"][ti, bi, ki]), 5) for ki, k in enumerate(ks)},
                                      "L": round(float(r["L"][ti, bi]), 5), "final": round(float(r["final"][ti, bi]), 5),
                                      "ood": round(float(r["ood"][ti, bi]), 5)})
                per_graph.append({"condition": cond, "graph": r["spec"], "family": family(r["spec"]), "task": t,
                                  "L": round(float(r["L"][ti].mean()), 5), "L_sd_blocks": round(float(r["L"][ti].std(ddof=1)), 5),
                                  "A0": round(float(r["A"][ti, :, 0].mean()), 5), "A_end": round(float(r["A"][ti, :, -1].mean()), 5),
                                  "final": round(float(r["final"][ti].mean()), 5), "ood": round(float(r["ood"][ti].mean()), 5)})
            rj = r["run"]
            meta.append({"condition": cond, "graph": r["spec"], "reused_reference": r["reused"],
                         "gamma": rj.get("gamma"), "gain": rj.get("gain"), "eta": rj.get("eta"), "tau": rj.get("tau"),
                         "wall_s": rj.get("wall_s"), "train_step_s_median": rj.get("train_step_s_median"),
                         "gpu_peak_mem_gb": rj.get("gpu_peak_mem_gb")})
        for t in tasks:
            pg = [x for x in per_graph if x["condition"] == cond and x["task"] == t]
            con = next(x for x in pg if x["family"] == "connectome")
            for f in ("S", "E"):
                nul = [x for x in pg if x["family"] == f]
                for m in ("L", "A0", "A_end", "final", "ood"):
                    v = np.array([x[m] for x in nul])
                    ranks.append({"condition": cond, "task": t, "family": f, "measure": m,
                                  "connectome": con[m], "null_mean": round(float(v.mean()), 5),
                                  "null_sd": round(float(v.std(ddof=1)), 5), "null_min": float(v.min()), "null_max": float(v.max()),
                                  "connectome_minus_null_mean": round(con[m] - float(v.mean()), 5),
                                  "n_nulls_at_or_above_connectome": int((v >= con[m]).sum()), "n_nulls": len(v),
                                  "rank_p_one_sided": round((int((v >= con[m]).sum()) + 1) / (len(v) + 1), 4)})
    wcsv(TAB / "per_run_block.csv", per_block)
    wcsv(TAB / "per_graph.csv", per_graph)
    wcsv(TAB / "connectome_vs_nulls.csv", ranks)
    wcsv(TAB / "run_meta.csv", meta)
    if rep:
        wcsv(TAB / "contrasts_G.csv", [{k: v for k, v in x.items()} for x in rep["contrasts"]])
        if rep["change_after_tuning"]:
            wcsv(TAB / "change_after_tuning_D.csv", rep["change_after_tuning"])
    sel = json.loads((ART / "tuning/selection.json").read_text())["selection"]
    trials = {t["trial"]: t for t in c["tuning"]["trials"]} if isinstance(c["tuning"]["trials"][0], dict) and "trial" in c["tuning"]["trials"][0] \
        else {i + 1: t for i, t in enumerate(c["tuning"]["trials"])}
    trows = []
    for g, v in sel.items():
        tr = trials.get(v["trial"], {})
        trows.append({"graph": g, "family": family(g), "selected_trial": v["trial"], "judgeable": v.get("judgeable"),
                      "gamma": tr.get("gamma"), "eta": tr.get("eta"), "tau": tr.get("tau"),
                      **{f"score_trial_{k}": round(float(s), 5) for k, s in sorted(v["scores"].items(), key=lambda x: int(x[0]))}})
    wcsv(TAB / "tuning_selection.csv", trows)

    # ------------------------------------------------------------------ F01 learning curves
    for cond, rows in data.items():
        fig, axs = plt.subplots(1, len(tasks), figsize=(12, 4.6), sharey=True)
        for ti, t in enumerate(tasks):
            ax = axs[ti]
            for f in ("S", "E"):
                fam = [r for r in rows if family(r["spec"]) == f]
                for r in fam:
                    ax.plot(ks, r["A"][ti].mean(0), color=COL[f], lw=0.6, alpha=0.35)
                ax.plot(ks, np.mean([r["A"][ti].mean(0) for r in fam], 0), color=COL[f], lw=2.2, ls="--",
                        label=f"moyenne famille {f} (10 nulls)")
            con = rows[0]
            ax.plot(ks, con["A"][ti].mean(0), color="black", lw=2.6, marker="o", ms=4, label="connectome")
            ax.axhline(CHANCE, color="grey", lw=0.8, ls=":", label="hasard 0,125")
            ax.set_title(f"{t}")
            ax.set_xlabel("décisions")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("précision greedy (banque de progrès, moyenne des 6 blocs)")
        axs[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"Courbes d'apprentissage — condition {FR[cond]}")
        fig.tight_layout()
        fig.savefig(FIG / f"F01_courbes_{cond}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ F02 G and D intervals
    if rep:
        rows = rep["contrasts"]
        fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.6))
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows, strict=True):
            col = COL[r["family"]]
            ax.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=col, lw=3 if r["primary"] else 1.8)
            ax.plot(100 * r["G"], yi, "o", color=col, ms=7 if r["primary"] else 5,
                    mfc=col if r["primary"] else "white")
        m = 100 * rep["margin"]
        ax.axvspan(-m, m, color="grey", alpha=0.12, label=f"marge ±{m:g} points")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{FR[r['condition']]} · {r['task']} · {r['family']} ({100 * r['level']:g} %)" for r in rows],
                           fontsize=8)
        ax.set_xlabel("G = L(connectome) − moyenne L(nulls de la famille), en points")
        ax.set_title("Écarts G et intervalles bootstrap (primaire : réglé, 98,75 %, traits épais)")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "F02_G_intervalles.png", dpi=150)
        plt.close(fig)
        if rep["change_after_tuning"]:
            rows = rep["change_after_tuning"]
            fig, ax = plt.subplots(figsize=(8, 3))
            for yi, r in enumerate(rows):
                ax.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=COL[r["family"]], lw=2.5)
                ax.plot(100 * r["D"], yi, "o", color=COL[r["family"]])
            ax.axvspan(-m, m, color="grey", alpha=0.12)
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{r['task']} · {r['family']}" for r in rows])
            ax.set_xlabel("D = G_réglé − G_référence, en points (95 %, apparié)")
            ax.set_title("Changement de l'écart après réglage à budget égal")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIG / "F02b_D_apres_reglage.png", dpi=150)
            plt.close(fig)

    # ------------------------------------------------------------------ F03 / F05 / F06 per-graph strips
    def strips(measure: str, fname: str, title: str, ylabel: str) -> None:
        fig, axs = plt.subplots(len(tasks), 1, figsize=(12, 7.5), sharex=True)
        for ti, t in enumerate(tasks):
            ax = axs[ti]
            for ci, cond in enumerate(CONDS):
                pg = [x for x in per_graph if x["condition"] == cond and x["task"] == t]
                for f, dx in (("S", -0.15), ("E", 0.15)):
                    v = [x[measure] for x in pg if x["family"] == f]
                    ax.scatter(np.full(len(v), ci + dx) + np.linspace(-0.05, 0.05, len(v)), v, s=16, color=COL[f],
                               alpha=0.75, label=f"nulls {f}" if ci == 0 else None)
                con = next(x for x in pg if x["family"] == "connectome")
                ax.scatter([ci], [con[measure]], marker="*", s=180, color="black", zorder=5,
                           label="connectome" if ci == 0 else None)
            if measure in ("A0", "A_end", "final", "ood", "L"):
                ax.axhline(CHANCE, color="grey", lw=0.8, ls=":")
            ax.set_ylabel(f"{t}\n{ylabel}")
            ax.grid(alpha=0.3)
        axs[-1].set_xticks(range(len(CONDS)))
        axs[-1].set_xticklabels([FR[x] for x in CONDS])
        axs[0].legend(fontsize=8, ncol=3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    strips("L", "F03_L_par_graphe.png", "L par graphe (moyenne des 6 blocs principaux)", "L (aire / horizon)")
    strips("final", "F05a_final_par_graphe.png", "Précision finale (banque finale, après 6 000 décisions)", "précision")
    strips("ood", "F05b_ood_par_graphe.png", "Précision hors distribution (banque ood, après 6 000 décisions)", "précision")
    strips("A0", "F06_A0_par_graphe.png", "Précision avant apprentissage A(0)", "précision")
    strips("A_end", "F06b_A6000_par_graphe.png", "Précision au dernier point de contrôle A(6000), banque de progrès", "précision")

    # ------------------------------------------------------------------ F04 per-block gap
    fig, axs = plt.subplots(len(tasks), len(CONDS), figsize=(16, 6), sharey=True)
    for ti, t in enumerate(tasks):
        for ci, cond in enumerate(CONDS):
            ax = axs[ti, ci]
            rows = data[cond]
            con = rows[0]["L"][ti]
            for f in ("S", "E"):
                fam = np.array([r["L"][ti] for r in rows if family(r["spec"]) == f])
                ax.plot(range(len(blocks)), 100 * (con - fam.mean(0)), "o-", color=COL[f], label=f"famille {f}")
            ax.axhline(0, color="grey", lw=0.8)
            ax.axhspan(-3, 3, color="grey", alpha=0.1)
            ax.set_xticks(range(len(blocks)))
            ax.set_xticklabels(blocks, fontsize=8)
            if ti == 0:
                ax.set_title(FR[cond], fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{t}\nécart par bloc (points)")
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Écart L(connectome) − moyenne L(famille), bloc par bloc")
    fig.tight_layout()
    fig.savefig(FIG / "F04_ecart_par_bloc.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ F07 tuning
    order = [r["graph"] for r in trows]
    S = np.array([[r[f"score_trial_{k}"] for k in range(1, 10)] for r in trows])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S, aspect="auto", cmap="viridis")
    for gi, r in enumerate(trows):
        ax.add_patch(plt.Rectangle((r["selected_trial"] - 1.5, gi - 0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{k}\nγ{trials[k]['gamma']}\nη{trials[k]['eta']}\nτ{trials[k]['tau']}" for k in range(1, 10)],
                       fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    fig.colorbar(im, ax=ax, label="aire de validation (blocs de réglage t0, t1)")
    ax.set_title("Réglage à budget égal : score de chaque essai (rouge = essai retenu)")
    fig.tight_layout()
    fig.savefig(FIG / "F07_reglage_scores.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for f, dx in (("connectome", 0), ("S", -0.25), ("E", 0.25)):
        cnt = [sum(1 for r in trows if r["family"] == f and r["selected_trial"] == k) for k in range(1, 10)]
        ax.bar(np.arange(1, 10) + dx, cnt, width=0.25, color=COL[f], label=f)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("essai retenu")
    ax.set_ylabel("nombre de graphes")
    ax.legend()
    ax.set_title("Essais retenus par famille")
    fig.tight_layout()
    fig.savefig(FIG / "F07b_essais_retenus.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ F08 paired scatter per graph
    pairs = [("reference", "tuned"), ("reference", "supervised"), ("reference", "lognorm"), ("reference", "autapse")]
    fig, axs = plt.subplots(len(tasks), len(pairs), figsize=(16, 7.5))
    for ti, t in enumerate(tasks):
        for pi, (a, b) in enumerate(pairs):
            ax = axs[ti, pi]
            xa = {x["graph"]: x["L"] for x in per_graph if x["condition"] == a and x["task"] == t}
            xb = [x for x in per_graph if x["condition"] == b and x["task"] == t]
            for x in xb:
                g = x["graph"].replace("nulllatL:", "nulllat:").replace("nullinL:", "nullin:")
                ax.scatter(xa[g], x["L"], color=COL[x["family"]], marker="*" if x["family"] == "connectome" else "o",
                           s=160 if x["family"] == "connectome" else 22, zorder=5 if x["family"] == "connectome" else 3)
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lim, lim, color="grey", lw=0.8, ls="--")
            ax.set_xlabel(f"L {FR[a]}", fontsize=8)
            ax.set_ylabel(f"L {FR[b]}", fontsize=8)
            ax.set_title(f"{t} : {FR[b]} vs {FR[a]}", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("L par graphe, une condition contre la référence (autapses : null apparié par graine ; ★ connectome)")
    fig.tight_layout()
    fig.savefig(FIG / "F08_appariements.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ F09 learning diagnostics
    keys = [k for k in ("reward", "entropy_pi", "max_pi", "active_out_frac", "v_norm", "upd_rms_post", "clipped_frac")
            if k in data["reference"][0]["diag"]]
    win = 200
    kern = np.ones(win) / win
    for cond, rows in data.items():
        fig, axs = plt.subplots(len(keys), len(tasks), figsize=(12, 2.1 * len(keys)), sharex=True)
        for ki, key in enumerate(keys):
            for ti, t in enumerate(tasks):
                ax = axs[ki, ti]
                cols = [r * len(tasks) + ti for r in range(len(blocks))]
                for f in ("S", "E", "connectome"):
                    fam = [r for r in rows if family(r["spec"]) == f]
                    y = np.mean([r["diag"][key][:, cols].mean(1) for r in fam], 0)
                    ax.plot(np.convolve(y, kern, mode="valid"), color=COL[f], lw=1.8 if f == "connectome" else 1.2,
                            label=f if (ki == 0 and ti == 0) else None)
                if ti == 0:
                    ax.set_ylabel(key, fontsize=8)
                if ki == 0:
                    ax.set_title(t)
                ax.grid(alpha=0.3)
        axs[-1, 0].set_xlabel(f"décision (moyenne glissante {win})")
        axs[-1, 1].set_xlabel(f"décision (moyenne glissante {win})")
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"Diagnostics d'apprentissage — {FR[cond]} (moyenne des 6 blocs ; familles : moyenne des 10 nulls)")
        fig.tight_layout()
        fig.savefig(FIG / f"F09_diagnostics_{cond}.png", dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------ F10 gap at each checkpoint
    fig, axs = plt.subplots(1, len(tasks), figsize=(12, 4.6), sharey=True)
    ccol = {"reference": "#4a5568", "tuned": "#2f855a", "supervised": "#805ad5", "lognorm": "#c53030", "autapse": "#b7791f"}
    for ti, t in enumerate(tasks):
        ax = axs[ti]
        for cond, rows in data.items():
            con = rows[0]["A"][ti].mean(0)
            for f, ls in (("S", "-"), ("E", "--")):
                fam = np.mean([r["A"][ti].mean(0) for r in rows if family(r["spec"]) == f], 0)
                ax.plot(ks, 100 * (con - fam), ls, marker="o", ms=3, color=ccol[cond],
                        label=f"{FR[cond]} · {f}" if ti == 0 else None)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhspan(-3, 3, color="grey", alpha=0.1)
        ax.set_title(t)
        ax.set_xlabel("décisions")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("A(connectome) − A moyen des nulls (points)")
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Écart de précision à chaque point de contrôle (trait plein : famille S ; tirets : famille E)")
    fig.tight_layout()
    fig.savefig(FIG / "F10_ecart_par_point_de_controle.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ F11 gain from tuning per graph
    fig, axs = plt.subplots(1, len(tasks), figsize=(12, 4.6), sharey=True)
    for ti, t in enumerate(tasks):
        ax = axs[ti]
        ref = {x["graph"]: x for x in per_graph if x["condition"] == "reference" and x["task"] == t}
        tun = [x for x in per_graph if x["condition"] == "tuned" and x["task"] == t]
        for xi, (f, dx) in enumerate((("connectome", 0), ("S", 1), ("E", 2))):
            v = [100 * (x["L"] - ref[x["graph"]]["L"]) for x in tun if x["family"] == f]
            ax.scatter(np.full(len(v), dx) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[f],
                       s=160 if f == "connectome" else 26, marker="*" if f == "connectome" else "o", zorder=4)
            if f != "connectome":
                ax.plot([dx - 0.25, dx + 0.25], [np.mean(v)] * 2, color=COL[f], lw=2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["connectome", "nulls S", "nulls E"])
        ax.set_title(t)
        ax.axhline(0, color="grey", lw=0.8)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("L(réglé) − L(référence), points")
    fig.suptitle("Gain apporté par le réglage à budget égal, graphe par graphe (trait : moyenne de la famille)")
    fig.tight_layout()
    fig.savefig(FIG / "F11_gain_du_reglage.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ F12 S vs E side by side
    if rep:
        fig, axs = plt.subplots(1, len(tasks), figsize=(12, 4.6), sharey=True)
        for ti, t in enumerate(tasks):
            ax = axs[ti]
            for ci, cond in enumerate(CONDS):
                for f, dx in (("S", -0.18), ("E", 0.18)):
                    r = next(x for x in rep["contrasts"] if x["condition"] == cond and x["task"] == t and x["family"] == f)
                    ax.bar(ci + dx, 100 * r["G"], width=0.34, color=COL[f], alpha=0.85, label=f"famille {f}" if ci == 0 else None)
                    ax.errorbar(ci + dx, 100 * r["G"], yerr=[[100 * (r["G"] - r["lo"])], [100 * (r["hi"] - r["G"])]],
                                color="black", capsize=3, lw=1)
            ax.axhspan(-3, 3, color="grey", alpha=0.12)
            ax.set_xticks(range(len(CONDS)))
            ax.set_xticklabels([FR[c] for c in CONDS], fontsize=8, rotation=15)
            ax.set_title(t)
            ax.grid(axis="y", alpha=0.3)
        axs[0].set_ylabel("G (points), intervalle de la lecture déclarée")
        axs[0].legend(fontsize=8)
        fig.suptitle("Écart G contre chaque famille : S (force sortante conservée) vs E (force entrante conservée)")
        fig.tight_layout()
        fig.savefig(FIG / "F12_familles_S_vs_E.png", dpi=150)
        plt.close(fig)
    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("*.png")),
                      "tables": sorted(p.name for p in TAB.glob("*.csv"))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
