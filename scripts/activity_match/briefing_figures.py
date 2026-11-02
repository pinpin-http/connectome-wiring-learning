#!/usr/bin/env python
"""Owner briefing figures for ACTIVITY_MATCH_V1, in French and simplified (the owner's request: every check-in comes
with pictures). Built from saved files only.

    .venv/bin/python scripts/activity_match/briefing_figures.py freeze
    .venv/bin/python scripts/activity_match/briefing_figures.py final --run main_am

Writes artifacts/activity_match/briefing/<stage>/*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTS = ROOT / "artifacts/activity_match"
CONN, NULL, NULL_DARK, BAND = "#c2410c", "#9ca3af", "#4b5563", "#e5e7eb"
TCOL = {"CL": "#1d4ed8", "CM": "#047857", "CH": "#7c3aed"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, stage: str, name: str) -> None:
    out = ARTS / "briefing" / stage
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f3f4f6", ec="#374151", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#374151"))


def freeze_schema() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.6), ax.axis("off")
    ax.text(5, 4.35, "L'expérience en une image : le même cerveau, un seul bouton tourné", ha="center",
            fontsize=12, weight="bold")
    box(ax, 0.1, 2.3, 1.7, 1.2, "Image\n(rayures orientées\n+ luminosité)")
    box(ax, 2.3, 2.3, 2.3, 1.2, "Réseau figé\n(connectome ou\nun des 40 témoins)", fc="#fff7ed", ec=CONN)
    box(ax, 5.1, 2.3, 1.9, 1.2, "2 227 neurones\nde sortie\n(nombre de spikes)")
    box(ax, 7.5, 2.3, 2.3, 1.2, "Petit décodeur\nexterne appris\n→ « quelle orientation ? »", fc="#ecfdf5", ec="#047857")
    for x0, x1 in ((1.8, 2.3), (4.6, 5.1), (7.0, 7.5)):
        arrow(ax, x0, 2.9, x1, 2.9)
    ax.add_patch(plt.Circle((3.45, 1.5), 0.38, fc="#fde68a", ec="#92400e", lw=1.5))
    ax.text(3.45, 1.5, "γ", ha="center", va="center", fontsize=16, weight="bold")
    arrow(ax, 3.45, 1.88, 3.45, 2.3)
    ax.text(3.45, 0.7, "bouton « gain » : multiplie la force\nde toutes les synapses internes", ha="center",
            fontsize=9)
    txt = ("C0 : bouton à 1 pour tous (comme la V1)\n"
           "CL / CM / CH : bouton réglé graphe par graphe\n"
           "pour que tous aient la même activité de sortie\n"
           "(basse / moyenne / haute), sans regarder les réponses")
    ax.text(7.55, 1.25, txt, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f9fafb", ec="#d1d5db"))
    save(fig, "freeze", "01_schema_experience")


def freeze_calibration() -> None:
    cal = json.loads((ARTS / "calibration/calibration.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in cal["graphs"]:
        pts = cal["search"][s]["points"]
        g = np.array(sorted(float(k) for k in pts))
        r = np.array([pts[str(x)]["rate_hz"] for x in g])
        is_c = s == "connectome"
        ax.plot(g, r, "-", color=CONN if is_c else NULL, lw=2.5 if is_c else 0.7, alpha=1 if is_c else 0.55,
                zorder=3 if is_c else 1)
    ax.axhspan(cal["common_range"]["r_min"], cal["common_range"]["r_max"], color=BAND, zorder=0)
    for k, t in cal["targets_hz"].items():
        ax.axhline(t, color=TCOL[k], lw=1.2, ls="--")
        lab = {"CL": "cible basse", "CM": "cible moyenne (principale)", "CH": "cible haute"}[k]
        ax.text(0.021, t + 0.6, f"{k} : {lab}, {t:.1f} Hz", color=TCOL[k], fontsize=9)
    ax.axvline(1, color="k", lw=0.7, ls=":")
    ax.text(1.05, 2, "réglage V1\n(γ = 1)", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("bouton de gain γ (échelle log)")
    ax.set_ylabel("activité moyenne des sorties (spikes / s)")
    ax.set_title("Au réglage de la V1 (γ = 1), le connectome (orange) est moins actif que les 40 témoins (gris).\n"
                 "Pour une même activité, il faut baisser fortement le gain de tous, celui des témoins encore plus.",
                 fontsize=10)
    ax.plot([], [], color=CONN, lw=2.5, label="connectome")
    ax.plot([], [], color=NULL, lw=0.7, label="40 témoins")
    ax.fill_between([], [], color=BAND, label="zone d'activité commune")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "freeze", "02_calibration")


def freeze_check() -> None:
    cal = json.loads((ARTS / "calibration/calibration.json").read_text())
    tg = list(cal["targets_hz"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for ti, k in enumerate(tg):
        for s in cal["graphs"]:
            e = 100 * cal["check"][s][k]["rel_err"]
            is_c = s == "connectome"
            ax.plot(ti + (0.18 if is_c else np.random.default_rng(abs(hash(s)) % 2**32).uniform(-0.12, 0.12)), e,
                    "D" if is_c else "o", ms=7 if is_c else 3, color=CONN if is_c else NULL, zorder=3 if is_c else 1)
    ax.axhspan(0, 10, color="#dcfce7", zorder=0)
    ax.text(len(tg) - 0.6, 9.2, "tolérance : 10 %", fontsize=8, ha="right")
    ax.set_xticks(range(len(tg)), tg)
    ax.set_ylabel("écart à la cible sur des images neuves (%)")
    ax.set_title("Vérification sur 128 images jamais vues")
    ax = axes[1]
    gm = {k: [cal["search"][s]["selected"][k]["gamma"] for s in cal["graphs"] if s != "connectome"] for k in tg}
    for ti, k in enumerate(tg):
        ax.plot(ti + np.random.default_rng(ti).uniform(-0.12, 0.12, len(gm[k])), gm[k], "o", ms=3, color=NULL)
        ax.plot(ti + 0.18, cal["search"]["connectome"]["selected"][k]["gamma"], "D", ms=7, color=CONN)
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(len(tg)), tg)
    ax.set_ylabel("γ retenu (log)")
    ax.set_title("Réglage choisi pour chaque graphe")
    save(fig, "freeze", "03_verification")


def _iv(res, k):
    lo, hi = res["bootstrap"]["intervals_95"][k]
    return 100 * res["point"][k], 100 * lo, 100 * hi


def final_scores(run: str) -> None:
    res = json.loads((ARTS / run / "metrics/analysis.json").read_text())
    man = json.loads((ARTS / run / "manifest.json").read_text())
    conds, graphs = res["conditions"], res["graphs"]
    rate = {c: np.mean([res["activity_test"][g][c]["rate_hz"] for g in graphs if g != "connectome"]) for c in conds}
    rate_c = {c: res["activity_test"]["connectome"][c]["rate_hz"] for c in conds}
    name = {"C0": "C0\nréglage V1", "CL": "CL\nactivité basse", "CM": "CM\nactivité moyenne\n(principal)",
            "CH": "CH\nactivité haute"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for ci, c in enumerate(conds):
        nv = [100 * res["M"][g][c]["orient"] for g in graphs if g != "connectome"]
        ax.plot(ci + np.random.default_rng(ci).uniform(-0.13, 0.13, len(nv)), nv, "o", ms=3, color=NULL, zorder=1)
        ax.plot([ci - 0.2, ci + 0.2], [np.mean(nv)] * 2, "-", color=NULL_DARK, lw=2.5)
        ax.plot(ci + 0.3, 100 * res["M"]["connectome"][c]["orient"], "D", ms=8, color=CONN, zorder=3)
        ax.text(ci, 22.6, f"{rate_c[c]:.0f} / {rate[c]:.0f} Hz", ha="center",
                fontsize=7.5, color="#374151")
    ax.axhline(25, color="k", lw=0.7, ls=":")
    ax.text(len(conds) - 0.45, 25.2, "hasard", fontsize=8, ha="right")
    ax.axhline(30, color="#b45309", lw=0.9, ls="--")
    ax.text(len(conds) - 0.45, 30.2, "seuil « informatif » 30 %", fontsize=8, ha="right", color="#b45309")
    ax.set_xticks(range(len(conds)), [name[c] for c in conds], fontsize=8.5)
    ax.set_ylabel("orientation devinée correctement (%)")
    ax.set_ylim(22, None)
    ax.set_title("Score de chaque cerveau selon le réglage\n(en bas : activité connectome / témoins)", fontsize=10)
    ax.plot([], [], "D", color=CONN, label="connectome")
    ax.plot([], [], "o", color=NULL, ms=3, label="chaque témoin")
    ax.plot([], [], "-", color=NULL_DARK, lw=2.5, label="moyenne des témoins")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    delta = float(man["config"]["analysis"]["practical_margin"])
    ax.axhspan(-100 * delta, 100 * delta, color="#dcfce7", zorder=0)
    ax.text(len(conds) - 0.5, 100 * delta - 0.45, "zone « pas de différence\nqui compte » (±2 pts)", fontsize=7.5,
            ha="right", va="top")
    ax.axhline(0, color="k", lw=0.6)
    for ci, c in enumerate(conds):
        p, lo, hi = _iv(res, f"G_{c}_orient")
        col = "#047857" if c == "CM" else NULL_DARK
        ax.errorbar(ci, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, ms=7, capsize=5, lw=1.8)
        ax.text(ci + 0.12, p, f"{p:+.1f}", fontsize=8.5, va="center")
    ax.set_xticks(range(len(conds)), [c for c in conds])
    ax.set_ylabel("retard (−) ou avance (+) du connectome, en points")
    ax.set_title("Écart connectome − moyenne des témoins\n(barre = intervalle de confiance à 95 %)", fontsize=10)
    save(fig, "final", "05_resultat_principal")


def final_decomposition(run: str) -> None:
    res = json.loads((ARTS / run / "metrics/analysis.json").read_text())
    keys = ["T_real_orient", "T_nulls_orient", "R_orient"]
    labels = ["Connectome\n(CM − C0)", "Témoins, en moyenne\n(CM − C0)", "Écart rattrapé\nR = différence des deux"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0, color="k", lw=0.6)
    for i, (k, col) in enumerate(zip(keys, (CONN, NULL_DARK, "#1f2937"))):
        p, lo, hi = _iv(res, k)
        ax.bar(i, p, color=col, width=0.55, alpha=0.85)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", color="k", capsize=5, lw=1.2)
        ax.text(i + 0.32, p / 2, f"{p:+.1f} pts\n[{lo:+.1f} ; {hi:+.1f}]", ha="left", va="center", fontsize=8.5)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("changement du score d'orientation (points)")
    ax.set_title("Quand on égalise l'activité (CM au lieu de C0), qui bouge ?", fontsize=10, pad=12)
    ax.set_xlim(-0.5, 2.9)
    save(fig, "final", "06_qui_bouge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "final"])
    ap.add_argument("--run", default="main_am")
    a = ap.parse_args()
    if a.stage == "freeze":
        freeze_schema()
        freeze_calibration()
        freeze_check()
    else:
        final_scores(a.run)
        final_decomposition(a.run)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
