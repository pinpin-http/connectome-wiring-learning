#!/usr/bin/env python
"""Owner briefing figures for EXPERIENCE_COMMUNE_V1, in French and simplified (one final report only, mission §3).

    .venv/bin/python scripts/experience_commune/briefing_figures.py schema
    .venv/bin/python scripts/experience_commune/briefing_figures.py final      # after the complete campaign only
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
ART = ROOT / "artifacts/experience_commune_v1"
CONN, B_COL, C_COL = "#c2410c", "#6b7280", "#2563eb"
CK = np.array([0, 250, 500, 1000, 2000, 4000, 6000])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name: str) -> None:
    out = ART / "briefing"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f3f4f6", ec="#374151", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#374151"))


def schema() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.set_xlim(0, 11), ax.set_ylim(-0.4, 5.4), ax.axis("off")
    ax.text(5.5, 5.15, "Deux façons d'apprendre la même tâche", ha="center", fontsize=12.5, weight="bold")
    for row, (y, mode, who, note, col) in enumerate((
            (3.3, "N : politique propre", "le cerveau choisit\nsa réponse lui-même",
             "chaque cerveau vit une\nhistoire différente", "#fff7ed"),
            (1.1, "U : expérience commune", "un tirage au sort externe\nimpose la réponse",
             "tous les cerveaux vivent\nexactement la même histoire", "#ecfdf5"))):
        ax.text(0.15, y + 1.35, mode, fontsize=11, weight="bold")
        box(ax, 0.1, y, 1.8, 1.1, "image\n(« Où ? » ou\n« Approche/recul ? »)")
        box(ax, 2.3, y, 2.3, 1.1, "cerveau\n(connectome ou témoin)\nqui apprend", fc=col, fs=8.5)
        box(ax, 4.9, y, 2.2, 1.1, who)
        box(ax, 7.6, y, 1.6, 1.1, "récompense\n1 si bonne\nréponse")
        for x0, x1 in ((1.9, 2.3), (4.6, 4.9), (7.1, 7.6)):
            arrow(ax, x0, y + 0.55, x1, y + 0.55)
        ax.add_patch(FancyArrowPatch((8.4, y - 0.02), (3.45, y - 0.02), connectionstyle="arc3,rad=-0.25",
                                     arrowstyle="-|>", mutation_scale=12, lw=1, color="#6b7280", ls="--"))
        ax.text(5.9, y - 0.82, "le cerveau apprend de la récompense", fontsize=8, color="#6b7280", ha="center")
        ax.text(9.4, y + 0.55, note, fontsize=8.5, va="center")
    ax.text(5.5, -0.35, "Dans les deux cas : on arrête régulièrement l'apprentissage et on fait passer un examen "
            "(le cerveau répond seul, sans tirage au sort).", ha="center", fontsize=9, style="italic")
    save(fig, "01_schema_N_U")


def final() -> None:
    a = json.loads((ART / "metrics/analysis.json").read_text())
    cur = 100 * np.array(a["curves_per_graph"])
    roles = np.array(a["roles"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    fig.subplots_adjust(wspace=0.28)
    for ti, (t, name) in enumerate((("where", "« Où ? »"), ("loom", "« Approche ou recul ? »"))):
        ax = axes[ti]
        for mi, (m, ls) in enumerate((("N", "--"), ("U", "-"))):
            for r, col in (("b", B_COL), ("c", C_COL)):
                ax.plot(CK, cur[roles == r, ti, mi].mean(0), ls, color=col, lw=1.6)
            ax.plot(CK, cur[roles == "connectome", ti, mi][0], ls, color=CONN, lw=2.4)
        ax.axhline(12.5, color="k", lw=0.5, ls=":")
        ax.set_title(f"{name} : examens au fil de l'apprentissage", fontsize=9.5)
        ax.set_xlabel("nombre d'essais d'apprentissage")
        if ti == 0:
            ax.set_ylabel("bonnes réponses à l'examen (%)")
    axes[0].plot([], [], "-", color=CONN, lw=2.4, label="connectome")
    axes[0].plot([], [], "-", color=B_COL, label="témoins b (moyenne)")
    axes[0].plot([], [], "-", color=C_COL, label="témoins c (moyenne)")
    axes[0].plot([], [], "--", color="k", label="pointillés : N (politique propre)")
    axes[0].plot([], [], "-", color="k", label="trait plein : U (expérience commune)")
    axes[0].legend(frameon=False, fontsize=7.5)
    ax = axes[2]
    ax.axhspan(-3, 3, color="#dcfce7", zorder=0)
    ax.axhline(0, color="k", lw=0.6)
    labels = []
    for i, (k, v) in enumerate(a["primary"].items()):
        t, f = k.split("/")
        for dx, key, ivk, col in ((-0.15, "G_U", "interval_98.75", "#047857"), (0.15, "G_N", "G_N_interval_95", "#9ca3af")):
            p = 100 * v[key]
            lo, hi = (100 * x for x in v[ivk])
            ax.errorbar(i + dx, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=4, ms=6)
            ax.text(i + dx + (0.07 if dx > 0 else -0.07), p, f"{p:+.1f}", fontsize=8, va="center",
                    ha="left" if dx > 0 else "right")
        labels.append(f"{'Où ?' if t == 'where' else 'Approche/recul'}\ncontre témoins {f}")
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_ylabel("avance du connectome (points)")
    ax.set_title("Avance du connectome : vert = U (principal),\ngris = N (référence) ; zone verte = ±3 points",
                 fontsize=10)
    save(fig, "02_resultat")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["schema", "final"])
    a = ap.parse_args()
    schema() if a.stage == "schema" else final()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
