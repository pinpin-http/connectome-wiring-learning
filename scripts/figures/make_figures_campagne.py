#!/usr/bin/env python
"""Paper figures other than visual tasks and play (see make_figures_taches.py).

    python -I scripts/figures/make_figures_campagne.py [--out <dir>]   # inputs read from the repository (make figures)
    python -I scripts/figures/make_figures_campagne.py --campagne <campaign folder> --complements <extracted folder>

The second form reads the same files from the owner's unpacked backups (paths in main()).

Forest-plot contrasts and schematic-figure numbers (nulls, weight transform) are copied from the frozen analysis
reports, not recomputed; each value is commented. Style: the connectome is the only accented series (blue), nulls
are context (light grey for S nulls, dark grey for E nulls).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# Figures go to artifacts/figures/ (not in git); --out <dir> writes them elsewhere.
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else REPO / "artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY_S, GREY_E = "#c9c7c0", "#7d7b76"
NB = " "
TASKS = {"where": f"«{NB}Où{NB}?{NB}»", "loom": f"«{NB}Approche ou recul{NB}?{NB}»"}
CHECKPOINTS = [0, 250, 500, 1000, 2000, 4000, 6000]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def fr(v: float, nd: int = 1, signed: bool = False) -> str:
    """French number format: decimal comma, true minus sign, explicit plus if requested."""
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "−").replace(".", ",")


def comma_axis(ax, axis: str = "y", nd: int = 0) -> None:
    fmt = FuncFormatter(lambda v, _: fr(v, nd))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands_axis(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def family(graph: str) -> str:
    if graph == "connectome":
        return "connectome"
    return "S" if graph.startswith("nulllat") else "E"


def legend_below(fig, handles, ncol, y=-0.06, rect_bottom=0.07) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))
    fig.tight_layout(rect=(0, rect_bottom, 1, 1))


# ------------------------------------------------------------------------------- nulls and instrument (methods)
def fig_temoins() -> None:
    # Values measured on the graphs (null analysis documents): direct eye -> output edges, share of edges between
    # same-side neurons, neurons reached within two connections from an eye.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4))
    names = ["connectome", "recâblage\nà degrés\nconservés", "stratifié", "stratifié,\nlatéralisé"]
    cols = [BLUE, ORANGE, GREY_E, GREY_S]
    panels = [
        ("Arêtes directes œil → sortie", [69, 11659, None, None], "{:,.0f}"),
        ("Arêtes entre neurones\ndu même côté (%)", [78.87, None, 46.8, 78.87], "{:.1f}"),
        ("Neurones atteints par un œil\nen deux connexions", [30948, None, 52419, 30948], "{:,.0f}"),
    ]
    for ax, (title, vals, fmt) in zip(axes, panels):
        xs = [i for i, v in enumerate(vals) if v is not None]
        ax.bar(xs, [vals[i] for i in xs], color=[cols[i] for i in xs], width=0.62, zorder=3)
        for i in xs:
            ax.text(i, vals[i] * 1.02, fmt.format(vals[i]).replace(",", " ").replace(".", ","), ha="center",
                    va="bottom", fontsize=6.8, color=INK2)
        ax.set_xticks(xs, [names[i] for i in xs], fontsize=6.6)
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_ylim(0, max(v for v in vals if v) * 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_temoins.pdf")
    plt.close(fig)


def fig_transformation() -> None:
    # Decodability gap connectome - nulls (4 nulls x 10 seeds x 10 scenes), two seed sets.
    trans = ["log_norm", "lin_norm", "log_raw", "lin_raw"]
    labels = ["logarithme\nnormalisé", "linéaire\nnormalisé", "logarithme\nbrut", "nombre brut\n(retenu)"]
    a = [0.175, -0.170, -0.175, -0.138]
    b = [0.210, -0.117, -0.205, -0.140]
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    x = np.arange(4)
    ax.bar(x - 0.18, a, width=0.34, color=[BLUE if v > 0 else GREY_E for v in a], zorder=3,
           label="graines 1 à 10")
    ax.bar(x + 0.18, b, width=0.34, color=[BLUE if v > 0 else GREY_E for v in b], alpha=0.55, zorder=3,
           label="graines 11 à 20")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("décodabilité :\nconnectome − témoins")
    comma_axis(ax, "y", 2)
    grid(ax)
    ax.text(3.45, 0.2, "barres pleines : graines 1 à 10\nbarres claires : graines 11 à 20", ha="right",
            fontsize=6.6, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig_transformation.pdf")
    plt.close(fig)


def fig_balayage(confirm_path: Path) -> None:
    c = json.loads(confirm_path.read_text())
    groups = [("règle locale", [x["acc"] for x in c["criterion"]["cells"]], BLUE),
              ("poids figés", [x["frozen"] for x in c["criterion"]["cells"]], GREY_E),
              ("étiquettes\nmélangées", [x["shuffled"] for x in c["criterion"]["cells"]], GREY_S),
              ("apprentissage\nsupervisé", [x["acc"] for x in c["criterion_supervised_reference"]["cells"]], ORANGE)]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    rng = np.random.default_rng(3)
    for i, (name, vals, col) in enumerate(groups):
        v = 100 * np.array(vals)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=col, zorder=3, linewidths=0)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK2, lw=1.2, zorder=4)
        ax.text(i + 0.28, v.mean(), fr(round(v.mean() + 1e-6, 1), 1), va="center", fontsize=6.8, color=INK2)
    ax.axhline(75, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(3.45, 76, "seuil de réussite", ha="right", fontsize=6.6, color=MUTED)
    ax.axhline(50, color=AXIS, lw=0.8)
    ax.set_xticks(range(4), [g[0] for g in groups])
    ax.set_ylabel("précision sur images réservées (%)")
    ax.set_ylim(25, 105)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_balayage.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- structural activity
def fig_regime(regime_path: Path) -> None:
    d = json.loads(regime_path.read_text())
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for g, v in d.items():
        xs = sorted((float(k) for k in v if float(k) >= 0.01), key=float)
        ys = [v[str(x) if str(x) in v else f"{x:g}"] for x in xs]
        if g == "connectome":
            ax.plot(xs, ys, color=BLUE, lw=1.8, marker="D", ms=3.5, zorder=4, label="connectome")
        else:
            ax.plot(xs, ys, color=GREY_E, lw=1.0, marker="o", ms=2.5, zorder=3,
                    label="témoins S" if g.endswith(":0") else None)
    ax.set_xscale("log")
    ax.axvline(1, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.06, 300, "réglage\ncommun", fontsize=6.6, color=MUTED)
    ax.set_xlabel("facteur multiplicatif du gain global (échelle logarithmique)")
    ax.set_ylabel("impulsions de sortie\npar décision")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 2 if v < 0.1 else 1 if v < 1 else 0)))
    thousands_axis(ax)
    grid(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_regime.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- play: button hold duration
def fig_appui(ceiling_path: Path) -> None:
    rows = json.loads(ceiling_path.read_text())
    pol = [("always_run", "toujours courir", GREY_S), ("label", "professeur,\nappui d'une décision", GREY_E),
           ("label_hold2", "appui prolongé\n× 2", BLUE), ("label_hold4", "× 4", "#5b9be3"),
           ("label_hold6", "× 6", "#9cc2ee")]
    levels = ["Level1-1", "Level4-1", "Level8-1"]
    val = {(r["level"], r["policy"]): r["furthest_x"] for r in rows}
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    w = 0.16
    for j, (p, name, col) in enumerate(pol):
        xs = np.arange(3) + (j - 2) * w
        ax.bar(xs, [val[(lv, p)] for lv in levels], width=w * 0.92, color=col, zorder=3, label=name.replace("\n", " "))
    ax.set_xticks(range(3), ["niveau 1-1", "niveau 4-1", "niveau 8-1"])
    ax.set_ylabel("distance maximale (pixels)")
    thousands_axis(ax)
    grid(ax)
    ax.legend(loc="upper right", fontsize=6.8, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_appui.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- permutation of the eye map
def fig_stimuli(png_path: Path) -> None:
    from PIL import Image
    im = np.asarray(Image.open(png_path).convert("RGB"))
    thumbs = [(14, 82, 243, 296), (305, 82, 535, 296), (596, 82, 826, 296), (888, 82, 1118, 296)]
    hexes = [(16, 420, 243, 686), (307, 420, 535, 686)]
    fig = plt.figure(figsize=(6.3, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.28, wspace=0.08)
    for i, (x0, y0, x1, y1) in enumerate(thumbs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im[y0:y1, x0:x1])
        ax.set_title(f"orientation {45 * i}°", fontsize=7.5)
        ax.axis("off")
    titles = ["entrée ON par colonne,\ncarte respectée", "même image,\ncolonnes permutées"]
    for i, (x0, y0, x1, y1) in enumerate(hexes):
        ax = fig.add_subplot(gs[1, 1 + i])
        ax.imshow(im[y0:y1, x0:x1])
        ax.set_title(titles[i], fontsize=7.5)
        ax.axis("off")
    fig.savefig(OUT / "fig_stimuli.pdf")
    plt.close(fig)


def fig_retino(path: Path) -> None:
    m = json.loads(path.read_text())["M"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for ax, key, title in [(axes[0], "orient", "Orientation (4 classes)"), (axes[1], "lum", "Luminosité (4 classes)")]:
        for g, v in m.items():
            if g == "connectome":
                continue
            o = 100 * v["ordered"][key]
            p = 100 * np.mean([v[f"perm{i}"][key] for i in (1, 2, 3)])
            ax.plot([0, 1], [o, p], color=GREY_E, lw=0.6, alpha=0.55, zorder=2)
            ax.scatter([0, 1], [o, p], s=7, color=GREY_E, alpha=0.7, zorder=3, linewidths=0)
        v = m["connectome"]
        o = 100 * v["ordered"][key]
        p = 100 * np.mean([v[f"perm{i}"][key] for i in (1, 2, 3)])
        ax.plot([0, 1], [o, p], color=BLUE, lw=1.8, zorder=4)
        ax.scatter([0, 1], [o, p], s=26, marker="D", color=BLUE, zorder=5, linewidths=0)
        ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks([0, 1], ["carte de l'œil\nrespectée", "colonnes\npermutées"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, loc="left")
        ax.set_ylabel("précision du décodeur (%)")
        comma_axis(ax)
        grid(ax)
    axes[0].text(1.05, 25.6, "hasard", color=MUTED, fontsize=7)
    handles = [Line2D([], [], color=BLUE, marker="D", lw=1.8, ms=5, label="connectome"),
               Line2D([], [], color=GREY_E, marker="o", lw=0.6, ms=3, label="40 témoins E")]
    legend_below(fig, handles, 2)
    fig.savefig(OUT / "fig_retino.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- matched activity, frozen network
def fig_calibration(cal_path: Path) -> None:
    c = json.loads(cal_path.read_text())
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    for g in c["graphs"]:
        pts = {}
        for src in (c["grid"].get(g, {}), c["search"].get(g, {}).get("points", {})):
            for k, v in src.items():
                pts[float(k)] = v["rate_hz"]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        if g == "connectome":
            ax.plot(xs, ys, color=BLUE, lw=1.8, zorder=4)
        elif g.startswith("nullin:") and int(g.split(":")[1]) < 40:
            ax.plot(xs, ys, color=GREY_E, lw=0.6, alpha=0.6, zorder=3)
    for name, hz in c["targets_hz"].items():
        ax.axhline(hz, color=ORANGE, lw=0.8, ls=(0, (3, 3)), zorder=2)
        ax.text(0.021, hz + 0.8, {"CL": "cible basse", "CM": "cible médiane", "CH": "cible haute"}[name],
                fontsize=6.5, color=ORANGE)
    ax.set_xscale("log")
    ax.set_xlabel("facteur multiplicatif du gain global $\\gamma$ (échelle logarithmique)")
    ax.set_ylabel("activité de sortie (Hz)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 2 if v < 0.1 else 1 if v < 1 else 0)))
    grid(ax)
    handles = [Line2D([], [], color=BLUE, lw=1.8, label="connectome"),
               Line2D([], [], color=GREY_E, lw=0.8, label="40 témoins E")]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.pdf")
    plt.close(fig)


def fig_activite(path: Path) -> None:
    rows = json.loads(path.read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharex=True)
    done = set()
    for ax, col, title in [(axes[0], 4, "Orientation"), (axes[1], 5, "Luminosité")]:
        for exp, g, cond, hz, ori, lum in rows:
            fam = family(g)
            if fam == "connectome":
                continue
            ax.scatter(hz, 100 * (ori if col == 4 else lum), s=6, color=GREY_S if fam == "S" else GREY_E,
                       alpha=0.75, linewidths=0, zorder=2)
        conn = {}
        for exp, g, cond, hz, ori, lum in rows:
            if g == "connectome":
                conn[(round(hz, 1), cond)] = (hz, 100 * (ori if col == 4 else lum))
        for (_, cond), (hz, v) in sorted(conn.items()):
            ax.scatter(hz, v, s=30, marker="D", color=BLUE, zorder=4, linewidths=0)
            names = {"CL": "basse", "CM": "médiane", "CH": "haute"}
            if (col, cond) not in done and cond in names:
                ax.annotate(names[cond], (hz, v), xytext=(0, -9 if col == 4 else 6), textcoords="offset points",
                            ha="center", fontsize=6.3, color=BLUE)
                done.add((col, cond))
        if col == 4:
            ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.text(60, 25.4, "hasard", color=MUTED, fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("activité moyenne des neurones de sortie (Hz)")
        ax.set_ylabel("précision du décodeur (%)")
        comma_axis(ax)
        grid(ax)
    axes[0].annotate("réglage commun", xy=(38.7, 29.3), xytext=(19, 33.2), fontsize=6.5, color=BLUE,
                     arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    handles = [Line2D([], [], color=BLUE, marker="D", ls="", ms=5, label="connectome (toutes conditions)"),
               Line2D([], [], color=GREY_E, marker="o", ls="", ms=3, label="témoins E"),
               Line2D([], [], color=GREY_S, marker="o", ls="", ms=3, label="témoins S")]
    legend_below(fig, handles, 3, y=-0.07)
    fig.savefig(OUT / "fig_activite.pdf")
    plt.close(fig)


def fig_synthese() -> None:
    # Contrasts "connectome - mean of nulls" in orientation decoding, frozen network (points, 95 % CI).
    rows = [
        ("Permutation : écart initial, réglage commun (E)", -6.0, -6.6, -5.5, None),
        ("Permutation : perte différentielle (E)", -0.65, -1.26, -0.05, 5),
        ("Gain abaissé pour tous : réglage commun (E)", -5.5, -6.0, -4.9, None),
        ("Gain abaissé pour tous : 25,9 Hz (E)", -1.8, -2.2, -1.3, 2),
        ("Témoins seuls ajustés : réglage commun (E)", -5.4, -6.0, -4.9, None),
        ("Témoins seuls ajustés : activité égale (E)", -0.7, -1.2, -0.1, 2),
        ("Témoins seuls ajustés : réglage commun (S)", -7.9, -8.6, -7.2, None),
        ("Témoins seuls ajustés : activité égale (S)", 1.6, 1.1, 2.1, 2),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for i, (label, est, lo, hi, margin) in enumerate(rows):
        y = -i
        if margin:
            ax.fill_between([-margin, margin], y - 0.38, y + 0.38, color=GRID, alpha=0.8, zorder=0, linewidths=0)
        color = BLUE if margin else GREY_E
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, zorder=3)
        ax.scatter(est, y, color=color, s=24, marker="D" if margin else "o", zorder=4, linewidths=0)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows])
    ax.set_xlabel("écart de précision, connectome − moyenne des témoins (points ; IC à 95 %)")
    ax.set_xlim(-9.5, 5.5)
    grid(ax, "x")
    comma_axis(ax, "x")
    fig.tight_layout()
    fig.savefig(OUT / "fig_synthese.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- common experiment
def load_runs(path: Path) -> dict:
    acc = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            acc[(r["graph"], r["task"], r["mode"])].setdefault(r["block"], {})[int(r["checkpoint"])] = float(r["accuracy"])
    return {k: np.array([[b[c] for c in CHECKPOINTS] for _, b in sorted(blocks.items())]) for k, blocks in acc.items()}


def area(curve: np.ndarray) -> float:
    return 100 * np.trapezoid(curve, CHECKPOINTS) / CHECKPOINTS[-1]


def fig_ec_courbes(runs: dict) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True, sharey=True)
    L = {}
    for i, task in enumerate(["where", "loom"]):
        for j, (mode, mname) in enumerate([("N", "actions choisies par le réseau"), ("U", "actions imposées")]):
            ax = axes[i, j]
            for fam, color in [("S", GREY_S), ("E", GREY_E)]:
                curves = np.array([100 * v.mean(0) for (g, t, m), v in runs.items()
                                   if t == task and m == mode and family(g) == fam])
                lo, hi = np.percentile(curves, [5, 95], axis=0)
                ax.fill_between(CHECKPOINTS, lo, hi, color=color, alpha=0.35, linewidths=0, zorder=2)
                ax.plot(CHECKPOINTS, np.median(curves, 0), color=color if fam == "E" else "#a9a7a0", lw=1.1, zorder=3)
            ax.plot(CHECKPOINTS, 100 * runs[("connectome", task, mode)].mean(0), color=BLUE, lw=1.8, marker="o",
                    ms=2.5, zorder=4)
            ax.axhline(12.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.set_title(f"{TASKS[task]}, {mname}", loc="left", fontsize=8)
            grid(ax)
            comma_axis(ax)
            for (g, t, m), v in runs.items():
                if t == task and m == mode:
                    L[(g, t, m)] = float(np.mean([area(b) for b in v]))
    for ax in axes[1]:
        ax.set_xlabel("décisions d'entraînement")
        thousands_axis(ax, "x")
    for ax in axes[:, 0]:
        ax.set_ylabel("précision aux examens (%)")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.8, ms=3, label="connectome (moyenne de 6 blocs)"),
               plt.Rectangle((0, 0), 1, 1, color=GREY_S, alpha=0.6, label="20 témoins S (médiane, 5e-95e centiles)"),
               plt.Rectangle((0, 0), 1, 1, color=GREY_E, alpha=0.6, label="20 témoins E")]
    legend_below(fig, handles, 3, y=-0.04, rect_bottom=0.04)
    fig.savefig(OUT / "fig_ec_courbes.pdf")
    plt.close(fig)
    return L


def fig_ec_contrastes() -> None:
    # Values published by the frozen analysis (chosen / imposed actions) and by the matched-activity analysis.
    rows = [
        ("« Où ? », témoins S", [("N", 19.0, 14.0, 23.7), ("U", 23.2, 17.2, 29.8), ("X", 11.2, 6.3, 16.2)]),
        ("« Où ? », témoins E", [("N", 13.3, 9.7, 16.8), ("U", 15.3, 10.2, 21.0), ("X", 14.6, 10.7, 18.6)]),
        ("« Approche ou recul ? », témoins S",
         [("N", 18.8, 14.3, 23.1), ("U", 18.8, 9.4, 29.3), ("X", 9.9, 3.5, 17.2)]),
        ("« Approche ou recul ? », témoins E",
         [("N", 17.0, 13.5, 20.6), ("U", 15.1, 7.5, 23.4), ("X", 15.0, 10.0, 20.2)]),
    ]
    styles = {"N": (GREY_E, "o", "actions choisies (IC à 95 %)"), "U": (BLUE, "D", "actions imposées (IC à 98,75 %)"),
              "X": (ORANGE, "s", "actions imposées, activité égalisée (IC à 95 %)")}
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y, ticks, labels = 0, [], []
    for label, items in rows:
        for k, (name, est, lo, hi) in enumerate(items):
            yy = y - 0.27 * (k - 1)
            color, marker, _ = styles[name]
            ax.plot([lo, hi], [yy, yy], color=color, lw=1.4, zorder=3)
            ax.scatter(est, yy, color=color, marker=marker, s=24, zorder=4, linewidths=0)
        ticks.append(y)
        labels.append(label)
        y -= 1.25
    ax.axvspan(-3, 3, color=GRID, alpha=0.6, zorder=0)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("écart d'aire d'apprentissage, connectome − moyenne des témoins (points)")
    ax.set_xlim(-5, 32)
    ax.set_ylim(ticks[-1] - 0.6, ticks[0] + 0.8)
    ax.text(0, ticks[0] + 0.55, "marge ±3", ha="center", fontsize=6.5, color=MUTED)
    grid(ax, "x")
    comma_axis(ax, "x")
    handles = [Line2D([], [], color=c, marker=m, lw=1.4, ms=4, label=t) for c, m, t in styles.values()]
    legend_below(fig, handles, 3, y=-0.05, rect_bottom=0.06)
    fig.savefig(OUT / "fig_ec_contrastes.pdf")
    plt.close(fig)


def fig_sonde(probe_path: Path, L: dict) -> None:
    probe = json.loads(probe_path.read_text())["graphs"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, task in zip(axes, ["where", "loom"]):
        xs, ys = defaultdict(list), defaultdict(list)
        for g, v in probe.items():
            xs[family(g)].append(100 * v[f"{task}_scores8"]["M"])
            ys[family(g)].append(L[(g, task, "U")])
        for fam, color in [("S", GREY_S), ("E", GREY_E)]:
            ax.scatter(xs[fam], ys[fam], s=14, color=color, linewidths=0, zorder=3)
        ax.scatter(xs["connectome"], ys["connectome"], s=38, color=BLUE, marker="D", linewidths=0, zorder=4)
        ax.set_title(TASKS[task], loc="left")
        ax.set_xlabel("décodage depuis les 8 scores (%)")
        ax.set_ylabel("aire d'apprentissage, actions imposées (%)")
        comma_axis(ax)
        comma_axis(ax, "x")
        grid(ax)
    handles = [Line2D([], [], color=BLUE, marker="D", ls="", ms=5, label="connectome"),
               Line2D([], [], color=GREY_S, marker="o", ls="", ms=4, label="témoins S"),
               Line2D([], [], color=GREY_E, marker="o", ls="", ms=4, label="témoins E")]
    legend_below(fig, handles, 3)
    fig.savefig(OUT / "fig_sonde.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- statistics (appendix)
def fig_validation(stat_path: Path, boot_path: Path) -> None:
    s = json.loads(stat_path.read_text())["cells"]
    b = json.loads(boot_path.read_text())["results"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    names = list(s)
    for i, n in enumerate(names):
        for k, (meth, col, dx) in enumerate([("historical", BLUE, -0.15), ("symmetric", GREY_E, 0.15)]):
            r = s[n][meth]
            ax.plot([i + dx] * 2, [100 * r["ci95"][0], 100 * r["ci95"][1]], color=col, lw=1.2, zorder=3)
            ax.scatter(i + dx, 100 * r["rate"], color=col, s=10, zorder=4, linewidths=0,
                       label=({"historical": "procédure utilisée", "symmetric": "variante symétrique"}[meth]
                              if i == 0 else None))
    ax.axhline(5, color=ORANGE, lw=0.8, ls=(0, (3, 3)))
    lab = {"normal": "normale", "normal_corr_0.8": "corrélée", "student_t3": "Student", "lognormal_sd1.5": "log-norm."}
    ax.set_xticks(range(len(names)), [f"{lab[n.split('/')[0]]} {n.split('/')[1]} {n.split('/')[2]}" for n in names],
                  rotation=90, fontsize=5.8)
    ax.set_ylabel("rejets erronés (%)")
    ax.set_ylim(2.5, 6)
    comma_axis(ax, "y", 1)
    grid(ax)
    ax.legend(loc="upper left", fontsize=6.6)
    ax.set_title("Erreur de première espèce, hypothèse nulle complète", loc="left", fontsize=7.8)
    ax = axes[1]
    names_b = ["format réel", "4 × moins\nd'images", "2 × moins\nde témoins"]
    for i, r in enumerate(b):
        cov, se = 100 * r["coverage_95"], 100 * 1.96 * r["mc_se_coverage"]
        ax.plot([i, i], [cov - se, cov + se], color=BLUE, lw=1.2)
        ax.scatter(i, cov, color=BLUE, s=14, zorder=4, linewidths=0)
    ax.axhline(95, color=ORANGE, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(3), names_b, fontsize=6.4)
    ax.set_ylabel("couverture (%)")
    ax.set_ylim(88, 100)
    ax.set_xlim(-0.5, 2.5)
    comma_axis(ax, "y", 0)
    grid(ax)
    ax.set_title("Couverture des intervalles", loc="left", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_validation.pdf")
    plt.close(fig)




def inputs_from_repository() -> dict[str, Path]:
    a = REPO / "artifacts"
    return {
        "balayage": a / "runs/gate_g_sweep_confirm-20260930.json",
        "regime": a / "runs/mission/regime_map.json",
        "appui": a / "runs/mission/speedrun_ceiling.json",
        "stimuli": a / "retino_align/pilot_e2e/figures/fig1_stimulus_permutation.png",
        "retino": a / "retino_align/main_v1/metrics/analysis.json",
        "calibration": a / "activity_match/calibration/calibration.json",
        "activite": a / "complements/synthesis_activity.json",
        "ec": a / "experience_commune_v1/metrics/runs_table.csv",
        "sonde": a / "experience_commune_v1/complements/probe_representation/probe.json",
        "validation": a / "statistical_audit_20261019/calibration.json",
        "couverture": a / "complements/bootstrap_coverage.json",
    }


def inputs_from_backups(c: Path, k: Path) -> dict[str, Path]:
    bk = str(c / "SMB_Flywire_Speedrun_backups-20261021T151848Z-1-0{}/SMB_Flywire_Speedrun_backups/C.55054955/"
                 "artifacts/runs")
    arch = c / "SMB_Flywire_Speedrun_final_2026-10-20_1036UTC-002/SMB_Flywire_Speedrun/artifacts"
    ec = c / "EXPERIENCE_COMMUNE_V1_complet_2026-10-21-004/SMB_Flywire_Speedrun_ec/artifacts/experience_commune_v1"
    return {
        "balayage": Path(bk.format("06")) / "gate_g_sweep_confirm-20260930.json",
        "regime": Path(bk.format("11")) / "mission/regime_map.json",
        "appui": Path(bk.format("06")) / "mission/speedrun_ceiling.json",
        "stimuli": c / ("SMB_Flywire_Speedrun_backups-20261021T151848Z-1-005/SMB_Flywire_Speedrun_backups/C.56757376/"
                        "artifacts/retino_align/pilot_e2e/figures/fig1_stimulus_permutation.png"),
        "retino": arch / "retino_align/main_v1/metrics/analysis.json",
        "calibration": arch / "activity_match/calibration/calibration.json",
        "activite": k / "SMB_Flywire_Speedrun/report/complements/synthesis_activity.json",
        "ec": ec / "metrics/runs_table.csv",
        "sonde": k / "SMB_Flywire_Speedrun_ec/artifacts/experience_commune_v1/complements/probe_representation/probe.json",
        "validation": arch / "statistical_audit_20261019/calibration.json",
        "couverture": k / "SMB_Flywire_Speedrun/report/complements/bootstrap_coverage.json",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campagne", type=Path, help="owner's campaign folder (default: read from the repository)")
    ap.add_argument("--complements", type=Path,
                    help="folder where the complements archive was extracted (JSON files only)")
    ap.add_argument("--out", type=Path, help="output folder (default: artifacts/figures/)")
    args = ap.parse_args()
    if (args.campagne is None) != (args.complements is None):
        ap.error("--campagne and --complements go together")
    p = inputs_from_repository() if args.campagne is None else inputs_from_backups(args.campagne, args.complements)
    missing = [str(f) for f in p.values() if not f.is_file()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    fig_temoins()
    fig_transformation()
    fig_balayage(p["balayage"])
    fig_regime(p["regime"])
    fig_appui(p["appui"])
    fig_stimuli(p["stimuli"])
    fig_retino(p["retino"])
    fig_calibration(p["calibration"])
    fig_activite(p["activite"])
    fig_synthese()
    L = fig_ec_courbes(load_runs(p["ec"]))
    for task in ("where", "loom"):
        for mode in ("N", "U"):
            vals = {f: [v for (g, t, m), v in L.items() if t == task and m == mode and family(g) == f] for f in "SE"}
            print(f"L {task} {mode}: connectome {L[('connectome', task, mode)]:.1f} ; "
                  f"S {np.mean(vals['S']):.1f} ; E {np.mean(vals['E']):.1f}")
    fig_ec_contrastes()
    fig_sonde(p["sonde"], L)
    fig_validation(p["validation"], p["couverture"])


if __name__ == "__main__":
    main()
