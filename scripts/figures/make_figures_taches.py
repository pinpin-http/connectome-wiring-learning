#!/usr/bin/env python
"""Figures of the visual tasks and of the game, recomputed from the raw result files.

    python -I scripts/figures/make_figures_taches.py --donnees <folder containing artifacts/runs> [--out <dir>]

Inputs: artifacts/runs/study_s/*.json (Study S) and artifacts/runs/mission/{readings-24,-26,-27,graph_activity}.json.
Labels are in French (the report's language) and decimals use a comma. The real wiring is blue, null
copies grey, the three earlier copies orange. Also prints the post hoc numbers quoted in the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

ROOT = Path(sys.argv[sys.argv.index("--donnees") + 1]) if "--donnees" in sys.argv else Path(".")
REPO = Path(__file__).resolve().parents[2]
# Figures go to artifacts/figures/ (not in git); --out <dir> writes them elsewhere.
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else REPO / "artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)
S_DIR = ROOT / "artifacts/runs/study_s"
M_DIR = ROOT / "artifacts/runs/mission"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY_B, GREY_C = "#c9c7c0", "#7d7b76"
NB = " "
TASKS = {"where": f"«{NB}Où{NB}?{NB}»", "loom": f"«{NB}Approche ou recul{NB}?{NB}»"}
CHANCE = 12.5

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


def fr(v: float, nd: int = 2, signed: bool = False) -> str:
    """French number: decimal comma, true minus sign, optional explicit plus."""
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "−").replace(".", ",")


def comma_axis(ax, axis: str = "y", nd: int = 1) -> None:
    fmt = FuncFormatter(lambda v, _: fr(v, nd))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands_axis(ax, axis: str = "x") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def ygrid(ax) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------------- Study S data
def load_study_s() -> dict:
    g = {"connectome": json.loads((S_DIR / "connectome.json").read_text())}
    for s in range(40):
        g[f"b{s}"] = json.loads((S_DIR / f"nulllat{s}.json").read_text())
        g[f"c{s}"] = json.loads((S_DIR / f"nullin{s}.json").read_text())
    return g


def val(d: dict, task: str, key: str) -> float:
    return d["summary"]["true/" + task][key]


def rung_vals(G: dict, rung: str, task: str, key: str) -> np.ndarray:
    return np.array([val(G[f"{rung}{s}"], task, key) for s in range(40)])


def curve(d: dict, task: str) -> np.ndarray:
    """Graph-level learning curve: mean over the 6 learners of the correct-choice rate per 100-trial bin."""
    return np.mean([lr["curve"] for lr in d["learners"] if lr["task"] == task and lr["kind"] == "true"], axis=0)


def dot(ax, x, y, color, s=14, z=2, marker="o", ring=0.6):
    ax.scatter(x, y, s=s, color=color, marker=marker, edgecolors="white", linewidths=ring, zorder=z)


# --------------------------------------------------------------------------------------- task illustration
def fig_tasks() -> None:
    """The two Study S tasks as drawn by scripts/diagnostics/study_s.py (amendment A1 geometry)."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    H, W, HUD = 240, 256, 32
    field_c = [(32 + 64 * i, 84 + 104 * j) for j in range(2) for i in range(4)]
    loom_c = [(64, 85), (192, 85), (64, 185), (192, 185)]
    dark = "#2b2a28"
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.05), gridspec_kw=dict(wspace=0.12))

    def frame(ax, title):
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(AXIS)
        ax.add_patch(Rectangle((0, 0), W, HUD, facecolor="#efeee9", edgecolor="none", zorder=5))
        ax.text(W / 2, HUD / 2, "bandeau masqué", ha="center", va="center", fontsize=6.5, color=MUTED, zorder=6)
        ax.set_title(title, loc="left", fontsize=8)

    ax = axes[0]
    frame(ax, f"«{NB}Où{NB}?{NB}» : 8 positions")
    for k, (cx, cy) in enumerate(field_c):
        filled = k == 5
        ax.add_patch(Rectangle((cx - 32, cy - 52), 64, 104, facecolor=dark if filled else "none",
                               edgecolor=dark if filled else AXIS, lw=0.8, zorder=2))
        ax.text(cx, cy, str(k + 1), ha="center", va="center", fontsize=7, color="white" if filled else INK2,
                zorder=3)
    for ax, (r0, r1, name) in zip(axes[1:], ((30, 90, "approche (rayon 30 → 90)"), (90, 30, "recul (rayon 90 → 30)")),
                                  strict=True):
        frame(ax, name)
        cx, cy = loom_c[3]
        for qx, qy in loom_c:
            ax.plot(qx, qy, marker="+", color=AXIS, ms=5, mew=0.8, zorder=1)
        ax.add_patch(Circle((cx, cy), r1, facecolor=dark, edgecolor="none", zorder=2))
        ax.add_patch(Circle((cx, cy), r0, facecolor="none", edgecolor="white" if r0 < r1 else dark, lw=1.0,
                            zorder=3))
        lo, hi = min(r0, r1), max(r0, r1)
        for ang in (np.pi, np.pi * 1.25):
            p0 = (cx + (lo + 6) * np.cos(ang), cy + (lo + 6) * np.sin(ang))
            p1 = (cx + (hi - 6) * np.cos(ang), cy + (hi - 6) * np.sin(ang))
            st, en = (p0, p1) if r0 < r1 else (p1, p0)          # outward when looming, inward when receding
            ax.add_patch(FancyArrowPatch(st, en, arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                         color="white" if r0 < r1 else INK2, zorder=4))
    fig.savefig(OUT / "fig_s_taches.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 1
def fig_mario() -> None:
    r = {k: json.loads((M_DIR / f"readings-{k}.json").read_text()) for k in (24, 26, 27)}
    act = json.loads((M_DIR / "graph_activity.json").read_text())
    fig = plt.figure(figsize=(6.3, 3.75))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], wspace=0.42, hspace=1.35)
    rng = np.random.default_rng(1)
    panels = [(24, "Flux expert, témoins S", GREY_B),
              (26, "Flux expert, témoins E", GREY_C),
              (27, "Flux scripté, témoins S", GREY_B)]
    for i, (k, title, col) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        d = r[k]
        x = np.array([d["null_S"][str(s)] for s in range(40)])
        y = rng.uniform(-1, 1, 40)
        cols = [ORANGE if (k == 24 and s < 3) else col for s in range(40)]
        order = np.argsort([c == ORANGE for c in cols])           # orange drawn last, on top
        dot(ax, x[order], y[order], [cols[j] for j in order])
        ax.axvline(d["connectome_S"], color=BLUE, lw=1.6, zorder=3)
        above = 41 - d["rank_from_bottom"]
        ax.text(0, 1.05, f"{above} témoins sur 40 plus rapides que le connectome "
                f"(p{NB}={NB}{fr(d['p_two_sided'], 2)})", transform=ax.transAxes,
                color=INK2, fontsize=7, va="bottom")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        comma_axis(ax, "x", 1)
        ax.set_title(title, loc="left", pad=15)
        if i == 2:
            ax.set_xlabel("vitesse d'apprentissage supervisé (AUC moyenne)")
    ax = fig.add_subplot(gs[:, 1])
    for k, kind, col in ((24, "nulllat", GREY_B), (26, "nullin", GREY_C)):
        xs = [act[f"{kind}:{s}"]["motor_spikes_per_decision"] for s in range(40)]
        ys = [r[k]["null_S"][str(s)] for s in range(40)]
        dot(ax, xs, ys, col)
    dot(ax, [act["connectome"]["motor_spikes_per_decision"]], [r[24]["connectome_S"]], BLUE, s=34, z=3, ring=0.8)
    ax.set_xlabel("impulsions par décision (avant apprentissage)")
    ax.set_ylabel("vitesse (flux expert)")
    ax.set_title("Activité et vitesse", loc="left")
    thousands_axis(ax, "x")
    comma_axis(ax, "y", 1)
    ax.xaxis.set_major_locator(MultipleLocator(1500))
    ygrid(ax)
    handles = [Line2D([], [], marker="o", ls="", color=GREY_B, mec="white", ms=5, label="témoins S"),
               Line2D([], [], marker="o", ls="", color=GREY_C, mec="white", ms=5, label="témoins E"),
               Line2D([], [], marker="o", ls="", color=ORANGE, mec="white", ms=5,
                      label="3 témoins d'une comparaison préliminaire"),
               Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=5, mec="white", label="connectome")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), handletextpad=0.4,
               columnspacing=1.4)
    fig.savefig(OUT / "fig_mario_copies.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 2
def fig_curves(G: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55), sharey=True)
    t = (np.arange(60) + 1) * 100
    for ax, task in zip(axes, TASKS, strict=True):
        for rung, col in (("b", GREY_B), ("c", GREY_C)):
            C = np.array([curve(G[f"{rung}{s}"], task) for s in range(40)]) * 100
            lo, med, hi = np.percentile(C, [5, 50, 95], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.35, lw=0, zorder=1)
            ax.plot(t, med, color=col, lw=1.0, zorder=2)
        ax.plot(t, curve(G["connectome"], task) * 100, color=BLUE, lw=1.6, zorder=3)
        ax.axhline(CHANCE, color=MUTED, lw=0.8, zorder=1)
        ax.text(6000, CHANCE - 0.6, f"hasard (12,5{NB}%)", ha="right", va="top", color=INK2, fontsize=7)
        ax.set_title(TASKS[task], loc="left")
        ax.set_xlabel("essais d'apprentissage")
        thousands_axis(ax, "x")
        ax.set_xlim(0, 6100)
        ax.set_ylim(8, 70)
        ygrid(ax)
    axes[0].set_ylabel("bonnes réponses (%)")
    handles = [Line2D([], [], color=BLUE, lw=1.6, label="connectome"),
               (Patch(color=GREY_C, alpha=0.35, lw=0), Line2D([], [], color=GREY_C, lw=1.0)),
               (Patch(color=GREY_B, alpha=0.35, lw=0), Line2D([], [], color=GREY_B, lw=1.0))]
    labels = ["connectome", "40 témoins E : médiane et 5e-95e centiles",
              "40 témoins S : médiane et 5e-95e centiles"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(OUT / "fig_s_courbes.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 3
def fig_endpoints(G: dict) -> None:
    learn = [("E1", "vitesse\nd'apprentissage"), ("A_test", "précision\nfinale"), ("A_gen", "variantes\nnon vues")]
    rob = [("E4", "connexions\nsupprimées"), ("E5", "bruit\ndoublé")]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.3), gridspec_kw=dict(width_ratios=[3, 2], hspace=0.55,
                                                                        wspace=0.28))
    rng = np.random.default_rng(2)
    for i, task in enumerate(TASKS):
        for j, eps in enumerate((learn, rob)):
            ax = axes[i, j]
            for k, (key, _lab) in enumerate(eps):
                for dx, rung, col in ((-0.22, "b", GREY_B), (0.22, "c", GREY_C)):
                    v = rung_vals(G, rung, task, key) * 100
                    dot(ax, k + dx + rng.uniform(-0.09, 0.09, 40), v, col, s=11, ring=0.5)
                dot(ax, [k], [val(G["connectome"], task, key) * 100], BLUE, s=36, z=3, marker="D", ring=0.8)
            ax.set_xticks(range(len(eps)), [lab for _, lab in eps])
            ax.set_xlim(-0.6, len(eps) - 0.4)
            ax.tick_params(axis="x", length=0)
            ygrid(ax)
            if j == 0:
                ax.set_ylim(15, 85)
                ax.set_ylabel("bonnes réponses (%)")
                ax.set_title(f"{TASKS[task]} : apprentissage", loc="left")
            else:
                ax.set_ylim(-2, 60)
                ax.set_ylabel("perte (points de %)")
                ax.set_title("robustesse (plus bas = moins de perte)", loc="left")
    handles = [Line2D([], [], marker="D", ls="", color=BLUE, mec="white", ms=6, label="connectome"),
               Line2D([], [], marker="o", ls="", color=GREY_B, mec="white", ms=5, label="40 témoins S"),
               Line2D([], [], marker="o", ls="", color=GREY_C, mec="white", ms=5, label="40 témoins E")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUT / "fig_s_mesures.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 4
def fig_exploratory(G: dict) -> dict:
    """Post hoc: does accuracy on variants (or after damage) follow accuracy on the training distribution?"""
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8), gridspec_kw=dict(hspace=0.55, wspace=0.36))
    out = {}
    line_col = {"b": "#a3a19a", "c": "#5f5d58"}
    cols = (("A_gen", "précision sur les variantes (%)", "généralisation"),
            ("A_rob", "précision après dommage (%)", "robustesse"))
    for i, task in enumerate(TASKS):
        cx = val(G["connectome"], task, "A_test") * 100
        for j, (ykey, ylab, head) in enumerate(cols):
            ax = axes[i, j]
            cy = val(G["connectome"], task, ykey) * 100
            zres = {}
            for rung, col in (("b", GREY_B), ("c", GREY_C)):
                x = rung_vals(G, rung, task, "A_test") * 100
                y = rung_vals(G, rung, task, ykey) * 100
                dot(ax, x, y, col, s=12, ring=0.5)
                slope, icpt = np.polyfit(x, y, 1)
                sd = (y - (slope * x + icpt)).std(ddof=2)
                xx = np.array([x.min() - 1, cx + 3])
                ax.plot(xx, slope * xx + icpt, color=line_col[rung], lw=1.0, zorder=1)
                zres[rung] = (cy - (slope * cx + icpt)) / sd
            dot(ax, [cx], [cy], BLUE, s=36, z=3, marker="D", ring=0.8)
            out[f"{ykey}/{task}/resid_z_b_c"] = [round(float(zres["b"]), 2), round(float(zres["c"]), 2)]
            txt = (f"connectome, écart aux droites :\n{fr(zres['b'], 1, True)}{NB}σ (témoins S) ; "
                   f"{fr(zres['c'], 1, True)}{NB}σ (témoins E)")
            pos = dict(x=0.98, y=0.04, ha="right", va="bottom") if ykey == "A_gen" else \
                dict(x=0.02, y=0.97, ha="left", va="top")
            ax.text(pos["x"], pos["y"], txt, transform=ax.transAxes, ha=pos["ha"], va=pos["va"], fontsize=6.8,
                    color=INK2)
            ax.set_xlabel("précision finale (%)")
            ax.set_ylabel(ylab)
            ax.set_title(f"{TASKS[task]} : {head}", loc="left", fontsize=8)
            if ykey == "A_rob":
                ax.set_ylim(10, max(cy + 6, 40))
            ygrid(ax)
    handles = [Line2D([], [], marker="D", ls="", color=BLUE, mec="white", ms=6, label="connectome"),
               Line2D([], [], marker="o", ls="", color=GREY_B, mec="white", ms=5, label="40 témoins S"),
               Line2D([], [], marker="o", ls="", color=GREY_C, mec="white", ms=5, label="40 témoins E"),
               Line2D([], [], color=line_col["b"], lw=1.0, label="droite (témoins S)"),
               Line2D([], [], color=line_col["c"], lw=1.0, label="droite (témoins E)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), columnspacing=0.9,
               handletextpad=0.4, fontsize=7)
    fig.savefig(OUT / "fig_s_exploratoire.pdf")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------ post hoc numbers
def posthoc_numbers(G: dict) -> dict:
    """The exploratory quantities quoted in the report (decided after the declared readings)."""
    res = {}
    for task in TASKS:
        c = G["connectome"]
        for rung in ("b", "c"):
            at, ag, ar = (rung_vals(G, rung, task, k) for k in ("A_test", "A_gen", "A_rob"))
            ratio = ag / at
            cr = val(c, task, "A_gen") / val(c, task, "A_test")
            keep = (ar - 0.125) / (at - 0.125)
            ck = (val(c, task, "A_rob") - 0.125) / (val(c, task, "A_test") - 0.125)
            for name, y in (("A_gen", ag), ("A_rob", ar)):
                sl, ic = np.polyfit(at, y, 1)
                sd = (y - (sl * at + ic)).std(ddof=2)
                res[f"{task}/{rung}/{name}_resid_z"] = round(float((val(c, task, name) - (sl * val(c, task, "A_test")
                                                                                        + ic)) / sd), 2)
            res[f"{task}/{rung}/gen_ratio"] = [round(float(cr), 3), round(float(ratio.mean()), 3),
                                               round(float(ratio.std(ddof=1)), 3)]
            res[f"{task}/{rung}/corr_Atest_Agen"] = round(float(np.corrcoef(at, ag)[0, 1]), 2)
            res[f"{task}/{rung}/corr_Atest_E4"] = round(float(np.corrcoef(at, at - ar)[0, 1]), 2)
            res[f"{task}/{rung}/keep_above_chance"] = [round(float(ck), 3), round(float(keep.mean()), 3),
                                                       int(1 + (keep < ck).sum())]
            res[f"{task}/{rung}/A_rob_mean"] = [round(float(val(c, task, "A_rob")), 3), round(float(ar.mean()), 3),
                                                round(float(ar.max()), 3)]
            # trials the real wiring needs to reach the copies' end level (mean of their last 1,000 trials)
            cm = np.mean([curve(G[f"{rung}{s}"], task) for s in range(40)], axis=0)
            final = cm[-10:].mean()
            sc = np.convolve(curve(c, task), np.ones(5) / 5, mode="valid")
            hit = (int(np.argmax(sc >= final)) + 5) * 100
            res[f"{task}/{rung}/trials_to_null_end_level"] = [round(float(final), 3), hit, round(6000 / hit, 1)]
            res[f"{task}/{rung}/end_slope_per_1000"] = round(float(np.polyfit(np.arange(40, 60) * 100, cm[40:60], 1)[0]
                                                                  * 1000), 3)
        res[f"{task}/connectome/end_slope_per_1000"] = round(float(np.polyfit(np.arange(40, 60) * 100,
                                                                               curve(c, task)[40:60], 1)[0] * 1000), 3)
    return res


def main() -> None:
    G = load_study_s()
    fig_tasks()
    fig_mario()
    fig_curves(G)
    fig_endpoints(G)
    pooled = fig_exploratory(G)
    nums = posthoc_numbers(G)
    nums.update(pooled)
    print(json.dumps(nums, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
