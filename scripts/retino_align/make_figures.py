#!/usr/bin/env python
"""RETINO_ALIGN_V1 figures (mission §18.2), generated from a run's saved data only (no simulation).

    .venv/bin/python scripts/retino_align/make_figures.py --run main_v1

Writes artifacts/retino_align/<run>/figures/fig1..fig7 (PNG). Nulls are drawn individually, never only as a mean
bar; intervals are labelled as conditional bootstrap intervals; every axis carries its unit and every panel its n.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CONNECTOME = "#eb6834"      # categorical slot 2 (orange): the measured connectome
NULLS = "#2a78d6"           # slot 1 (blue): family-c nulls, one mark each
AQUA = "#1baf7a"            # slot 3: brightness task where both tasks share a panel
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TASK_NAME = {"orient": "orientation", "lum": "brightness"}


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                         "savefig.bbox": "tight", "legend.frameon": False, "lines.linewidth": 1.5})


def load(run: str) -> dict:
    d = ROOT / "artifacts/retino_align" / run
    man = json.loads((d / "manifest.json").read_text())
    res = json.loads((d / "metrics/analysis.json").read_text())
    return {"dir": d, "man": man, "res": res}


def g0_label(R) -> str:
    return "connectome" if not R["man"]["pilot"] else f"{R['man']['graphs'][0]} (pseudo-g0, pipeline test)"


def fig1(R, out: Path) -> None:
    import matplotlib.pyplot as plt

    from flymario.retino_align.stimuli import ORIENT_DEG
    d = R["dir"]
    inp = np.load(d / "inputs/inputs.npz")
    pz = np.load(d / "permutations/permutations.npz")
    imgs = np.load(d / "inputs/example_images.npz")["train"]
    orient = inp["orient"][inp["part"] == 0][: len(imgs)]
    lum = inp["lum"][inp["part"] == 0][: len(imgs)]
    xy, col = pz["col_xy"], pz["col"]
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.35, wspace=0.25)
    shown = []
    for o in range(4):
        i = int(np.flatnonzero(orient == o)[0])
        shown.append(i)
        ax = fig.add_subplot(gs[0, o])
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.axhline(31.5, color=CONNECTOME, lw=1)
        ax.set_title(f"orientation {ORIENT_DEG[o]:.0f}°, brightness class {lum[i]}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    fig.text(0.5, 0.93, "a. Stimuli (240 × 256 px; rows above the orange line are the cropped HUD band)",
             ha="center", fontsize=10)
    X = inp["X"][inp["part"] == 0][shown[1]]
    on = np.isin(pz["types"], ("L1", "L5"))
    v_col = np.array([X[(col == c) & on].mean() for c in range(len(xy))])
    s1 = pz["perm1_sigma"]
    Xp = X[s1]
    v_colp = np.array([Xp[(col == c) & on].mean() for c in range(len(xy))])
    for k, (vals, title) in enumerate(((v_col, "b. ON input per column, ordered"),
                                       (v_colp, "c. same image, permutation 1"))):
        ax = fig.add_subplot(gs[1, k])
        ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="gray", s=9, vmin=v_col.min(), vmax=v_col.max())
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        ax.set_title(title, fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    pi = pz["perm1_pi"]
    hue = xy[:, 0] / xy[:, 0].max()
    ax.scatter(xy[:, 0], xy[:, 1], c=hue[pi], cmap="viridis", s=9)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    ax.set_title("d. source column of each column\n(colour = source's horizontal position)", fontsize=9)
    ax = fig.add_subplot(gs[1, 3])
    for k, cond in enumerate(("perm1", "perm2", "perm3")):
        rep = R["man"]["permutations"][cond]
        disp = np.linalg.norm(xy[pz[f"{cond}_pi"]] - xy, axis=1)
        ax.hist(disp, bins=30, histtype="step", lw=1.5, color=(NULLS, CONNECTOME, AQUA)[k],
                label=f"{cond}: moved {100 * rep['frac_cells_moved']:.1f} % of cells")
    ax.set_xlabel("displacement (column spacings)")
    ax.set_ylabel("columns")
    ax.set_title("e. displacement of the 892 columns", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(out / "fig1_stimulus_permutation.png")
    plt.close(fig)


def fig2(R, out: Path) -> None:
    import matplotlib.pyplot as plt
    res, man = R["res"], R["man"]
    curves = np.array(res["curves"])                                  # (G, C, T, K)
    graphs = res["graphs_complete"]
    budgets = 4 * np.array(man["config"]["decoder"]["per_combination"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ti, t in enumerate(("orient", "lum")):
        ax = axes[ti]
        for gi in range(len(graphs)):
            col, lw, z = (CONNECTOME, 2.5, 3) if gi == 0 else (NULLS, 0.8, 1)
            a = 1.0 if gi == 0 else 0.45
            ax.plot(budgets, curves[gi, 0, ti], color=col, lw=lw, alpha=a, zorder=z, marker="o" if gi == 0 else None,
                    ms=5)
            ax.plot(budgets, curves[gi, 1:, ti].mean(0), color=col, lw=lw, alpha=a, ls="--", zorder=z)
        ax.axhline(0.25, color=INK2, lw=1, ls=":")
        ax.text(budgets[0], 0.255, "chance", color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("training examples per class")
        ax.set_ylabel("test accuracy (4 classes)")
        ax.set_title(f"{TASK_NAME[t]}\nsolid = ordered, dashed = mean of the 3 permutations", fontsize=9)
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color=CONNECTOME, lw=2.5), Line2D([], [], color=NULLS, lw=0.8)],
                   [g0_label(R), f"family-c nulls (n = {len(graphs) - 1}), one line each"], loc="lower right",
                   fontsize=8)
    n_te = man["n_images"]["test"]
    fig.suptitle(f"Decoding curves: mean over 2 dynamic seeds × 10 subset draws; {n_te} test images", fontsize=10, y=1.04)
    fig.savefig(out / "fig2_curves.png")
    plt.close(fig)


def fig3_4(R, out: Path) -> None:
    import matplotlib.pyplot as plt
    res = R["res"]
    graphs = res["graphs_complete"]
    M = res["M"]
    conds = R["man"]["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ti, t in enumerate(("orient", "lum")):
        ax = axes[ti]
        for gi, g in enumerate(graphs):
            o = M[g]["ordered"][t]
            ps = [M[g][c][t] for c in conds[1:]]
            col = CONNECTOME if gi == 0 else NULLS
            x = 0 if gi == 0 else 1 + 0.6 * (gi - 1) / max(1, len(graphs) - 2)
            ax.plot([x, x], [o, np.mean(ps)], color=col, lw=1.0, alpha=0.8)
            ax.scatter([x], [o], color=col, s=28, zorder=3, edgecolor="white", linewidth=1)
            ax.scatter([x] * 3, ps, color=col, s=12, marker="x", zorder=3)
        ax.set_xticks([0, 1.3], [g0_label(R).split(" (")[0], f"nulls (n = {len(graphs) - 1})"])
        ax.set_ylabel("M: accuracy averaged over budgets, seeds, draws")
        ax.set_title(f"{TASK_NAME[t]}: ● ordered, × each permutation")
    fig.suptitle("Ordered vs permuted scores, every graph drawn individually", fontsize=10)
    fig.savefig(out / "fig3_ordered_vs_permuted.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rng = np.random.default_rng(0)
    for ti, t in enumerate(("orient", "lum")):
        ax = axes[ti]
        L = np.array([res["L"][g][t] for g in graphs])
        ax.scatter(L[1:], rng.uniform(-0.15, 0.15, len(L) - 1), color=NULLS, s=22, alpha=0.8,
                   label=f"nulls (n = {len(L) - 1})")
        ax.axvline(L[0], color=CONNECTOME, lw=2.5, label=g0_label(R))
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xlabel("L = M(ordered) − mean M(permutations)  (proportion correct)")
        rk = res["rank_L_orientation" if t == "orient" else "rank_L_brightness"]
        ax.set_title(f"{TASK_NAME[t]}: loss under permutation; nulls with L ≥ g0: {rk['n_nulls_ge']} / {rk['K']}")
        ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig4_loss_distribution.png")
    plt.close(fig)


def fig5(R, out: Path) -> None:
    import matplotlib.pyplot as plt
    res = R["res"]
    delta = R["man"]["config"]["analysis"]["practical_margin"]
    keys = ["D_orientation", "G", "D_brightness", "S"]
    labels = ["D orientation\n(primary)", "G (initial\nadvantage)", "D brightness", "S = D_or − D_br"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(-delta, delta, color=GRID, alpha=0.7, lw=0)
    ax.text(3.45, delta, f" ±δ = {delta}", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    for i, k in enumerate(keys):
        lo, hi = res["bootstrap"]["intervals_95"][k]
        p = res["point"][k]
        col = CONNECTOME if i == 0 else INK2
        ax.plot([i, i], [lo, hi], color=col, lw=2.5, solid_capstyle="round")
        ax.scatter([i], [p], color=col, s=50, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(i + 0.08, p, f"{p:+.3f}\n[{lo:+.3f}, {hi:+.3f}]", fontsize=8, va="center", color=INK)
    ax.set_xticks(range(4), labels)
    ax.set_xlim(-0.5, 3.9)
    ax.set_ylabel("accuracy difference (proportion)")
    nrep = res["bootstrap"]["replicates"]
    ax.set_title(f"Contrasts with 95 % conditional bootstrap intervals ({nrep} replicates;\n"
                 "test images resampled within cells, nulls as whole graphs; not a population interval)",
                 fontsize=9)
    fig.savefig(out / "fig5_contrasts.png")
    plt.close(fig)


def fig6(R, out: Path) -> None:
    import matplotlib.pyplot as plt
    d, man = R["dir"], R["man"]
    conds = man["conditions"]
    keys = [("mean_count", "mean readout count / 16 ticks"), ("frac_silent_entries", "fraction of silent counts"),
            ("frac_near_ceiling_entries_ge7", "fraction of counts ≥ 7 (max 8)"), ("rate_all", "global firing rate")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for gi, g in enumerate(man["graphs"]):
        for ci, c in enumerate(conds):
            for s in man["dynamic_seeds"]:
                p = d / "features" / g.replace(":", "") / f"{c}_s{s}.json"
                if not p.exists():
                    continue
                a = json.loads(p.read_text())["activity"]
                for k, (key, _lab) in enumerate(keys):
                    x = ci + (0.18 if s == man["dynamic_seeds"][-1] else -0.18) + \
                        (np.random.default_rng(gi).uniform(-0.08, 0.08) if gi else 0)
                    axes[k].scatter([x], [a[key]], s=26 if gi == 0 else 9,
                                    color=CONNECTOME if gi == 0 else NULLS, alpha=1 if gi == 0 else 0.5,
                                    zorder=3 if gi == 0 else 2)
    for k, (_key, lab) in enumerate(keys):
        axes[k].set_xticks(range(len(conds)), conds)
        axes[k].set_title(lab, fontsize=9)
    fig.suptitle(f"Activity by condition (left/right of each tick: dynamic seed; orange = {g0_label(R)})", fontsize=10)
    fig.savefig(out / "fig6_activity.png")
    plt.close(fig)


def fig7(R, out: Path) -> None:
    import matplotlib.pyplot as plt
    sens = R["res"]["sensitivity_secondary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, (("per_permutation", "by permutation"), ("per_seed", "by dynamic seed"),
                                       ("per_budget", "by budget (examples per class)")), strict=True):
        names = list(sens[key])
        for j, stat in enumerate(("D_orientation", "D_brightness")):
            vals = [sens[key][n][stat] for n in names]
            ax.plot(range(len(names)), vals, marker="o", color=(CONNECTOME, AQUA)[j], lw=1.5,
                    label=stat.replace("_", " "))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(names)), names)
        ax.set_title(f"D {title} (secondary, point estimates)", fontsize=9)
    axes[0].set_ylabel("accuracy difference")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "fig7_sensitivity.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    style()
    R = load(a.run)
    out = R["dir"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3_4, fig5, fig6, fig7):
        f(R, out)
        print("wrote", f.__name__, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
