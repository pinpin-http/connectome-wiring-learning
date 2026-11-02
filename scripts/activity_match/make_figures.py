#!/usr/bin/env python
"""ACTIVITY_MATCH_V1 figures (mission §19), from saved files only (no simulation, no refit).

    .venv/bin/python scripts/activity_match/make_figures.py --calibration calibration            # fig1, fig2
    .venv/bin/python scripts/activity_match/make_figures.py --calibration calibration --run main_am  # fig1-fig7

Writes into artifacts/activity_match/<run or calibration>/figures/. Nulls are drawn individually; the connectome is
the one coloured mark. Units: rates in Hz, accuracies and contrasts in percentage points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTS = ROOT / "artifacts/activity_match"
CONN, NULL, NULL_DARK, BAND = "#c2410c", "#9ca3af", "#4b5563", "#e5e7eb"
TCOL = {"CL": "#1d4ed8", "CM": "#047857", "CH": "#7c3aed"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 10})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig1(cal: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    graphs = cal["graphs"]
    for s in graphs:
        pts = cal["search"].get(s, {}).get("points") or cal["grid"][s]
        g = np.array(sorted(float(k) for k in pts))
        r = np.array([pts[str(x) if str(x) in pts else repr(x)]["rate_hz"] for x in g])
        v = np.array([pts[str(x) if str(x) in pts else repr(x)]["valid"] for x in g])
        is_c = s == "connectome"
        ax.plot(g, r, "-", color=CONN if is_c else NULL, lw=2.0 if is_c else 0.7, alpha=1 if is_c else 0.6,
                zorder=3 if is_c else 1, label="connectome" if is_c else None)
        ax.plot(g[~v], r[~v], "x", color=CONN if is_c else NULL_DARK, ms=5, zorder=4)
    ax.plot([], [], "-", color=NULL, lw=0.7, label=f"{len(graphs) - 1 if 'connectome' in graphs else len(graphs)} nulls")
    ax.plot([], [], "x", color=NULL_DARK, label="invalid point (silent)")
    handles_extra = []
    if cal.get("common_range", {}).get("exists"):
        ax.axhspan(cal["common_range"]["r_min"], cal["common_range"]["r_max"], color=BAND, zorder=0)
        handles_extra.append(Patch(color=BAND, label="common range"))
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for k, t in cal["targets_hz"].items():
            ax.axhline(t, color=TCOL[k], lw=1, ls="--")
            ax.text(1.01, t, f"{k} {t:.1f} Hz", color=TCOL[k], va="center", fontsize=8, transform=tr)
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("gain multiplier gamma (log scale; 1 = RETINO_ALIGN_V1)")
    ax.set_ylabel("mean readout rate on CAL_FIT (Hz)")
    ax.set_title("Readout activity against the global gain (label-free calibration)")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + handles_extra, frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig1_activity_vs_gamma")


def fig2(cal: dict, out: Path, res: dict | None) -> None:
    n = 2 if res else 1
    fig, axes = plt.subplots(1, n, figsize=(6.4 if n == 1 else 9.6, 3.6), squeeze=False)
    tol = 0.10
    tg = list(cal["targets_hz"])
    for pi_, (title, src) in enumerate([("CAL_CHECK (independent calibration images)", "check"),
                                        ("main test images (transport)", "test")][:n]):
        ax = axes[0, pi_]
        for ti, k in enumerate(tg):
            for s in cal["graphs"]:
                if src == "check":
                    e = (cal["check"][s][k]["rate_hz"] - cal["targets_hz"][k]) / cal["targets_hz"][k]
                else:
                    a = res["activity_test"][s][k]
                    e = (a["rate_hz"] - a["target_hz"]) / a["target_hz"]
                is_c = s == "connectome"
                ax.plot(ti + (0.15 if is_c else np.random.default_rng(hash(s) % 2**32).uniform(-0.1, 0.1)),
                        100 * e, "o", ms=6 if is_c else 3, color=CONN if is_c else NULL, zorder=3 if is_c else 1)
        ax.axhspan(-100 * tol, 100 * tol, color=BAND, zorder=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(tg)), [f"{k}\n{cal['targets_hz'][k]:.1f} Hz" for k in tg])
        ax.set_ylabel("rate error relative to target (%)")
        ax.set_title(title)
    axes[0, 0].plot([], [], "o", color=CONN, label="connectome")
    axes[0, 0].plot([], [], "o", color=NULL, ms=3, label="nulls")
    h, _ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles=h + [Patch(color=BAND, label="tolerance +-10 %")], frameon=False, fontsize=8)
    save(fig, out, "fig2_matching_quality")


def load_run(run: str) -> tuple[dict, dict]:
    d = ARTS / run
    return json.loads((d / "metrics/analysis.json").read_text()), json.loads((d / "manifest.json").read_text())


def fig3(res: dict, man: dict, out: Path) -> None:
    conds, graphs = res["conditions"], res["graphs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ti, (t, name) in enumerate((("orient", "orientation (primary)"), ("lum", "brightness (secondary)"))):
        ax = axes[ti]
        for ci, c in enumerate(conds):
            nv = [100 * res["M"][g][c][t] for g in graphs if g != "connectome"]
            x = ci + np.random.default_rng(ci).uniform(-0.12, 0.12, len(nv))
            ax.plot(x, nv, "o", ms=3, color=NULL, zorder=1)
            ax.plot([ci - 0.2, ci + 0.2], [np.mean(nv)] * 2, "-", color=NULL_DARK, lw=2)
            if "connectome" in graphs:
                ax.plot(ci + 0.28, 100 * res["M"]["connectome"][c][t], "D", ms=7, color=CONN, zorder=3)
        ax.axhline(25, color="k", lw=0.6, ls=":")
        ax.text(len(conds) - 0.5, 25.3, "chance 25 %", fontsize=7, ha="right")
        if t == "orient":
            ax.axhline(30, color="k", lw=0.6, ls="--")
            ax.text(len(conds) - 0.5, 30.3, "informative threshold 30 %", fontsize=7, ha="right")
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel("test accuracy M (%)")
        ax.set_title(name)
    axes[0].plot([], [], "o", color=NULL, ms=3, label="each null")
    axes[0].plot([], [], "-", color=NULL_DARK, lw=2, label="null mean")
    axes[0].plot([], [], "D", color=CONN, label="connectome")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, out, "fig3_absolute_scores")


def _iv(res, k):
    lo, hi = res["bootstrap"]["intervals_95"][k]
    return 100 * res["point"][k], 100 * lo, 100 * hi


def fig4(res: dict, man: dict, out: Path, delta: float) -> None:
    conds = res["conditions"]
    keys = [f"G_{c}_orient" for c in conds] + (["R_orient"] if "R_orient" in res["point"] else [])
    labels = [f"G at {c}" for c in conds] + (["R = G_CM - G_C0"] if "R_orient" in res["point"] else [])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(-100 * delta, 100 * delta, color=BAND, zorder=0, label=f"practical margin +-{100 * delta:.0f} pts")
    ax.axhline(0, color="k", lw=0.5)
    for i, k in enumerate(keys):
        p, lo, hi = _iv(res, k)
        col = "#047857" if k in ("G_CM_orient",) else ("#1f2937" if k == "R_orient" else NULL_DARK)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, ms=6, capsize=4, lw=1.5)
        ax.text(i + 0.1, p, f"{p:+.1f}", fontsize=8, va="center")
    ax.set_xticks(range(len(keys)), labels, rotation=15)
    ax.set_ylabel("percentage points (95 % bootstrap interval)")
    ax.set_title("Orientation: connectome minus null mean (G), and R; primary = G at CM")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "fig4_contrasts")


def fig5(res: dict, man: dict, out: Path) -> None:
    if "T_real_orient" not in res["point"]:
        return
    keys = ["T_real_orient", "T_nulls_orient", "R_orient"]
    labels = ["T_real: connectome\nCM - C0", "T_nulls: null mean\nCM - C0", "R = T_real - T_nulls"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(0, color="k", lw=0.5)
    for i, (k, col) in enumerate(zip(keys, (CONN, NULL_DARK, "#1f2937"))):
        p, lo, hi = _iv(res, k)
        ax.bar(i, p, color=col, width=0.55, alpha=0.85)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", color="k", capsize=4, lw=1)
        ax.text(i, hi + 0.3, f"{p:+.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("change in orientation accuracy (pts)")
    ax.set_title("Where does the change of G come from?")
    save(fig, out, "fig5_decomposition")


def fig6(res: dict, man: dict, out: Path) -> None:
    conds, graphs = res["conditions"], res["graphs"]
    cur = 100 * np.array(res["curves"])                          # (G, C, T, K)
    bpc = 4 * np.array(man["config"]["decoder"]["per_combination"])
    fig, axes = plt.subplots(1, len(conds) + 1, figsize=(3.1 * (len(conds) + 1), 3.3), sharey=False)
    fig.subplots_adjust(wspace=0.35)
    for ax in axes[1:len(conds)]:
        ax.sharey(axes[0])
    gi_c = graphs.index("connectome") if "connectome" in graphs else None
    for ci, c in enumerate(conds):
        ax = axes[ci]
        for gi in range(len(graphs)):
            if gi == gi_c:
                continue
            ax.plot(bpc, cur[gi, ci, 0], "-", color=NULL, lw=0.6, alpha=0.6)
        if gi_c is not None:
            ax.plot(bpc, cur[gi_c, ci, 0], "-o", color=CONN, lw=2, ms=3)
        ax.axhline(25, color="k", lw=0.5, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bpc, [str(b) for b in bpc])
        ax.set_title(f"{c}: orientation")
        ax.set_xlabel("training examples per class")
        if ci == 0:
            ax.set_ylabel("test accuracy (%)")
    ax = axes[-1]
    seeds = list(res["sensitivity_secondary"]["per_seed"])
    for si, s in enumerate(seeds):
        v = [100 * res["sensitivity_secondary"]["per_seed"][s][f"G_{c}_orient"] for c in conds]
        ax.plot(range(len(conds)), v, "-o", ms=4, label=f"noise seed {s}", color=("#1f2937", "#6b7280")[si % 2])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_title("G (orientation) per noise seed")
    ax.set_ylabel("pts")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "fig6_learning_curves_and_seeds")


def fig7(res: dict, man: dict, out: Path, run_dir: Path) -> None:
    conds, graphs = res["conditions"], res["graphs"]
    seed = man["noise_seeds"][0]
    fig, axes = plt.subplots(2, len(conds), figsize=(3.0 * len(conds), 5.6), squeeze=False)
    for ax in axes[1, 1:]:
        ax.sharex(axes[1, 0]), ax.sharey(axes[1, 0])
    fig.suptitle(f"Readout activity per condition (top: spike-count distribution, noise seed {seed}; "
                 "bottom: each graph on the test images)", fontsize=10)
    for ci, c in enumerate(conds):
        ax = axes[0, ci]
        hs = {}
        for g in graphs:
            h = json.loads((run_dir / "features" / g.replace(":", "") / f"{c}_s{seed}.json").read_text())
            hist = np.array(h["activity"]["all"]["count_hist"], float)
            hs[g] = hist / hist.sum()
        nulls = np.array([hs[g] for g in graphs if g != "connectome"])
        x = np.arange(nulls.shape[1])
        ax.fill_between(x, nulls.min(0), nulls.max(0), color=NULL, alpha=0.5, step="mid", label="nulls (range)")
        if "connectome" in hs:
            ax.step(x, hs["connectome"], where="mid", color=CONN, lw=1.8, label="connectome")
        ax.set_title(c)
        ax.set_xlabel("spikes in 16 ticks")
        if ci == 0:
            ax.set_ylabel("fraction of entries")
            ax.legend(frameon=False, fontsize=7)
        ax = axes[1, ci]
        act = res["activity_test"]
        for g in graphs:
            is_c = g == "connectome"
            ax.plot(act[g][c]["rate_hz"], 100 * act[g][c]["active_frac_mean"], "D" if is_c else "o",
                    ms=7 if is_c else 3, color=CONN if is_c else NULL, zorder=3 if is_c else 1)
        if c in man["targets_hz"]:
            ax.axvline(man["targets_hz"][c], color=TCOL.get(c, "k"), lw=1, ls="--")
        ax.set_xlabel("readout rate, test images (Hz)")
        if ci == 0:
            ax.set_ylabel("active outputs per image (%)")
    save(fig, out, "fig7_activity_distribution")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calibration")
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    cal = json.loads((ARTS / a.calibration / "calibration.json").read_text())
    res = man = None
    if a.run:
        res, man = load_run(a.run)
    out = ARTS / (a.run or a.calibration) / "figures"
    fig1(cal, out)
    if cal.get("check"):
        fig2(cal, out, res if res and "activity_test" in res else None)
    if res and "point" in res:
        delta = float(man["config"]["analysis"]["practical_margin"])
        fig3(res, man, out)
        fig4(res, man, out, delta)
        fig5(res, man, out)
        fig6(res, man, out)
        fig7(res, man, out, ARTS / a.run)
    print("figures in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
