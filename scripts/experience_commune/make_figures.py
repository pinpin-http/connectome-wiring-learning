#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 figures (mission §19), from metrics/analysis.json and the per-run files only.

    .venv/bin/python scripts/experience_commune/make_figures.py [--metrics metrics]

Individual nulls stay visible; the connectome is the coloured mark. Accuracies in %, contrasts in points of L.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experience_commune_v1"
CONN, B_COL, C_COL, BAND = "#c2410c", "#6b7280", "#2563eb", "#e5e7eb"
CK = np.array([0, 250, 500, 1000, 2000, 4000, 6000])
TASK_NAME = {"where": "Où ? (where)", "loom": "Approche ou recul ? (loom)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_curves(a: dict, out: Path) -> None:
    cur = 100 * np.array(a["curves_per_graph"])                   # (G, T, M, C)
    roles = np.array(a["roles"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey="row")
    fig.subplots_adjust(hspace=0.42)
    for ti, t in enumerate(("where", "loom")):
        for mi, m in enumerate(("N", "U")):
            ax = axes[ti, mi]
            for gi, r in enumerate(roles):
                if r == "connectome":
                    continue
                ax.plot(CK, cur[gi, ti, mi], "-", color=B_COL if r == "b" else C_COL, lw=0.5, alpha=0.45)
            for r, col in (("b", B_COL), ("c", C_COL)):
                ax.plot(CK, cur[roles == r, ti, mi].mean(0), "--", color=col, lw=2)
            ax.plot(CK, cur[roles == "connectome", ti, mi][0], "-o", color=CONN, lw=2.2, ms=3.5)
            ax.axhline(12.5, color="k", lw=0.5, ls=":")
            ax.set_title(f"{TASK_NAME[t]}, mode {m} ({'own policy' if m == 'N' else 'common experience'})")
            ax.set_xlabel("training decisions")
            if mi == 0:
                ax.set_ylabel("greedy accuracy on the progress bank (%)")
    axes[0, 0].plot([], [], "-o", color=CONN, label="connectome")
    axes[0, 0].plot([], [], "--", color=B_COL, lw=2, label="family b (mean; thin: each graph)")
    axes[0, 0].plot([], [], "--", color=C_COL, lw=2, label="family c (mean; thin: each graph)")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Greedy learning curves (mean over blocks); L = normalised area under each curve", fontsize=10)
    save(fig, out, "fig1_learning_curves")


def fig_contrasts(a: dict, out: Path, delta: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhspan(-100 * delta, 100 * delta, color=BAND, zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    labels = []
    for i, (k, v) in enumerate(a["primary"].items()):
        t, f = k.split("/")
        for dx, mode, key, ivk, col in ((-0.15, "U", "G_U", "interval_98.75", "#047857"),
                                        (0.15, "N", "G_N", "G_N_interval_95", "#9ca3af")):
            p = 100 * v[key]
            lo, hi = (100 * x for x in v[ivk])
            ax.errorbar(i + dx, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=4, ms=6, lw=1.6)
            ax.text(i + dx + 0.05, p, f"{p:+.1f}", fontsize=7.5, va="center")
        labels.append(f"{TASK_NAME[t].split(' (')[0]}\nvs family {f}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("connectome - family mean, L (points)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="#047857", ls="", label="G_U (primary, 98.75 % interval)"),
                       plt.Line2D([], [], marker="o", color="#9ca3af", ls="", label="G_N (reference, 95 % interval)"),
                       Patch(color=BAND, label=f"practical margin +-{100 * delta:.0f} points")],
              frameon=False, fontsize=8, loc="best")
    ax.set_title("Primary contrasts under common experience (U) and the own-policy reference (N)")
    save(fig, out, "fig2_contrasts")


def fig_interaction(a: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axhline(0, color="k", lw=0.5)
    labels = []
    for i, (k, v) in enumerate(a["primary"].items()):
        p = 100 * v["I"]
        lo, hi = (100 * x for x in v["I_interval_95"])
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="s", color="#1f2937", capsize=4, ms=6)
        ax.text(i + 0.08, p, f"{p:+.1f}", fontsize=8, va="center")
        t, f = k.split("/")
        labels.append(f"{t}\nvs {f}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("I = G_U - G_N (points of L)")
    ax.set_title("Interaction: does the relative advantage change under common experience? (95 %)")
    save(fig, out, "fig3_interaction")


def fig_final(a: dict, out: Path, thr: float, metrics: Path) -> None:
    roles = np.array(a["roles"])
    tab = np.genfromtxt(metrics / "runs_table.csv", delimiter=",", names=True, dtype=None, encoding=None)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    graphs = a["graphs"]
    for ti, t in enumerate(("where", "loom")):
        ax = axes[ti]
        for mi, m in enumerate(("N", "U")):
            sel = (tab["task"] == t) & (tab["mode"] == m) & (tab["checkpoint"] == 6000)
            fin = {}
            for g in graphs:
                s2 = sel & (tab["graph"] == g)
                fin[g] = 100 * tab["final_accuracy"][s2].mean()
            for r, col, dx in (("b", B_COL, -0.18), ("c", C_COL, 0.0)):
                v = [fin[g] for g, rr in zip(graphs, roles) if rr == r]
                ax.plot(mi * 1.2 + dx + np.random.default_rng(mi).uniform(-0.05, 0.05, len(v)), v, "o", ms=3,
                        color=col, alpha=0.7)
            ax.plot(mi * 1.2 + 0.18, fin["connectome"], "D", ms=8, color=CONN)
        ax.axhline(100 * thr, color="#b45309", lw=0.9, ls="--")
        ax.axhline(12.5, color="k", lw=0.5, ls=":")
        ax.set_xticks([0, 1.2], ["N (own policy)", "U (common experience)"])
        ax.set_title(TASK_NAME[t])
        if ti == 0:
            ax.set_ylabel("final greedy accuracy, independent bank (%)")
    axes[0].legend(handles=[plt.Line2D([], [], marker="D", color=CONN, ls="", label="connectome"),
                            plt.Line2D([], [], marker="o", color=B_COL, ls="", label="family b"),
                            plt.Line2D([], [], marker="o", color=C_COL, ls="", label="family c"),
                            plt.Line2D([], [], color="#b45309", ls="--", label=f"informative threshold {100 * thr:.0f} %")],
                   frameon=False, fontsize=7.5)
    save(fig, out, "fig4_final_accuracy")


def fig_acquisition(a: dict, out: Path) -> None:
    p, iv = a["point"], a["intervals_95"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ai, (name, lab) in enumerate((("A0", "initial accuracy A(0)"), ("gain", "progress A(6000) - A(0)"),
                                      ("Lgain", "area of progress L - A(0)"))):
        ax = axes[ai]
        xt = []
        for i, (t, m) in enumerate([(t, m) for t in ("where", "loom") for m in ("N", "U")]):
            for dx, r, col in ((-0.2, "connectome", CONN), (0.0, "b", B_COL), (0.2, "c", C_COL)):
                k = f"{name}_{r}_{t}_{m}"
                lo, hi = (100 * x for x in iv[k])
                ax.errorbar(i + dx, 100 * p[k], yerr=[[100 * p[k] - lo], [hi - 100 * p[k]]], fmt="o", color=col,
                            ms=4, capsize=2)
            xt.append(f"{t}\n{m}")
        ax.set_xticks(range(4), xt)
        ax.set_title(lab)
        ax.set_ylabel("%")
    axes[0].legend(handles=[plt.Line2D([], [], marker="o", color=c, ls="", label=l)
                            for c, l in ((CONN, "connectome"), (B_COL, "family b"), (C_COL, "family c"))],
                   frameon=False, fontsize=7.5)
    fig.suptitle("Performance versus acquisition (secondary, 95 % intervals)", fontsize=10)
    save(fig, out, "fig5_initial_and_progress")


def fig_diagnostics(a: dict, out: Path) -> None:
    d = a["diagnostics"]
    items = [("entropy_first500", "target-policy entropy, first 500 (nats)"),
             ("entropy_last500", "target-policy entropy, last 500 (nats)"),
             ("reward_rate", "reward rate during training"), ("rho_p99", "rho, 99th percentile"),
             ("upd_pre_mean", "update RMS before normalisation"), ("active_out_frac", "active readout fraction")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, lab) in zip(axes.ravel(), items):
        xt = []
        for i, (t, m) in enumerate([(t, m) for t in ("where", "loom") for m in ("N", "U")]):
            for dx, r, col in ((-0.2, "connectome", CONN), (0.0, "b", B_COL), (0.2, "c", C_COL)):
                ax.bar(i + dx, d[f"{t}_{m}"][k][r], width=0.2, color=col)
            xt.append(f"{t}\n{m}")
        ax.set_xticks(range(4), xt)
        ax.set_title(lab, fontsize=9)
    axes[0, 0].legend(handles=[Patch(color=c, label=l) for c, l in ((CONN, "connectome"), (B_COL, "family b"),
                                                                     (C_COL, "family c"))], frameon=False, fontsize=7.5)
    fig.suptitle("Diagnostics (descriptive; the behaviour of U is uniform by construction, its target policy is not)",
                 fontsize=10)
    save(fig, out, "fig6_diagnostics")


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics")
    ap.add_argument("--out", default="figures")
    a_ = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/experience_commune_v1.yaml").read_text())
    a = json.loads((ART / a_.metrics / "analysis.json").read_text())
    out = ART / a_.out
    fig_curves(a, out)
    fig_contrasts(a, out, float(cfg["analysis"]["practical_margin"]))
    fig_interaction(a, out)
    fig_final(a, out, float(cfg["analysis"]["informative_final_accuracy_threshold"]), ART / a_.metrics)
    fig_acquisition(a, out)
    fig_diagnostics(a, out)
    print("figures in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
