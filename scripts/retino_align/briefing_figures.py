#!/usr/bin/env python
"""Owner-facing figures for the protocol-freeze check-in (labels in French: they accompany messages to the owner;
the study's own figures, make_figures.py, are in English). Built from saved pilot data only.

    .venv/bin/python scripts/retino_align/briefing_figures.py --out artifacts/retino_align/briefing/freeze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/retino_align"))
from make_figures import AQUA, CONNECTOME, GRID, INK, INK2, NULLS, style  # noqa: E402

ARTS = ROOT / "artifacts/retino_align"


def schema(out: Path) -> None:
    """The experiment in one picture: what flows where, and what is compared."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14), ax.set_ylim(0, 6.2), ax.axis("off")

    def box(x, y, w, h, text, col=INK2, fill="#f4f3f0", fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=fill, ec=col,
                                    lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)

    def arrow(x0, y0, x1, y1, col=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=col))
    y = 3.6
    box(0.1, y, 1.9, 1.5, "Image fixe\nréseau de bandes\n4 orientations ×\n4 luminosités")
    box(2.5, y, 2.1, 1.5, "Œil artificiel\n892 colonnes\n3 575 cellules\nL1 L2 L3 L5")
    box(5.1, y, 2.0, 1.5, "Permutation ?\nordonné, ou\ncolonnes mélangées\n(3 tirages fixes)", col=CONNECTOME,
        fill="#fbe9e1")
    box(7.6, y, 2.4, 1.5, "Réseau figé\n169 984 neurones\n16 pas depuis le repos\naucun apprentissage")
    box(10.5, y, 1.6, 1.5, "2 227\ncomptes\nde sortie")
    box(12.4, y, 1.5, 1.5, "Décodeur\nlinéaire\n(seul à\napprendre)", col=NULLS, fill="#e3edf9")
    for x0, x1 in ((2.0, 2.5), (4.6, 5.1), (7.1, 7.6), (10.0, 10.5), (12.1, 12.4)):
        arrow(x0, y + 0.75, x1, y + 0.75)
    ax.text(8.8, y + 1.75, "le connectome mesuré  OU  l'un des 40 témoins recâblés (famille c)", ha="center",
            fontsize=9, color=INK2)
    ax.text(0.1, 2.75, "Ce qu'on mesure", fontsize=11, color=INK, weight="bold")
    lines = [
        "M  = précision du décodeur sur 2 560 images test, moyennée sur 5 budgets (8 à 128 exemples par classe),",
        "       2 graines de bruit et 10 tirages d'exemples d'entraînement",
        "L   = M(ordonné) − M(permuté)   →  combien le décodage perd quand on brouille la carte de l'œil",
        "D  = L(connectome) − moyenne des L(témoins)   →  le connectome dépend-il plus de l'alignement que les témoins ?",
        "      D > +0,05 : effet soutenu ;  D dans [−0,05 ; +0,05] : pas de différence importante ;  sinon : indéterminé",
        "Contrôle : la même chose avec la luminosité (une tâche qui ne demande pas de carte spatiale)",
    ]
    for i, t in enumerate(lines):
        ax.text(0.25, 2.3 - 0.36 * i, t, fontsize=9.5, color=INK if i % 2 == 0 or i == 3 else INK2,
                family="DejaVu Sans")
    fig.savefig(out / "01_schema_experience.png")
    plt.close(fig)


def permutation(out: Path, run: str) -> None:
    import matplotlib.pyplot as plt
    d = ARTS / run
    inp = np.load(d / "inputs/inputs.npz")
    pz = np.load(d / "permutations/permutations.npz")
    imgs = np.load(d / "inputs/example_images.npz")["train"]
    o = inp["orient"][inp["part"] == 0][: len(imgs)]
    i = int(np.flatnonzero(o == 1)[0])
    X = inp["X"][inp["part"] == 0][i]
    xy, col = pz["col_xy"], pz["col"]
    on = np.isin(pz["types"], ("L1", "L5"))
    v = np.array([X[(col == c) & on].mean() for c in range(len(xy))])
    vp = np.array([X[pz["perm1_sigma"]][(col == c) & on].mean() for c in range(len(xy))])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("1. l'image (bandes à 45°)", fontsize=10)
    for k, (vals, t) in enumerate(((v, "2. ce que reçoit l'œil, colonne par colonne\n(carte rétinotopique intacte)"),
                                   (vp, "3. même image, colonnes permutées\n(mêmes valeurs, autres places)"))):
        axes[k + 1].scatter(xy[:, 0], xy[:, 1], c=vals, cmap="gray", s=13, vmin=v.min(), vmax=v.max())
        axes[k + 1].set_aspect("equal")
        axes[k + 1].set_title(t, fontsize=10)
    for a in axes:
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    rep = json.loads((d / "manifest.json").read_text())["permutations"]["perm1"]
    fig.text(0.5, 0.02, f"Une colonne entière (ses 4 types de cellules ensemble) est déplacée : {100 * rep['frac_cells_moved']:.1f} % "
             f"des cellules bougent, de {rep['displacement_hex_units']['mean_moved']:.1f} colonnes en moyenne ; "
             f"{100 * rep['frac_neighbour_pairs_kept']:.1f} % des voisins restent voisins.",
             ha="center", fontsize=9.5, color=INK2)
    fig.savefig(out / "02_permutation.png")
    plt.close(fig)


def contrast_grid(out: Path) -> None:
    import matplotlib.pyplot as plt
    r = json.loads((ARTS / "pilot/P1.json").read_text())["results"]
    sets = list(r)
    amp = [f"{s}\n{r[s]['gen']['amp_range'][0]:.2f}-{r[s]['gen']['amp_range'][1]:.2f}" for s in sets]

    def m(s, k):
        return np.mean([r[s]["graphs"][g][k]["mean"] for g in r[s]["graphs"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(len(sets))
    ax = axes[0]
    ax.plot(x, [m(s, "orient_128") for s in sets], marker="o", color=NULLS, label="128 exemples par classe")
    ax.plot(x, [m(s, "orient_8") for s in sets], marker="o", color=NULLS, ls="--", alpha=0.7,
            label="8 exemples par classe")
    ax.axhline(0.40, color=CONNECTOME, lw=1.5)
    ax.text(-0.1, 0.405, "seuil déclaré avant : 0,40", color=CONNECTOME, fontsize=9)
    ax.axhline(0.25, color=INK2, ls=":", lw=1)
    ax.text(-0.1, 0.255, "hasard", color=INK2, fontsize=8)
    ax.scatter([2], [m("C", "orient_128")], s=160, facecolor="none", edgecolor=CONNECTOME, lw=2, zorder=4)
    ax.set_xticks(x, amp)
    ax.set_xlabel("réglage du contraste (amplitude des bandes)")
    ax.set_ylabel("précision (test)")
    ax.set_title("Orientation à la sortie du réseau (2 graphes réservés)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax = axes[1]
    ax.plot(x, [m(s, "lum_128") for s in sets], marker="o", color=AQUA, label="128 exemples par classe")
    ax.plot(x, [m(s, "lum_8") for s in sets], marker="o", color=AQUA, ls="--", alpha=0.7, label="8 exemples par classe")
    ax.axhline(0.25, color=INK2, ls=":", lw=1)
    ax.set_ylim(0.2, 0.9)
    ax.set_xticks(x, amp)
    ax.set_xlabel("réglage du contraste")
    ax.set_title("Luminosité : déjà bien transmise", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Pilote : avec le contraste proposé (A), l'orientation n'arrive pas à la sortie ; "
                 "C est le premier réglage qui passe la règle", fontsize=10)
    fig.savefig(out / "03_pilote_contraste.png")
    plt.close(fig)


def controls(out: Path) -> None:
    import matplotlib.pyplot as plt
    v = json.loads((ARTS / "pilot_e2e/validation/validation.json").read_text())
    c = json.loads((ARTS / "pilot/P2_controls.json").read_text())
    a = json.loads((ARTS / "pilot_e2e/metrics/analysis.json").read_text())
    curves = np.array(a["curves"])                                    # (G, C, T, K)
    rows = [
        ("Pixels (méthode indépendante, sans label)", v["pixel_orientation_control"]["accuracy"], INK2),
        ("Entrée de l'œil, carte intacte (décodeur direct)", v["direct_decoder_accuracy"]["orient"]["mean_by_budget"][-1], INK2),
        ("Entrée de l'œil, mêmes valeurs triées (sans carte)", c["non_spatial"]["sorted_values_per_type"]["orient"][-1], AQUA),
        ("Entrée : moyenne, variance, quantiles (sans carte)", c["non_spatial"]["global_descriptors"]["orient"][-1], AQUA),
        ("Sortie du réseau, témoins pilotes, ordonné", curves[:, 0, 0, -1].mean(), NULLS),
        ("Sortie du réseau, témoins pilotes, permuté", curves[:, 1:, 0, -1].mean(), NULLS),
        ("Labels d'entraînement mélangés (contrôle négatif)", c["label_shuffle"]["orient"]["selected_true_val"]["mean_by_budget"][-1], INK2),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (lab, val, col) in enumerate(rows[::-1]):
        ax.plot([0.25, val], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.scatter([val], [i], color=col, s=60, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(val + 0.012, i, f"{val:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9)
    ax.axvline(0.25, color=INK2, ls=":", lw=1)
    ax.text(0.252, len(rows) - 0.45, "hasard", fontsize=8, color=INK2)
    ax.set_xlim(0.2, 1.08)
    ax.set_xlabel("précision d'orientation (128 exemples par classe ; 4 classes)")
    ax.set_title("Contrôles : où se trouve l'information d'orientation (pilote, graphes réservés)", fontsize=10)
    fig.savefig(out / "04_controles.png")
    plt.close(fig)


def results(out: Path, run: str) -> None:
    """End-of-mission check-in: the main result, in four panels the owner can read on a phone."""
    import matplotlib.pyplot as plt
    d = ARTS / run
    r = json.loads((d / "metrics/analysis.json").read_text())
    ex = json.loads((d / "metrics/exploratory.json").read_text())
    curves = np.array(r["curves"])
    budgets = 4 * np.array(json.loads((d / "manifest.json").read_text())["config"]["decoder"]["per_combination"])
    graphs = r["graphs_complete"]
    # R1 curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ti, (t, name) in enumerate((("orient", "orientation"), ("lum", "luminosité"))):
        ax = axes[ti]
        for gi in range(1, len(graphs)):
            ax.plot(budgets, curves[gi, 0, ti], color=NULLS, lw=0.7, alpha=0.35)
        ax.plot(budgets, curves[1:, 0, ti].mean(0), color=NULLS, lw=2.2, label="40 témoins (moyenne en gras)")
        ax.plot(budgets, curves[1:, 1:, ti].mean(axis=(0, 1)), color=NULLS, lw=2.2, ls="--",
                label="témoins, colonnes permutées")
        ax.plot(budgets, curves[0, 0, ti], color=CONNECTOME, lw=2.8, marker="o", label="connectome")
        ax.plot(budgets, curves[0, 1:, ti].mean(0), color=CONNECTOME, lw=2.2, ls="--",
                label="connectome, colonnes permutées")
        ax.axhline(0.25, color=INK2, ls=":", lw=1)
        ax.text(budgets[0], 0.255, "hasard", color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("exemples d'entraînement par classe")
        ax.set_ylabel("précision sur 2 560 images test")
        ax.set_title(name, fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Ce que le décodeur lit à la sortie du réseau (trait plein : œil normal ; tirets : colonnes permutées)",
                 fontsize=10, y=1.02)
    fig.savefig(out / "05_courbes.png")
    plt.close(fig)
    # R2 contrasts with the margin and the connectome's headroom
    delta = 0.05
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhspan(-delta, delta, color=GRID, alpha=0.8, lw=0)
    ax.text(2.55, 0.044, "zone « pas de différence\nimportante » (±5 points)", fontsize=8, color=INK2, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    keys = [("D_orientation", "D orientation\n(résultat principal)"), ("G", "G : avantage initial\nen orientation"),
            ("D_brightness", "D luminosité\n(contrôle)")]
    for i, (k, lab) in enumerate(keys):
        lo, hi = r["bootstrap"]["intervals_95"][k]
        p = r["point"][k]
        col = CONNECTOME if i == 0 else INK2
        ax.plot([i, i], [lo, hi], color=col, lw=3, solid_capstyle="round")
        ax.scatter([i], [p], color=col, s=60, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(i + 0.1, p, f"{100 * p:+.1f} pts", fontsize=9, va="center")
    mx = ex["E1_headroom"]["max_D_if_connectome_fell_to_chance"]
    ax.plot([-0.3, 0.3], [mx, mx], color=CONNECTOME, ls=":", lw=1.5)
    ax.text(-0.45, mx + 0.003, f"D maximal possible pour ce connectome : +{100 * mx:.1f} pts", fontsize=8,
            color=CONNECTOME)
    ax.set_xticks(range(3), [k[1] for k in keys])
    ax.set_xlim(-0.6, 3.2)
    ax.set_ylabel("différence de précision (proportion)")
    ax.set_title("Barres = intervalles à 95 % (bootstrap conditionnel, 5 000 tirages)", fontsize=9)
    fig.savefig(out / "06_resultat_principal.png")
    plt.close(fig)
    # R3 losses and activity (one small panel per activity measure: different units, never one shared axis)
    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.4, 0.15, 1, 1, 1], wspace=0.45)
    rng = np.random.default_rng(0)
    Lo = np.array([r["L"][g]["orient"] for g in graphs])
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(100 * Lo[1:], rng.uniform(-0.2, 0.2, len(Lo) - 1), color=NULLS, s=30, alpha=0.8,
               label="un point = un témoin")
    ax.axvline(100 * Lo[0], color=CONNECTOME, lw=3, label="connectome")
    ax.axvline(0, color=INK2, ls=":", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.32, 0.42)
    ax.set_xlabel("perte d'orientation quand on permute les colonnes (points de %)")
    ax.set_title("Le connectome ne perd pas plus que les témoins", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    act = ex["E4_activity_ordered"]
    items = [("mean_count", "décharges par neurone\nde sortie (sur 16 pas)", 1),
             ("frac_near_ceiling_entries_ge7", "sorties saturées\n(≥ 7 sur 8), %", 100),
             ("rate_all", "taux global\nde décharge, %", 100)]
    for i, (k, lab, sc) in enumerate(items):
        ax = fig.add_subplot(gs[0, 2 + i])
        a = act[k]
        ax.plot([0, 0], [sc * a["nulls_min"], sc * a["nulls_max"]], color=NULLS, lw=14, alpha=0.5,
                solid_capstyle="butt")
        ax.scatter([0], [sc * a["connectome"]], color=CONNECTOME, s=80, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(0.08, sc * a["connectome"], f"{sc * a['connectome']:.2f}", fontsize=8, va="center")
        ax.set_xlim(-0.3, 0.4)
        ax.set_xticks([])
        ax.set_title(lab, fontsize=9)
    fig.text(0.72, -0.02, "barre bleue = plage des 40 témoins ; point orange = connectome", ha="center", fontsize=9,
             color=INK2)
    fig.savefig(out / "07_pertes_et_activite.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ARTS / "briefing/freeze")
    ap.add_argument("--run", default="main_v1")
    ap.add_argument("--results", action="store_true", help="the end-of-mission figures only")
    a = ap.parse_args()
    style()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.results:
        results(a.out, a.run)
        print("wrote", sorted(p.name for p in a.out.glob("*.png")))
        return 0
    schema(a.out)
    permutation(a.out, a.run)
    contrast_grid(a.out)
    controls(a.out)
    print("wrote", sorted(p.name for p in a.out.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
