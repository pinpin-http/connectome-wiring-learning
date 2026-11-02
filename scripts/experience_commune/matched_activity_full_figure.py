"""Figure for the exploratory X3 (French): learning curves of the connectome (gamma 1) and of the nulls at the common
gain and at the connectome's activity, modes U and N, both tasks; contrasts with their intervals."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X3 = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parents[2]
                                                  / "artifacts/experience_commune_v1/complements/matched_activity_full")
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
A = {m: json.loads((X3 / f"analysis_{m}.json").read_text()) for m in ("U", "N")}
fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
TN = {"where": "« Où ? »", "loom": "« Approche ou recul ? »"}
MN = {"U": "expérience commune (U)", "N": "propre politique (N)"}
for mi, m in enumerate(("U", "N")):
    a = A[m]
    ck = a["checkpoints"]
    for ti, t in enumerate(("where", "loom")):
        ax = axes[mi, ti]
        c = a["curves"]
        ax.plot(ck, [100 * x for x in c[f"{t}/connectome/common"]], color="#c2410c", lw=2.4, label="connectome")
        for f, col in (("b", "#6b7280"), ("c", "#2563eb")):
            ax.plot(ck, [100 * x for x in c[f"{t}/{f}/common"]], color=col, lw=1.4, ls="--",
                    label=f"témoins {f}, réglage commun")
            ax.plot(ck, [100 * x for x in c[f"{t}/{f}/matched"]], color=col, lw=2, label=f"témoins {f}, activité égale")
        ax.axhline(12.5, color="k", lw=0.6, ls=":")
        ax.set_ylim(0, 100)
        ax.set_title(f"{TN[t]}, {MN[m]}", fontsize=10)
        if ti == 0:
            ax.set_ylabel("précision sur images nouvelles (%)")
        if mi == 1:
            ax.set_xlabel("décisions d'apprentissage")

    ax = axes[mi, 2]
    rows = [(f"{TN[t]} · {f}", a["contrasts"][f"{t}/{f}"]) for t in ("where", "loom") for f in ("b", "c")]
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        for k, (key, col, dy) in enumerate((("G_common", "#9ca3af", 0.15), ("G_matched", "#2563eb", -0.15))):
            p, lo, hi = (100 * x for x in v[key])
            ax.errorbar(p, y + dy, xerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=3, ms=5,
                        label=("réglage commun" if k == 0 else "activité égale") if i == 0 else None)
            if key == "G_matched":
                ax.text(hi + 0.8, y + dy, f"{p:+.1f} [{lo:+.1f} ; {hi:+.1f}]", va="center", fontsize=7.5)
    ax.axvspan(-3, 3, color="#dcfce7", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlim(-5, 45)
    ax.set_title(f"avance du connectome (aire L, points), {m}", fontsize=10)
    if mi == 0:
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
n = len(A["U"]["graphs"]) - 1
fig.suptitle(f"Complément exploratoire X3 : les témoins ramenés à l'activité du connectome apprennent-ils aussi bien que lui ? "
             f"({n} témoins, 6 blocs ; IC 95 % descriptifs ; zone verte : marge de 3 points)", fontsize=10.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=5, bbox_to_anchor=(0.36, -0.035))
fig.tight_layout()
fig.savefig(X3 / "fig_x3_activite_egale.png", bbox_inches="tight")
print("saved", X3 / "fig_x3_activite_egale.png")
