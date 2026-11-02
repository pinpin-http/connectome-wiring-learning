"""Figure for the exploratory X2 (French): learning area L, connectome at gamma 1 versus the nulls at the common gain and
at the connectome's activity (mode N, blocks b0-b2)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = Path(__file__).resolve().parents[2] / "artifacts/experience_commune_v1/complements/matched_activity"
a = json.loads((ART / "analysis.json").read_text())["contrasts"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ti, (t, name) in enumerate((("where", "« Où ? »"), ("loom", "« Approche ou recul ? »"))):
    ax = axes[ti]
    xs, labels = [], []
    for fi, f in enumerate(("b", "c")):
        v = a[f"{t}/{f}"]
        x0 = fi * 3.2
        bars = [(x0, 100 * v["L_nulls_common"], "#9ca3af", "réglage commun"),
                (x0 + 1, 100 * v["L_nulls_matched"], "#2563eb", "activité du connectome")]
        for x, h, c, lab in bars:
            ax.bar(x, h, width=0.9, color=c, label=lab if fi == 0 and ti == 0 else None)
            ax.text(x, h + 0.8, f"{h:.0f}", ha="center", fontsize=8.5)
        xs.append(x0 + 0.5)
        labels.append(f"témoins {f}")
    lc = 100 * a[f"{t}/b"]["L_connectome"]
    ax.axhline(lc, color="#c2410c", lw=2.2, label="connectome (réglage normal)" if ti == 0 else None)
    ax.text(4.6, lc + 0.8, f"connectome {lc:.0f}", color="#c2410c", ha="right", fontsize=8.5)
    ax.set_xticks(xs, labels)
    ax.set_title(name)
    if ti == 0:
        ax.set_ylabel("aire d'apprentissage L (%), mode N, 3 blocs")
fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Complément exploratoire : ramener les témoins à l'activité du connectome ne leur fait pas rattraper "
             "son avance d'apprentissage", fontsize=10, y=1.03)
fig.savefig(ART / "fig_apprentissage_activite_egale.png", bbox_inches="tight")
print("saved")
