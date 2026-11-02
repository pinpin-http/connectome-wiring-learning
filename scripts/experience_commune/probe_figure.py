"""Figure for the exploratory representation probe (French): class information in the untrained readout (8 fixed
scores) against the learning area L under common experience (U), each graph."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from flymario.experience_commune.analysis import trapezoid_weights  # noqa: E402

ART = ROOT / "artifacts/experience_commune_v1"
p = json.loads((ART / "complements/probe_representation/probe.json").read_text())["graphs"]
a = json.loads((ART / "metrics/analysis.json").read_text())
graphs, roles = a["graphs"], np.array(a["roles"])
L = np.array(a["curves_per_graph"]) @ trapezoid_weights()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
col = {"connectome": "#c2410c", "b": "#6b7280", "c": "#2563eb"}
for ti, (t, name) in enumerate((("where", "« Où ? »"), ("loom", "« Approche ou recul ? »"))):
    ax = axes[ti]
    x = np.array([100 * p[g][f"{t}_scores8"]["M"] for g in graphs])
    y = 100 * L[:, ti, 1]
    for r, lab in (("b", "témoins b"), ("c", "témoins c"), ("connectome", "connectome")):
        m = roles == r
        ax.scatter(x[m], y[m], s=80 if r == "connectome" else 22, marker="D" if r == "connectome" else "o",
                   color=col[r], label=lab, zorder=3 if r == "connectome" else 2, edgecolor="white")
    ax.set_xlabel("information de classe AVANT apprentissage\n(décodage des 8 scores de départ, %)")
    if ti == 0:
        ax.set_ylabel("apprentissage à expérience commune\n(aire L en mode U, %)")
    ax.set_title(name, fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Complément exploratoire : les témoins b « voient » au départ aussi bien que le connectome pour certains, "
             "mais apprennent bien moins bien.\nL'information disponible avant apprentissage n'explique pas l'avantage "
             "d'apprentissage (tous les graphes atteignent ~100 % depuis les 2 227 neurones).", fontsize=9.5, y=1.04)
out = ART / "complements/probe_representation/fig_sonde_representation.png"
fig.savefig(out, bbox_inches="tight")
print(out)
