#!/usr/bin/env python
"""EXPLORATORY X2: learning (mode N, blocks b0-b2) of the P2/P3 nulls at matched activity vs at the common gain.

Contrasts on L (greedy-learning area): G_common = L(connectome) - mean L(nulls, gamma 1); G_matched = the same
with nulls at their matched gamma; D_nulls = their difference. Descriptive 95 % bootstrap (2,000) over blocks,
nulls and images. Also reports each graph's untrained readout rate on block b0's final bank.

    PYTHONPATH=src .venv/bin/python scripts/experience_commune/matched_activity_analysis.py
    PYTHONPATH=src .venv/bin/python scripts/experience_commune/matched_activity_analysis.py --x3 [N|U]
        X3: nulls rerun at matched gain with 24 learners (complements/matched_activity_full), blocks b0-b5;
        graphs without a complete run are left out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"
X2 = ART / "complements/matched_activity"
CK = (0, 250, 500, 1000, 2000, 4000, 6000)
TASKS = ("where", "loom")
BLOCKS = ("b0", "b1", "b2")
X3 = "--x3" in sys.argv
if X3:
    X2 = ART / "complements/matched_activity_full"
    BLOCKS = ("b0", "b1", "b2", "b3", "b4", "b5")
    MODE = sys.argv[sys.argv.index("--x3") + 1] if len(sys.argv) > sys.argv.index("--x3") + 1 else "U"
else:
    MODE = "N"


def correctness(run_dir: Path, mode="N"):
    """prog (T, R, C, n), final (T, R, n') for the learners of one mode and the selected blocks."""
    r = json.loads((run_dir / "run.json").read_text())
    z = np.load(run_dir / "predictions.npz")
    prog = np.zeros((2, len(BLOCKS), len(CK), z["labels_progress"].shape[1]), np.uint8)
    fin = np.zeros((2, len(BLOCKS), z["labels_final"].shape[1]), np.uint8)
    found = 0
    for j, l in enumerate(r["learners"]):
        if l["mode"] != mode or l["block"] not in BLOCKS or l.get("kind", "main") != "main":
            continue
        ti, ri = TASKS.index(l["task"]), BLOCKS.index(l["block"])
        for ci, c in enumerate(CK):
            prog[ti, ri, ci] = z[f"progress_{c}"][j] == z["labels_progress"][j]
        fin[ti, ri] = z["final"][j] == z["labels_final"][j]
        found += 1
    if found != 2 * len(BLOCKS):
        raise RuntimeError(f"{run_dir}: {found} learners")
    return prog, fin, None


def rates(graphs, gammas):
    """Readout rate (Hz) of the untrained network on block b0's final bank (both tasks) at the given gain."""
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.retino_align import simulate
    ticks = int(config.get("timing.ticks_per_decision"))
    data = {t: dict(np.load(ART / "data/b0" / f"{t}.npz")) for t in TASKS}
    out = {}
    for g in graphs:
        _g, W, motor = graphsel.load(g)
        p = LIFParams(gain=float(config.get("dynamics.gain")) * gammas[g], noise_std=float(config.get("dynamics.noise_std")))
        net = simulate.FrozenNet(W, motor, data["where"]["idx"], p, ticks, batch=128)
        out[g] = {t: float(net.run(data[t]["final_x"][:512], data[t]["final_uid"][:512], 31337).counts.mean()
                           / (ticks * 3.33e-3)) for t in TASKS}
        del net
        torch.cuda.empty_cache()
    return out


def main() -> int:
    from flymario.experience_commune import analysis as AN
    plan = json.loads((X2 / "plan.json").read_text())
    nulls = plan["graphs"]
    if X3:
        nulls = [g for g in nulls if (X2 / g.replace(":", "") / "run.json").exists()
                 and json.loads((X2 / g.replace(":", "") / "run.json").read_text()).get("complete")]
    graphs = ["connectome"] + nulls
    roles = np.array(["connectome"] + ["b" if g.startswith("nulllat") else "c" for g in nulls])
    n_p, n_f = 512, 2048
    R = len(BLOCKS)
    prog = np.zeros((len(graphs), 2, 2, R, len(CK), n_p), np.uint8)       # mode axis: 0 common gain, 1 matched
    fin = np.zeros((len(graphs), 2, 2, R, n_f), np.uint8)
    for gi, g in enumerate(graphs):
        pc, fc, _ = correctness(ART / "runs" / g.replace(":", ""), MODE)
        prog[gi, :, 0], fin[gi, :, 0] = pc, fc
        if g == "connectome":
            prog[gi, :, 1], fin[gi, :, 1] = pc, fc                     # the connectome stays at gamma 1
        else:
            pm, fm, _ = correctness(X2 / g.replace(":", ""), MODE)
            prog[gi, :, 1], fin[gi, :, 1] = pm, fm
    z = np.load(ART / "runs/connectome/predictions.npz")
    r = json.loads((ART / "runs/connectome/run.json").read_text())
    cls_f = np.stack([[z["labels_final"][[j for j, l in enumerate(r["learners"]) if l["mode"] == MODE and l["task"] == t
                                          and l["block"] == b][0]] for t in TASKS] for b in BLOCKS])
    cls_pr = np.stack([[z["labels_progress"][[j for j, l in enumerate(r["learners"]) if l["mode"] == MODE and
                                              l["task"] == t and l["block"] == b][0]] for t in TASKS] for b in BLOCKS])
    Ap, Af = prog.mean(-1), fin.mean(-1)
    pt = AN.stats(Ap, Af, roles)
    bs = AN.bootstrap(prog, fin, cls_pr, cls_f, roles, 2000, 20261022)
    iv = {k: AN.interval(v, 0.95) for k, v in bs.items()}
    res = {"label": "EXPLORATORY (decided after EXPERIENCE_COMMUNE_V1's result)", "graphs": graphs, "mode": MODE,
           "blocks": list(BLOCKS),
           "gammas": plan["gammas"], "note": "mode index 0 = nulls at gamma 1 (campaign runs), 1 = nulls at their matched "
                                             "gamma; the connectome is at gamma 1 in both", "contrasts": {}}
    for t in TASKS:
        for f in ("b", "c"):
            k0, k1 = f"L_G_{f}_{t}_N", f"L_G_{f}_{t}_U"
            res["contrasts"][f"{t}/{f}"] = {
                "G_common": [pt[k0], *iv[k0]], "G_matched": [pt[k1], *iv[k1]],
                "L_connectome": pt[f"L_connectome_{t}_N"], "L_nulls_common": pt[f"L_{f}_{t}_N"],
                "L_nulls_matched": pt[f"L_{f}_{t}_U"],
                "final_connectome": pt[f"final_connectome_{t}_N"], "final_nulls_common": pt[f"final_{f}_{t}_N"],
                "final_nulls_matched": pt[f"final_{f}_{t}_U"],
                "A0_nulls_common": pt[f"A0_{f}_{t}_N"], "A0_nulls_matched": pt[f"A0_{f}_{t}_U"]}
    # mean greedy accuracy per checkpoint (descriptive curves): role x condition, averaged over graphs and blocks
    res["checkpoints"] = list(CK)
    res["curves"] = {f"{t}/{r}/{cond}": Ap[roles == r, ti, ci].mean(axis=(0, 1)).tolist()
                     for ti, t in enumerate(TASKS) for r in ("connectome", "b", "c")
                     for ci, cond in enumerate(("common", "matched"))}
    if X3:
        res["rates_hz_b0_untrained"] = json.loads((ART / "complements/matched_activity/analysis.json").read_text()
                                                  )["rates_hz_b0_untrained"]
    else:
        res["rates_hz_b0_untrained"] = rates(graphs, {**{g: plan["gammas"][g] for g in nulls}, "connectome": 1.0})
    (X2 / (f"analysis_{MODE}.json" if X3 else "analysis.json")).write_text(json.dumps(res, indent=1))
    for k, v in res["contrasts"].items():
        print(k, "G_common %+.1f [%+.1f, %+.1f]" % tuple(100 * x for x in v["G_common"]),
              "G_matched %+.1f [%+.1f, %+.1f]" % tuple(100 * x for x in v["G_matched"]),
              "L nulls %.1f -> %.1f" % (100 * v["L_nulls_common"], 100 * v["L_nulls_matched"]))
    rr = res["rates_hz_b0_untrained"]
    for t in TASKS:
        print(t, "rate conn %.1f" % rr["connectome"][t], "b %.1f-%.1f" % (min(rr[g][t] for g in nulls if g.startswith("nulllat")),
              max(rr[g][t] for g in nulls if g.startswith("nulllat"))), "c %.1f-%.1f" % (
              min(rr[g][t] for g in nulls if g.startswith("nullin")), max(rr[g][t] for g in nulls if g.startswith("nullin"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
