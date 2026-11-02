#!/usr/bin/env python
"""REVISION48 declared reading: G per condition, task and family; D after tuning; intervals and readings.

    analyse.py [--partial] [--replicates N]   -> artifacts/revision48/analysis/report.json and report.md

Refuses while a planned run is incomplete, unless --partial (then every output is labelled PARTIAL and lists the
missing runs). The autapse condition's nulls are ordered by the seeds of the analysed nulls so that the paired
bootstrap pairs them; its connectome is the reference run (same numerical plan).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/revision48"


def complete(d: Path) -> bool:
    if (d / "reuse.json").exists():
        d = d.parents[1] / "reference" / d.name
    rj = d / "run.json"
    return rj.exists() and bool(json.loads(rj.read_text()).get("complete")) and (d / "predictions.npz").exists()


def main() -> int:
    from flymario.revision48 import analysis as A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--replicates", type=int, default=None)
    a = ap.parse_args()
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    an = c["analysis"]
    analysed = json.loads((ROOT / c["graphs"]["manifest"]).read_text())["order"]
    autapse = ["connectome"] + [s.replace("nulllat:", "nulllatL:").replace("nullin:", "nullinL:") for s in analysed[1:]]
    plan = {}
    for cond in ("reference", "tuned", "supervised", "lognorm", "autapse"):
        specs = autapse if cond == "autapse" else analysed
        dirs = [ART / "runs/main" / ("reference" if (cond == "autapse" and s == "connectome") else cond) / s.replace(":", "")
                for s in specs]
        plan[cond] = (specs, dirs)
    missing = [f"{cond}:{s}" for cond, (specs, dirs) in plan.items() for s, d in zip(specs, dirs) if not complete(d)]
    if missing and not a.partial:
        print(f"refused: {len(missing)} planned runs incomplete (first: {missing[:5]}); --partial labels a partial reading")
        return 2
    conds = {}
    for cond, (specs, dirs) in plan.items():
        keep = [i for i, d in enumerate(dirs) if complete(d)]
        if not keep or specs[0] != "connectome" or 0 not in keep:
            continue
        conds[cond] = A.load_condition([dirs[i] for i in keep], [specs[i] for i in keep], c["tasks"],
                                       c["blocks"]["main"], c["checkpoints"])
    rep = A.report(conds, c["tasks"], c["checkpoints"], a.replicates or an["bootstrap"]["replicates"],
                   an["bootstrap"]["seed"], an["margin"], primary=an["primary_condition"])
    rep["status"] = "PARTIAL" if missing else "complete"
    rep["missing_runs"] = missing
    rep["graphs"] = {k: v[0] for k, v in plan.items()}
    out = ART / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    A.save_report(out / "report.json", rep)
    lines = [f"# REVISION48 reading ({rep['status']})", "", "| condition | task | family | G (points) | interval | level | reading |",
             "|---|---|---|---|---|---|---|"]
    for r in rep["contrasts"]:
        lines.append(f"| {r['condition']}{' (primary)' if r['primary'] else ''} | {r['task']} | {r['family']} | "
                     f"{100 * r['G']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {r['level']:.4g} | {r['reading']} |")
    lines += ["", "| change after tuning D = G_tuned - G_reference | task | family | D (points) | 95 % interval |",
              "|---|---|---|---|---|"]
    for d in rep["change_after_tuning"]:
        lines.append(f"| | {d['task']} | {d['family']} | {100 * d['D']:+.1f} | [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |")
    if missing:
        lines += ["", f"PARTIAL: {len(missing)} planned runs missing: " + ", ".join(missing)]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
