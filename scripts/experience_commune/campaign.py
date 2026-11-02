#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 main campaign launcher (frozen protocol only).

One process per graph (all its tasks, modes and blocks in one batch), `--workers` processes at a time. A graph whose
run.json says complete is skipped; a failed process is restarted once from its last checkpoint (technical retry,
mission §20); no new graph starts if it could not finish before the campaign deadline (H20). No score is read or
printed here: the progress file holds counts and times only (mission §3).

    .venv/bin/python scripts/experience_commune/campaign.py --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experience_commune_v1"
PY = "/workspace/SMB_Flywire_Speedrun/.venv/bin/python"


def utc() -> datetime:
    return datetime.now(UTC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/experience_commune_v1.yaml").read_text())
    if cfg["protocol_status"] != "frozen" or not cfg["design"]["selected_plan"]:
        print("refused: protocol not frozen or plan not selected")
        return 2
    plan = cfg["design"]["selected_plan"]
    sel = json.loads((ROOT / cfg["design"]["graph_manifest"]).read_text())
    if plan == "P1":
        b, c = sel["families"]["b"]["P1"], sel["families"]["c"]["P1"]
    else:
        b, c = sel["families"]["b"]["P2_P3"], sel["families"]["c"]["P2_P3"]
    graphs = ["connectome"] + [g for pair in zip(b, c) for g in pair]        # alternate families
    blocks = cfg["design"]["blocks_P3" if plan == "P3" else "blocks_P1_P2"]
    banks = "final,ood" if cfg["design"]["include_ood"] else "final"
    deadline = datetime.fromisoformat(cfg["budget"]["main_campaign_must_end_utc"].replace("Z", "+00:00"))
    per_graph_s = float(cfg["design"]["projected_graph_wall_s"])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    status_p = ART / "campaign_status.json"
    st = {"plan": plan, "graphs": graphs, "blocks": blocks, "banks": banks, "workers": a.workers,
          "started_utc": utc().isoformat(timespec="seconds"), "done": [], "failed": [], "retried": [],
          "skipped_deadline": [], "running": {}}

    def complete(g):
        p = ART / "runs" / g.replace(":", "") / "run.json"
        return p.exists() and json.loads(p.read_text()).get("complete")

    def save():
        st["updated_utc"] = utc().isoformat(timespec="seconds")
        st["n_done"], st["n_total"] = len(st["done"]), len(graphs)
        tmp = status_p.with_name(status_p.name + ".part")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, status_p)

    queue = [g for g in graphs if not complete(g)]
    st["done"] = [g for g in graphs if complete(g)]
    procs: dict[str, tuple] = {}
    tries: dict[str, int] = {}
    while queue or procs:
        for g, (p, t0) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[g]
            st["running"].pop(g, None)
            if rc == 0 and complete(g):
                st["done"].append(g)
            elif tries.get(g, 0) < 2:
                st["retried"].append({"graph": g, "rc": rc, "utc": utc().isoformat(timespec="seconds")})
                queue.insert(0, g)
            else:
                st["failed"].append({"graph": g, "rc": rc})
            save()
        while queue and len(procs) < a.workers:
            g = queue.pop(0)
            if (deadline - utc()).total_seconds() < per_graph_s:
                st["skipped_deadline"].append(g)
                save()
                continue
            tries[g] = tries.get(g, 0) + 1
            log = open(ART / "logs" / f"main_{g.replace(':', '')}.log", "a")
            p = subprocess.Popen([PY, str(ROOT / "scripts/experience_commune/ec.py"), "run", "--graph", g,
                                  "--blocks", ",".join(blocks), "--banks", banks],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            procs[g] = (p, time.time())
            st["running"][g] = utc().isoformat(timespec="seconds")
            save()
        time.sleep(15)
    st["finished_utc"] = utc().isoformat(timespec="seconds")
    save()
    return 0 if not st["failed"] and not st["skipped_deadline"] else 1


if __name__ == "__main__":
    sys.exit(main())
