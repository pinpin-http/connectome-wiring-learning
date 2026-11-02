#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 orchestrator (docs/experience_commune/MISSION.md §17).

    ec.py prepare --blocks d0,b0,b1,b2,b3,b4,b5 [--workers 2]   shared stimuli, U journals, N streams, banks
    ec.py run --graph G --blocks b0,... [--tag T] [--kinds main] [--stop-at N]   one GPU process per graph
    ec.py status                                                  runs done / running, time per run
    ec.py analyse --plan P                                        the frozen analysis (only once every run is done)

Data: artifacts/experience_commune_v1/data/<block>/<task>.npz (immutable, sha256 in manifests/data.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
ART = ROOT / "artifacts/experience_commune_v1"


def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, p)


def _prepare_one(block: str, task: str) -> dict:
    from study_s import stimulus

    from flymario.connectome import datasets
    from flymario.env.vision import RetinotopicVision
    from flymario.experience_commune import data as D
    t = time.time()
    g = datasets.load(datasets.DEFAULT)
    vis = RetinotopicVision(g)
    idx = np.unique(np.r_[vis.on_idx, vis.off_idx]).astype(np.int64)
    d = D.build_block_task(block, task, vis, idx, stimulus)
    p = ART / "data" / block / f"{task}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, idx=idx, **d)
    os.replace(tmp, p)
    j = ART / "journals" / f"{block}_{task}_U.csv"
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(D.journal_csv(d))
    return {"block": block, "task": task, "file": str(p.relative_to(ROOT)), "sha256": D.sha256(p),
            "journal": str(j.relative_to(ROOT)), "journal_sha256": D.sha256(j), "seconds": round(time.time() - t, 1),
            "n_train": int(len(d["train_cls"])), "U_reward_rate": float(d["U_reward"].mean()),
            "U_action_counts": np.bincount(d["U_action"], minlength=8).tolist(),
            "train_class_counts": np.bincount(d["train_cls"], minlength=8).tolist(),
            "n_progress": int(len(d["progress_cls"])), "n_final": int(len(d["final_cls"])),
            "n_ood": int(len(d["ood_cls"])), "seeds": json.loads(str(d["seeds_json"]))}


def cmd_prepare(a) -> int:
    from flymario.experience_commune.data import TASKS
    man_p = ART / "manifests/data.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"created_utc": utc(), "entries": {}}
    jobs = [(b, t) for b in a.blocks.split(",") for t in TASKS if f"{b}/{t}" not in man["entries"]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_prepare_one, b, t): (b, t) for b, t in jobs}
        for f in as_completed(futs):
            r = f.result()
            man["entries"][f"{r['block']}/{r['task']}"] = r
            man["updated_utc"] = utc()
            write_json(man_p, man)
            print(utc(), r["block"], r["task"], r["seconds"], "s", "U reward rate %.4f" % r["U_reward_rate"], flush=True)
    return 0


def tag(spec: str) -> str:
    return spec.replace(":", "")


def learners_for(blocks: list[str], kinds: str = "main"):
    from flymario.experience_commune.data import TASKS
    from flymario.experience_commune.learn import Learner
    out = [Learner(t, m, b) for b in blocks for t in TASKS for m in ("N", "U")]
    if kinds == "pilot":                      # development controls (mission §9), block d0 only
        out += [Learner(t, "N", blocks[0], "supervised") for t in TASKS]
        out += [Learner(t, "U", blocks[0], "shuffled") for t in TASKS]
        out += [Learner(t, "U", blocks[0], "frozen") for t in TASKS]
    return out


def cmd_run(a) -> int:
    from flymario.experience_commune import learn
    blocks = a.blocks.split(",")
    ls = learners_for(blocks, a.kinds)
    out = ART / ("runs" if not a.tag else f"runs_{a.tag}") / tag(a.graph)
    logf = out / "log.txt"
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{utc()} {msg}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")
    log(f"start {a.graph} learners={len(ls)} blocks={blocks} kinds={a.kinds} stop_at={a.stop_at}")
    r = learn.run(a.graph, ls, ART / "data", out, stop_at=a.stop_at, eval_cols=a.eval_cols,
                  banks_final=tuple(b for b in a.banks.split(",") if b), log=log, log_accuracy=a.log_accuracy,
                  pair_u_to_n=a.pair_u_to_n)
    log(f"end k={r['k']} complete={r['complete']} wall={r['wall_s']}s train_step_median={r['train_step_s_median']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--blocks", required=True)
    p.add_argument("--workers", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--graph", required=True)
    r.add_argument("--blocks", required=True)
    r.add_argument("--kinds", default="main")
    r.add_argument("--tag", default="")
    r.add_argument("--stop-at", type=int, default=None)
    r.add_argument("--eval-cols", type=int, default=128)
    r.add_argument("--banks", default="final,ood")
    r.add_argument("--log-accuracy", action="store_true", help="development runs only")
    r.add_argument("--pair-u-to-n", action="store_true", help="test hook: U takes N's action with rho = 1")
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
