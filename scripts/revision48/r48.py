#!/usr/bin/env python
"""REVISION48 command line (configs/revision48/campaign.yaml; protocol docs/revision48/PROTOCOL_FROZEN.md).

    r48.py prepare --blocks r0,...,t0,t1,rd0 [--workers 4]   shared stimuli, U journals, banks (sha256 manifest)
    r48.py tune --graph G --gamma X [--stop-at N]            one tuning batch: the 3 trials of that gain level,
                                                             2 tasks x 2 tuning blocks (12 learners, 1,500 decisions)
    r48.py select [--seal]                                   per graph, the trial with the best mean validation area
                                                             (tuning blocks only); --seal writes the sealed file
    r48.py run --graph G --condition C [--stop-at N]         one condition on the 6 main blocks (12 learners)
    r48.py refs --graph D                                    references on a development graph, block rd0
    r48.py status                                            runs done, times (no score)

Common options: --slow (PlasticLIF block path: ROCm / CPU tests), --eval-cols, --base-gain (lognorm, development),
--frozen-product csr|segment, --root DIR (outputs; default artifacts/revision48).
Each run directory holds run.json, predictions.npz, diagnostics.npz, final_weights.npz and log.txt (resume.pt
while incomplete).
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
CFG_P = ROOT / "configs/revision48/campaign.yaml"
TASKS = ("where", "loom")


def cfg() -> dict:
    return yaml.safe_load(CFG_P.read_text())


def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, p)


def tag(spec: str) -> str:
    return spec.replace(":", "")


def art(a) -> Path:
    return Path(a.root) if a.root else ROOT / "artifacts/revision48"


# ---------------------------------------------------------------------------------------------------- data blocks
def _prepare_one(block: str, task: str, root: str) -> dict:
    from study_s import stimulus

    from flymario.connectome import datasets
    from flymario.env.vision import RetinotopicVision
    from flymario.experience_commune import data as D
    t = time.time()
    g = datasets.load(datasets.DEFAULT)
    vis = RetinotopicVision(g)
    idx = np.unique(np.r_[vis.on_idx, vis.off_idx]).astype(np.int64)
    d = D.build_block_task(block, task, vis, idx, stimulus)
    base = Path(root)
    p = base / "data" / block / f"{task}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, idx=idx, **d)
    os.replace(tmp, p)
    j = base / "journals" / f"{block}_{task}_U.csv"
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(D.journal_csv(d))
    return {"block": block, "task": task, "file": str(p.relative_to(base)), "sha256": D.sha256(p),
            "journal_sha256": D.sha256(j), "seconds": round(time.time() - t, 1),
            "U_reward_rate": float(d["U_reward"].mean()), "U_action_counts": np.bincount(d["U_action"], minlength=8).tolist(),
            "train_class_counts": np.bincount(d["train_cls"], minlength=8).tolist(), "seeds": json.loads(str(d["seeds_json"]))}


def cmd_prepare(a) -> int:
    base = art(a)
    man_p = base / "manifests/data.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"created_utc": utc(), "entries": {}}
    jobs = [(b, t) for b in a.blocks.split(",") for t in TASKS if f"{b}/{t}" not in man["entries"]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_prepare_one, b, t, str(base)): (b, t) for b, t in jobs}
        for f in as_completed(futs):
            r = f.result()
            man["entries"][f"{r['block']}/{r['task']}"] = r
            man["updated_utc"] = utc()
            write_json(man_p, man)
            print(utc(), r["block"], r["task"], r["seconds"], "s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- learners
def learners_main(c: dict, kind: str, hp: dict | None):
    from flymario.experience_commune.learn import Learner
    eta = None if hp is None else hp["eta"]
    tau = None if hp is None else hp["tau"]
    return [Learner(t, c["mode"], b, kind, eta, tau) for b in c["blocks"]["main"] for t in c["tasks"]]


def learners_tuning(c: dict, gamma: float):
    from flymario.experience_commune.learn import Learner
    trials = [tr for tr in c["tuning"]["trials"] if abs(tr["gamma"] - gamma) < 1e-12]
    if len(trials) != 3:
        raise SystemExit(f"gamma {gamma}: expected 3 trials, found {len(trials)}")
    out = []
    for tr in trials:
        ref = tr["eta"] == c["reference"]["eta"] and tr["tau"] == c["reference"]["tau"]
        for b in c["blocks"]["tuning"]:
            for t in c["tasks"]:
                out.append(Learner(t, c["mode"], b, "main", None if ref else tr["eta"], None if ref else tr["tau"]))
    return trials, out


def _run(a, spec: str, learners, out: Path, horizon: int, checkpoints, gamma: float, weights: str = "raw",
         banks=("final", "ood"), base_gain=None) -> dict:
    from flymario.experience_commune import learn
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "log.txt"

    def log(msg):
        line = f"{utc()} {msg}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")
    log(f"start {spec} learners={len(learners)} gamma={gamma} weights={weights} horizon={horizon} stop_at={a.stop_at}")
    r = learn.run(spec, learners, art(a) / "data", out, stop_at=a.stop_at, eval_cols=a.eval_cols,
                  banks_final=banks, log=log, gamma=gamma, horizon=horizon, checkpoints=checkpoints,
                  weights=weights, frozen_product=a.frozen_product or "csr", base_gain=base_gain, slow=a.slow,
                  stop_file=art(a) / "STOP")
    log(f"end k={r['k']} complete={r['complete']} stopped={r.get('stopped')} wall={r['wall_s']}s")
    return r


def cmd_tune(a) -> int:
    c = cfg()
    trials, ls = learners_tuning(c, a.gamma)
    out = art(a) / "runs/tuning" / f"g{a.gamma}" / tag(a.graph)
    tu = c["tuning"]
    r = _run(a, a.graph, ls, out, tu["horizon"], tu["checkpoints"], a.gamma, banks=())
    write_json(out / "trials.json", {"trials": trials, "learner_trial": [tr["trial"] for tr in trials
                                                                         for _b in c["blocks"]["tuning"] for _t in c["tasks"]]})
    return 0 if r["complete"] or r.get("stopped") else 1


def area(acc: np.ndarray, checkpoints) -> np.ndarray:
    """Trapezoid of accuracy over the checkpoints, divided by the horizon: acc (..., C)."""
    n = np.asarray(checkpoints, float)
    w = np.zeros(len(n))
    d = np.diff(n)
    w[:-1] += d / 2
    w[1:] += d / 2
    return acc @ (w / (n[-1] - n[0]))


def tuning_scores(base: Path, c: dict, spec: str) -> dict | None:
    """Validation area per trial for one graph, or None if a tuning batch is missing / incomplete."""
    tu = c["tuning"]
    per_trial: dict[int, list[float]] = {}
    for gamma in sorted({tr["gamma"] for tr in tu["trials"]}):
        d = base / "runs/tuning" / f"g{gamma}" / tag(spec)
        rj, pz = d / "run.json", d / "predictions.npz"
        if not (rj.exists() and pz.exists() and json.loads(rj.read_text()).get("complete")):
            return None
        tr_of = json.loads((d / "trials.json").read_text())["learner_trial"]
        z = np.load(pz)
        lab = z["labels_progress"]
        acc = np.stack([(z[f"progress_{k}"] == lab).mean(1) for k in tu["checkpoints"]], 1)   # (B, C)
        L = area(acc, tu["checkpoints"])
        w = np.load(d / "final_weights.npz")["m"]
        for j, trial in enumerate(tr_of):
            v = float(L[j]) if np.isfinite(L[j]) and np.isfinite(w[j]).all() else float("nan")
            per_trial.setdefault(trial, []).append(v)
    return {t: (float(np.mean(v)) if np.isfinite(v).all() else None) for t, v in sorted(per_trial.items())}


def cmd_select(a) -> int:
    c = cfg()
    base = art(a)
    man = json.loads((ROOT / c["graphs"]["manifest"]).read_text())
    graphs = man["order"] + (a.extra_graphs.split(",") if a.extra_graphs else [])
    sel, missing = {}, []
    for spec in graphs:
        sc = tuning_scores(base, c, spec)
        if sc is None:
            missing.append(spec)
            continue
        valid = {t: v for t, v in sc.items() if v is not None}
        best = max(valid, key=lambda t: (valid[t], -t)) if valid else None      # ties: lowest trial number
        sel[spec] = {"trial": best, "scores": sc, "judgeable": best is not None}
    out = {"what": "equal-budget tuning selection (validation areas on the tuning blocks only)", "utc": utc(),
           "selection": sel, "missing": missing}
    if a.seal:
        if missing:
            print("refused: tuning incomplete for", ", ".join(missing))
            return 2
        p = base / "tuning/selection.json"
        if p.exists():
            print(f"refused: {p} is already sealed")
            return 2
        write_json(p, out)
        from flymario.experience_commune.data import sha256
        write_json(base / "tuning/selection.sha256.json", {"file": str(p.relative_to(base)), "sha256": sha256(p),
                                                          "sealed_utc": utc()})
        print("sealed", p)
    else:
        print(json.dumps({k: (v["trial"] if isinstance(v, dict) else v) for k, v in sel.items()}, indent=1))
        print("missing:", missing)
    return 0


def selected_hp(base: Path, c: dict, spec: str) -> tuple[dict | None, float, int]:
    p = base / "tuning/selection.json"
    if not p.exists():
        raise SystemExit("tuned condition refused: no sealed selection (r48.py select --seal)")
    s = json.loads(p.read_text())["selection"].get(spec)
    if s is None or not s["judgeable"]:
        raise SystemExit(f"{spec}: no valid tuned setting (not judgeable)")
    tr = next(t for t in c["tuning"]["trials"] if t["trial"] == s["trial"])
    ref = tr["eta"] == c["reference"]["eta"] and tr["tau"] == c["reference"]["tau"]
    return (None if ref else {"eta": tr["eta"], "tau": tr["tau"]}), float(tr["gamma"]), int(tr["trial"])


def cmd_run(a) -> int:
    c = cfg()
    base = art(a)
    cond = c["conditions"][a.condition]
    out = base / "runs/main" / a.condition / tag(a.graph)
    weights = cond.get("weights", "raw")
    base_gain = None
    if weights == "lognorm":
        base_gain = a.base_gain if a.base_gain is not None else c["lognorm"]["base_gain"]
        if not isinstance(base_gain, (int, float)):
            raise SystemExit("lognorm refused: lognorm.base_gain is not frozen (calibrate_gain.py)")
        # the frozen product follows the protocol (checks: csr is not repeatable with fractional weights here)
        a.frozen_product = a.frozen_product or c["lognorm"]["frozen_product"]
        if a.frozen_product not in ("csr", "segment", "exact64"):
            raise SystemExit("lognorm refused: lognorm.frozen_product is not frozen (checks.py)")
    if a.condition == "tuned":
        hp, gamma, trial = selected_hp(base, c, a.graph)
        if trial == 1:                    # the reference setting: the identical computation already exists
            ref = base / "runs/main/reference" / tag(a.graph) / "run.json"
            write_json(out / "reuse.json", {"trial": 1, "reused": str(ref.relative_to(base)),
                                            "complete": ref.exists() and json.loads(ref.read_text()).get("complete"),
                                            "why": "trial 1 is the reference setting; same computation, platform and "
                                                   "batch composition"})
            return 0
    else:
        hp, gamma = None, float(c["reference"]["gamma"])
    ls = learners_main(c, cond["kind"], hp)
    r = _run(a, a.graph, ls, out, c["horizon"], c["checkpoints"], gamma, weights=weights, base_gain=base_gain)
    return 0 if r["complete"] or r.get("stopped") else 1


def cmd_refs(a) -> int:
    """Instrument references on a development graph (block rd0): main U learners, supervised, frozen, shuffled."""
    from flymario.experience_commune.learn import Learner
    c = cfg()
    b = c["blocks"]["dev"][0]
    ls = [Learner(t, c["mode"], b, k) for k in ("main", "supervised", "frozen", "shuffled") for t in c["tasks"]]
    out = art(a) / "runs/refs" / tag(a.graph)
    r = _run(a, a.graph, ls, out, c["horizon"], c["checkpoints"], float(c["reference"]["gamma"]), banks=("final",))
    if not r["complete"]:
        return 0 if r.get("stopped") else 1
    z = np.load(out / "predictions.npz")
    acc_final = (z["final"] == z["labels_final"]).mean(1)
    prog = [z[f"progress_{k}"] for k in c["checkpoints"]]
    res = {}
    for j, l in enumerate(ls):
        res[l.key] = {"final_accuracy": float(acc_final[j]),
                      "identical_predictions_all_checkpoints": bool(all((p[j] == prog[0][j]).all() for p in prog))}
    sup = [res[f"{t}-{c['mode']}-{b}-supervised"]["final_accuracy"] for t in c["tasks"]]
    frz = [res[f"{t}-{c['mode']}-{b}-frozen"]["identical_predictions_all_checkpoints"] for t in c["tasks"]]
    shf = [res[f"{t}-{c['mode']}-{b}-shuffled"]["final_accuracy"] for t in c["tasks"]]
    verdict = {"supervised_must_learn": all(x >= 0.80 for x in sup), "frozen_must_not_learn": all(frz),
               "shuffled_reward_near_chance": all(x <= 0.20 for x in shf)}
    write_json(out / "references.json", {"graph": a.graph, "per_learner": res, "verdict": verdict,
                                         "pass": all(verdict.values()), "utc": utc()})
    print(json.dumps(verdict))
    return 0


def cmd_status(a) -> int:
    base = art(a)
    rows = []
    for rj in sorted(base.glob("runs/*/*/*/run.json")) + sorted(base.glob("runs/refs/*/run.json")):
        r = json.loads(rj.read_text())
        rows.append({"run": str(rj.parent.relative_to(base)), "k": r["k"], "complete": r["complete"],
                     "wall_s": r.get("wall_s"), "step_s": r.get("train_step_s_median")})
    for rr in sorted(base.glob("runs/main/tuned/*/reuse.json")):
        rows.append({"run": str(rr.parent.relative_to(base)), "reused": json.loads(rr.read_text())["reused"]})
    print(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default="")
        p.add_argument("--slow", action="store_true", help="PlasticLIF block path (ROCm / tests)")
        p.add_argument("--eval-cols", type=int, default=128)
        p.add_argument("--stop-at", type=int, default=None)
        p.add_argument("--frozen-product", default=None, choices=["csr", "segment", "exact64"],
                       help="default: lognorm.frozen_product for the lognorm condition, else csr")
        p.add_argument("--base-gain", type=float, default=None)
        return p
    p = common(sub.add_parser("prepare"))
    p.add_argument("--blocks", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = common(sub.add_parser("tune"))
    p.add_argument("--graph", required=True)
    p.add_argument("--gamma", type=float, required=True)
    p = common(sub.add_parser("select"))
    p.add_argument("--seal", action="store_true")
    p.add_argument("--extra-graphs", default="", help="development graphs (tests)")
    p = common(sub.add_parser("run"))
    p.add_argument("--graph", required=True)
    p.add_argument("--condition", required=True, choices=["reference", "tuned", "supervised", "lognorm", "autapse"])
    p = common(sub.add_parser("refs"))
    p.add_argument("--graph", required=True)
    common(sub.add_parser("status"))
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "tune": cmd_tune, "select": cmd_select, "run": cmd_run, "refs": cmd_refs,
            "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
