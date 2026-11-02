#!/usr/bin/env python
"""Gate G follow-up tests before Mario (docs/GATE_G_PLAN.md): same task, regime and rule B as the Gate G pass.

Tests (seed sets): delayed-explore (231, 232), delayed-confirm (241, 242), delayed-long (271, 272),
reversal (251, 252), reversal-adaptive (281, 282), reversal-floor (291, 292), generalise (261, 262).
Criterion as Gate G: (i) candidate >= .75 in all 12 cells; (ii) beats both named controls by > 2u in all 12;
(iii) shuffled does not beat frozen on average. A failing supervised reference makes a test uninformative.

    .venv/bin/python scripts/gates/gate_g_stage1.py delayed-explore
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_g_sweep import READOUTS, gpu_temps

from flymario import config
from flymario.connectome import datasets
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.sweep import RuleConfig, SweepLearner, SweepTask, evaluate
from flymario.training.two_cue import cue_currents

CKPT = RUNS / "checkpoints"
SEED_SETS = {"delayed-explore": (231, 232), "delayed-confirm": (241, 242),
             "reversal": (251, 252), "generalise": (261, 262),
             "delayed-long": (271, 272), "reversal-adaptive": (281, 282),
             "reversal-floor": (291, 292)}
RULE_B = {"eta": 0.001, "kappa": 0.2}
GAMMAS = {"G5": 0.5, "G8": 0.8}
DELAY = 2


def learners(conds: dict, sets, flip: bool = False) -> list[SweepLearner]:
    out = []
    for name, over in conds.items():
        for ss in sets:
            for r in READOUTS:
                for mp in (0, 1):
                    out.append(SweepLearner(f"{name}|s{ss}|r{r}|m{mp}", ss, 1 - mp if flip else mp,
                                            r, **over))
    return out


def cond(l) -> str:
    return l.name.split("|")[0]


def base_mapping(l) -> int:
    return int(l.name.split("|m")[1])


def rows(ls, ev, rel) -> list[dict]:
    return [{"learner": l.name, "condition": cond(l), "seed_set": l.seed_set,
             "readout": l.readout_seed, "mapping": base_mapping(l), "task_mapping": l.mapping,
             "accuracy": float(ev["accuracy"][b]),
             "left_given_cue0": float(ev["left_given_cue0"][b]),
             "left_given_cue1": float(ev["left_given_cue1"][b]),
             "relative_rms_weight_change": float(rel[b])} for b, l in enumerate(ls)]


def summary(rs) -> dict:
    out = {}
    for c in sorted({r["condition"] for r in rs}):
        xs = [r["accuracy"] for r in rs if r["condition"] == c]
        out[c] = {"mean": round(float(np.mean(xs)), 4), "min": round(float(np.min(xs)), 4),
                  "n>=.75": int(sum(x >= .75 for x in xs)), "n": len(xs)}
    return out


def criterion(rs, cand, controls, n_test) -> dict:
    """Gate G criteria (i) and (ii) for `cand` against the two named controls."""
    get = lambda c, s, r, m: next(x["accuracy"] for x in rs if x["condition"] == c
                                  and x["seed_set"] == s and x["readout"] == r and x["mapping"] == m)
    sets = sorted({x["seed_set"] for x in rs})
    cells, ok_i, ok_ii = [], True, True
    for r in READOUTS:
        for m in (0, 1):
            px = [get(cand, s, r, m) for s in sets]
            ses = [np.sqrt(p * (1 - p) / n_test + q * (1 - q) / n_test)
                   for s, p in zip(sets, px) for q in (get(c, s, r, m) for c in controls)]
            u = max(max(ses), abs(px[0] - px[1]) / 2)
            for s, p in zip(sets, px):
                ctl = {c: get(c, s, r, m) for c in controls}
                ci, cii = p >= .75, all(p - q > 2 * u for q in ctl.values())
                ok_i &= ci
                ok_ii &= cii
                cells.append({"readout": r, "mapping": m, "seed_set": s, "acc": p, **ctl,
                              "u": round(float(u), 4), "i": bool(ci), "ii": bool(cii)})
    return {"condition": cand, "controls": list(controls), "pass": bool(ok_i and ok_ii),
            "i": bool(ok_i), "ii": bool(ok_ii), "cells": cells}


def controls_check(rs, shuf, frozen) -> dict:
    d = np.array([x["accuracy"] - next(y["accuracy"] for y in rs if y["condition"] == frozen
                                       and y["seed_set"] == x["seed_set"]
                                       and y["readout"] == x["readout"]
                                       and y["mapping"] == x["mapping"])
                  for x in rs if x["condition"] == shuf])
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean_shuffled_minus_frozen": float(d.mean()), "se": se,
            "pass": bool(abs(d.mean()) <= 2 * se)}


class Runner:
    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.run_id = f"gate_g_stage1_{args.stage}-{args.tag}"
        self.out = RUNS / f"{self.run_id}.json"
        rep = json.loads(self.out.read_text()) if self.out.exists() else None
        if rep is not None and rep.get("complete"):
            raise FileExistsError(self.out)
        g = datasets.load(datasets.DEFAULT)
        self.W, self.motor = g.to_sparse(), datasets.motor_index(g, datasets.DEFAULT)
        pair = json.loads((RUNS / "gate_g_pair.json").read_text())["chosen"]["pair"]
        self.pair, self.cues = pair, cue_currents(g, pair)
        self.g = g
        self.key = (self.run_id, "operator")
        self.report = rep or {"run_id": self.run_id, "stage": args.stage, "pair": pair,
                              "seed_sets": SEED_SETS[args.stage], "readouts": READOUTS,
                              "tasks": {}, "evaluations": {}, "complete": False,
                              "started_utc": datetime.now(UTC).isoformat(timespec="seconds")}
        CKPT.mkdir(parents=True, exist_ok=True)

    def save(self):
        part = self.out.with_suffix(".json.part")
        part.write_text(json.dumps(self.report, indent=2, allow_nan=False))
        part.replace(self.out)

    def train(self, tag, task, n, ctl):
        """Train `task` to n trials in chunks, checkpointing; publish curves."""
        ck = CKPT / f"{self.run_id}_{tag}.pt"
        names = sorted({cond(l) for l in task.learners})
        ctl.chart(f"acc_{tag}", f"{tag}: greedy accuracy on training trials (last 50)", "trial",
                  "accuracy", names, yfmt="pct",
                  refs=[{"axis": "y", "value": .5, "label": "chance"},
                        {"axis": "y", "value": .75, "label": "0.75"}])

        def publish(t):
            k = t.trial
            if k % 25 == 0 and k >= 50:
                gr = t.log["correct_greedy"][k - 50:k].mean(0)
                for c in names:
                    idx = [b for b, l in enumerate(t.learners) if cond(l) == c]
                    ctl.point(f"acc_{tag}", c, k, float(gr[idx].mean()))
        ctl.begin(tag)
        t0 = time.time()
        while task.trial < n:
            task.run(min(100, n - task.trial), ctl=ctl, every=10, on_trial=publish)
            task.save(ck)
        lg = task.log["correct_greedy"][:n]
        self.report["tasks"][tag] = {
            "rule": asdict(task.rule), "purpose": task.purpose, "trials": n,
            "learners": [asdict(l) for l in task.learners],
            "training_greedy_by_100": {c: [round(float(lg[i:i + 100, [b for b, l in
                                       enumerate(task.learners) if cond(l) == c]].mean()), 4)
                                       for i in range(0, n, 100)]
                                       for c in sorted({cond(l) for l in task.learners})},
            "seconds": round(time.time() - t0, 1), "gpu_temps_c": gpu_temps()}
        self.save()
        ctl.done({"tag": tag, "trials": n})

    def task(self, tag, ls, purpose, n, rule, frames=None, pool=None):
        t = SweepTask(self.W, self.motor, self.cues, ls, purpose, n, rule=rule,
                      cache_key=self.key, frames=frames, frame_pool=pool)
        ck = CKPT / f"{self.run_id}_{tag}.pt"
        if ck.exists():
            t.load(ck)
        return t

    def evaluate(self, tag, ls, weights, rel, n_test, frames=None, pool=None):
        if tag in self.report["evaluations"]:
            return self.report["evaluations"][tag]["rows"]
        ev = evaluate(self.W, self.motor, self.cues, ls, weights, n_test, cache_key=self.key,
                      frames=frames, frame_pool=pool)
        rs = rows(ls, ev, rel)
        self.report["evaluations"][tag] = {"n_test": n_test, "rows": rs, "summary": summary(rs),
                                           "frame_pool": pool}
        self.save()
        print(f"[{tag}] " + "  ".join(f"{c} {v['mean']:.3f}/{v['min']:.2f}/{v['n>=.75']}of{v['n']}"
                                      for c, v in summary(rs).items()), flush=True)
        return rs


def stage_delayed_explore(R, ctl):
    a, sets, out = R.args, SEED_SETS["delayed-explore"], []
    for gname, gamma in GAMMAS.items():
        conds = {gname: RULE_B}
        if gname == "G8":
            conds.update({"SUPERVISED": {"eta": .001, "reward": "supervised"},
                          "FROZEN": {"eta": 0.0}, "SHUFFLED": {**RULE_B, "reward": "shuffled"}})
        ls = learners(conds, sets)
        rule = RuleConfig(delay=DELAY, trace_gamma=gamma)
        t = R.task(gname, ls, "explore", a.n_train, rule)
        R.train(gname, t, a.n_train, ctl)
        out += R.evaluate(f"{gname}@{a.n_train}", ls, t.weights(), t.relative_rms(), a.n_test)
        del t
        R.torch.cuda.empty_cache()
    rank = sorted(({"variant": v, "gamma": GAMMAS[v],
                    "mean": float(np.mean([r["accuracy"] for r in out if r["condition"] == v])),
                    "min": float(np.min([r["accuracy"] for r in out if r["condition"] == v]))}
                   for v in GAMMAS), key=lambda x: (-round(x["mean"], 2), -x["min"], x["gamma"]))
    R.report["selection"] = {"ranking": rank, "selected": rank[0]["variant"]}
    R.report["criterion_exploratory"] = {v: criterion(out, v, ("FROZEN", "SHUFFLED"), a.n_test)
                                         for v in list(GAMMAS) + ["SUPERVISED"]}
    R.report["controls"] = controls_check(out, "SHUFFLED", "FROZEN")


def stage_delayed_confirm(R, ctl):
    a = R.args
    if a.variant not in GAMMAS:
        raise SystemExit(f"--variant must be one of {list(GAMMAS)}")
    conds = {a.variant: RULE_B, "SUPERVISED": {"eta": .001, "reward": "supervised"},
             "FROZEN": {"eta": 0.0}, "SHUFFLED": {**RULE_B, "reward": "shuffled"}}
    ls = learners(conds, SEED_SETS[a.stage])
    t = R.task("confirm", ls, "confirm", a.n_train, RuleConfig(delay=DELAY,
                                                                trace_gamma=GAMMAS[a.variant]))
    R.train("confirm", t, a.n_train, ctl)
    rs = R.evaluate(f"confirm@{a.n_train}", ls, t.weights(), t.relative_rms(), a.n_test)
    sup = criterion(rs, "SUPERVISED", ("FROZEN", "SHUFFLED"), a.n_test)
    R.report.update({"reference_supervised": sup,
                     "criterion": criterion(rs, a.variant, ("FROZEN", "SHUFFLED"), a.n_test),
                     "controls": controls_check(rs, "SHUFFLED", "FROZEN"),
                     "informative": sup["i"]})
    R.report["pass"] = bool(R.report["criterion"]["pass"] and R.report["controls"]["pass"])


def stage_reversal(R, ctl):
    a, sets = R.args, SEED_SETS[R.args.stage]
    rule = {"reversal-adaptive": RuleConfig(tau_adapt=0.5),
            "reversal-floor": RuleConfig(epsilon=0.1)}.get(a.stage, RuleConfig())
    p1 = {"RULE": RULE_B, "FROZEN2": RULE_B, "SHUF2": RULE_B,
          "SUPERVISED": {"eta": .001, "reward": "supervised"}, "FROZEN": {"eta": 0.0}}
    l1 = learners(p1, sets)
    t1 = R.task("phase1", l1, "confirm", a.n_train, rule)
    R.train("phase1", t1, a.n_train, ctl)
    R.evaluate("phase1@mapping", l1, t1.weights(), t1.relative_rms(), a.n_test)
    p2 = {"RULE": RULE_B, "FROZEN2": {"eta": 0.0}, "SHUF2": {**RULE_B, "reward": "shuffled"},
          "SUPERVISED": {"eta": .001, "reward": "supervised"}, "FROZEN": {"eta": 0.0}}
    l2 = learners(p2, sets, flip=True)
    # the phase-1 task's learner order is the same, so its state transfers row by row
    fresh = not (CKPT / f"{R.run_id}_phase2.pt").exists()
    t2 = R.task("phase2", l2, "phase2", a.n_train, rule)
    if fresh:
        t2.copy_learning_state(t1)
    R.evaluate("phase2@start(reversed)", l2, t2.weights(), t2.relative_rms(), a.n_test)
    del t1
    R.torch.cuda.empty_cache()
    R.train("phase2", t2, a.n_train, ctl)
    rs = R.evaluate("phase2@end(reversed)", l2, t2.weights(), t2.relative_rms(), a.n_test)
    sup = criterion(rs, "SUPERVISED", ("FROZEN2", "SHUF2"), a.n_test)
    R.report.update({"reference_supervised": sup,
                     "criterion": criterion(rs, "RULE", ("FROZEN2", "SHUF2"), a.n_test),
                     "controls": controls_check(rs, "SHUF2", "FROZEN2"), "informative": sup["i"]})
    R.report["pass"] = bool(R.report["criterion"]["pass"])


def stage_generalise(R, ctl):
    from flymario.env import stimulus_bank
    from flymario.env.vision import RetinotopicVision
    a = R.args
    bank = stimulus_bank.build(which="calibration")
    vision = RetinotopicVision(R.g)
    frames = []
    for nm in R.pair:
        vision.reset()
        frames.append(np.stack([vision.encode(f) for f in bank.sequences[nm]]))
    frames = np.stack(frames).astype(np.float32)                    # (2, 48, n), in order
    train_pool, test_pool = list(range(24)), list(range(24, 48))
    conds = {"RULE": RULE_B, "SUPERVISED": {"eta": .001, "reward": "supervised"},
             "FROZEN": {"eta": 0.0}, "SHUFFLED": {**RULE_B, "reward": "shuffled"}}
    ls = learners(conds, SEED_SETS["generalise"])
    t = R.task("train", ls, "confirm", a.n_train, RuleConfig(), frames=frames, pool=train_pool)
    R.train("train", t, a.n_train, ctl)
    w, rel = t.weights(), t.relative_rms()
    del t
    R.torch.cuda.empty_cache()
    R.evaluate("train_frames", ls, w, rel, max(40, a.n_test // 2), frames=frames, pool=train_pool)
    rs = R.evaluate("heldout_frames", ls, w, rel, a.n_test, frames=frames, pool=test_pool)
    sup = criterion(rs, "SUPERVISED", ("FROZEN", "SHUFFLED"), a.n_test)
    R.report.update({"frames": {"train": train_pool, "test": test_pool,
                                "encoding": "each scene's 48-decision sequence, in order"},
                     "reference_supervised": sup,
                     "criterion": criterion(rs, "RULE", ("FROZEN", "SHUFFLED"), a.n_test),
                     "controls": controls_check(rs, "SHUFFLED", "FROZEN"), "informative": sup["i"]})
    R.report["pass"] = bool(R.report["criterion"]["pass"] and R.report["controls"]["pass"])


STAGES = {"delayed-explore": stage_delayed_explore, "delayed-confirm": stage_delayed_confirm,
          "reversal": stage_reversal, "generalise": stage_generalise,
          "delayed-long": stage_delayed_confirm, "reversal-adaptive": stage_reversal,
          "reversal-floor": stage_reversal}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--tag", default="20261001")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("declared GPU unavailable")
    torch.cuda.set_per_process_memory_fraction(.6)
    R = Runner(args)
    R.report["design"] = {**vars(args), "delay": DELAY, "gammas": GAMMAS, "rule_B": RULE_B,
                          "ticks_per_decision": int(config.get("timing.ticks_per_decision"))}
    manifest = RunManifest(R.run_id, "Gate G", f"stage-1 follow-up: {args.stage}", R.report["design"],
                           seeds={"seed_sets": list(SEED_SETS[args.stage]), "readouts": READOUTS},
                           dataset=str(config.get("connectome.dataset")),
                           input_files={**dataset_files(), "script": Path(__file__),
                                        "sweep": ROOT / "src/flymario/training/sweep.py",
                                        "plan": ROOT / "docs/GATE_G_PLAN.md",
                                        "config": ROOT / "configs/base.yaml"},
                           output_files={"report": R.out})
    ctl = RunControl(R.run_id, total=2, checkpoint=lambda: None, unit="task",
                     meta={"question": f"Gate G follow-up: {args.stage}"},
                     fresh=R.report["tasks"] == {})
    try:
        STAGES[args.stage](R, ctl)
        R.report.update({"complete": True, "manifest": str(manifest.path),
                         "finished_utc": datetime.now(UTC).isoformat(timespec="seconds")})
        R.save()
        ctl.close("complete")
        manifest.finish("complete")
        for k in ("selection", "pass", "informative", "controls"):
            if k in R.report:
                print(k, json.dumps(R.report[k]))
    except StopRequested:
        ctl.close("stopped")
        manifest.finish("stopped")
        raise SystemExit(0)
    except BaseException as exc:
        ctl.close("failed")
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
