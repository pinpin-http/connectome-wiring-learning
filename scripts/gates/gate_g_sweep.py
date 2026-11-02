#!/usr/bin/env python
"""Gate G in the sweep regime (docs/GATE_G_PLAN.md, amendment 2026-09-30).

Three stages, each on its own seed sets so no stage can inform the next through
shared noise, cues or test trials:

    explore    exploratory comparison of rule variants     seed sets 201, 202
    reference  the pass criterion applied to references    seed sets 211, 212
               first: supervised must PASS; frozen, shuffled
               reward and anti-reward must FAIL
    confirm    the selected rule, frozen before this stage, seed sets 221, 222
               with its matched controls

Task (all stages): the Gate G cue pair's first frames (gate_g_pair.json), one
sweep per decision (training/sweep.py): reset, 16 ticks, motor spike counts,
the unchanged FixedRandomProjection restricted to RIGHT vs LEFT. Readouts:
policy.readout.block_seeds(3). Mappings 0 and 1. Cues balanced 5/5 per block of
10. Held-out evaluation: learning off, greedy action, the 'test' streams of the
same seed set (fresh noise, fresh cue order), `--n-test` trials.

Pass criterion (declared in the amendment, applied identically everywhere).
For a condition X, per cell (readout, mapping) and seed set s, with n test
trials and matched controls F (frozen, eta = 0) and S (shuffled reward):

    (i)   acc_X >= 0.75 in all 12 (cell, seed set) pairs
    (ii)  acc_X - acc_F > 2u and acc_X - acc_S > 2u in all 12, where for the
          cell u = max(binomial SE of each difference in either seed set,
                       half the gap between the two seed sets' acc_X)
    (iii) (run-level, controls) the mean over the 12 pairs of acc_S - acc_F is
          within 2 standard errors of 0: shuffled reward does not learn.

`2u` is a readability threshold for an engineering gate, not a calibrated
confidence bound (WORKING_AGREEMENT §3).

Exploratory selection rule (declared before `explore` runs): among the rule
variants, the highest mean held-out accuracy over its 12 learners after the
last training block; ties within 0.01 go to the higher minimum, then to the
simpler variant (fewer non-default options). The number of confirmatory
training trials equals the exploratory length.

    .venv/bin/python scripts/gates/gate_g_sweep.py explore
    .venv/bin/python scripts/gates/gate_g_sweep.py reference
    .venv/bin/python scripts/gates/gate_g_sweep.py confirm --variant NAME
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

from flymario import config
from flymario.connectome import datasets
from flymario.paths import RUNS
from flymario.policy.readout import block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.sweep import RuleConfig, SweepLearner, SweepTask, evaluate
from flymario.training.two_cue import cue_currents

CKPT = RUNS / "checkpoints"
SEED_SETS = {"explore": (201, 202), "reference": (211, 212), "confirm": (221, 222)}
READOUTS = block_seeds(3)

#: Rule variants for `explore`. name -> (learner overrides, RuleConfig overrides).
#: Declared before the exploratory run; each is motivated in the amendment.
VARIANTS = {
    "A_reinforce":        ({"eta": 0.001}, {}),
    "B_centred":          ({"eta": 0.001, "kappa": 0.2}, {}),
    "C_adaptive_tau":     ({"eta": 0.001}, {"tau_adapt": 0.5}),
    "D_centred_adaptive": ({"eta": 0.002, "kappa": 0.2}, {"tau_adapt": 0.5}),
}
#: references share the default RuleConfig
REFERENCES = {
    "SUPERVISED": {"eta": 0.001, "reward": "supervised"},
    "FROZEN":     {"eta": 0.0},
    "SHUFFLED":   {"eta": 0.001, "reward": "shuffled"},
    "ANTI":       {"eta": 0.001, "reward": "anti"},
}


def gpu_temps() -> dict:
    out = {}
    for f in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/temp*_input"):
        lab = f.with_name(f.name.replace("_input", "_label"))
        try:
            out[lab.read_text().strip() if lab.exists() else f.name] = int(f.read_text()) / 1000
        except OSError:
            pass
    return out


def learners_for(conditions: dict, stage: str) -> list[SweepLearner]:
    out = []
    for cname, over in conditions.items():
        for ss in SEED_SETS[stage]:
            for r in READOUTS:
                for mp in (0, 1):
                    out.append(SweepLearner(f"{cname}|s{ss}|r{r}|m{mp}", ss, mp, r, **over))
    return out


def cname(l: SweepLearner) -> str:
    return l.name.split("|")[0]


def criterion(rows: list[dict], cand: str, n_test: int) -> dict:
    """(i) and (ii) for condition `cand` against FROZEN and SHUFFLED in `rows`."""
    def acc(c, ss, r, mp):
        return next(x["accuracy"] for x in rows if x["condition"] == c and x["seed_set"] == ss
                    and x["readout"] == r and x["mapping"] == mp)
    sets = sorted({x["seed_set"] for x in rows})
    cells, ok_i, ok_ii = [], True, True
    for r in READOUTS:
        for mp in (0, 1):
            px = [acc(cand, s, r, mp) for s in sets]
            gap = abs(px[0] - px[1]) / 2 if len(px) == 2 else 0.0
            ses = []
            for s, p in zip(sets, px):
                for ctrl in ("FROZEN", "SHUFFLED"):
                    q = acc(ctrl, s, r, mp)
                    ses.append(np.sqrt(p * (1 - p) / n_test + q * (1 - q) / n_test))
            u = max(max(ses), gap)
            for s, p in zip(sets, px):
                f, sh = acc("FROZEN", s, r, mp), acc("SHUFFLED", s, r, mp)
                c_i = p >= 0.75
                c_ii = (p - f > 2 * u) and (p - sh > 2 * u)
                ok_i &= c_i
                ok_ii &= c_ii
                cells.append({"readout": r, "mapping": mp, "seed_set": s, "acc": p,
                              "frozen": f, "shuffled": sh, "u": round(float(u), 4),
                              "i_acc>=.75": bool(c_i), "ii_beats_controls_by_2u": bool(c_ii)})
    return {"condition": cand, "pass": bool(ok_i and ok_ii), "i": bool(ok_i), "ii": bool(ok_ii),
            "cells": cells}


def controls_check(rows: list[dict]) -> dict:
    """(iii): shuffled reward does not improve on the frozen twin, on average."""
    d = []
    for x in rows:
        if x["condition"] != "SHUFFLED":
            continue
        f = next(y["accuracy"] for y in rows if y["condition"] == "FROZEN"
                 and y["seed_set"] == x["seed_set"] and y["readout"] == x["readout"]
                 and y["mapping"] == x["mapping"])
        d.append(x["accuracy"] - f)
    d = np.asarray(d)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean_shuffled_minus_frozen": float(d.mean()), "se": se,
            "pass": bool(abs(d.mean()) <= 2 * se) if np.isfinite(se) else False,
            "values": d.round(4).tolist()}


def rows_from(learners, ev, rel) -> list[dict]:
    rows = []
    for b, l in enumerate(learners):
        rows.append({"learner": l.name, "condition": cname(l), "seed_set": l.seed_set,
                     "readout": l.readout_seed, "mapping": l.mapping, "eta": l.eta,
                     "reward": l.reward, "kappa": l.kappa,
                     "accuracy": float(ev["accuracy"][b]),
                     "left_given_cue0": float(ev["left_given_cue0"][b]),
                     "left_given_cue1": float(ev["left_given_cue1"][b]),
                     "d_cue0": float(ev["d_cue0"][b]), "d_cue1": float(ev["d_cue1"][b]),
                     "relative_rms_weight_change": float(rel[b])})
    return rows


def train_and_eval(tag, stage, learners, rule, n_train, n_test, milestones, W, motor, cues,
                   report, out_path, ctl, key):
    """Train one task with checkpoints; evaluate at milestones. Mutates `report`."""
    import torch
    ck = CKPT / f"{report['run_id']}_{tag}.pt"
    task = SweepTask(W, motor, cues, learners, stage, n_train, rule=rule, cache_key=key)
    if ck.exists():
        task.load(ck)
    rec = report["tasks"].setdefault(tag, {"rule": asdict(rule), "evaluations": {},
                                          "learners": [asdict(l) for l in learners]})
    ctl.chart(f"acc_{tag}", f"{tag}: greedy accuracy on training trials (last 50)",
              "trial", "accuracy", sorted({cname(l) for l in learners}), yfmt="pct",
              refs=[{"axis": "y", "value": 0.5, "label": "chance"},
                    {"axis": "y", "value": 0.75, "label": "0.75"}])

    def publish(t):
        k = t.trial
        if k % 25 == 0 and k >= 50:
            g = t.log["correct_greedy"][k - 50:k].mean(0)
            for c in sorted({cname(l) for l in learners}):
                idx = [b for b, l in enumerate(learners) if cname(l) == c]
                ctl.point(f"acc_{tag}", c, k, float(g[idx].mean()))

    while task.trial < n_train:
        nxt = min([m for m in milestones if m > task.trial] + [n_train])
        ctl.begin(f"{tag} trials {task.trial}-{nxt}")
        t0 = time.time()
        while task.trial < nxt:
            chunk = min(100, nxt - task.trial)
            task.run(chunk, ctl=ctl, every=10, on_trial=publish)
            task.save(ck)
        sec = time.time() - t0
        if nxt in milestones and str(nxt) not in rec["evaluations"]:
            nt = n_test if nxt == n_train else max(40, n_test // 2)
            ev = evaluate(W, motor, cues, learners, task.weights(), nt, cache_key=key)
            rows = rows_from(learners, ev, task.relative_rms())
            summ = {}
            for c in sorted({cname(l) for l in learners}):
                xs = [r["accuracy"] for r in rows if r["condition"] == c]
                summ[c] = {"mean": round(float(np.mean(xs)), 4), "min": round(float(np.min(xs)), 4),
                           "n>=.75": int(sum(x >= .75 for x in xs)), "n": len(xs)}
            rec["evaluations"][str(nxt)] = {"n_test": nt, "rows": rows, "summary": summ,
                                            "gpu_temps_c": gpu_temps()}
            print(f"[{tag}] trial {nxt}: " + "  ".join(f"{c} {v['mean']:.3f}/{v['min']:.2f}/"
                  f"{v['n>=.75']}of{v['n']}" for c, v in summ.items()), flush=True)
        rec["training_log"] = {k: task.log[k][:task.trial].mean(1).round(4).tolist()
                               for k in ("correct_greedy", "rate_motor")}
        rec["training_greedy_by_learner_last100"] = task.log["correct_greedy"][
            max(0, task.trial - 100):task.trial].mean(0).round(3).tolist()
        rec["trials_done"] = task.trial
        part = out_path.with_suffix(".json.part")
        part.write_text(json.dumps(report, indent=2, allow_nan=False))
        part.replace(out_path)
        ctl.done({"tag": tag, "trial": task.trial, "seconds": round(sec, 1)})
    del task
    torch.cuda.empty_cache()
    return rec


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(SEED_SETS))
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--variant", default=None, help="confirm: the selected rule variant")
    ap.add_argument("--tag", default="20260930")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("declared GPU unavailable")
    torch.cuda.set_per_process_memory_fraction(.6)
    run_id = f"gate_g_sweep_{args.stage}-{args.tag}"
    out = RUNS / f"{run_id}.json"
    report = json.loads(out.read_text()) if out.exists() else None
    if report is not None and report.get("complete"):
        raise FileExistsError(out)
    CKPT.mkdir(parents=True, exist_ok=True)
    g = datasets.load(datasets.DEFAULT)
    W, motor = g.to_sparse(), datasets.motor_index(g, datasets.DEFAULT)
    pair_file = RUNS / "gate_g_pair.json"
    pair = json.loads(pair_file.read_text())["chosen"]["pair"]
    cues = cue_currents(g, pair)
    key = (run_id, "operator")
    milestones = sorted({200, 400, 600, args.n_train}) if args.stage == "explore" else [args.n_train]
    milestones = [m for m in milestones if m <= args.n_train]

    if args.stage == "explore":
        groups = {"default": (RuleConfig(), {k: v[0] for k, v in VARIANTS.items() if not v[1]}
                              | {k: REFERENCES[k] for k in ("SUPERVISED", "FROZEN", "SHUFFLED")}),
                  "tau_adapt": (RuleConfig(tau_adapt=0.5),
                                {k: v[0] for k, v in VARIANTS.items() if v[1]})}
    elif args.stage == "reference":
        groups = {"references": (RuleConfig(), dict(REFERENCES))}
    else:
        if args.variant not in VARIANTS:
            raise SystemExit(f"--variant must be one of {list(VARIANTS)}")
        over, rover = VARIANTS[args.variant]
        conds = {args.variant: over, **{k: REFERENCES[k] for k in ("FROZEN", "SHUFFLED", "SUPERVISED")}}
        groups = {"confirm": (RuleConfig(**rover), conds)}
        # the controls must run under the candidate's RuleConfig too (same tau)

    design = {"stage": args.stage, "seed_sets": SEED_SETS[args.stage], "readouts": READOUTS,
              "mappings": [0, 1], "pair": pair, "n_train": args.n_train, "n_test": args.n_test,
              "milestones": milestones, "ticks_per_decision": int(config.get("timing.ticks_per_decision")),
              "regime": "sweep: reset before every decision, motor spike counts",
              "groups": {k: {"rule": asdict(v[0]), "conditions": v[1]} for k, v in groups.items()},
              "variants": {k: {"learner": v[0], "rule": v[1]} for k, v in VARIANTS.items()},
              "selection_rule": "highest mean held-out accuracy at the last milestone; ties "
                                "within .01 -> higher minimum -> simpler variant",
              "criterion": "(i) acc>=.75 all 12; (ii) beats frozen and shuffled by >2u all 12; "
                           "(iii) mean(shuffled-frozen) within 2 SE of 0"}
    if report is None:
        report = {"run_id": run_id, "what": f"Gate G sweep regime, stage {args.stage}",
                  "design": design, "tasks": {}, "complete": False,
                  "started_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    manifest = RunManifest(run_id, "Gate G", f"sweep regime: {args.stage}", design,
                           seeds={"seed_sets": list(SEED_SETS[args.stage]), "readouts": READOUTS},
                           dataset=str(config.get("connectome.dataset")),
                           input_files={**dataset_files(), "script": Path(__file__),
                                        "sweep": ROOT / "src/flymario/training/sweep.py",
                                        "plastic": ROOT / "src/flymario/training/plastic.py",
                                        "pair": pair_file, "plan": ROOT / "docs/GATE_G_PLAN.md",
                                        "config": ROOT / "configs/base.yaml"},
                           output_files={"report": out})
    n_steps = sum(len(milestones) for _ in groups)
    ctl = RunControl(run_id, total=n_steps, checkpoint=lambda: None, unit="block",
                     meta={"question": f"Gate G sweep regime: {args.stage}"}, fresh=report["tasks"] == {})
    try:
        for tag, (rule, conds) in groups.items():
            learners = learners_for(conds, args.stage)
            train_and_eval(tag, args.stage, learners, rule, args.n_train, args.n_test, milestones,
                           W, motor, cues, report, out, ctl, key)
        # analysis on the final evaluation
        final = str(args.n_train)
        rows = [r for rec in report["tasks"].values() for r in rec["evaluations"][final]["rows"]]
        if args.stage == "explore":
            refs = [r for r in rows if r["condition"] in ("FROZEN", "SHUFFLED")]
            ranking = []
            for v in VARIANTS:
                xs = [r["accuracy"] for r in rows if r["condition"] == v]
                ranking.append({"variant": v, "mean": float(np.mean(xs)), "min": float(np.min(xs)),
                                "n>=.75": int(sum(x >= .75 for x in xs)),
                                "options": sum(bool(o) for o in VARIANTS[v][1].values())
                                + (VARIANTS[v][0].get("kappa", 1.0) != 1.0)})
            ranking.sort(key=lambda x: (-round(x["mean"], 2), -x["min"], x["options"]))
            report["selection"] = {"ranking": ranking, "selected": ranking[0]["variant"]}
            report["criterion_exploratory"] = {v: criterion(refs + [r for r in rows if r["condition"] == v],
                                                            v, args.n_test)
                                               for v in list(VARIANTS) + ["SUPERVISED"]}
            report["controls"] = controls_check(rows)
        elif args.stage == "reference":
            report["criterion_on_references"] = {
                c: criterion(rows, c, args.n_test) for c in ("SUPERVISED", "SHUFFLED", "ANTI", "FROZEN")}
            report["controls"] = controls_check(rows)
            exp = {"SUPERVISED": True, "SHUFFLED": False, "ANTI": False, "FROZEN": False}
            report["criterion_validated"] = all(
                report["criterion_on_references"][c]["pass"] == v for c, v in exp.items())
        else:
            report["criterion"] = criterion(rows, args.variant, args.n_test)
            report["criterion_supervised_reference"] = criterion(rows, "SUPERVISED", args.n_test)
            report["controls"] = controls_check(rows)
            report["gate_g_pass"] = bool(report["criterion"]["pass"] and report["controls"]["pass"])
        report.update({"complete": True, "manifest": str(manifest.path),
                       "finished_utc": datetime.now(UTC).isoformat(timespec="seconds")})
        part = out.with_suffix(".json.part")
        part.write_text(json.dumps(report, indent=2, allow_nan=False))
        part.replace(out)
        ctl.close("complete")
        manifest.finish("complete")
        for k in ("selection", "criterion_validated", "criterion", "controls", "gate_g_pass"):
            if k in report:
                v = report[k]
                print(k, json.dumps(v if k != "criterion" else {kk: v[kk] for kk in ("pass", "i", "ii")}))
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
