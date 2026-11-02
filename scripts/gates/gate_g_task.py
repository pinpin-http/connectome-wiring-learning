#!/usr/bin/env python
"""Gate G on the connectome: the two-cue task (docs/GATE_G_PLAN.md §2-§8).

    python scripts/gates/gate_g_task.py resume-check   # bit-identical resume, real brain
    python scripts/gates/gate_g_task.py scale          # the η scale (§7), no learning
    python scripts/gates/gate_g_task.py criterion      # §6.2: test the criterion first
    python scripts/gates/gate_g_task.py pilot          # §7: choose η per rule
    python scripts/gates/gate_g_task.py evaluate       # §8: the gate itself

Training checkpoints every 50 trials and on request (make follow / stop / checkpoint); a relaunch resumes
bit-identically. Results: artifacts/runs/gate_g_*.json. `criterion()` (§8) serves both the check and the gate.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flymario import config
from flymario.connectome import datasets
from flymario.paths import ROOT, RUNS
from flymario.policy.readout import block_seeds
from flymario.runctl import RunControl, StopRequested
from flymario.run_manifest import RunManifest, dataset_files
from flymario.training.two_cue import (LEFT, RIGHT, MASTER, PURPOSES, Learner, TwoCue,
                                       cue_currents, streams)  # noqa: F401

READOUTS = block_seeds(3)
SETS = (1, 2)
# Selected for the next immediate-reward TwoCue stage from the completed v3
# directional screen. Its eligibility resets each decision; it is not the
# delayed-credit rule for Mario.
RULES = ("node_perturbation",)
N_TRAIN, N_TEST, CHUNK = 2000, 200, 50
GRID = (1e-4, 1e-3, 1e-2)
CKPT = RUNS / "checkpoints"


# -- setup ---------------------------------------------------------------------
def setup():
    ds = datasets.DEFAULT
    g = datasets.load(ds)
    W = g.to_sparse()
    motor = datasets.motor_index(g, ds)
    pair = json.loads((RUNS / "gate_g_pair.json").read_text())["chosen"]["pair"]
    return {"g": g, "W": W, "motor": motor, "cues": cue_currents(g, pair), "pair": pair}


def surviving_rules() -> list[str]:
    """Require a powered, positively aligned signal before a feasibility pilot.

    Exact scale calibration remains a fidelity diagnostic. This screen only
    requires useful direction in every declared feedback/noise configuration.
    """
    verdict_path = RUNS / "gate_g_estimator_verdict_v3.json"
    if not verdict_path.exists():
        raise SystemExit("run the amended four-configuration estimator check first")
    rep = json.loads(verdict_path.read_text())
    if rep.get("protocol_version") != 3:
        raise SystemExit("learning feasibility requires the amended estimator protocol v3")
    if not rep.get("complete") or not rep.get("controls_behaved"):
        raise SystemExit("the estimator check (§6.1) is incomplete, or its negative controls misbehaved")
    expected = {"no_feedback_s0.1", "feedback_s0.1",
                "no_feedback_s0.02", "feedback_s0.02"}
    if set(rep.get("configurations", {})) != expected:
        raise SystemExit("the estimator verdict must cover feedback/no-feedback at σ=0.1 and σ=0.02")
    passed = []
    for rule in RULES:
        configs = list(rep["configurations"].values())
        if len(configs) != 4:
            continue
        usable = all(
            (entry := c.get("rules", {}).get(rule)) is not None
            and entry.get("direction_screen_pass", False)
            for c in configs
        )
        if usable:
            passed.append(rule)
    if not passed:
        raise SystemExit("no candidate has an independently validated positive direction in all four "
                         "feedback × noise configurations")
    return passed


def require_bptt_control() -> None:
    """Do not interpret a failed local rule before checking controllability."""
    path = RUNS / "gate_g_bptt_control.json"
    if not path.exists():
        raise SystemExit("run the BPTT controllability positive control before the learning pilot")
    rep = json.loads(path.read_text())
    if not rep.get("pass"):
        raise SystemExit("the BPTT control did not establish controllability; review the "
                         "plastic subset, fixed readout or task before testing a local rule")


def require_bptt_witness() -> dict:
    """Permit only the exploratory §7 pilot after the staged one-readout witness."""
    path = RUNS / "gate_g_bptt_pilot.json"
    if not path.exists():
        raise SystemExit("run the staged BPTT controllability witness before the exploratory learning pilot")
    rep = json.loads(path.read_text())
    witness = rep.get("staged_witness", {})
    if not witness.get("scope", "").startswith("one seed set, one readout, two mappings"):
        raise SystemExit("the BPTT pilot does not contain the declared staged witness")
    if witness.get("block_len") != 1 or not witness.get("pass"):
        raise SystemExit("the staged BPTT witness did not establish one-decision controllability")
    return rep


# -- accuracy from recorded decision traces ------------------------------------------
def readout_rows(M: int, seed: int) -> np.ndarray:
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.policy.readout import FixedRandomProjection
    return np.asarray(FixedRandomProjection(M, len(ACTION_NAMES), seed).W)[[RIGHT, LEFT]]


def accuracy(features: np.ndarray, cues: np.ndarray, readout_seed: int, mapping: int) -> float:
    """Fraction of trials where the fixed readout's RIGHT-vs-LEFT choice is right.
    features: (trials, M) decision-time traces."""
    rows = readout_rows(features.shape[1], readout_seed)
    s = features @ rows.T                                     # (trials, 2)
    action = (s[:, 1] > s[:, 0]).astype(int)                  # ties -> RIGHT, as in TwoCue
    return float(np.mean(action == (cues ^ mapping)))


def positive_reference(Xtr, ytr, Xte, yte) -> float:
    """The two task rows replaced by a supervised logistic regression on the
    motor features -- deliberately cheating: a ceiling, never a result (§5).
    Same decoder settings as the Gate D/E instrument."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Continuous motor traces include decayed float32 subnormals. A float32
    # transform can round their nonzero fitted scales to zero and create NaNs.
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=2000, random_state=0).fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


# -- the criterion, as formalised before §6.2 -----------------------------------------
def criterion(cand: dict, eta0: dict, shuf: dict, n: dict) -> dict:
    """docs/GATE_G_PLAN.md §8. Arguments map a block (seed_set, readout_seed) to test accuracy; `n` to its test count.

    u = max over candidate and controls of max(largest binomial SE, half the gap between seed-set means).
    Pass iff in every block: candidate >= 0.75; candidate − η0 > 2u; candidate − shuffled > 2u; |η0 − 0.5|,
    |shuffled − 0.5| <= 2u.
    """
    blocks = sorted(cand)

    def u_of(x):
        se = max(np.sqrt(max(x[b] * (1 - x[b]), 1e-12) / n[b]) for b in blocks)
        sets = sorted({b[0] for b in blocks})
        means = [np.mean([x[b] for b in blocks if b[0] == s]) for s in sets]
        gap = abs(means[0] - means[1]) / 2 if len(means) == 2 else 0.0
        return float(max(se, gap)), float(se), float(gap)

    parts = {k: u_of(x) for k, x in (("candidate", cand), ("eta0", eta0), ("shuffled", shuf))}
    u = max(p[0] for p in parts.values())
    checks = {
        "candidate_at_least_0.75": all(cand[b] >= 0.75 for b in blocks),
        "beats_eta0_by_2u": all(cand[b] - eta0[b] > 2 * u for b in blocks),
        "beats_shuffled_by_2u": all(cand[b] - shuf[b] > 2 * u for b in blocks),
        "eta0_within_2u_of_half": all(abs(eta0[b] - 0.5) <= 2 * u for b in blocks),
        "shuffled_within_2u_of_half": all(abs(shuf[b] - 0.5) <= 2 * u for b in blocks),
    }
    return {"pass": all(checks.values()), "checks": checks, "u": u,
            "u_parts": {k: {"u": v[0], "binomial_se": v[1], "half_set_gap": v[2]}
                        for k, v in parts.items()},
            "blocks": {f"set{b[0]}|ro{b[1]}": {"candidate": cand[b], "eta0": eta0[b],
                                               "shuffled": shuf[b], "n": n[b]} for b in blocks}}


# -- running learners ------------------------------------------------------------------
def train(S, learners, n_trials, name, ctl, record=(), chunk=CHUNK, gpu_duty=0.6):
    """Train with checkpoints every `chunk` trials; resumes from the last one."""
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / f"{name}.pt"
    run = TwoCue(S["W"], S["motor"], S["cues"], learners, "train", n_trials,
                 record=list(record), cache_key=("gate_g", 0))
    if path.exists():
        try:
            run.load(path)
            print(f"resumed {name} at trial {run.trial}", flush=True)
        except ValueError:
            print(f"checkpoint {path.name} belongs to another run: starting fresh", flush=True)
    ctl._checkpoint = lambda: run.save(path)
    while run.trial < n_trials:
        t0 = time.time()
        k0 = run.trial
        ctl.begin(f"{name}: trials {k0}-{min(k0 + chunk, n_trials)}")
        run.run(chunk, ctl=ctl, gpu_duty=gpu_duty)
        run.save(path)
        c = run.log["correct"][max(0, run.trial - chunk):run.trial]
        for b, l in enumerate(learners):
            ctl.point("train", l.name.split("|")[0], run.trial, float(c[:, b].mean()), line=l.name)
        ctl.done({"trial": run.trial,
                  "correct": round(float(run.log["correct"][k0:run.trial].mean()), 3)},
                 seconds=time.time() - t0, kind="train")
    return run


def test(S, learners, weights, name, ctl, gpu_duty=0.6):
    """Learning frozen; N_TEST trials on the test streams (noise never used in
    training); decision traces recorded for every brain."""
    frozen = [Learner(l.name, l.seed_set, None, 0.0, "true", l.mapping, l.readout_seed)
              for l in learners]
    ctl.begin(f"{name}: test")
    run = TwoCue(S["W"], S["motor"], S["cues"], frozen, "test", N_TEST, record=True,
                 cache_key=("gate_g", 0))
    run.brain.set_weights(weights)
    run.run(N_TEST, ctl=ctl, gpu_duty=gpu_duty)
    ctl.done({"test": "done"}, kind="test")
    return run


def summary_train(run) -> dict:
    out = {}
    for b, l in enumerate(run.learners):
        c = run.log["correct"][:run.trial, b]
        out[l.name] = {"learner": l.__dict__, "correct_by_50": [
            float(c[i:i + 50].mean()) for i in range(0, len(c), 50)],
            "update_rms_mean": float(run.log["update_rms"][:run.trial, b].mean()),
            "motor_rate_mean": float(run.log["rate_motor"][:run.trial, b].mean())}
    return out


def trials_to(correct: np.ndarray, level: float = 0.75, window: int = 50) -> int | None:
    """First trial at which the mean of the previous `window` trials reaches `level`."""
    c = np.convolve(correct, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(c >= level)
    return int(hit[0] + window) if len(hit) else None


def meta(question, why, method):
    return {"question": question, "why": why, "method": method}


def write(path: Path, rep: dict) -> None:
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rep, indent=2, default=float))
    tmp.replace(path)


# -- modes -----------------------------------------------------------------------------
def mode_resume_check(S, args) -> None:
    """On the real connectome: 30 trials straight vs 12 + checkpoint + a fresh
    process-equivalent object + 18. Must be bit-identical."""
    import torch
    L = [Learner("np", 1, "node_perturbation", 0.01, "true", 0, READOUTS[0]),
         Learner("sr", 2, "spike_reinforce", 0.3, "true", 1, READOUTS[1]),
         Learner("eta0", 1),
         Learner("np_full", 1, "node_perturbation_full_trajectory", 0.01,
                 "true", 0, READOUTS[0])]
    mk = lambda: TwoCue(S["W"], S["motor"], S["cues"], L, "train", 30, record=[2],  # noqa: E731
                        cache_key=("gate_g", 0))
    a = mk()
    a.run(30)
    b = mk()
    b.run(12)
    CKPT.mkdir(parents=True, exist_ok=True)
    b.save(CKPT / "resume_check.pt")
    del b
    c = mk()
    c.load(CKPT / "resume_check.pt")
    c.run(18)
    same = {"weights": bool(torch.equal(a.brain.m, c.brain.m)),
            "baselines": bool(torch.equal(a.Rbar, c.Rbar)),
            "membranes": bool(torch.equal(a.brain.v, c.brain.v)),
            "rewards": bool(np.array_equal(a.log["reward"], c.log["reward"])),
            "features": bool(np.array_equal(a.features, c.features)),
            "full_trajectory_score": bool(torch.equal(
                a.rules["node_perturbation_full_trajectory"][0].e,
                c.rules["node_perturbation_full_trajectory"][0].e)),
            "pending_block_update": bool(torch.equal(
                a.pending["node_perturbation_full_trajectory"],
                c.pending["node_perturbation_full_trajectory"]))}
    moved = (a.brain.m - a.brain.m0[None, :]).abs().sum(1).cpu().numpy().tolist()
    rep = {"what": "bit-identical resume on the connectome", "identical": same,
           "all_identical": all(same.values()), "weight_change_L1_per_learner": moved,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rep["manifest"] = args.manifest_path
    write(RUNS / "gate_g_resume_check.json", rep)
    (CKPT / "resume_check.pt").unlink(missing_ok=True)
    print(json.dumps(rep, indent=1))


def mode_scale(S, args) -> None:
    """§7's scale: the η at which one update's RMS change equals 10 x the median initial synapse count,
    so the grid {1e-4, 1e-3, 1e-2} x scale moves a typical connection by 0.1 %, 1 %, 10 %. Measured at η = 0."""
    rules = surviving_rules()
    L = [Learner(f"{r}|m{mp}", 1, r, 0.0, "true", mp, READOUTS[0]) for r in rules for mp in (0, 1)]
    ctl = RunControl("gate_g_scale", total=1, checkpoint=lambda: None, unit="step", meta=meta(
        "How big is one update of each rule, per unit of learning rate?",
        "The pilot's three learning rates are multiples of a scale that must be measured first.",
        "50 trials without learning; RMS of (R − R̄)e over the 769,342 plastic connections."))
    ctl.begin("50 trials, no learning")
    run = TwoCue(S["W"], S["motor"], S["cues"], L, "train", 50, cache_key=("gate_g", 0))
    run.run(50, ctl=ctl)
    ctl.done({"ok": True})
    med = float(run.brain.m0.median())
    rep = {"median_initial_count": med, "definition": "scale = 10 * median m0 / median trial RMS",
           "rules": {}}
    for r in rules:
        idx = [b for b, l in enumerate(L) if l.rule == r]
        if r == "node_perturbation_full_trajectory":
            values = run.log["block_update_rms"][:, idx]
            values = values[values > 0]
        else:
            values = run.log["update_rms"][:, idx]
            values = values[values > 0]
        rms = float(np.median(values))
        scale = 10.0 * med / rms
        rep["rules"][r] = {"update_rms_per_unit_eta": rms, "scale": scale,
                           "grid": [g * scale for g in GRID]}
    rep["motor_rate"] = float(run.log["rate_motor"].mean())
    rep["correct_without_learning"] = float(run.log["correct"].mean())
    rep["manifest"] = args.manifest_path
    write(RUNS / "gate_g_scale.json", rep)
    ctl.close("complete")
    print(json.dumps(rep, indent=1))


def controls_and_reference(S, rules, etas, n_train, name, ctl):
    """η = 0 (features recorded, for the positive reference) and shuffled reward
    per rule, on both seed sets; train then test."""
    L = [Learner(f"eta0|set{s}", s) for s in SETS]
    L += [Learner(f"shuffled_{r}|set{s}", s, r, etas[r], "shuffled", 0, None)
          for r in rules for s in SETS]
    return L


def evaluate_blocks(test_run, train_run, learners, designs=("single", "counterbalanced")):
    """Per design, per learner family, a block -> accuracy table."""
    cues_te = {s: test_run.cues[[b for b, l in enumerate(test_run.learners)
                                 if l.seed_set == s][0]].cpu().numpy() for s in SETS}
    out = {d: {} for d in designs}
    feats = test_run.features                                          # (trials, M, B)
    for d in designs:
        maps = (0,) if d == "single" else (0, 1)
        fam, fam_by_mapping = {}, {}
        for b, l in enumerate(learners):
            key = l.name.split("|")[0]
            if l.reward == "true" and l.rule is not None:
                # Keep each trained mapping visible as well as its balanced
                # aggregate; averaging alone can hide a failed direction.
                if l.mapping not in maps:
                    continue
                blk = (l.seed_set, l.readout_seed)
                acc = accuracy(feats[:, :, b], cues_te[l.seed_set], l.readout_seed, l.mapping)
                fam.setdefault(key, {}).setdefault(blk, []).append(acc)
                fam_by_mapping.setdefault(key, {}).setdefault(blk, {})[l.mapping] = acc
            else:
                # controls: one brain per seed set, read by every block's readout
                for r in READOUTS:
                    blk = (l.seed_set, r)
                    accs = {mp: accuracy(feats[:, :, b], cues_te[l.seed_set], r, mp)
                            for mp in maps}
                    fam.setdefault(key, {})[blk] = list(accs.values())
                    fam_by_mapping.setdefault(key, {})[blk] = accs
        out[d] = {k: {blk: float(np.mean(v)) for blk, v in x.items()} for k, x in fam.items()}
        out[d]["_by_mapping"] = fam_by_mapping
        # Mapped learners reuse the same held-out cue/noise trials; count the
        # paired evaluation conservatively as N_TEST, not 2*N_TEST.
        out[d]["_n"] = {blk: N_TEST for blk in
                        {(s, r) for s in SETS for r in READOUTS}}
    # positive reference: mapping-invariant (labels are the cue)
    pos = {}
    tr_idx = [i for i, l in enumerate(train_run.learners) if l.name.startswith("eta0")]
    for j, i in enumerate(tr_idx):
        l = train_run.learners[i]
        Xtr = train_run.features[:train_run.trial, :, j]
        ytr = train_run.cues[i, :train_run.trial].cpu().numpy()
        bte = [b for b, t in enumerate(test_run.learners) if t.name == l.name][0]
        acc = positive_reference(Xtr, ytr, feats[:, :, bte], cues_te[l.seed_set])
        for r in READOUTS:
            pos[(l.seed_set, r)] = acc
    for d in designs:
        out[d]["positive_reference"] = pos
    return out


def mode_criterion(S, args) -> None:
    """§6.2: the positive reference must PASS; η = 0 and shuffled must FAIL."""
    rules = surviving_rules()
    sc = json.loads((RUNS / "gate_g_scale.json").read_text())["rules"]
    etas = {r: GRID[-1] * sc[r]["scale"] for r in rules}         # the pilot's largest η
    L = controls_and_reference(S, rules, etas, N_TRAIN, "criterion", None)
    rec = [i for i, l in enumerate(L) if l.name.startswith("eta0")]
    chunks = -(-N_TRAIN // CHUNK) + 1
    ctl = RunControl("gate_g_criterion", total=chunks, checkpoint=lambda: None, unit="chunk",
                     meta=meta("Does the pass criterion separate what must pass from what must fail?",
                               "A criterion never run on known cases can be wrong either way "
                               "(the owner's condition, 2026-09-23).",
                               "Run the controls exactly like learners; the cheating supervised "
                               "decoder must PASS, no-learning and shuffled reward must FAIL."))
    ctl.chart("train", "Training accuracy of the controls", "trial", "fraction correct",
              ["eta0"] + [f"shuffled_{r}" for r in rules], yfmt="pct",
              refs=[{"axis": "y", "value": 0.5, "label": "chance"}],
              note="Neither control can learn the task: both should hover around chance.")
    try:
        tr = train(S, L, N_TRAIN, "gate_g_criterion", ctl, record=rec,
                   gpu_duty=args.gpu_duty)
        te = test(S, L, tr.brain.m, "gate_g_criterion", ctl, args.gpu_duty)
    except StopRequested:
        ctl.close("stopped")
        raise SystemExit(0)
    tabs = evaluate_blocks(te, tr, L)
    rep = {"what": "Gate G §6.2 criterion check", "etas_shuffled": etas, "pair": S["pair"],
           "n_train": N_TRAIN, "n_test": N_TEST, "readouts": READOUTS, "designs": {}}
    rep["manifest"] = args.manifest_path
    for d, tab in tabs.items():
        maps = (0,) if d == "single" else (0, 1)
        n = tab["_n"]
        e0 = tab["eta0"]
        res = {"positive_reference": criterion(tab["positive_reference"], e0,
                                                tab[f"shuffled_{rules[0]}"], n)}
        res["positive_reference"]["checks"]["each_mapping_at_least_0.75"] = all(
            tab["positive_reference"][blk] >= 0.75
            for blk in tab["positive_reference"] for _ in maps
        )
        res["positive_reference"]["pass"] = bool(
            res["positive_reference"]["pass"]
            and res["positive_reference"]["checks"]["each_mapping_at_least_0.75"]
        )
        res["eta0_as_candidate"] = criterion(e0, e0, tab[f"shuffled_{rules[0]}"], n)
        eta0_map_floor = all(
            tab["_by_mapping"]["eta0"][blk].get(mapping, -np.inf) >= 0.75
            for blk in e0 for mapping in maps
        )
        res["eta0_as_candidate"]["checks"]["each_mapping_at_least_0.75"] = bool(eta0_map_floor)
        res["eta0_as_candidate"]["pass"] = bool(
            res["eta0_as_candidate"]["pass"] and eta0_map_floor
        )
        for r in rules:
            name = f"shuffled_{r}"
            item = criterion(tab[name], e0, tab[name], n)
            map_floor = all(
                tab["_by_mapping"][name][blk].get(mapping, -np.inf) >= 0.75
                for blk in tab[name] for mapping in maps
            )
            item["checks"]["each_mapping_at_least_0.75"] = bool(map_floor)
            item["pass"] = bool(item["pass"] and map_floor)
            res[f"{name}_as_candidate"] = item
        sep = (res["positive_reference"]["pass"] and not res["eta0_as_candidate"]["pass"]
               and not any(res[f"shuffled_{r}_as_candidate"]["pass"] for r in rules))
        # Self-comparisons force the negative candidates to fail. Preserve
        # that algebraic check without certifying empirical false positives.
        rep["designs"][d] = {"separates": False,
                             "algebraic_separation": sep,
                             "empirical_negative_controls_validated": False,
                             "validation_note": "independent negative pseudo-candidates required",
                             "results": res}
    rep["train"] = summary_train(tr)
    rep["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write(RUNS / "gate_g_criterion.json", rep)
    ctl.close("complete")
    for d, v in rep["designs"].items():
        print(d, "separates:", v["separates"],
              {k: x["pass"] for k, x in v["results"].items()})


def ols(y: np.ndarray) -> dict:
    """Slope of y on trial index, with its 95 % interval."""
    from scipy import stats
    x = np.arange(len(y))
    r = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(y) - 2)
    return {"slope": float(r.slope), "ci95": [float(r.slope - t * r.stderr),
                                              float(r.slope + t * r.stderr)]}


def mode_pilot(S, args) -> None:
    """§7: seed set 1, readout seed 1, both mappings; each rule at three η;
    500 trials; η chosen by the training-reward slope (declared in §9)."""
    bptt_witness = require_bptt_witness()
    rules = surviving_rules()
    sc = json.loads((RUNS / "gate_g_scale.json").read_text())["rules"]
    n = 500
    L = [Learner(f"{r}|x{g:g}|m{mp}", 1, r, g * sc[r]["scale"], "true", mp, READOUTS[0])
         for r in rules for g in GRID for mp in (0, 1)]
    L.append(Learner("eta0|set1", 1))
    ctl = RunControl("gate_g_pilot", total=n // CHUNK, checkpoint=lambda: None, unit="chunk",
                     meta=meta("Which learning rate should each rule use?",
                               "Too small learns nothing in the budget; too large is noise. "
                               "Chosen on training trials only, never on the evaluation seeds.",
                               "Seed set 1, readout 1, both cue->action mappings; three rates "
                               "per rule (0.1 %, 1 %, 10 % of a typical synapse per update)."))
    ctl.chart("train", "Pilot: fraction correct per 50 trials", "trial", "fraction correct",
              list(rules) + ["eta0"], yfmt="pct",
              refs=[{"axis": "y", "value": 0.5, "label": "chance"},
                    {"axis": "y", "value": 0.75, "label": "gate"}],
              note="One line per learner; the gate asks for 75 % after learning.")
    try:
        tr = train(S, L, n, "gate_g_pilot", ctl, gpu_duty=args.gpu_duty)
    except StopRequested:
        ctl.close("stopped")
        raise SystemExit(0)
    rep = {"what": "Gate G §7 pilot (exploratory)", "n": n, "scale": sc,
           "gpu_duty_target": args.gpu_duty,
           "gpu_memory_fraction": args.gpu_memory_fraction,
           "bptt_witness_manifest": bptt_witness.get("manifest"),
           "bptt_witness_pass": bool(bptt_witness["staged_witness"]["pass"]),
           "bptt_witness_scope": bptt_witness["staged_witness"]["scope"],
           "rules": {}}
    rep["manifest"] = args.manifest_path
    any_signal = False
    for r in rules:
        rows = {}
        for g in GRID:
            idx = [b for b, l in enumerate(L) if l.rule == r and l.name.split("|")[1] == f"x{g:g}"]
            reward = tr.log["reward"][:n, idx].mean(1)
            fit = ols(reward)
            any_signal |= fit["ci95"][0] > 0
            rows[f"{g:g}"] = {"eta": g * sc[r]["scale"], **fit,
                              "correct_first100": float(tr.log["correct"][:100, idx].mean()),
                              "correct_last100": float(tr.log["correct"][n - 100:n, idx].mean()),
                              "correct_last100_by_mapping": tr.log["correct"][n - 100:n, idx].mean(0).tolist(),
                              "clipped_low": int(tr.log["clipped"][:n, 0].sum())}
        best = max(rows, key=lambda k: rows[k]["slope"])
        rep["rules"][r] = {"grid": rows, "chosen_multiplier": float(best),
                           "chosen_eta": rows[best]["eta"]}
    rep["eta0_correct"] = float(tr.log["correct"][:n, -1].mean())
    rep["usable_gradient_signal"] = bool(any_signal)
    rep["fallback_triggered"] = not any_signal
    rep["train"] = summary_train(tr)
    rep["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write(RUNS / "gate_g_pilot.json", rep)
    ctl.close("complete")
    print(json.dumps({r: {k: {kk: v[kk] for kk in ("slope", "ci95", "correct_last100")}
                          for k, v in x["grid"].items()} | {"chosen": x["chosen_multiplier"]}
                      for r, x in rep["rules"].items()}, indent=1))
    print("usable gradient signal:", any_signal)


def mode_evaluate(S, args) -> None:
    """§8: the gate. Chosen η per rule; both seed sets, three readouts; the
    design (single or counterbalanced) fixed by the §6.2 check."""
    criterion_report = json.loads((RUNS / "gate_g_criterion.json").read_text())
    if not any(d.get("empirical_negative_controls_validated", False)
               for d in criterion_report.get("designs", {}).values()):
        raise SystemExit("§6.2 negative self-comparisons are algebraic; validate independent "
                         "negative candidates before a Gate G evaluation")
    require_bptt_control()
    rules = surviving_rules()
    pilot = json.loads((RUNS / "gate_g_pilot.json").read_text())
    if pilot["fallback_triggered"]:
        raise SystemExit("the pilot found no usable gradient signal: §4's fallback first")
    crit = json.loads((RUNS / "gate_g_criterion.json").read_text())["designs"]
    design = ("single" if crit["single"]["separates"] else
              "counterbalanced" if crit["counterbalanced"]["separates"] else None)
    if design is None:
        raise SystemExit("the criterion does not separate its references (§6.2)")
    maps = (0,) if design == "single" else (0, 1)
    etas = {r: pilot["rules"][r]["chosen_eta"] for r in rules}
    L = [Learner(f"{r}|set{s}|ro{i + 1}|m{mp}", s, r, etas[r], "true", mp, READOUTS[i])
         for r in rules for s in SETS for i in range(len(READOUTS)) for mp in maps]
    L += [Learner(f"eta0|set{s}", s) for s in SETS]
    L += [Learner(f"shuffled_{r}|set{s}", s, r, etas[r], "shuffled", 0, None)
          for r in rules for s in SETS]
    rec = [i for i, l in enumerate(L) if l.name.startswith("eta0")]
    ctl = RunControl("gate_g_evaluate", total=N_TRAIN // CHUNK + 1, checkpoint=lambda: None,
                     unit="chunk", meta=meta(
        "Does a learning rule make the fly brain solve the two-cue task?",
        "Gate G (protocol §12.4): the rule must learn a simple task under controlled "
        "conditions before it is trusted with Mario.",
        f"Each rule at its pilot learning rate; 2 seed sets x 3 fixed readouts"
        f"{' x 2 mappings' if len(maps) == 2 else ''}; no-learning and shuffled-reward "
        f"controls; 2,000 training trials, then 200 test trials with learning frozen."))
    ctl.chart("train", "Fraction correct per 50 training trials", "trial", "fraction correct",
              list(rules) + ["eta0"] + [f"shuffled_{r}" for r in rules], yfmt="pct",
              refs=[{"axis": "y", "value": 0.5, "label": "chance"},
                    {"axis": "y", "value": 0.75, "label": "gate"}],
              note="Learners should climb; the two controls must stay at chance.")
    try:
        tr = train(S, L, N_TRAIN, "gate_g_evaluate", ctl, record=rec,
                   gpu_duty=args.gpu_duty)
        te = test(S, L, tr.brain.m, "gate_g_evaluate", ctl, args.gpu_duty)
    except StopRequested:
        ctl.close("stopped")
        raise SystemExit(0)
    tab = evaluate_blocks(te, tr, L, designs=(design,))[design]
    n = tab["_n"]
    rep = {"what": "Gate G §8 evaluation", "design": design, "etas": etas, "pair": S["pair"],
           "readouts": READOUTS, "n_train": N_TRAIN, "n_test": N_TEST, "rules": {}}
    rep["manifest"] = args.manifest_path
    for r in rules:
        v = criterion(tab[r], tab["eta0"], tab[f"shuffled_{r}"], n)
        candidate_maps = tab["_by_mapping"][r]
        map_floor = all(
            candidate_maps[blk].get(mapping, -np.inf) >= 0.75
            for blk in tab[r] for mapping in maps
        )
        v["checks"]["each_mapping_at_least_0.75"] = bool(map_floor)
        v["pass"] = bool(v["pass"] and map_floor)
        tt = [trials_to(tr.log["correct"][:, b]) for b, l in enumerate(L)
              if l.rule == r and l.reward == "true"]
        tt_num = [x if x is not None else N_TRAIN + 1 for x in tt]
        rep["rules"][r] = {"criterion": v, "trials_to_0.75": tt,
                           "accuracy_by_mapping": {
                               f"set{blk[0]}|ro{blk[1]}": {
                                   f"mapping{mapping}": float(candidate_maps[blk][mapping])
                                   for mapping in maps
                               } for blk in tab[r]
                           },
                           "median_trials_to_0.75": float(np.median(tt_num))}
    passing = [r for r in rules if rep["rules"][r]["criterion"]["pass"]]
    rep["passing"] = passing
    rep["chosen"] = (min(passing, key=lambda r: rep["rules"][r]["median_trials_to_0.75"])
                     if passing else None)
    rep["gate"] = "PASS" if passing else "FAIL (learning-rule feasibility, §12.4)"
    rep["positive_reference"] = {f"set{b[0]}|ro{b[1]}": v
                                 for b, v in tab["positive_reference"].items()}
    rep["train"] = summary_train(tr)
    rep["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write(RUNS / "gate_g_evaluate.json", rep)
    CKPT.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"learners": [l.__dict__ for l in L], "m": tr.brain.m.cpu()},
               CKPT / "gate_g_final_weights.pt")
    ctl.close("complete")
    print(json.dumps({r: {"pass": x["criterion"]["pass"], "u": x["criterion"]["u"],
                          "median_trials_to_0.75": x["median_trials_to_0.75"]}
                      for r, x in rep["rules"].items()}, indent=1))
    print("GATE G:", rep["gate"], "chosen:", rep["chosen"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["resume-check", "scale", "criterion", "pilot", "evaluate"])
    ap.add_argument("--gpu-duty", type=float, default=0.6,
                    help="target mean GPU duty per simulation tick (default: 60%%)")
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.50,
                    help="cap this process's PyTorch GPU allocator (default: 50%%)")
    args = ap.parse_args()
    if not 0.0 < args.gpu_duty <= 1.0:
        ap.error("--gpu-duty must be in (0, 1]")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        ap.error("--gpu-memory-fraction must be in (0, 1]")
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,
                                                   device=torch.cuda.current_device())
    elif config.get("dynamics.backend") == "gpu":
        raise RuntimeError("Gate G is configured for GPU execution, but no PyTorch GPU is available")
    S = setup()
    outputs = {
        "resume-check": {"report": RUNS / "gate_g_resume_check.json"},
        "scale": {"report": RUNS / "gate_g_scale.json",
                  "live_status": RUNS / "live/gate_g_scale/status.json",
                  "live_series": RUNS / "live/gate_g_scale/series.jsonl"},
        "criterion": {"report": RUNS / "gate_g_criterion.json",
                      "checkpoint": CKPT / "gate_g_criterion.pt",
                      "live_status": RUNS / "live/gate_g_criterion/status.json",
                      "live_series": RUNS / "live/gate_g_criterion/series.jsonl"},
        "pilot": {"report": RUNS / "gate_g_pilot.json",
                  "checkpoint": CKPT / "gate_g_pilot.pt",
                  "live_status": RUNS / "live/gate_g_pilot/status.json",
                  "live_series": RUNS / "live/gate_g_pilot/series.jsonl"},
        "evaluate": {"report": RUNS / "gate_g_evaluate.json",
                     "checkpoint": CKPT / "gate_g_evaluate.pt",
                     "final_weights": CKPT / "gate_g_final_weights.pt",
                     "live_status": RUNS / "live/gate_g_evaluate/status.json",
                     "live_series": RUNS / "live/gate_g_evaluate/series.jsonl"},
    }[args.mode]
    dataset = str(config.get("connectome.dataset"))
    min_syn = int(config.get("connectome.min_syn"))
    input_files = dataset_files(dataset, min_syn)
    input_files.update({"gate_task": Path(__file__),
                        "two_cue": ROOT / "src/flymario/training/two_cue.py",
                        "learning_rules": ROOT / "src/flymario/training/rules.py",
                        "base_config": ROOT / "configs/base.yaml",
                        "gate_plan": ROOT / "docs/GATE_G_PLAN.md",
                        "gate_pair": RUNS / "gate_g_pair.json",
                        "estimator_verdict": RUNS / "gate_g_estimator_verdict_v3.json"})
    if args.mode == "pilot":
        input_files["staged_bptt_witness"] = RUNS / "gate_g_bptt_pilot.json"
    if args.mode == "evaluate":
        input_files["bptt_control"] = RUNS / "gate_g_bptt_control.json"
    if args.mode in ("criterion", "pilot", "evaluate"):
        input_files["scale_report"] = RUNS / "gate_g_scale.json"
    if args.mode == "evaluate":
        input_files.update({"criterion_report": RUNS / "gate_g_criterion.json",
                            "pilot_report": RUNS / "gate_g_pilot.json"})
    seed_registry = {
        "master": MASTER, "purpose_ids": PURPOSES, "seed_sets": list(SETS),
        "readout_seeds": list(READOUTS), "stream_derivation":
        "SeedSequence([MASTER, seed_set, purpose]); spawn noise, cue, shuffled",
        "calibration_seed_sets": {str(s): {
            purpose: int(streams(s, purpose, N_TRAIN)["noise_seed"])
            for purpose in ("train", "test")} for s in SETS},
    }
    config_record = {"mode": args.mode, "dataset": dataset, "min_syn": min_syn,
                     "pair": S["pair"], "nodes": int(S["W"].shape[0]),
                     "edges": int(S["W"].nnz), "plastic_edges": int(S["W"][S["motor"]].nnz),
                     "readout_seeds": list(READOUTS), "seed_sets": list(SETS),
                     "rules": list(RULES), "n_train": N_TRAIN, "n_test": N_TEST,
                     "grid": list(GRID), "arguments": vars(args)}
    manifest = RunManifest(run_id=f"gate_g_connectome_{args.mode}", study="Gate G",
                           phase=f"connectome two-cue {args.mode}",
                           configuration=config_record, seeds=seed_registry,
                           dataset=dataset, input_files=input_files,
                           output_files=outputs)
    args.manifest_path = str(manifest.path)
    try:
        {"resume-check": mode_resume_check, "scale": mode_scale,
         "criterion": mode_criterion, "pilot": mode_pilot,
         "evaluate": mode_evaluate}[args.mode](S, args)
    except SystemExit as exc:
        status = "stopped" if exc.code in (None, 0) else "failed"
        manifest.finish(status, error=None if status == "stopped" else str(exc))
        raise
    except BaseException as exc:
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    manifest.finish("complete")


if __name__ == "__main__":
    main()
