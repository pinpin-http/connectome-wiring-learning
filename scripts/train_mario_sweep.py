#!/usr/bin/env python
"""Train the fly brain in the game, sweep regime -- pilot (docs/PROTOCOL.md §20).

    python scripts/train_mario_sweep.py --resume-check      # before any launch
    python scripts/train_mario_sweep.py                      # start or resume
    make follow / make stop / make checkpoint

Settings: docs/MARIO_PILOT_PLAN.md; trainer: src/flymario/training/mario_sweep.py. Learners come in
groups sharing noise and random draws (e.g. learner, frozen twin, shuffled-reward twin per readout).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_mario import grid, png, publish

from flymario.connectome import datasets
from flymario.paths import RUNS
from flymario.policy.readout import block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.telemetry import live
from flymario.training.mario_sweep import (
    MarioSweepTrainer,
    SweepMarioLearner,
    SweepMarioSettings,
)

CKPT = RUNS / "checkpoints"
ETA = 0.001                         # the Gate G rule B rate


TRAIN_LEVELS = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")   # generalist's pool (day)


def settings_for(version: int) -> SweepMarioSettings:
    """v1: MARIO_PILOT_PLAN amendment 2026-10-02; v2: amendment 2026-10-03;
    3 = the generalisation study (docs/GENERALISATION_STUDY.md), v2's rule."""
    if version == 1:
        return SweepMarioSettings()
    if version == 14:     # RL_STUDY §6: v2 + behaviour-policy factor + truncation bootstrap
        return SweepMarioSettings(algo="actor_critic", epsilon=0.1, death_extra=-75.0, sample_seed=20261016,
                                  pg_factor="behaviour", bootstrap_truncation=True)
    if version == 16:     # RL_STUDY §9: the v3 procedure, RL started from Stage D weights
        return SweepMarioSettings(algo="actor_critic", epsilon=0.1, death_extra=-75.0, sample_seed=20261017,
                                  pg_factor="behaviour", bootstrap_truncation=True)
    if version == 17:     # RL_STUDY §9.1: version 16, actor never updated = frozen Stage D policy
        return SweepMarioSettings(algo="actor_critic", epsilon=0.1, death_extra=-75.0, sample_seed=20261017,
                                  pg_factor="behaviour", bootstrap_truncation=True,
                                  actor_warmup=10 ** 9)
    if version == 19:     # RL_STUDY §11.1: Stage D learners + scripted references, for --eval-from
        return settings_for(13)
    if version == 18:     # RL_STUDY §9.2: Stage D (13) resumed with every update off
        from dataclasses import replace
        return replace(settings_for(13), frozen=True)
    return SweepMarioSettings(algo="actor_critic", epsilon=0.1, death_extra=-75.0,
                              sample_seed={2: 20261003, 3: 20261004, 5: 20261007, 6: 20261007,
                                           7: 20261008, 9: 20261010, 10: 20261011, 11: 20261012,
                                           12: 20261013, 13: 20261015}[version])


def generalisation_learners() -> list[SweepMarioLearner]:
    """Per fixed readout, a specialist (1-1 only) and a generalist (four levels,
    random start delay, palette jitter), paired: same noise, same random draws."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        out += [SweepMarioLearner(f"specialist_ro{i + 1}", r, 7300 + i, ETA, group=i),
                SweepMarioLearner(f"generalist_ro{i + 1}", r, 7300 + i, ETA, group=i,
                                  levels=TRAIN_LEVELS, max_delay=22, jitter=True)]
    return out


def teacher_learners() -> list[SweepMarioLearner]:
    """Version 5: the positive reference of criterion D (PERCEPTION_CREDIT_STUDY §5):
    supervised by the RAM danger label, eye on the game, generalist conditions."""
    return [SweepMarioLearner(f"teacher_ro{i + 1}", r, 7500 + i, ETA, reward="teacher", group=i,
                              levels=TRAIN_LEVELS, max_delay=22, jitter=True)
            for i, r in enumerate(block_seeds(3))]


def shuffled_teacher_learners() -> list[SweepMarioLearner]:
    """Version 6: the negative reference of criterion D -- version 5 exactly, but the
    teacher's target is a coin (jump with p 0.25), independent of the frame."""
    return [SweepMarioLearner(f"tshuffled_ro{i + 1}", r, 7500 + i, ETA, reward="teacher_shuffled", group=i,
                              levels=TRAIN_LEVELS, max_delay=22, jitter=True)
            for i, r in enumerate(block_seeds(3))]


def reference_trios() -> list[SweepMarioLearner]:
    """Version 7: fresh references for criterion E (PERCEPTION_CREDIT_STUDY §5). Per readout,
    a trio sharing noise and random draws (one group): a RAM-taught learner and two
    shuffled-target twins whose coins differ. Positive pair: teacher vs A; negative: B vs A."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"teacher2_ro{i + 1}", r, 7500 + i, ETA, reward="teacher", **kw),
                SweepMarioLearner(f"tshufA_ro{i + 1}", r, 7500 + i, ETA, reward="teacher_shuffled", **kw),
                SweepMarioLearner(f"tshufB_ro{i + 1}", r, 7500 + i, ETA, reward="teacher_shuffled", **kw)]
    return out


def oracle_study_learners() -> list[SweepMarioLearner]:
    """Version 9: the perception-or-credit study (PERCEPTION_CREDIT_STUDY §4, amended).
    Per readout, two arms, each a learner with its shuffled-reward twin (one group per
    pair: shared noise and draws): the eye sees the RAM schematic, or the game."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        sch = {"levels": TRAIN_LEVELS, "max_delay": 22, "view": "schematic", "group": 2 * i}
        pix = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": 2 * i + 1}
        out += [SweepMarioLearner(f"oracle_ro{i + 1}", r, 7400 + i, ETA, **sch),
                SweepMarioLearner(f"oracleshuf_ro{i + 1}", r, 7400 + i, ETA, reward="shuffled", **sch),
                SweepMarioLearner(f"pixels_ro{i + 1}", r, 7400 + i, ETA, **pix),
                SweepMarioLearner(f"pixelsshuf_ro{i + 1}", r, 7400 + i, ETA, reward="shuffled", **pix)]
    return out


def stage_c_learners() -> list[SweepMarioLearner]:
    """Version 12, Stage C (PERCEPTION_CREDIT_STUDY §14): per readout, a learner taught the
    take-off label and its coin twin (p TAKEOFF_P), one group; noise 7800 + i."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"tko_ro{i + 1}", r, 7800 + i, ETA, reward="teacher_takeoff", **kw),
                SweepMarioLearner(f"tkoshuf_ro{i + 1}", r, 7800 + i, ETA, reward="teacher_takeoff_shuffled", **kw)]
    return out


def stage_d_learners() -> list[SweepMarioLearner]:
    """Version 13, Stage D (PERCEPTION_CREDIT_STUDY §17): Stage C with the clean take-off
    label (ground threats only) and its coin twin; noise 7900 + i."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"tko2_ro{i + 1}", r, 7900 + i, ETA, reward="teacher_takeoff2", **kw),
                SweepMarioLearner(f"tko2shuf_ro{i + 1}", r, 7900 + i, ETA, reward="teacher_takeoff2_shuffled", **kw)]
    return out


def stage_d_eval_learners() -> list[SweepMarioLearner]:
    """Version 19 (RL_STUDY §11.1): the Stage D learners and twins plus, per readout group, the
    scripted take-off v2 reference (same group, so the same starts under stable streams);
    used with --eval-from for frozen competence."""
    out = stage_d_learners()
    for i, r in enumerate(block_seeds(3)):
        out.append(SweepMarioLearner(f"script_tko2_ro{i + 1}", r, 7900 + i, 0.0, reward="script_takeoff2",
                                     levels=TRAIN_LEVELS, max_delay=22, jitter=True, group=i))
    return out


def rl_pilot_learners() -> list[SweepMarioLearner]:
    """Version 14, the RL pilot (docs/RL_STUDY.md §6): per readout an RL learner on the
    game's reward and its shuffled-reward twin (same noise and draws), generalist
    conditions (four training levels, 0-22 NOOP start, palette jitter); noise 8100 + i."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"rl_ro{i + 1}", r, 8100 + i, ETA, reward="true", **kw),
                SweepMarioLearner(f"rlshuf_ro{i + 1}", r, 8100 + i, ETA, reward="shuffled", **kw)]
    return out


def srl_learners() -> list[SweepMarioLearner]:
    """Version 16, supervised + RL (docs/RL_STUDY.md §9): as version 14 (RL learner and
    shuffled-reward twin per readout, generalist conditions), fresh noise 8200 + i; both
    start from the readout's Stage D taught learner (`--init-from`)."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"srl_ro{i + 1}", r, 8200 + i, ETA, reward="true", **kw),
                SweepMarioLearner(f"srlshuf_ro{i + 1}", r, 8200 + i, ETA, reward="shuffled", **kw)]
    return out


def stage_b_confirm_learners() -> list[SweepMarioLearner]:
    """Version 11: Stage B confirmation (PERCEPTION_CREDIT_STUDY §13), fresh noise 7700 + i."""
    return [SweepMarioLearner(l.name, l.readout_seed, l.noise_seed + 100, l.eta, reward=l.reward, group=l.group,
                              levels=l.levels, max_delay=l.max_delay, jitter=l.jitter) for l in stage_b_learners()]


def stage_b_learners() -> list[SweepMarioLearner]:
    """Version 10, Stage B (PERCEPTION_CREDIT_STUDY §12): per readout, a learner taught
    by the full RAM label (enemy, wall or pit ahead) and its coin-taught twin (p 0.35),
    one group (shared noise and draws); eye on the game, generalist conditions."""
    out = []
    for i, r in enumerate(block_seeds(3)):
        kw = {"levels": TRAIN_LEVELS, "max_delay": 22, "jitter": True, "group": i}
        out += [SweepMarioLearner(f"tfull_ro{i + 1}", r, 7600 + i, ETA, reward="teacher_full", **kw),
                SweepMarioLearner(f"tfullshuf_ro{i + 1}", r, 7600 + i, ETA, reward="teacher_full_shuffled", **kw)]
    return out


def trios(n_readouts: int = 3, version: int = 1) -> list[SweepMarioLearner]:
    out = []
    for i, r in enumerate(block_seeds(n_readouts)):
        noise = (7100 if version == 1 else 7200) + i
        out += [SweepMarioLearner(f"learner_ro{i + 1}", r, noise, ETA, group=i),
                SweepMarioLearner(f"frozen_ro{i + 1}", r, noise, 0.0, group=i),
                SweepMarioLearner(f"shuffled_ro{i + 1}", r, noise, ETA, reward="shuffled", group=i)]
    return out


def resume_check(version: int = 1) -> None:
    """40 decisions straight vs 15 + checkpoint + a fresh trainer (new emulator
    processes) + 25: weights, trace, screens, positions, episodes identical."""
    import torch
    g = datasets.load(datasets.DEFAULT)
    W, motor = g.to_sparse(), datasets.motor_index(g, datasets.DEFAULT)
    L = trios(1, version)
    mk = lambda: MarioSweepTrainer(g, W, motor, L, settings_for(version), cache_key=("mario_sweep", 0))
    a = mk()
    for _ in range(40):
        a.decide()
    b = mk()
    for _ in range(15):
        b.decide()
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = CKPT / f"mario_sweep_resume_check_v{version}.pt"
    b.save(ck)
    b.close()
    c = mk()
    c.load(ck)
    for _ in range(25):
        c.decide()
    same = {"weights": bool(torch.equal(a.brain.m, c.brain.m)),
            "critic": bool(torch.equal(a.wv, c.wv)),
            "eligibility_trace": bool(torch.equal(a.E, c.E)),
            "screens": all(np.array_equal(x, y) for x, y in zip(a.obs, c.obs)),
            "positions": [i.get("mario_x") for i in a.info] == [i.get("mario_x") for i in c.info],
            "episodes": a.episodes == c.episodes}
    rel = a.relative_rms()
    rep = {"what": "Mario sweep trainer: bit-identical resume, emulators included",
           "identical": same, "all_identical": all(same.values()),
           "relative_rms_weight_change": dict(zip([l.name for l in L], rel.round(6).tolist())),
           "positions_after_40": [i.get("mario_x") for i in a.info],
           "mean_policy_entropy": float(np.mean(a.stats["entropy"])),
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    name = "mario_sweep_resume_check.json" if version == 1 else f"mario_sweep_resume_check_v{version}.json"
    (RUNS / name).write_text(json.dumps(rep, indent=2, default=float))
    ck.unlink(missing_ok=True)
    a.close()
    c.close()
    print(json.dumps(rep, indent=1, default=float))


def live_snapshot(tr: MarioSweepTrainer, dps: float, pi=None) -> dict:
    """Per-decision snapshot for the live channel (training and frozen play)."""
    L, vis = tr.last, tr.visions[0]
    f = L["focus"]
    return {"decision": np.int64(tr.decisions), "time": np.float64(time.time()), "dps": np.float64(dps),
            "frames": np.stack([o[::2, ::2] for o in tr.obs]).astype(np.uint8),
            "focus": np.int64(f), "focus_frame": tr.obs[f].astype(np.uint8),
            "pi": (L["pi"] if pi is None else pi).astype(np.float32),
            "scores": L["scores"].astype(np.float32),
            "actions": L["actions"].astype(np.int64), "reward": L["reward"].astype(np.float32),
            "entropy": L["entropy"].astype(np.float32),
            "x": np.array([i.get("mario_x", 0.0) for i in tr.info], np.float32),
            "episode": np.array([sum(1 for e in tr.episodes if e["b"] == b) for b in range(tr.B)], np.int64),
            "ep_len": np.array([e["decisions"] for e in tr.ep], np.int64),
            "ep_best": np.array([e["best_x"] for e in tr.ep], np.float32),
            "spikes": np.packbits(L["spikes"].astype(bool), axis=1),
            "motor_counts": L["motor_counts"].astype(np.uint8),
            "eye_on": L["current"][vis.on_idx].astype(np.float16),
            "eye_off": L["current"][vis.off_idx].astype(np.float16),
            "value": L["value"].astype(np.float32), "td": L["td"].astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, default=2, choices=[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19])
    ap.add_argument("--name", default=None, help="default: mario_sweep_pilot (v1), mario_sweep_v2")
    ap.add_argument("--max-decisions", type=int, default=300_000)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--screens-every", type=float, default=2.0)
    ap.add_argument("--resume-check", action="store_true")
    ap.add_argument("--graph", default="connectome", help="connectome | null:<seed> (depth-stratified, cached)")
    ap.add_argument("--init-from", default=None,
                    help="versions 16-17: Stage D checkpoint whose tko2_ro<i> weights start srl_ro<i> and srlshuf_ro<i>")
    ap.add_argument("--eval-from", default=None,
                    help="frozen evaluation (RL_STUDY §11 item 5): every learner of this version takes the "
                         "weights of the same-named learner in this checkpoint; no weight update, stable "
                         "random streams (same starts for a group in every arm); run name <name>_frozeneval")
    args = ap.parse_args()
    if args.resume_check:
        return resume_check(args.version)
    if args.graph != "connectome" and not args.name:
        args.name = {13: "mario_stageD", 14: "mario_rl3", 16: "mario_srl", 17: "mario_srl_frozen",
                     19: "mario_stageD_eval"}.get(args.version, f"mario_v{args.version}") + "_" + args.graph.replace(":", "")
    args.name = args.name or {1: "mario_sweep_pilot", 2: "mario_sweep_v2", 3: "mario_gen",
                              5: "mario_teacher", 6: "mario_teacher_shuffled", 7: "mario_ref2",
                              9: "mario_oracle", 10: "mario_stageB", 11: "mario_stageB2",
                              12: "mario_stageC", 13: "mario_stageD", 14: "mario_rl3",
                              16: "mario_srl", 17: "mario_srl_frozen",
                              18: "mario_stageD_frozen", 19: "mario_stageD_eval"}[args.version]
    settings = settings_for(args.version)
    if args.eval_from:
        from dataclasses import replace
        settings = replace(settings, frozen=True, stable_streams=True)
        args.name = args.name + "_frozeneval"
    learners = {3: generalisation_learners, 5: teacher_learners, 6: shuffled_teacher_learners,
                7: reference_trios, 9: oracle_study_learners,
                10: stage_b_learners, 11: stage_b_confirm_learners,
                12: stage_c_learners, 13: stage_d_learners, 14: rl_pilot_learners, 16: srl_learners,
                17: srl_learners, 18: stage_d_learners, 19: stage_d_eval_learners}.get(args.version, lambda: trios(3, args.version))()
    from flymario.connectome import graphsel
    g, W, motor = graphsel.load(args.graph)
    tr = MarioSweepTrainer(g, W, motor, learners, settings,
                           cache_key=("mario_sweep", 0) if args.graph == "connectome" else ("mario_sweep", args.graph))
    CKPT.mkdir(parents=True, exist_ok=True)
    path, out = CKPT / f"{args.name}.pt", RUNS / f"{args.name}.json"
    if path.exists():
        tr.load(path)
        print(f"resumed at decision {tr.decisions}, {len(tr.episodes)} episodes", flush=True)
    elif args.eval_from:
        import torch
        src = torch.load(args.eval_from, map_location="cpu", weights_only=False)
        names = [x["name"] for x in src["learners"]]
        m = tr.brain.m.clone()
        for b, lr in enumerate(learners):
            if lr.reward != "script_takeoff2":              # scripted slots: weights never used
                m[b] = src["brain"]["m"][names.index(lr.name)].to(m.device)
        tr.brain.set_weights(m)
        print(f"frozen evaluation of {args.eval_from} (weights by learner name)", flush=True)
    elif args.init_from:
        import torch
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if src["settings"]["sample_seed"] != 20261015 or "graph" in src and src["graph"] != args.graph:
            raise ValueError("--init-from must be a Stage D checkpoint")
        names = [x["name"] for x in src["learners"]]
        m = tr.brain.m.clone()
        for b, lr in enumerate(learners):
            m[b] = src["brain"]["m"][names.index("tko2_ro" + lr.name[-1])].to(m.device)
        tr.brain.set_weights(m)
        print(f"initialised from {args.init_from} (tko2_ro<i> weights)", flush=True)
    manifest = RunManifest(args.name, "Mario pilot", "sweep regime, eprop_readout",
                           {"settings": asdict(settings), "learners": [asdict(l) for l in learners],
                            "max_decisions": args.max_decisions, "graph": args.graph},
                           dataset="malecns", output_files={"report": out},
                           input_files={**dataset_files(), "script": Path(__file__),
                                        "trainer": ROOT / "src/flymario/training/mario_sweep.py",
                                        "plan": ROOT / "docs/MARIO_PILOT_PLAN.md"})

    def flush():
        tr.save(path)
        last = {l.name: [e["best_x"] for e in tr.episodes if e["learner"] == l.name][-20:]
                for l in learners}
        rep = {"what": "Mario training PILOT, sweep regime (protocol §20) -- not confirmatory",
               "settings": asdict(settings), "learners": [asdict(l) for l in learners],
               "decisions": tr.decisions, "episodes": tr.episodes,
               "rolling20_best_x": {k: (float(np.mean(v)) if v else None) for k, v in last.items()},
               "relative_rms_weight_change": dict(zip([l.name for l in learners],
                                                      tr.relative_rms().round(6).tolist())),
               "mean_entropy_last1000": np.mean(tr.stats["entropy"][-1000:], axis=0).round(4).tolist()
               if tr.stats["entropy"] else None,
               "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, default=float))
        tmp.replace(out)

    step = args.checkpoint_every
    ctl = RunControl(args.name, total=args.max_decisions // step, checkpoint=flush,
                     unit="1000 decisions", already_done=tr.decisions // step,
                     fresh=tr.decisions == 0, meta={
        "question": "Does the Gate G learner learn to play? (PILOT, sweep regime)",
        "method": "3 trios (learner | frozen twin | shuffled-reward twin), one per fixed readout. "
                  "Each decision: one fresh 16-tick look at the frame, motor spike counts, fixed "
                  "readout, sampled action. Reward: progress, time, death -25, flag +100."})
    names = [l.name for l in learners]
    # the live channel (flymario.telemetry.live)
    vis = tr.visions[0]
    live.write_meta(args.name, {
        "names": names, "readouts": [l.readout_seed for l in learners],
        "groups": [l.group for l in learners], "eta": [l.eta for l in learners],
        "rewards": [l.reward for l in learners], "ticks": tr.K, "n": int(tr.brain.n),
        "motor": [int(i) for i in tr.brain.motor_np], "on_idx": [int(i) for i in vis.on_idx],
        "off_idx": [int(i) for i in vis.off_idx], "run_dir": str(ctl.dir),
        "ckpt": str(path), "report": str(out), "max_decisions": args.max_decisions,
        "settings": asdict(settings), "version": args.version})
    tr.focus = live.get_focus(args.name, 0)
    rate = {"t": time.time(), "d": tr.decisions, "dps": 0.0}
    ent_acc = []

    def publish_live():
        live.write(args.name, **live_snapshot(tr, rate["dps"]))
    ctl.chart("x", "How far each episode got (Mario's x, pixels)", "episode", "furthest x", names,
              note="One dot per finished episode; a learner should rise above both its twins.")
    ctl.chart("mean", "Rolling mean of the last 20 episodes", "decisions (thousands)",
              "furthest x", names, note="Learner vs frozen vs shuffled, per readout.")
    last_screens = 0.0
    frames = {b: [] for b in range(len(learners))}
    last_replay = {b: 0.0 for b in range(len(learners))}
    try:
        while tr.decisions < args.max_decisions:
            ctl.begin(f"decisions {tr.decisions}-{tr.decisions + step}")
            t0 = time.time()
            target = (tr.decisions // step + 1) * step
            while tr.decisions < target:
                tr.focus = live.get_focus(args.name, tr.focus or 0)
                ended = tr.decide()
                if time.time() - rate["t"] >= 2.0:
                    rate["dps"] = (tr.decisions - rate["d"]) / (time.time() - rate["t"])
                    rate["t"], rate["d"] = time.time(), tr.decisions
                publish_live()
                ent_acc.append(tr.last["entropy"])
                if tr.decisions % 50 == 0:
                    e50 = np.mean(ent_acc, axis=0)
                    ent_acc.clear()
                    wc = tr.relative_rms()
                    for b, nm in enumerate(names):
                        ctl.point("entropy", nm, tr.decisions, float(e50[b]))
                        ctl.point("weights", nm, tr.decisions, float(wc[b]))
                # a stop or checkpoint request is honoured at the next decision:
                # between decisions the trainer is in a consistent state
                ctl.poll()
                done_b = {e["b"] for e in ended}
                for e in ended:
                    b = e["b"]
                    if frames[b] and time.time() - last_replay[b] >= 20:
                        publish(ctl, "Latest finished episode, at game speed", f"replay_{b}",
                                png(frames[b], duration_ms=67),
                                f"{e['learner']} -- episode {e['episode']}, {e['end']}, "
                                f"x {e['best_x']:.0f}, {e['decisions']} decisions")
                        last_replay[b] = time.time()
                    frames[b] = []
                    ctl.point("x", e["learner"], e["episode"], e["best_x"], end=e["end"])
                    mine = [x["best_x"] for x in tr.episodes if x["learner"] == e["learner"]][-20:]
                    ctl.point("mean", e["learner"], tr.decisions / 1000, float(np.mean(mine)))
                for b in range(len(learners)):
                    if b not in done_b:
                        frames[b].append(tr.obs[b][::2, ::2].copy())
                if time.time() - last_screens >= args.screens_every:
                    labels = [f"{l.name}  x {tr.info[b].get('mario_x', 0):.0f}"
                              for b, l in enumerate(learners)]
                    publish(ctl, "Live: every screen", "live", png([grid(tr.obs, labels, cols=3)]),
                            f"decision {tr.decisions} -- columns: learner | frozen | shuffled")
                    last_screens = time.time()
                if tr.decisions % 50 == 0:
                    ctl.sub(tr.decisions - (target - step), step, "decisions")
            flush()
            ctl.done({"episodes": len(tr.episodes)}, seconds=time.time() - t0, kind="decisions")
    except StopRequested:
        flush()
        ctl.close("stopped")
        manifest.finish("stopped")
        tr.close()
        print("stopped cleanly; relaunch to resume", flush=True)
        return
    flush()
    ctl.close("complete")
    manifest.finish("complete")
    tr.close()


if __name__ == "__main__":
    main()
