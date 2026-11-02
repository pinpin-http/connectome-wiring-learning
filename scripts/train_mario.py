#!/usr/bin/env python
"""Train the fly brain in the game with the Gate G rule (pilot; settings in docs/MARIO_PILOT_PLAN.md).

    python scripts/train_mario.py                  # start or resume
    make follow / make stop / make checkpoint      # follow, stop, save now

Each learner has its own emulator and a no-learning twin (same readout, same noise). Pilot
results are labelled pilot and never pooled with confirmatory ones.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from flymario.connectome import datasets
from flymario.paths import RUNS
from flymario.policy.readout import block_seeds
from flymario.runctl import RunControl, StopRequested
from flymario.training.mario import MarioLearner, MarioSettings, MarioTrainer

CKPT = RUNS / "checkpoints"


def png(frames, scale=1, duration_ms=None) -> bytes:
    """One frame -> PNG; several -> animated PNG (plays in every browser)."""
    from PIL import Image
    ims = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    if scale != 1:
        ims = [im.resize((im.width * scale // 2, im.height * scale // 2), Image.NEAREST) for im in ims]
    buf = io.BytesIO()
    if len(ims) == 1:
        ims[0].save(buf, format="PNG")
    else:
        ims[0].save(buf, format="PNG", save_all=True, append_images=ims[1:],
                    duration=duration_ms, loop=0, optimize=False)
    return buf.getvalue()


def grid(frames, labels, cols=3) -> np.ndarray:
    """The learners' screens side by side, each with its name burnt in."""
    from PIL import Image, ImageDraw
    h, w = frames[0].shape[:2]
    rows = -(-len(frames) // cols)
    canvas = Image.new("RGB", (cols * w, rows * (h + 16)), (20, 20, 24))
    d = ImageDraw.Draw(canvas)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + 16)
        canvas.paste(Image.fromarray(np.asarray(f, dtype=np.uint8)), (x, y + 16))
        d.text((x + 4, y + 2), lab, fill=(235, 235, 235))
    return np.asarray(canvas)


def publish(ctl, group, name, data, caption) -> None:
    """Replace a group's picture with a new file name, so the page reloads it."""
    old = [i for i in ctl.images if i.get("group") == group and i["file"].startswith(name)]
    for i in old:
        (ctl.dir / "img" / i["file"]).unlink(missing_ok=True)
    ctl.images = [i for i in ctl.images if i not in old]
    ctl.image(f"{name}_{int(time.time() * 10) % 10**9}.png", data, caption=caption, group=group)


def resume_check() -> None:
    """40 decisions straight vs 15 + checkpoint + a fresh trainer (new emulator
    processes) + 25. Weights, eligibility, screens, episodes: must be identical."""
    import torch
    ds = datasets.DEFAULT
    g = datasets.load(ds)
    W, motor = g.to_sparse(), datasets.motor_index(g, ds)
    r = block_seeds(1)[0]
    L = [MarioLearner("learner", r, 7000, "node_perturbation", 0.02),
         MarioLearner("frozen", r, 7000, None, 0.0)]
    mk = lambda: MarioTrainer(g, W, motor, L, MarioSettings(), cache_key=("mario", 0))  # noqa: E731
    a = mk()
    for _ in range(40):
        a.decide()
    b = mk()
    for _ in range(15):
        b.decide()
    CKPT.mkdir(parents=True, exist_ok=True)
    b.save(CKPT / "mario_resume_check.pt")
    b.close()
    c = mk()
    c.load(CKPT / "mario_resume_check.pt")
    for _ in range(25):
        c.decide()
    same = {"weights": bool(torch.equal(a.brain.m, c.brain.m)),
            "eligibility_trace": bool(torch.equal(a.E, c.E)),
            "screens": all(np.array_equal(x, y) for x, y in zip(a.obs, c.obs)),
            "positions": [i.get("mario_x") for i in a.info] == [i.get("mario_x") for i in c.info],
            "episodes": a.episodes == c.episodes}
    rep = {"what": "Mario trainer: bit-identical resume, emulators included", "identical": same,
           "all_identical": all(same.values()),
           "learner_weight_change_L1": float((a.brain.m[0] - a.brain.m0).abs().sum()),
           "twin_weight_change_L1": float((a.brain.m[1] - a.brain.m0).abs().sum()),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (RUNS / "mario_resume_check.json").write_text(json.dumps(rep, indent=2))
    (CKPT / "mario_resume_check.pt").unlink(missing_ok=True)
    a.close()
    c.close()
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="mario_pilot")
    ap.add_argument("--max-decisions", type=int, default=300_000,
                    help="per learner; the run can be stopped at any point before")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--screens-every", type=float, default=2.0, help="seconds")
    ap.add_argument("--resume-check", action="store_true")
    args = ap.parse_args()
    if args.resume_check:
        return resume_check()

    gate = json.loads((RUNS / "gate_g_evaluate.json").read_text())
    if gate.get("chosen") is None:
        raise SystemExit("Gate G has not passed: no rule to train with (docs/GATE_G_PLAN.md §8)")
    # docs/MARIO_PILOT_PLAN.md §4: the Gate G rule at its Gate G rate scaled by sqrt(1 − γ),
    # since the trace sums ~1/(1 − γ) decisions of eligibility whose noise adds in quadrature.
    settings = MarioSettings()
    rule = gate["chosen"]
    eta = float(gate["etas"][rule]) * float(np.sqrt(1.0 - settings.gamma))
    plan = {"rule": rule, "eta_gate_g": float(gate["etas"][rule]), "eta": eta,
            "readouts": 3, "settings": asdict(settings), "doc": "docs/MARIO_PILOT_PLAN.md"}
    readouts = block_seeds(plan["readouts"])
    learners = []
    for i, r in enumerate(readouts):
        noise = 7000 + i
        learners.append(MarioLearner(f"learner_ro{i + 1}", r, noise, rule, eta))
        learners.append(MarioLearner(f"frozen_ro{i + 1}", r, noise, None, 0.0))
    ds = datasets.DEFAULT
    g = datasets.load(ds)
    tr = MarioTrainer(g, g.to_sparse(), datasets.motor_index(g, ds), learners, settings,
                      cache_key=("mario", 0))
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / f"{args.name}.pt"
    out = RUNS / f"{args.name}.json"
    if path.exists():
        tr.load(path)
        print(f"resumed at decision {tr.decisions}, {len(tr.episodes)} episodes", flush=True)

    def flush():
        tr.save(path)
        rep = {"what": "Mario training PILOT (protocol §20) -- not confirmatory",
               "rule": rule, "eta": eta, "settings": asdict(settings), "plan": plan,
               "learners": [asdict(l) for l in learners], "decisions": tr.decisions,
               "episodes": tr.episodes, "gate_g": {"chosen": gate["chosen"], "etas": gate["etas"]},
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, default=float))
        tmp.replace(out)

    step = args.checkpoint_every
    total = args.max_decisions // step
    ctl = RunControl(args.name, total=total, checkpoint=flush, unit="1000 decisions",
                     already_done=tr.decisions // step, fresh=tr.decisions == 0, meta={
        "question": "Can the fly brain learn to play, with the rule that passed Gate G? (PILOT)",
        "why": ("Gate G showed the rule learns a two-cue task. Mario is the real benchmark: "
                "vision, timing, delayed consequences. The pilot finds out whether learning "
                "happens at all here, how fast, and how much it varies -- before anything is frozen."),
        "method": (f"{len(readouts)} learners ({rule}), each with a frozen twin (same readout, "
                   "same noise, no learning). The only difference inside a pair is learning. "
                   "Reward: forward progress, time penalty, death −25, flag +100.")})
    names = [l.name for l in learners]
    ctl.chart("x", "How far each episode got (Mario's x, pixels)", "episode", "furthest x",
              names, note="One dot per finished episode. Learners vs their frozen twins: "
                          "same readout, same noise, only learning differs.")
    ctl.chart("mean", "Rolling mean of the last 20 episodes", "decisions (thousands)",
              "furthest x", names, note="If learning works, each learner rises above its twin.")
    for i in range(len(readouts)):
        ctl.task(f"ro{i + 1}", "running")
    last_screens = 0.0
    frames = {b: [] for b in range(len(learners))}         # this episode, half size
    last_replay = {b: 0.0 for b in range(len(learners))}
    try:
        while tr.decisions < args.max_decisions:
            ctl.begin(f"decisions {tr.decisions}-{tr.decisions + step}")
            t0 = time.time()
            target = (tr.decisions // step + 1) * step
            while tr.decisions < target:
                ended = tr.decide()
                done_b = {e["b"] for e in ended}
                for e in ended:
                    b = e["b"]
                    # a replay at game speed (15 fps), at most one per learner every 20 s
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
                    labels = [f"{l.name}  ep {tr.ep[b]['decisions']} dec  x {tr.info[b].get('mario_x', 0):.0f}"
                              for b, l in enumerate(learners)]
                    publish(ctl, "Live: every learner's screen", "live",
                            png([grid(tr.obs, labels, cols=2)]),
                            f"decision {tr.decisions} -- left column learns, right column is its frozen twin")
                    last_screens = time.time()
                if tr.decisions % 50 == 0:
                    ctl.sub(tr.decisions - (target - step), step, "decisions")
            flush()
            ends = tr.episodes[-50:]
            ctl.done({"episodes": len(tr.episodes),
                      "best_x_last50": round(float(np.mean([e["best_x"] for e in ends])), 1) if ends else None},
                     seconds=time.time() - t0, kind="decisions")
    except StopRequested:
        flush()
        ctl.close("stopped")
        tr.close()
        print("stopped cleanly; relaunch to resume", flush=True)
        return
    flush()
    ctl.close("complete")
    tr.close()


if __name__ == "__main__":
    main()
