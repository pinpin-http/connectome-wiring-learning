#!/usr/bin/env python
"""Watch the trained flies play, weights frozen -- no learning, nothing saved.

Loads a stopped run's last checkpoint and plays every learner at once on the live channel (flymario.telemetry.live).
--level: own (each learner's training levels) or one savestate for everyone (Level1-1 ... Level8-1).
--policy: sampled (draw from the probabilities) or greedy (most likely button). Stop with Ctrl-C or SIGTERM.

    .venv/bin/python scripts/play_mario.py --run mario_gen                     # own levels, sampled
    .venv/bin/python scripts/play_mario.py --run mario_gen --level Level5-1 --policy greedy
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_mario_sweep import CKPT, live_snapshot

from flymario import runctl
from flymario.connectome import datasets
from flymario.telemetry import live
from flymario.training.mario_sweep import MarioSweepTrainer, SweepMarioLearner, SweepMarioSettings

PLAY_FLAG = "PLAYING"                 # in the run's control dir while a frozen play runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="mario_gen")
    ap.add_argument("--level", default="own")
    ap.add_argument("--policy", default="sampled", choices=["sampled", "greedy"])
    ap.add_argument("--max-decisions", type=int, default=200_000)
    a = ap.parse_args()
    import torch
    st = torch.load(CKPT / f"{a.run}.pt", map_location="cpu", weights_only=False)
    learners = [SweepMarioLearner(**d) for d in st["learners"]]
    learners = [replace(l, eta=0.0, levels=tuple(l.levels)) for l in learners]
    if a.level != "own":
        learners = [replace(l, levels=(a.level,), max_delay=0, jitter=False) for l in learners]
    settings = SweepMarioSettings(**st["settings"])
    settings = replace(settings, epsilon=0.0, critic_lr=0.0, tau_adapt=0.0,
                       sample_seed=int(time.time()) % 100_000,
                       tau=1e-3 if a.policy == "greedy" else settings.tau)
    g = datasets.load(datasets.DEFAULT)
    tr = MarioSweepTrainer(g, g.to_sparse(), datasets.motor_index(g, datasets.DEFAULT), learners,
                           settings, cache_key=("mario_sweep", 0))
    tr.brain.set_weights(st["brain"]["m"].to(tr.dev))
    for k in ("wv", "bv"):                                   # the critic's expectation, display only
        getattr(tr, k).copy_(st["critic"][k].to(tr.dev))
    trained_at = int(st["decisions"])
    del st
    flag = runctl.LIVE / a.run / PLAY_FLAG
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(f"{os.getpid()} {a.level} {a.policy} {trained_at}\n")
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    meta = live.read_meta(a.run) or {}
    vis = tr.visions[0]
    meta.update({"names": [l.name for l in learners], "readouts": [l.readout_seed for l in learners],
                 "groups": [l.group for l in learners], "ticks": tr.K, "n": int(tr.brain.n),
                 "motor": [int(i) for i in tr.brain.motor_np], "on_idx": [int(i) for i in vis.on_idx],
                 "off_idx": [int(i) for i in vis.off_idx],
                 "play": {"level": a.level, "policy": a.policy, "trained_decisions": trained_at}})
    live.write_meta(a.run, meta)
    print(f"playing {a.run} frozen at decision {trained_at}: level {a.level}, policy {a.policy}", flush=True)
    rate = {"t": time.time(), "d": 0, "dps": 0.0}
    try:
        while not stop["now"] and tr.decisions < a.max_decisions:
            tr.focus = live.get_focus(a.run, tr.focus or 0)
            for e in tr.decide():
                print(f"{e['learner']}: {e.get('level', '')} x {e['best_x']:.0f}, {e['end']}", flush=True)
            if time.time() - rate["t"] >= 2.0:
                rate["dps"] = (tr.decisions - rate["d"]) / (time.time() - rate["t"])
                rate["t"], rate["d"] = time.time(), tr.decisions
            pi = None
            if a.policy == "greedy":                         # show the probabilities, not the one-hot
                z = tr.last["scores"] / 2.0
                pi = np.exp(z - z.max(1, keepdims=True))
                pi /= pi.sum(1, keepdims=True)
            live.write(a.run, **live_snapshot(tr, rate["dps"], pi))
    except KeyboardInterrupt:
        pass
    finally:
        flag.unlink(missing_ok=True)
        try:
            tr.close()
        except OSError:                                      # emulator workers already gone
            pass
        print("play stopped", flush=True)


if __name__ == "__main__":
    main()
