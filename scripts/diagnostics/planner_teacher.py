#!/usr/bin/env python
"""RL_STUDY §29 (exploratory, speedrun stage S0): a savestate lookahead planner (random-shooting MPC on the
emulator) on the route's development levels. The planner reads RAM and restores emulator states; it is a
candidate *teacher*, never a policy input. Writes artifacts/runs/mission/planner_teacher.json.

    .venv/bin/python scripts/diagnostics/planner_teacher.py [--k 24] [--h 24] [--exec 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LEVELS = ("Level1-1", "Level4-1", "Level8-1")
FPS = 60.0988
MAX_DEC = 3000


def plan_level(args):
    level, li, K, H, E, full = args
    from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
    RUN, JUMP, NOOP, LEFT = (ACTION_NAMES.index(a) for a in ("RIGHT+B", "RIGHT+A+B", "NOOP", "LEFT"))
    v2 = H >= 40 and not full                  # §29.1 variant: no NOOP / LEFT (time is the enemy); --full: v4
    choices, probs = ((np.array([RUN, JUMP]), np.array([0.6, 0.4])) if v2 else
                      (np.array([RUN, JUMP, NOOP, LEFT]), np.array([0.55, 0.35, 0.05, 0.05])))
    rng = np.random.default_rng(20261800 + li)
    env = MarioEnv(EnvConfig(state=level, max_frames=10 ** 9))
    env.reset()
    em = env.env.unwrapped.em

    def sample_seq():
        seq = []
        while len(seq) < H:
            a = int(rng.choice(choices, p=probs))
            seq += [a] * int(rng.integers(1, 9))
        return seq[:H]

    def rollout(state, seq):
        em.set_state(state)
        best = -1.0
        for t, a in enumerate(seq):
            _o, _r, _te, _tr, info = env.step(a)
            best = max(best, float(info["mario_x"]))
            if info["cleared"]:
                return 1e6 - t, True
            if info["died"]:
                return best - 1e5, False
        return best, False

    actions, xs, n, end = [], [], 0, None
    while end is None:
        root = em.get_state()
        cands = [sample_seq() for _ in range(K)]
        if actions:                            # keep the tail of the previous plan as one candidate
            cands[0] = (prev_tail + sample_seq())[:H]
        scores = [rollout(root, s)[0] for s in cands]
        best = cands[int(np.argmax(scores))]
        prev_tail = best[E:]
        em.set_state(root)
        for a in best[:E]:
            _o, _r, _te, _tr, info = env.step(a)
            actions.append(int(a))
            xs.append(float(info["mario_x"]))
            n += 1
            if info["cleared"]:
                end = "flag"
                break
            if info["died"]:
                end = "death"
                break
            if n >= MAX_DEC:
                end = "time"
                break
    env.close()
    return {"level": level, "end": end, "furthest_x": max(xs), "decisions": n,
            "game_seconds": round(n * 4 / FPS, 1), "jump_share": round(float(np.mean(np.array(actions) == JUMP)), 3),
            "actions": actions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--h", type=int, default=24)
    ap.add_argument("--exec", type=int, default=4)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--full", action="store_true", help="v4: keep NOOP / LEFT (run-ups) with a long horizon")
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(plan_level, [(lv, LEVELS.index(lv), a.k, a.h, a.exec, a.full) for lv in a.levels]))
    for r in res:
        print({k: v for k, v in r.items() if k != "actions"}, flush=True)
    out = ROOT / f"artifacts/runs/mission/planner_teacher-k{a.k}-h{a.h}{'-full' if a.full else ''}{'-' + '-'.join(a.levels) if a.levels != list(LEVELS) else ''}.json"
    out.write_text(json.dumps({"what": "RL_STUDY §29 savestate lookahead planner", "args": vars(a), "results": res}, indent=1))


if __name__ == "__main__":
    main()
