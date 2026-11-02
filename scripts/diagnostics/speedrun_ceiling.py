#!/usr/bin/env python
"""RL_STUDY §28 (a): ceiling of the label / interface paradigm on the development levels of the any% route
(1-1, 4-1, 8-1), scripted policies only (RAM label, no brain), from each level's start state until flag,
death or 2,400 decisions. Writes artifacts/runs/mission/speedrun_ceiling.json.

    .venv/bin/python scripts/diagnostics/speedrun_ceiling.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

LEVELS = ("Level1-1", "Level4-1", "Level8-1")
POLICIES = ("always_run", "label", "label_hold2", "label_hold4", "label_hold6")
MAX_DEC, STALL = 2400, 300
FPS = 60.0988


def run(args):
    level, pol = args
    from play_harness_v2 import JUMP, RUN, Hold6
    from flymario.env.retro_env import EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    if pol == "always_run":
        policy = lambda ram: RUN  # noqa: E731
    elif pol == "label":
        policy = lambda ram: JUMP if takeoff_label_v2(ram) else RUN  # noqa: E731
    else:
        policy = Hold6(lambda ram: bool(takeoff_label_v2(ram)), int(pol.split("hold")[1]))
    env = MarioEnv(EnvConfig(state=level))
    env.reset()
    ram = env.env.unwrapped.get_ram()
    best, since, n, end, jumps = 0.0, 0, 0, None, 0
    while end is None:
        a = policy(ram)
        jumps += a == JUMP
        _o, _r, term, trunc, info = env.step(a)
        ram = env.env.unwrapped.get_ram()
        n += 1
        if info["mario_x"] > best:
            best, since = info["mario_x"], 0
        else:
            since += 1
        end = ("flag" if info["cleared"] else "death" if info["died"] else "stall" if since >= STALL
               else "time" if (term or trunc or n >= MAX_DEC) else None)
    env.close()
    return {"level": level, "policy": pol, "end": end, "furthest_x": float(best), "decisions": n,
            "game_seconds": round(n * 4 / FPS, 1), "jump_rate": round(jumps / max(1, n), 3)}


def main():
    jobs = [(lv, p) for lv in LEVELS for p in POLICIES]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(run, jobs))
    for r in res:
        print(r, flush=True)
    (ROOT / "artifacts/runs/mission/speedrun_ceiling.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
