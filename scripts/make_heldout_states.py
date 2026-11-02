#!/usr/bin/env python
"""Held-out level starts for the confirmation (docs/RL_STUDY.md §7), for levels never opened by any agent.

From the 1-1 savestate: write the target world/level/area in RAM (0x075F, 0x075C, 0x0760), set the timer
to 001 so Mario dies of time-up, let the game reload the target level, and save the first controllable
frame. Water levels (2-2, 7-2) and castles (x-4) are left out. Verification reads RAM only.

    .venv/bin/python scripts/make_heldout_states.py
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flymario.env.retro_env import INTEGRATION_DIR, EnvConfig, MarioEnv
from flymario.paths import RUNS

#: (world, level) -> (area number in the world's area list, expected area type)
TARGETS = {(1, 2): (2, 2), (1, 3): (3, 1), (2, 3): (3, 1), (3, 2): (1, 1), (3, 3): (2, 1), (4, 2): (2, 2),
           (4, 3): (3, 1), (5, 2): (1, 1), (5, 3): (2, 1), (6, 2): (1, 1), (6, 3): (2, 1), (7, 3): (3, 1),
           (8, 2): (1, 1), (8, 3): (2, 1)}
GAME = INTEGRATION_DIR / "SuperMarioBros-Nes"


def ram_view(r):
    return {"world": int(r[0x75F]) + 1, "level": int(r[0x75C]) + 1, "area": int(r[0x760]), "area_type": int(r[0x74E]),
            "player_state": int(r[0x0E]), "x": int(r[0x6D]) * 256 + int(r[0x86]),
            "timer": int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10 + int(r[0x7FA])}


def make(world, level, area):
    env = MarioEnv(EnvConfig(state="Level1-1"))
    env.reset()
    u = env.env.unwrapped
    m = u.data.memory
    for addr, val in ((0x75F, world - 1), (0x75C, level - 1), (0x760, area), (0x75A, 5),
                      (0x7F8, 0), (0x7F9, 0), (0x7FA, 1)):
        m.assign(addr, "|u1", val)
    died = False
    for _ in range(1500):
        env.env.step(env._table[0])
        v = ram_view(u.get_ram())
        died = died or v["player_state"] in (6, 11)
        if died and v["player_state"] == 8 and v["timer"] >= 100:
            state = env.env.em.get_state()
            env.close()
            return state
    env.close()
    return None


def main():
    out = {}
    for (w, lv), (area, kind) in TARGETS.items():
        name = f"Held{w}-{lv}"
        st = make(w, lv, area)
        if st is None:
            out[name] = {"ok": False, "reason": "no reload within 1,500 frames"}
            print(name, out[name], flush=True)
            continue
        (GAME / f"{name}.state").write_bytes(gzip.compress(st))
        env = MarioEnv(EnvConfig(state=name))
        env.reset()
        v = ram_view(env.env.unwrapped.get_ram())
        env.close()
        ok = (v["world"], v["level"], v["area"], v["area_type"], v["player_state"]) == (w, lv, area, kind, 8)
        out[name] = {"ok": bool(ok), "ram_after_reload": v, "expected": {"area": area, "area_type": kind}}
        print(name, out[name], flush=True)
    rep = {"what": "held-out level starts (RL_STUDY §7): RAM-only verification, no agent run", "states": out,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / "heldout_states-20261007.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
