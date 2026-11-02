#!/usr/bin/env python
"""Play harness v2 (docs/RL_STUDY.md §13): starts that differ.

Per level, 16 starts = the savestate plus a random 5-40 decision prefix (RIGHT+B or NOOP, no
jump), redrawn if it ends the episode. Suites: training levels and unseen development levels.

    starts    build and store the starts (furthest x and RAM hash of each start state)
    validate [--rule hold6] [--out PATH]  scripted references only: the take-off v2 expert must
              pass 3/3; a second random jumper and always-run must fail 3/3

    .venv/bin/python scripts/gates/play_harness_v2.py starts
    .venv/bin/python scripts/gates/play_harness_v2.py validate
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from oracle_criterion_check import criterion

from flymario.env.retro_env import ACTION_NAMES
from flymario.paths import RUNS

SUITES = {"training": ("Level1-1", "Level2-1", "Level4-1", "Level7-1"),
          "unseen": ("Level5-1", "Level8-1", "Level3-1", "Level6-1")}
LEVELS = SUITES["training"] + SUITES["unseen"]
N_STARTS, K = 16, 5
NOOP, RUN, JUMP = (ACTION_NAMES.index(a) for a in ("NOOP", "RIGHT+B", "RIGHT+A+B"))
MAX_DEC, STALL = 600, 150
OUT = RUNS / "harness_v2"
STARTS = OUT / "starts-20261009.json"


def _env(level):
    from flymario.env.retro_env import EnvConfig, MarioEnv
    return MarioEnv(EnvConfig(state=level))


def make_start(level: str, i: int) -> dict:
    """One start: a prefix redrawn until it neither kills Mario nor ends the episode."""
    rng = np.random.default_rng(20261100 + 1000 * LEVELS.index(level) + i)
    for _attempt in range(20):
        n = int(rng.integers(5, 41))
        prefix = [RUN if rng.random() < 0.6 else NOOP for _ in range(n)]
        env = _env(level)
        env.reset()
        ok, info = True, {}
        for a in prefix:
            _o, _r, term, trunc, info = env.step(a)
            if term or trunc or info.get("died"):
                ok = False
                break
        ram = env.env.unwrapped.get_ram()
        env.close()
        if ok:
            return {"level": level, "start": i, "prefix": prefix, "x": float(info.get("mario_x", 0.0)),
                    "ram_md5": hashlib.md5(bytes(ram)).hexdigest()}
    raise RuntimeError(f"no valid prefix for {level} start {i}")


def play(level: str, prefix: list, policy) -> dict:
    """The harness episode: reset, prefix, then the policy until flag, death, stall
    (150 decisions without a new furthest x) or 600 decisions; same as EnvPool.reset_to."""
    env = _env(level)
    env.reset()
    for a in prefix:
        env.step(a)
    ram = env.env.unwrapped.get_ram()
    best, since, jumps, n, end = 0.0, 0, 0, 0, None
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
    return {"best_x": float(best), "end": end, "decisions": n, "jump_rate": jumps / max(1, n)}


class Hold6:
    """§13.2: a trigger presses RIGHT+A+B for 6 decisions (the trigger decision and 5 more),
    then RIGHT+B for one; `trigger(ram)` is asked only on eligible decisions (not holding,
    not releasing). Counts triggers and eligible decisions."""

    def __init__(self, trigger, k: int = 6):         # k: hold length (§20), 6 = §13.2
        self.trigger, self.hold, self.release, self.k = trigger, 0, False, int(k)
        self.triggers, self.eligible = 0, 0

    def __call__(self, ram):
        if self.hold > 0:
            self.hold -= 1
            self.release = self.hold == 0
            return JUMP
        if self.release:
            self.release = False
            return RUN
        self.eligible += 1
        if self.trigger(ram):
            self.triggers += 1
            self.hold = self.k - 1
            return JUMP
        return RUN


def _hold6_play(level, prefix, trigger, k=6):
    pol = Hold6(trigger, k)
    out = play(level, prefix, pol)
    out["trigger_rate"] = pol.triggers / max(1, pol.eligible)
    return out


def _expert_hold6(args):
    level, prefix, k = (*args, 6)[:3]
    from flymario.env.schematic import takeoff_label_v2
    return _hold6_play(level, prefix, lambda ram: bool(takeoff_label_v2(ram)), k)


def _random_hold6(args):
    level, prefix, q, seed, k = (*args, 6)[:5]
    g = np.random.default_rng(seed)
    return _hold6_play(level, prefix, lambda ram: g.random() < q, k)["best_x"]


def _expert(args):
    level, prefix = args
    from flymario.env.schematic import takeoff_label_v2
    return play(level, prefix, lambda ram: JUMP if takeoff_label_v2(ram) else RUN)


def _always_run(args):
    level, prefix = args
    return play(level, prefix, lambda ram: RUN)


def _random(args):
    level, prefix, q, seed = args
    g = np.random.default_rng(seed)
    return play(level, prefix, lambda ram: JUMP if g.random() < q else RUN)["best_x"]


def starts_cmd():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        st = list(pool.map(make_start, [lv for lv in LEVELS for _ in range(N_STARTS)],
                           [i for _ in LEVELS for i in range(N_STARTS)]))
    distinct = {lv: len({s["ram_md5"] for s in st if s["level"] == lv}) for lv in LEVELS}
    STARTS.write_text(json.dumps({"what": "play harness v2 starts (RL_STUDY §13)", "starts": st,
                                  "distinct_start_states": distinct,
                                  "utc": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1))
    print("distinct start states per level:", distinct)


def validate_cmd(rule=None, out_path=None):
    """rule "hold6" (§13.2): the expert and the random jumpers follow the hold6 rule; the
    random jumpers trigger at the expert's trigger rate per eligible decision."""
    st = json.loads(STARTS.read_text())["starts"]
    k = int(rule[4:]) if rule else None             # "hold6", "hold2", ... (§13.2, §20)
    with ProcessPoolExecutor(max_workers=8) as pool:
        ex = list(pool.map(_expert_hold6 if rule else _expert,
                           [(s["level"], s["prefix"], k) if rule else (s["level"], s["prefix"]) for s in st]))
        ar = list(pool.map(_always_run, [(s["level"], s["prefix"]) for s in st]))
        jobs = []
        for si, (s, e) in enumerate(zip(st, ex)):
            for d in range(3):
                for role in (0, 1):                     # 0: the reference jumpers, 1: the second random jumper
                    for j in range(K):
                        jobs.append((s["level"], s["prefix"], e["trigger_rate"] if rule else e["jump_rate"],
                                     20261200 + 100000 * d + 10000 * role + 50 * si + j) + ((k,) if rule else ()))
        xs = list(pool.map(_random_hold6 if rule else _random, jobs, chunksize=8))
    it = iter(xs)
    ref = {d: [] for d in range(3)}
    rb = {d: [] for d in range(3)}
    for _s in st:
        for d in range(3):
            ref[d].append(float(np.mean([next(it) for _ in range(K)])))
            rb[d].append(float(np.mean([next(it) for _ in range(K)])))
    res = {}
    for suite, lvls in SUITES.items():
        idx = [i for i, s in enumerate(st) if s["level"] in lvls]
        distinct_ep = {lv: len({(ex[i]["best_x"], ex[i]["decisions"], ex[i]["end"]) for i in idx if st[i]["level"] == lv})
                       for lv in lvls}
        out = {"distinct_expert_episodes": distinct_ep, "draws": []}
        for d in range(3):
            r = [ref[d][i] for i in idx]

            def test(cand, r=r, idx=idx, lvls=lvls):
                c = criterion(cand, r)
                carried = any(np.mean([cand[k] - r[k] for k, i in enumerate(idx) if st[i]["level"] != lv]) <= 0
                              for lv in lvls)
                return {**c, "carried_by_one_level": bool(carried), "counts": bool(c["pass"] and not carried),
                        "per_level": {lv: round(float(np.mean([cand[k] - r[k] for k, i in enumerate(idx)
                                                                if st[i]["level"] == lv])), 1) for lv in lvls}}
            out["draws"].append({"expert": test([ex[i]["best_x"] for i in idx]),
                                 "random_b": test([rb[d][i] for i in idx]),
                                 "always_run": test([ar[i]["best_x"] for i in idx])})
        enough = sum(v >= 12 for v in distinct_ep.values()) >= 3
        out["checks"] = {"distinct": bool(enough),
                         "expert_passes_3_of_3": all(x["expert"]["counts"] for x in out["draws"]),
                         "random_b_fails_3_of_3": not any(x["random_b"]["pass"] for x in out["draws"]),
                         "always_run_fails_3_of_3": not any(x["always_run"]["pass"] for x in out["draws"])}
        out["valid"] = all(out["checks"].values())
        res[suite] = out
        print(suite, "valid" if out["valid"] else "NOT valid", out["checks"], "| distinct expert episodes", distinct_ep)
        for d, x in enumerate(out["draws"]):
            print(f"  draw {d}: expert {x['expert']['wins']}/{x['expert']['losses']} {x['expert']['mean_diff']:+.0f} px "
                  f"p {x['expert']['p_greater']:.3f} carried {x['expert']['carried_by_one_level']} {x['expert']['per_level']} | "
                  f"random_b p {x['random_b']['p_greater']:.3f} | always-run p {x['always_run']['p_greater']:.3f}")
    path = Path(out_path) if out_path else OUT / f"validation{'-' + rule if rule else ''}-20261009.json"
    path.write_text(json.dumps(
        {"what": "play harness v2 validation (RL_STUDY §13), scripted only", "rule": rule, "results": res,
         "expert": ex, "always_run": ar, "utc": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "validate":       # --out: write elsewhere (e.g. a cross-platform check of the reference file)
        validate_cmd(a[a.index("--rule") + 1] if "--rule" in a else None, a[a.index("--out") + 1] if "--out" in a else None)
    else:
        starts_cmd()
