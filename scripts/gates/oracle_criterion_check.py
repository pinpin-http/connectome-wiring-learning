#!/usr/bin/env python
"""Check the PERCEPTION_CREDIT_STUDY criterion on scripted references before any learner runs.

A controller "times its jumps" if it gets further than random jumping at its own rate (paired by
level and start, one-sided Wilcoxon on furthest x, p < 0.05). ram_jumper (jumps when an enemy is
8-64 px ahead) must PASS; random_b and always_run must FAIL, each against `random_a`.
Levels 5-1, 8-1, 3-1, 6-1; starts after 0, 3, ..., 21 NOOP decisions (32 paired episodes).

    .venv/bin/python scripts/gates/oracle_criterion_check.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
from flymario.env.schematic import danger_ahead, jump_label, takeoff_label, takeoff_label_v2
from flymario.paths import RUNS

RUN, JUMP, NOOP = (ACTION_NAMES.index(a) for a in ("RIGHT+B", "RIGHT+A+B", "NOOP"))
LEVELS = ("Level5-1", "Level8-1", "Level3-1", "Level6-1")
DELAYS = tuple(range(0, 22, 3))
MAX_DEC, STALL = 600, 150


def play(level, delay, policy):
    env = MarioEnv(EnvConfig(state=level))
    env.reset()
    for _ in range(delay):
        env.step(NOOP)
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
    return {"best_x": float(best), "end": end, "jump_rate": jumps / max(1, n)}


def criterion(cand, ref):
    """The declared test: candidate beats the rate-matched random reference."""
    from scipy.stats import wilcoxon
    c, r = np.asarray(cand, float), np.asarray(ref, float)
    nz = c != r
    p = float(wilcoxon(c[nz], r[nz], alternative="greater").pvalue) if nz.sum() >= 1 else 1.0
    return {"n": len(c), "wins": int((c > r).sum()), "losses": int((c < r).sum()),
            "mean_diff": round(float((c - r).mean()), 1), "p_greater": round(p, 5), "pass": p < 0.05}


def full_expert():
    """--full: jumps when the full RAM label (enemy, wall or pit ahead) asks for it, holding A
    at most 6 decisions then releasing one (a held A does not re-jump in SMB)."""
    held = [0]

    def policy(ram):
        a = JUMP if jump_label(ram) and held[0] < 6 else RUN
        held[0] = held[0] + 1 if a == JUMP else 0
        return a
    return policy


def hold_expert(ram):
    """--hold: the same full label, but A is held for as long as the label says jump."""
    return JUMP if jump_label(ram) else RUN


def main():
    full = "--full" in sys.argv
    hold = "--hold" in sys.argv
    takeoff = "--takeoff" in sys.argv or "--takeoff2" in sys.argv
    label = takeoff_label_v2 if "--takeoff2" in sys.argv else takeoff_label   # v2: Stage D's label
    rows = []
    for level in LEVELS:
        for d in DELAYS:
            seed = 20261006 + 100 * LEVELS.index(level) + d
            jr = play(level, d, (lambda ram: JUMP if label(ram) else RUN) if takeoff else
                      hold_expert if hold else full_expert() if full else
                      (lambda ram: JUMP if 8 <= danger_ahead(ram) <= 64 else RUN))
            q = jr["jump_rate"]
            K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 1

            def rand_mean(base, q=q, level=level, d=d, K=K):
                """Mean furthest x of K random jumpers at rate q (one draw is too noisy).
                K = 1 with base = seed / seed + 1 reproduces single draws."""
                gens = [np.random.default_rng(base + 7 * j) for j in range(K)]
                xs = [play(level, d, lambda ram, g=g, q=q: JUMP if g.random() < q else RUN) for g in gens]
                return {"best_x": float(np.mean([x["best_x"] for x in xs])), "end": "mean",
                        "jump_rate": float(np.mean([x["jump_rate"] for x in xs]))}
            row = {"level": level, "delay": d, "ram_jumper": jr,
                   "random_a": rand_mean(seed),
                   "random_b": rand_mean(seed + 1 if K == 1 else seed + 1000),
                   "always_run": play(level, d, lambda ram: RUN)}
            rows.append(row)
            print(f"{level} delay {d:2d} rate {q:.2f}: " + "  ".join(
                f"{k}={v['best_x']:.0f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
    ref = [r["random_a"]["best_x"] for r in rows]
    checks = {k: criterion([r[k]["best_x"] for r in rows], ref) for k in ("ram_jumper", "random_b", "always_run")}
    ok = checks["ram_jumper"]["pass"] and not checks["random_b"]["pass"] and not checks["always_run"]["pass"]
    rep = {"what": "criterion references for docs/PERCEPTION_CREDIT_STUDY.md (scripted, no brain)",
           "levels": LEVELS, "delays": DELAYS, "checks": checks, "criterion_validated": ok,
           "episodes": rows, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rep["positive"] = (f"take-off label{' v2' if label is takeoff_label_v2 else ''}, memoryless" if takeoff else
                       "full RAM label, A held through the window (mechanism test)" if hold else
                       "full RAM expert (enemy, wall, pit; press-release)" if full else "enemy-only RAM jumper")
    kk = f"-k{sys.argv[sys.argv.index('--k') + 1]}" if "--k" in sys.argv else ""
    name = (f"oracle_criterion_check-takeoff{'2' if label is takeoff_label_v2 else ''}{kk}-20261004.json" if takeoff else
            "oracle_criterion_check-hold-20261004.json" if hold else
            f"oracle_criterion_check-full{kk}-20261004.json" if full else "oracle_criterion_check-20261003.json")
    (RUNS / name).write_text(json.dumps(rep, indent=1))
    print(json.dumps(checks, indent=1), "\ncriterion validated:", ok)


if __name__ == "__main__":
    main()
