#!/usr/bin/env python
"""Exploratory: where do frozen learners start jumps, how long do they hold, what kills them?

Every brain of a checkpoint plays greedily on unseen 5-1 and 8-1 (starts after 0/6/12/18 NOOPs).
A jump onset is a jump action after a non-jump one; onsets are compared with the RAM danger distance
and `schematic.takeoff_label`. Per brain: onset rate, share in the take-off window, hold length,
unexplained onsets, deaths by enemy vs other.

    .venv/bin/python scripts/diagnostics/jump_timing_trace.py artifacts/runs/checkpoints/mario_stageC_40k.pt
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))

from mario_generalisation_eval import JUMP, Evaluator

from flymario.env.schematic import takeoff_label
from flymario.paths import RUNS

LEVELS = ("Level5-1", "Level8-1")
DELAYS = (0, 6, 12, 18)


def main():
    ck = Path(sys.argv[1])
    ev = Evaluator(str(ck), 1, 600)
    logs = {n: [] for n in ev.names}
    for level in LEVELS:
        for d in DELAYS:
            B = ev.B
            obs, info = [None] * B, [None] * B
            for b in range(B):
                obs[b], info[b] = ev.pool.reset_to(b, level, d)
                ev.visions[b].reset()
            active = np.ones(B, bool)
            frames = [ev.view_frame(b, obs[b], info[b]) for b in range(B)]
            best = [info[b]["mario_x"] for b in range(B)]
            since = [0] * B
            ep = {b: [] for b in range(B)}
            n_dec = [0] * B
            while active.any():
                act = ev.scores(frames).argmax(1)
                for b in np.flatnonzero(active):
                    ev.pool.conns[b].send(("step", int(act[b])))
                for b in np.flatnonzero(active):
                    rec = {"a": int(act[b]), "dx": float(info[b].get("danger_dx", 999)),
                           "tko": bool(takeoff_label(info[b]["ram"])) if "ram" in info[b] else False}
                    o, _r, term, trunc, inf = ev.pool.conns[b].recv()
                    n_dec[b] += 1
                    if inf["mario_x"] > best[b]:
                        best[b], since[b] = inf["mario_x"], 0
                    else:
                        since[b] += 1
                    rec["died"] = bool(inf.get("died"))
                    ep[b].append(rec)
                    info[b] = inf
                    frames[b] = ev.view_frame(b, o, inf)
                    if inf.get("died") or inf.get("cleared") or term or trunc or since[b] >= 150 or n_dec[b] >= 600:
                        active[b] = False
            for b in range(B):
                logs[ev.names[b]].append({"level": level, "delay": d, "best_x": best[b], "trace": ep[b]})
            print(level, d, " ".join(f"{ev.names[b][:5]}{ev.names[b][-1]}={best[b]:.0f}" for b in range(B)), flush=True)
    summary = {}
    for name, eps in logs.items():
        onsets, holds, n, dist, reasonless, in_win, d_enemy, d_other = 0, [], 0, [], 0, 0, 0, 0
        for e in eps:
            tr = e["trace"]
            n += len(tr)
            run = 0
            for i, r in enumerate(tr):
                jump = r["a"] in JUMP
                if jump and (i == 0 or tr[i - 1]["a"] not in JUMP):
                    onsets += 1
                    in_win += r["tko"]
                    if r["dx"] < 999:
                        dist.append(r["dx"])
                    reasonless += (r["dx"] > 64 and not r["tko"])
                if jump:
                    run += 1
                elif run:
                    holds.append(run)
                    run = 0
            if run:
                holds.append(run)
            if tr and tr[-1]["died"]:
                if tr[-1]["dx"] <= 40:
                    d_enemy += 1
                else:
                    d_other += 1
        summary[name] = {
            "onsets_per_100": round(100 * onsets / max(1, n), 2), "onsets": onsets,
            "share_in_takeoff_window": round(in_win / max(1, onsets), 3),
            "share_without_reason": round(reasonless / max(1, onsets), 3),
            "enemy_dx_at_onset_median": float(np.median(dist)) if dist else None,
            "enemy_dx_at_onset_q25_q75": [float(np.percentile(dist, 25)), float(np.percentile(dist, 75))] if dist else None,
            "mean_hold": round(float(np.mean(holds)), 2) if holds else None,
            "deaths_enemy_near": d_enemy, "deaths_other": d_other,
            "mean_best_x": round(float(np.mean([e["best_x"] for e in eps])), 1)}
        print(name, summary[name], flush=True)
    rep = {"what": "closed-loop jump timing of frozen learners (exploratory)", "checkpoint": ck.name,
           "levels": LEVELS, "delays": DELAYS, "summary": summary,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"jump_timing_trace-{ck.stem}.json").write_text(json.dumps(rep, indent=1))
    ev.pool.close()


if __name__ == "__main__":
    main()
