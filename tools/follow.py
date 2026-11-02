#!/usr/bin/env python
"""Live view of a long run: progress, ETA, last results, and whether it is alive.

    python tools/follow.py                # most recently updated run
    python tools/follow.py transform_exposure_malecns
    python tools/follow.py --once         # print once and exit
    python tools/follow.py --list         # every live run and its state

Read-only. It never touches the run; stopping and checkpointing go through
tools/runctl.py (or `make stop` / `make checkpoint`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flymario.runctl import live_runs, read_status  # noqa: E402

GREEN, RED, AMBER, DIM, BOLD, OFF = ("\033[32m", "\033[31m", "\033[33m",
                                     "\033[2m", "\033[1m", "\033[0m")


def hms(seconds) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def bar(frac: float, width: int = 40) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full)


def render(s: dict) -> str:
    colour = {"RUNNING": GREEN, "COMPLETE": GREEN, "STOPPING": AMBER,
              "STOPPED": AMBER, "DEAD": RED, "FAILED": RED,
              "ABORTED": RED}.get(s["verdict"], OFF)
    total = max(1, int(s["total"]))
    done = int(s["completed"])
    out = [f"{BOLD}{s['name']}{OFF}   {colour}{s['verdict']}{OFF}"
           f"   pid {s['pid']}{'' if s['pid_alive'] else ' (gone)'}",
           f"  {bar(done / total)}  {done}/{total} {s['unit']}s  "
           f"({100 * done / total:.0f}%)",
           f"  started {s['started']}   step ≈ {hms(s['seconds_per_step'])}"
           f"   ETA {'≥ ' if s.get('eta_note') else ''}{hms(s['eta_seconds'])}"]
    if s.get("eta_note"):
        out.append(f"  {DIM}({s['eta_note']}){OFF}")

    age = s["heartbeat_age"]
    beat = f"{int(age)}s ago"
    if s["verdict"] == "DEAD":
        beat += f"  {RED}— process is gone; resume by relaunching the same command{OFF}"
    out.append(f"  heartbeat {beat}")

    if s.get("current"):
        line = f"  now: {s['current']}"
        sp = s.get("sub_progress")
        if sp and sp.get("n"):
            line += f"   [{sp['what']} {sp['i']}/{sp['n']}]"
        if s.get("step_elapsed") is not None:
            line += f"   {hms(s['step_elapsed'])} into this step"
        out.append(line)

    ck = s.get("last_checkpoint")
    if ck:
        out.append(f"  {DIM}last checkpoint {ck.get('at')}"
                   f"{'  ' + ck['note'] if ck.get('note') else ''}{OFF}")

    if s.get("last_results"):
        out.append(f"\n  {BOLD}latest results{OFF}")
        for r in list(s["last_results"])[-8:]:
            fields = "  ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ("label", "at"))
            out.append(f"    {r['label']:<46} {fields}")

    if s["verdict"] in ("RUNNING", "STOPPING"):
        out.append(f"\n  {DIM}make checkpoint   save now, keep running"
                   f"\n  make stop         save, then exit cleanly (resumable){OFF}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--every", type=float, default=2.0, help="refresh period, s")
    a = p.parse_args()

    runs = live_runs()
    if a.list:
        if not runs:
            print("no live runs")
        for d in runs:
            s = read_status(d)
            print(f"{s['name']:<40} {s['verdict']:<9} {s['completed']}/{s['total']}"
                  f"  heartbeat {int(s['heartbeat_age'])}s ago")
        return
    if a.name:
        target = next((d for d in runs if d.name == a.name), None)
    else:
        target = runs[0] if runs else None
    if target is None:
        sys.exit("no such live run (try --list)")

    while True:
        s = read_status(target)
        text = render(s)
        if a.once:
            print(text)
            return
        sys.stdout.write("\033[2J\033[H" + text + "\n")
        sys.stdout.flush()
        if s["verdict"] in ("COMPLETE", "STOPPED", "DEAD", "FAILED", "ABORTED"):
            return
        try:
            time.sleep(a.every)
        except KeyboardInterrupt:
            return   # Ctrl-C here stops the viewer, never the run


if __name__ == "__main__":
    main()
