#!/usr/bin/env python
"""Ask a running job to checkpoint or to stop.

    python tools/runctl.py checkpoint          # most recent run: save now
    python tools/runctl.py stop                # most recent run: save and exit
    python tools/runctl.py stop transform_exposure_malecns

This only creates a request file in the run's live directory. The run acts on
it at its next poll -- within seconds for runs that poll inside their steps --
and reports what it did in status.json, which `tools/follow.py` shows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flymario.runctl import live_runs, read_status, request  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["checkpoint", "stop"])
    p.add_argument("name", nargs="?", default=None)
    a = p.parse_args()

    runs = live_runs()
    target = (next((d for d in runs if d.name == a.name), None) if a.name
              else (runs[0] if runs else None))
    if target is None:
        sys.exit("no such live run (python tools/follow.py --list)")
    s = read_status(target)
    if s["verdict"] not in ("RUNNING", "STOPPING"):
        sys.exit(f"{s['name']} is {s['verdict']}: nothing to {a.action}")
    f = request(target, a.action.upper())
    print(f"{a.action} requested for {s['name']} (pid {s['pid']}) -> {f}")
    print("watch it land with: make follow")


if __name__ == "__main__":
    main()
