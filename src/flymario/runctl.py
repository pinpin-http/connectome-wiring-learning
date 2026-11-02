"""Follow, checkpoint and stop a long run from outside the process.

A long run publishes artifacts/runs/live/<name>/: status.json (progress, ETA, heartbeat, pid;
rewritten atomically), series.jsonl (live curve points), and watches for CHECKPOINT (save now) and
STOP (save, then exit) files. From a terminal: `make follow`, `make checkpoint`, `make stop`, `make runs`.
Signals: SIGUSR1 = checkpoint, SIGTERM or Ctrl-C = checkpoint and stop, a second Ctrl-C aborts.
A stale heartbeat or a dead pid marks the run DEAD. Stops land at `poll()` calls (see `sub()`).
"""
from __future__ import annotations

import json
import os
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from flymario.paths import RUNS

LIVE = RUNS / "live"


class StopRequested(Exception):
    """Raised inside a step when a stop was asked for; the caller checkpoints."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunControl:
    """Progress, checkpoints and stops for one long run.

    Typical loop:

        ctl = RunControl("my_run", total=len(work), checkpoint=save)
        try:
            for item in work:
                ctl.begin(label_of(item))
                result = do(item, progress=ctl.sub)   # ctl.sub may raise
                save()
                ctl.done(result_summary)
        except StopRequested:
            ctl.close("stopped")        # already checkpointed by poll()
            raise SystemExit(0)
        ctl.close("complete")
    """

    def __init__(self, name: str, total: int, checkpoint: Callable[[], None],
                 unit: str = "step", meta: dict | None = None,
                 already_done: int = 0, root: Path | None = None,
                 fresh: bool = True):
        self.dir = (root or LIVE) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        # A resumed run keeps the curves it already drew; a new one starts clean.
        if fresh:
            (self.dir / "series.jsonl").unlink(missing_ok=True)
        self.charts: dict[str, dict] = {}
        # Stale control files from a previous run must not stop this one.
        for f in ("STOP", "CHECKPOINT"):
            (self.dir / f).unlink(missing_ok=True)
        self.name, self.total, self.unit = name, int(total), unit
        self._checkpoint = checkpoint
        self.completed = int(already_done)
        self.started = time.time()
        self._step_started = None
        self._durations: list[float] = []     # seconds per completed step
        #: the same, per kind of step (the part of a task key before its first "|"),
        #: since steps of different kinds can differ tenfold in cost
        self._dur_by_kind: dict[str, list[float]] = {}
        self.eta_note: str | None = None
        self.last = deque(maxlen=12)          # most recent result summaries
        self.current = None
        self.sub_progress = None
        self.checkpoints = 0
        self.last_checkpoint = None
        self._stop = False
        self._ckpt = False
        self._interrupts = 0
        self.meta = meta or {}
        #: steps running at once; the ETA divides the remaining work by it
        self.parallel = 1
        #: every step by name -> "todo" | "running" | "done" | "infeasible",
        #: drawn as a map of the whole run by the web view
        self.tasks: dict[str, str] = {}
        #: pictures the run wants shown, e.g. the stimuli the network sees
        self.images: list[dict] = []
        self._install_signals()
        self._write("running")

    # -- signals -----------------------------------------------------------
    def _install_signals(self) -> None:
        def on_usr1(signum, frame):
            self._ckpt = True

        def on_stop(signum, frame):
            self._interrupts += 1
            if self._interrupts >= 2:
                # Second Ctrl-C: abort; the step in flight is abandoned, checkpoints are kept.
                self._write("aborted")
                raise KeyboardInterrupt
            self._stop = True

        try:
            signal.signal(signal.SIGUSR1, on_usr1)
            signal.signal(signal.SIGTERM, on_stop)
            signal.signal(signal.SIGINT, on_stop)
        except ValueError:
            pass  # not the main thread (e.g. under a test runner); files still work

    # -- the calls a run makes -----------------------------------------------
    def begin(self, label: str) -> None:
        """A new step starts. Also a safe point to act on requests."""
        self.poll()
        self.current = label
        self.sub_progress = None
        self._step_started = time.time()
        self._write("running")

    def sub(self, i: int, n: int, what: str = "") -> None:
        """Progress inside a long step. Polls, so a stop lands within seconds."""
        self.sub_progress = {"i": int(i), "n": int(n), "what": what}
        self.poll(in_step=True)
        # Heartbeat at most once a second: a tight inner loop must not spend
        # its time rewriting a JSON file.
        if time.time() - getattr(self, "_last_write", 0.0) >= 1.0:
            self._write("running")

    def beat(self, current: str | None = None) -> None:
        """Heartbeat from a loop that is waiting on others (parallel workers); at most once a second."""
        if current is not None:
            self.current = current
        if time.time() - getattr(self, "_last_write", 0.0) >= 1.0:
            self._write("running")

    def done(self, summary: dict | None = None,
             seconds: float | None = None, label: str | None = None,
             kind: str | None = None) -> None:
        """A step finished and its result is saved.

        `seconds`: for parallel runs, the step's own duration as reported by the worker.
        """
        if seconds is None and self._step_started is not None:
            seconds = time.time() - self._step_started
        if seconds is not None:
            self._durations.append(float(seconds))
            if kind is not None:
                self._dur_by_kind.setdefault(kind, []).append(float(seconds))
        self.completed += 1
        if summary is not None:
            self.last.append({"label": label or self.current, "at": _now(), **summary})
        self.current, self.sub_progress, self._step_started = None, None, None
        self._write("running")

    def poll(self, in_step: bool = False) -> None:
        """Act on a pending checkpoint or stop request.

        Between steps a stop checkpoints and raises; inside a step it raises without
        checkpointing (the caller's last checkpoint is the consistent state).
        """
        if (self.dir / "CHECKPOINT").exists() or self._ckpt:
            (self.dir / "CHECKPOINT").unlink(missing_ok=True)
            self._ckpt = False
            if not in_step:
                self.checkpoint()
            else:
                # Mid-step, the consistent state is the last saved one; say so
                # instead of pretending a new checkpoint was made.
                self.last_checkpoint = {"at": _now(), "note":
                                        "requested mid-step: last completed step "
                                        "is already on disk"}
                self._write("running")
        if (self.dir / "STOP").exists() or self._stop:
            (self.dir / "STOP").unlink(missing_ok=True)
            self._stop = False
            if not in_step:
                self.checkpoint()
            self._write("stopping")
            raise StopRequested

    # -- live curves ----------------------------------------------------------
    def chart(self, key: str, title: str, x: str, y: str, series: list[str],
              logx: bool = False, kind: str = "line",
              refs: list[dict] | None = None, note: str = "",
              yfmt: str = "num") -> None:
        """Declare a chart. `series` fixes the colour order: first = slot 1.

        kind  "line"    numeric x, one polyline per `line` id within a series
              "points"  categorical x (strings), one dot per point
        refs  reference rules, e.g. {"axis": "y", "value": 0.1, "label": "chance"}
        note  one sentence on how to read it, shown under the title
        yfmt  "num" or "pct" (values stored as fractions, shown as percentages)
        """
        self.charts[key] = {"title": title, "x": x, "y": y, "series": series,
                            "logx": logx, "kind": kind, "refs": refs or [],
                            "note": note, "yfmt": yfmt}
        self._write("running")

    def point(self, chart: str, series: str, x, y: float,
              line: str | None = None, **extra) -> None:
        """Append one data point to a declared chart."""
        rec = {"c": chart, "s": series, "l": line or series, "x": x,
               "y": float(y), "t": round(time.time(), 1), **extra}
        with open(self.dir / "series.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def task(self, key: str, state: str) -> None:
        """Set one step's state on the run map (no disk write by itself)."""
        self.tasks[key] = state

    def image(self, name: str, png: bytes, caption: str = "",
              group: str = "") -> None:
        """Publish a picture next to the run (served by the web view)."""
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        (self.dir / "img").mkdir(exist_ok=True)
        (self.dir / "img" / safe).write_bytes(png)
        self.images = [i for i in self.images if i["file"] != safe]
        self.images.append({"file": safe, "caption": caption, "group": group})
        self._write("running")

    def checkpoint(self) -> None:
        self._checkpoint()
        self.checkpoints += 1
        self.last_checkpoint = {"at": _now(), "completed": self.completed}
        self._write("running")

    def close(self, state: str) -> None:
        """'complete', 'stopped' or 'failed'."""
        self.current = None
        self.sub_progress = None
        self._write(state)

    # -- status file ---------------------------------------------------------
    def eta_seconds(self) -> float | None:
        """Remaining wall time; a flagged lower bound when some kind of remaining step
        has not been timed yet."""
        self.eta_note = None
        if not self._durations:
            return None
        in_flight = (time.time() - self._step_started) if self._step_started else 0.0
        if self._dur_by_kind and self.tasks:
            left: dict[str, int] = {}
            for k, st in self.tasks.items():
                if st in ("todo", "running"):
                    kind = k.split("|", 1)[0]
                    left[kind] = left.get(kind, 0) + 1
            total, untimed = 0.0, []
            for kind, n in left.items():
                d = self._dur_by_kind.get(kind)
                if d:
                    total += n * sum(d) / len(d)
                else:
                    untimed.append(f"{n} {kind}")
            if untimed:
                self.eta_note = ("lower bound: " + ", ".join(untimed)
                                 + " task(s) not timed yet")
            return max(0.0, total / max(1, self.parallel) - in_flight)
        per = sum(self._durations) / len(self._durations)
        left_n = self.total - self.completed
        return max(0.0, left_n * per / max(1, self.parallel) - in_flight)

    def _write(self, state: str) -> None:
        per = (sum(self._durations) / len(self._durations)) if self._durations else None
        status = {
            "name": self.name, "state": state, "pid": os.getpid(),
            "unit": self.unit, "completed": self.completed, "total": self.total,
            "started": datetime.fromtimestamp(self.started, timezone.utc)
                               .isoformat(timespec="seconds"),
            "heartbeat": _now(), "heartbeat_unix": time.time(),
            "seconds_per_step": per, "eta_seconds": self.eta_seconds(),
            "eta_note": self.eta_note,
            "seconds_per_kind": {k: sum(v) / len(v) for k, v in self._dur_by_kind.items()},
            "current": self.current, "sub_progress": self.sub_progress,
            "step_elapsed": (time.time() - self._step_started)
                            if self._step_started else None,
            "checkpoints": self.checkpoints, "last_checkpoint": self.last_checkpoint,
            "last_results": list(self.last), "meta": self.meta,
            "charts": getattr(self, "charts", {}),
            "tasks": getattr(self, "tasks", {}),
            "images": getattr(self, "images", []),
            "control": {"stop": str(self.dir / "STOP"),
                        "checkpoint": str(self.dir / "CHECKPOINT")},
        }
        tmp = self.dir / "status.json.part"
        tmp.write_text(json.dumps(status, indent=2, default=str))
        tmp.replace(self.dir / "status.json")
        self._last_write = time.time()


# -- the outside view: used by tools/follow.py and tools/runctl.py -----------
def live_runs(root: Path | None = None) -> list[Path]:
    """Live directories, most recently updated first."""
    base = root or LIVE
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if (d / "status.json").exists()]
    return sorted(dirs, key=lambda d: (d / "status.json").stat().st_mtime,
                  reverse=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_status(run_dir: Path) -> dict:
    """status.json plus a verdict the file cannot give about itself."""
    s = json.loads((run_dir / "status.json").read_text())
    age = time.time() - float(s.get("heartbeat_unix", 0))
    s["heartbeat_age"] = age
    alive = pid_alive(int(s["pid"])) if s.get("pid") else False
    s["pid_alive"] = alive
    if s["state"] in ("running", "stopping") and not alive:
        s["verdict"] = "DEAD"   # killed without being able to say so
    elif s["state"] in ("running", "stopping"):
        s["verdict"] = s["state"].upper()
    else:
        s["verdict"] = s["state"].upper()
    return s


def read_series(run_dir: Path, since: int = 0) -> tuple[list[dict], int]:
    """Data points from line `since` on, and the next offset to ask for (incremental reads)."""
    f = run_dir / "series.jsonl"
    if not f.exists():
        return [], 0
    lines = f.read_text().splitlines()
    out = []
    for ln in lines[since:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            break   # a line being written right now; picked up next time
    return out, since + len(out)


def request(run_dir: Path, what: str) -> Path:
    """Create the STOP or CHECKPOINT file for a run."""
    if what not in ("STOP", "CHECKPOINT"):
        raise ValueError(what)
    f = run_dir / what
    f.write_text(_now())
    return f
