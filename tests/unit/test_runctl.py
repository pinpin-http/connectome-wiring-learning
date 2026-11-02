"""Run control: a STOP must stop, a CHECKPOINT must save, a dead run must read dead.

Each test is built so it can fail: a control file that is ignored, a checkpoint
callback that is never called, or a stale status that still reads RUNNING
would each break one assertion below.
"""
from __future__ import annotations

import json

import pytest

from flymario.runctl import RunControl, StopRequested, read_status, request


def make(tmp_path, saved):
    return RunControl("t", total=5, checkpoint=lambda: saved.append(1),
                      unit="measurement", root=tmp_path)


def test_progress_is_published(tmp_path):
    saved = []
    ctl = make(tmp_path, saved)
    ctl.begin("first")
    ctl.done({"decode": 0.5})
    s = json.loads((tmp_path / "t" / "status.json").read_text())
    assert s["completed"] == 1 and s["total"] == 5
    assert s["last_results"][0]["label"] == "first"
    assert s["last_results"][0]["decode"] == 0.5
    assert s["seconds_per_step"] is not None


def test_checkpoint_file_saves_and_keeps_running(tmp_path):
    saved = []
    ctl = make(tmp_path, saved)
    request(tmp_path / "t", "CHECKPOINT")
    ctl.begin("next")                       # a safe point: acts on the request
    assert saved == [1]
    assert not (tmp_path / "t" / "CHECKPOINT").exists()   # consumed, not sticky
    ctl.done()
    ctl.begin("after")                      # no new request -> no new save
    assert saved == [1]


def test_stop_between_steps_saves_then_raises(tmp_path):
    saved = []
    ctl = make(tmp_path, saved)
    request(tmp_path / "t", "STOP")
    with pytest.raises(StopRequested):
        ctl.begin("never runs")
    assert saved == [1], "a clean stop must checkpoint first"


def test_stop_inside_a_step_raises_without_a_torn_checkpoint(tmp_path):
    saved = []
    ctl = make(tmp_path, saved)
    ctl.begin("long step")
    request(tmp_path / "t", "STOP")
    with pytest.raises(StopRequested):
        ctl.sub(3, 10, "class")
    # Mid-step there is nothing coherent to save; the last completed step is
    # already on disk, so no half-state may be written.
    assert saved == []


def test_stale_stop_file_does_not_stop_a_new_run(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "STOP").write_text("left over")
    saved = []
    ctl = make(tmp_path, saved)
    ctl.begin("runs")                       # must not raise
    assert saved == []


def test_a_run_whose_process_is_gone_reads_dead(tmp_path):
    saved = []
    make(tmp_path, saved)
    f = tmp_path / "t" / "status.json"
    s = json.loads(f.read_text())
    s["pid"] = 2 ** 22 + 12345              # beyond pid_max: never a live process
    f.write_text(json.dumps(s))
    assert read_status(tmp_path / "t")["verdict"] == "DEAD"


def test_series_are_read_incrementally(tmp_path):
    from flymario.runctl import read_series
    ctl = make(tmp_path, [])
    ctl.chart("c", "t", "x", "y", series=["a", "b"])
    ctl.point("c", "a", 1, 0.5)
    ctl.point("c", "b", 2, 0.7)
    pts, nxt = read_series(tmp_path / "t")
    assert [p["s"] for p in pts] == ["a", "b"] and nxt == 2
    ctl.point("c", "a", 3, 0.9)
    pts, nxt = read_series(tmp_path / "t", since=nxt)
    assert len(pts) == 1 and pts[0]["x"] == 3 and nxt == 3
    s = json.loads((tmp_path / "t" / "status.json").read_text())
    assert s["charts"]["c"]["series"] == ["a", "b"]
