"""common_stream.py checkpoint / resume (deployment, 2026-10-09): an interrupted and
resumed run equals an uninterrupted one. Short streams (--steps / --test-steps, --tag),
one graph; needs the connectome cache and the ROM, a few minutes -> marked slow.

    .venv/bin/python -m pytest -m slow tests/unit/test_common_stream_resume.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/diagnostics/common_stream.py"
CK = ROOT / "artifacts/runs/checkpoints"
RUNS = ROOT / "artifacts/runs"


def _run(tag, *extra):
    cmd = [sys.executable, str(SCRIPT), "--graphs", "connectome", "--n", "40", "--every", "20", "--save",
           "--tag", tag, "--steps", "120", "--test-steps", "120", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)


def _cleanup(tag):
    for p in (CK / f"common_stream_{tag}_connectome.pt", CK / f"common_stream_state_{tag}_connectome.pt",
              RUNS / f"common_stream-connectome-{tag}-20261009.json"):
        p.unlink(missing_ok=True)


@pytest.mark.slow
def test_interrupted_and_resumed_equals_uninterrupted():
    import torch
    for t in ("resumetestA", "resumetestB"):
        _cleanup(t)
    try:
        a = _run("resumetestA")
        assert a.returncode == 0, a.stderr[-2000:]
        b1 = _run("resumetestB", "--stop-after", "30")
        assert b1.returncode == 75, b1.stderr[-2000:]
        assert (CK / "common_stream_state_resumetestB_connectome.pt").exists()
        b2 = _run("resumetestB")
        assert b2.returncode == 0, b2.stderr[-2000:]
        assert "resumed at frame 30" in b2.stdout
        assert not (CK / "common_stream_state_resumetestB_connectome.pt").exists()
        ma = torch.load(CK / "common_stream_resumetestA_connectome.pt", weights_only=False)["brain"]["m"]
        mb = torch.load(CK / "common_stream_resumetestB_connectome.pt", weights_only=False)["brain"]["m"]
        assert torch.equal(ma, mb)
        ca = json.loads((RUNS / "common_stream-connectome-resumetestA-20261009.json").read_text())["results"]
        cb = json.loads((RUNS / "common_stream-connectome-resumetestB-20261009.json").read_text())["results"]
        assert ca == cb
    finally:
        for t in ("resumetestA", "resumetestB"):
            _cleanup(t)
