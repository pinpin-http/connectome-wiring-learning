"""The live channel between a training run and a live viewer.

The trainer writes one snapshot per decision to `/dev/shm` (temporary name, then rename, so a
reader always sees a whole one); a viewer reads the latest at its own pace. Contents:

    decision, time, names       counters and the brains' names
    frames                      every learner's screen, half size (B, 112, 120, 3)
    focus_frame                 the followed brain's screen, full size
    pi, scores, actions         the policy over the 8 actions, per brain
    reward, x, episode, ep_len  what the game returned
    spikes                      the followed brain's sweep, tick by tick, bit-packed
                                (ticks, ceil(n/8)) -- every neuron of the connectome
    motor_counts                its 2,227 motor neurons' spike counts
    eye_on, eye_off             the current injected into its ON and OFF lamina cells

A viewer picks the followed brain by writing its index to `focus` in the same directory.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

LIVE_DIR = Path(os.environ.get("FLYMARIO_LIVE", "/dev/shm/flymario_live"))


def channel(name: str) -> Path:
    d = LIVE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(name: str, **arrays) -> None:
    d = channel(name)
    tmp = d / "state.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, d / "state.npz")


def read(name: str):
    """The latest snapshot as a dict, or None."""
    f = LIVE_DIR / name / "state.npz"
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError):
        return None                       # replaced while being read: next time


def write_meta(name: str, meta: dict) -> None:
    d = channel(name)
    tmp = d / "meta.tmp.json"
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, d / "meta.json")


def read_meta(name: str) -> dict | None:
    f = LIVE_DIR / name / "meta.json"
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def set_focus(name: str, index: int) -> None:
    (channel(name) / "focus").write_text(str(int(index)))


def get_focus(name: str, default: int = 0) -> int:
    try:
        return int((LIVE_DIR / name / "focus").read_text())
    except (OSError, ValueError):
        return default


def now() -> float:
    return time.time()
