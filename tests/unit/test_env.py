"""Emulator-side setup guarantees. None of these need the game ROM."""
from __future__ import annotations

import json

import numpy as np
import pytest

from flymario.env.actions import DEFAULT_MACROS, macros_from_expert, to_button_array
from flymario.paths import INTEGRATIONS


def test_custom_integration_is_present():
    d = INTEGRATIONS / "SuperMarioBros-Nes"
    assert (d / "data.json").exists() and (d / "scenario.json").exists()
    assert (d / "Level1-1.state").exists()


def test_integration_exposes_position_and_death_ram():
    info = json.loads((INTEGRATIONS / "SuperMarioBros-Nes" / "data.json").read_text())["info"]
    # Needed for reward and metrics; the stock stable-retro integration ships
    # only a wrapping xscrollLo, which is useless as a progress signal.
    for var in ("player_page", "player_x_page", "player_state", "world", "area"):
        assert var in info, f"missing RAM variable {var}"


def test_action_macros_are_distinct():
    combos = [tuple(sorted(c)) for _, c in DEFAULT_MACROS]
    assert len(set(combos)) == len(combos)


def test_button_array_encoding():
    buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
    a = to_button_array(buttons, ("RIGHT", "B"))
    assert a.sum() == 2 and a[buttons.index("RIGHT")] == 1 and a[buttons.index("B")] == 1


def test_macros_from_expert_drops_unused_combinations():
    expert = [("RIGHT", "B")] * 50 + [("RIGHT", "A", "B")] * 20 + [("LEFT",)] * 2
    out = macros_from_expert(expert, min_count=5)
    names = {n for n, _ in out}
    assert "RIGHT_B" in names and "RIGHT_A_B" in names
    assert "LEFT" not in names, "rare expert combo should not get a class"
    assert "NOOP" in names


@pytest.mark.parametrize("bad", [("NOPE",)])
def test_unknown_button_is_ignored(bad):
    assert to_button_array(["A", "B"], bad).sum() == 0


def test_emulator_runs_headless():
    """Regression guard for the torch-before-retro GL crash (HEADLESS_NOTE)."""
    import torch  # noqa: F401  -- import order is the point of this test
    import retro

    env = retro.make("Airstriker-Genesis-v0", render_mode=None)
    obs, _ = env.reset()
    obs, *_ = env.step(env.action_space.sample())
    env.close()
    assert obs.ndim == 3 and obs.dtype == np.uint8


def _rom_present() -> bool:
    return any((INTEGRATIONS / "SuperMarioBros-Nes").glob("rom.*"))


@pytest.mark.skipif(not _rom_present(), reason="needs the game ROM")
def test_reset_info_matches_step_info():
    """`reset()` must report the same RAM fields as `step()`.

    stable-retro returns an empty info dict from reset. Passing that through
    made `info["mario_x"]` None on the first decision of every episode; cast
    into a float array it became NaN, and a single NaN turns np.median into
    NaN, so `x > median(x)` was False everywhere. Gate F's positive control
    died on "the data contains only one class" -- a statistic poisoned three
    steps upstream, not a broken classifier.
    """
    from flymario.env.retro_env import EnvConfig, MarioEnv

    env = MarioEnv(EnvConfig())
    try:
        _, reset_info = env.reset()
        _, _, _, _, step_info = env.step(0)
        assert set(reset_info) == set(step_info)
        for key in ("mario_x", "progress", "score", "time", "lives", "coins"):
            assert reset_info.get(key) is not None, f"{key} missing at reset"
    finally:
        env.close()


@pytest.mark.skipif(not _rom_present(), reason="needs the game ROM")
def test_first_step_earns_no_phantom_progress():
    """The progress baseline starts where Mario stands, not at zero.

    It used to be seeded from reset's empty info, i.e. progress 0 while Mario
    is really at x=40, so every episode handed out a free +40 of "forward
    progress" on its first step regardless of the action taken.
    """
    from flymario.env.retro_env import EnvConfig, MarioEnv

    env = MarioEnv(EnvConfig())
    try:
        _, info = env.reset()
        assert env._best_progress == info["progress"] > 0.0
        # Action 0 is NOOP: one frame-skip of standing still cannot move Mario
        # forward, so the shaped reward must not be positive.
        _, reward, _, _, _ = env.step(0)
        assert reward <= 0.0, f"phantom reward {reward:+.1f} on a NOOP"
    finally:
        env.close()
