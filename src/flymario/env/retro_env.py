"""Super Mario Bros as a Gymnasium environment: the policy sees pixels only.

`step()` returns the framebuffer as the observation; everything read out of RAM
(reward, termination, metrics) goes into `info`, which the agent must not consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import retro
from retro import Actions

from flymario.paths import ROOT

HEADLESS_NOTE = """Always pass render_mode=None.

Importing torch before stable-retro breaks pyglet's GL config selection on this
Wayland session: retro's reset() calls render(), pyglet tries to open an X11
shadow window, and screen.get_best_config() raises IndexError. Verified:

    import retro; retro.make(g).reset()               -> works
    import torch, retro; retro.make(g).reset()        -> IndexError
    import torch, retro; retro.make(g, render_mode=None).reset()  -> works

Since every training script imports torch, render_mode must be explicit.
"""

GAME = "SuperMarioBros-Nes"
INTEGRATION_DIR = ROOT / "integrations"

#: Discrete action set. Index order is the readout's class order.
ACTIONS: list[tuple[str, ...]] = [
    (),                        # 0 NOOP
    ("RIGHT",),                # 1 walk right
    ("RIGHT", "B"),            # 2 run right
    ("RIGHT", "A"),            # 3 jump right
    ("RIGHT", "A", "B"),       # 4 long jump right
    ("A",),                    # 5 jump in place
    ("LEFT",),                 # 6 back up
    ("DOWN",),                 # 7 enter pipe / crouch
]
ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+B", "RIGHT+A", "RIGHT+A+B", "A", "LEFT", "DOWN"]

# Mario is dying/dead in these player_state values.
DYING_STATES = {0x06, 0x0B}
# 0x03 in player_float means "sliding down the flagpole" = level cleared.
FLAGPOLE = 0x03


@dataclass
class EnvConfig:
    state: str = "Level1-1"
    frame_skip: int = 4          # 15 policy decisions/s, the usual SMB setting
    sticky_actions: bool = False
    max_frames: int = 20_000
    death_penalty: float = -25.0
    finish_bonus: float = 100.0
    time_penalty: float = -0.02  # this is what turns "finish" into "finish fast"
    render_mode: str | None = None   # keep None; see HEADLESS_NOTE


def register_integration() -> None:
    """Point stable-retro at our extended data.json (extra RAM variables)."""
    retro.data.Integrations.add_custom_path(str(INTEGRATION_DIR))


def _discrete_to_buttons(buttons: list[str], combo: tuple[str, ...]) -> np.ndarray:
    a = np.zeros(len(buttons), dtype=np.uint8)
    for name in combo:
        a[buttons.index(name)] = 1
    return a


class MarioEnv:
    """Thin wrapper; deliberately not a gym.Env subclass to keep the loop obvious."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        register_integration()
        self.env = retro.make(
            GAME,
            state=self.cfg.state,
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=Actions.ALL,
            render_mode=self.cfg.render_mode,
        )
        self.buttons = self.env.buttons
        self._table = [_discrete_to_buttons(self.buttons, c) for c in ACTIONS]
        self.n_actions = len(ACTIONS)
        self._reset_tracking()

    # -- tracking ----------------------------------------------------------
    def _reset_tracking(self) -> None:
        self._frames = 0
        self._best_progress = 0.0
        self._prev_progress = 0.0

    @staticmethod
    def _mario_x(info: dict) -> float:
        """Absolute x within the current area; page counter avoids the 256 wrap."""
        return float(info.get("player_page", 0)) * 256.0 + float(info.get("player_x_page", 0))

    @staticmethod
    def progress(info: dict) -> float:
        """Monotone route progress across areas, so warps do not reset the reward."""
        world = int(info.get("world", 0))
        area = int(info.get("area", 0))
        return (world * 4 + area) * 4000.0 + MarioEnv._mario_x(info)

    def enemy_ahead(self) -> tuple[float, int, int, int]:
        """Distance (px, along the level) to the nearest active enemy ahead of Mario
        and its type; (999, -1) if none. From RAM, for metrics only (enemy flags 0x0F-0x13,
        types 0x16-0x1A, page 0x6E-0x72, x 0x87-0x8B; Mario page 0x6D, x 0x86)."""
        ram = self.env.unwrapped.get_ram().astype(int)
        mario = ram[0x6D] * 256 + ram[0x86]
        best, kind, sx, sy = 999.0, -1, -1, -1
        for i in range(5):
            if ram[0x0F + i]:
                dx = ram[0x6E + i] * 256 + ram[0x87 + i] - mario
                if 0 < dx < best:
                    # screen position of the enemy (0x3AE + i) and its y (0xCF + i)
                    best, kind, sx, sy = float(dx), int(ram[0x16 + i]), int(ram[0x3AE + i]), int(ram[0xCF + i])
        return best, kind, sx, sy

    def set_level(self, state: str) -> None:
        """Make the next reset() start from another savestate of the integration."""
        self.env.load_state(state, retro.data.Integrations.ALL)
        self.cfg.state = state

    def _annotate(self, info: dict, *, died: bool, cleared: bool) -> dict:
        """Add the derived fields every consumer expects on top of raw RAM.

        Used by both `reset()` and `step()`, so the two return the same keys
        (stable-retro returns an empty dict from reset).
        """
        from flymario.env.schematic import danger_ahead
        info = dict(info)
        dx, kind, sx, sy = self.enemy_ahead()
        ram = self.env.unwrapped.get_ram()
        info.update(
            mario_x=self._mario_x(info),
            progress=self.progress(info),
            died=died,
            cleared=cleared,
            frames=self._frames,
            enemy_dx=dx,
            enemy_type=kind,
            enemy_sx=sx,
            enemy_sy=sy,
            mario_sx=int(ram[0x3AD]),
            mario_sy=int(ram[0xCE]),
            danger_dx=danger_ahead(ram),       # on-screen enemies only (schematic.py)
            ram=np.asarray(ram, np.uint8).copy(),   # for the schematic view; never a pixel input
        )
        return info

    # -- api ---------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, dict]:
        obs, _ = self.env.reset()
        self._reset_tracking()
        # Read the RAM ourselves rather than trusting reset's info: emulator
        # state exists at t=0, stable-retro just does not hand it back.
        info = self._annotate(self.env.unwrapped.data.lookup_all(),
                              died=False, cleared=False)
        # Seed the progress baseline from the true starting position (Mario starts
        # at x=40), so the first step collects no phantom progress.
        self._prev_progress = self._best_progress = info["progress"]
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_r = 0.0
        obs = None
        info: dict = {}
        terminated = truncated = False

        #: every emulator frame of this step, for viewers that want to show the
        #: motion between decisions; the agent itself only ever sees `obs`
        self.last_frames = []
        for _ in range(self.cfg.frame_skip):
            obs, _, term, trunc, info = self.env.step(self._table[action])
            self.last_frames.append(obs.copy())   # the emulator may reuse its buffer
            self._frames += 1
            if term or trunc:
                terminated, truncated = term, trunc
                break

        c = self.cfg
        prog = self.progress(info)
        # Reward only forward progress past the best point reached, so an agent
        # cannot farm reward by oscillating left and right.
        gained = max(0.0, prog - self._best_progress)
        self._best_progress = max(self._best_progress, prog)
        total_r += gained + c.time_penalty

        died = int(info.get("player_state", 0)) in DYING_STATES
        cleared = int(info.get("player_float", 0)) == FLAGPOLE
        if died:
            total_r += c.death_penalty
            terminated = True
        if cleared:
            total_r += c.finish_bonus
            terminated = True
        if self._frames >= c.max_frames:
            truncated = True

        info = self._annotate(info, died=died, cleared=cleared)
        self._prev_progress = prog
        return obs, total_r, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
