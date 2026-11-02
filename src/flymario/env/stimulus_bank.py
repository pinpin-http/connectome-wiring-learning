"""A bank of real sensory sequences, for characterising the model.

Input chain: real frames -> RetinotopicVision.encode -> connectome. Sequences, not single frames,
because `encode` is dominated by the temporal derivative; several levels, because luminance differs
more between levels than within one. This module only builds the bank: the inputs characterise the
regime a parameter set produces, they are not a target to tune against.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv

#: Action index in `ACTION_NAMES`, by intent. Named so a scenario reads as a
#: description of what the fly would be looking at, not as button indices.
HOLD = ACTION_NAMES.index("NOOP")          # standing still: no optic flow
RUN = ACTION_NAMES.index("RIGHT+B")        # forward scrolling, the common case
JUMP = ACTION_NAMES.index("RIGHT+A+B")     # vertical flow, sky and ground swap
BACK = ACTION_NAMES.index("LEFT")          # reverse flow, rare but in-distribution


@dataclass(frozen=True)
class Scenario:
    """One recording condition: where to start, and what to do there."""

    name: str
    state: str
    actions: tuple[int, ...]
    skip: int = 0
    why: str = ""
    #: What to do while skipping into position. Separate from `actions` because the
    #: scenario may be unreachable under its own action (LEFT from a level start does not scroll).
    skip_action: int | None = None

    def action_at(self, k: int) -> int:
        return self.actions[k % len(self.actions)]

    def skip_action_at(self, k: int) -> int:
        if self.skip_action is not None:
            return self.skip_action
        return self.action_at(k)


#: Calibration / characterisation bank, chosen for input coverage before any activity was
#: measured. The only bank methodological choices are allowed to see.
CALIBRATION_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("1-1_open", "Level1-1", (RUN,), 20,
             "bright sky over ground, steady forward scrolling"),
    Scenario("1-1_obstacles", "Level1-1", (RUN,), 90,
             "pipes and blocks entering frame, occlusion and edges"),
    Scenario("1-1_jumping", "Level1-1", (JUMP, JUMP, RUN, RUN), 40,
             "vertical optic flow; sky and ground exchange position"),
    Scenario("1-1_still", "Level1-1", (HOLD,), 30,
             "no self-motion. Only enemies move, so the temporal term is small"),
    Scenario("1-1_backward", "Level1-1", (BACK,), 120,
             "reverse flow; sign of the derivative inverts", skip_action=RUN),
    Scenario("1-4_castle", "Level1-4", (RUN,), 20,
             "dark background, bright foreground: inverted luminance statistics"),
    Scenario("2-1_clouds", "Level2-1-clouds", (RUN,), 20,
             "bright uniform sky, little contrast, weak edges"),
    Scenario("3-1_night", "Level3-1", (RUN,), 20,
             "night palette, a third luminance regime"),
    Scenario("4-1", "Level4-1", (RUN,), 20, "further level geometry"),
    Scenario("8-1", "Level8-1", (RUN,), 20, "late-level geometry and density"),
)

#: Held-out validation bank: checks that a choice made on the calibration bank reproduces.
#: Nothing may be selected or tuned from it. All states are absent from the calibration bank
#: (two entries revisit World 1-1 far past any offset it uses, to unconfound level identity).
HELDOUT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("2-1_ground", "Level2-1", (RUN,), 20,
             "unseen level, ordinary scrolling; the calibration bank only has "
             "its cloud variant"),
    Scenario("5-1", "Level5-1", (RUN,), 20, "unseen geometry and palette"),
    Scenario("6-1", "Level6-1", (RUN,), 20, "unseen geometry and palette"),
    Scenario("7-1", "Level7-1", (RUN,), 20, "unseen geometry and palette"),
    Scenario("2-1_clouds_easy", "Level2-1-clouds-easy", (RUN,), 20,
             "bright sky again, from a different savestate"),
    Scenario("5-1_jumping", "Level5-1", (JUMP, JUMP, RUN, RUN), 30,
             "vertical flow, matched to 1-1_jumping but elsewhere",
             skip_action=RUN),
    Scenario("6-1_still", "Level6-1", (HOLD,), 40,
             "zero self-motion control, matched to 1-1_still but elsewhere",
             skip_action=RUN),
    Scenario("1-1_late", "Level1-1", (RUN,), 260,
             "World 1-1 far past every calibration offset, near the flagpole",
             skip_action=RUN),
    Scenario("1-4_late", "Level1-4", (RUN,), 150,
             "castle interior, past the calibration offset", skip_action=RUN),
)

#: Kept for callers that predate the split.
DEFAULT_SCENARIOS = CALIBRATION_SCENARIOS

BANKS = {"calibration": CALIBRATION_SCENARIOS, "heldout": HELDOUT_SCENARIOS}


@dataclass
class Bank:
    """Frame sequences plus the provenance needed to reproduce them."""

    sequences: dict[str, np.ndarray]          # name -> (T, 224, 240, 3) uint8
    scenarios: tuple[Scenario, ...]
    length: int
    frame_skip: int
    which: str = "calibration"
    rom_sha: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def digest(self) -> str:
        """Content hash, so a report can name exactly which bank produced it."""
        h = hashlib.sha256()
        for k in sorted(self.sequences):
            h.update(k.encode())
            h.update(self.sequences[k].tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> dict:
        return {
            "bank": self.which,
            "n_sequences": len(self.sequences),
            "length": self.length,
            "frame_skip": self.frame_skip,
            "digest": self.digest(),
            "scenarios": [
                {"name": s.name, "state": s.state, "skip": s.skip,
                 "actions": [ACTION_NAMES[a] for a in s.actions], "why": s.why}
                for s in self.scenarios
            ],
            "luminance": self.luminance_stats(),
        }

    def luminance_stats(self) -> dict:
        """Mean and spread of frame luminance per sequence, HUD excluded (the bank's measured coverage)."""
        from flymario.env.vision import HUD_ROWS, _LUMA

        out = {}
        for k, seq in self.sequences.items():
            gray = seq[:, HUD_ROWS:, :, :3].astype(np.float32) @ _LUMA
            out[k] = {
                "mean": round(float(gray.mean()) / 255.0, 4),
                "std": round(float(gray.std()) / 255.0, 4),
                "frame_to_frame_abs_delta": round(
                    float(np.abs(np.diff(gray, axis=0)).mean()) / 255.0, 5),
            }
        return out


def build(scenarios=None, length: int = 48, frame_skip: int = 4,
          verbose: bool = False, which: str = "calibration") -> Bank:
    """Record each scenario as `length` consecutive decisions (the temporal derivative matters).

    A scenario whose savestate is missing is skipped and reported, never replaced. Scrolling
    scenarios start after a short run-up, since Mario needs about ten decisions to reach speed.
    """
    if scenarios is None:
        if which not in BANKS:
            raise KeyError(f"unknown bank {which!r}; choose from {sorted(BANKS)}")
        scenarios = BANKS[which]
    sequences: dict[str, np.ndarray] = {}
    kept: list[Scenario] = []
    for sc in scenarios:
        try:
            env = MarioEnv(EnvConfig(state=sc.state, frame_skip=frame_skip))
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                print(f"  {sc.name:<16} unavailable ({sc.state}): {exc}")
            continue
        obs, _ = env.reset()
        for k in range(sc.skip):
            obs, _, term, trunc, _ = env.step(sc.skip_action_at(k))
            if term or trunc:
                obs, _ = env.reset()
        frames = []
        for k in range(length):
            frames.append(obs.copy())
            obs, _, term, trunc, _ = env.step(sc.action_at(sc.skip + k))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        sequences[sc.name] = np.stack(frames)
        kept.append(sc)
        if verbose:
            print(f"  {sc.name:<16} {length} decisions from {sc.state}")

    return Bank(sequences=sequences, scenarios=tuple(kept),
                length=length, frame_skip=frame_skip, which=which)


def input_statistics(bank: Bank, vision, sample: int | None = None) -> dict:
    """What the connectome actually receives per sequence: statistics of `vision.encode(frame_t)`.

    The static/temporal split is exact: re-encoding a frame right after itself gives d = 0 and
    isolates the static term; the difference from the real step is the temporal one.
    """
    inj = np.union1d(vision.on_idx, vision.off_idx)
    out: dict[str, dict] = {}

    for name, frames in bank.sequences.items():
        if sample:
            frames = frames[:sample]
        vision.reset()
        cur = np.stack([vision.encode(f)[inj] for f in frames])

        # static-only: encode each frame twice, keep the second
        vision.reset()
        static = []
        for f in frames:
            vision.encode(f)
            static.append(vision.encode(f)[inj])
        static = np.stack(static)
        temporal = cur - static

        nz = cur != 0
        amp = np.abs(cur[nz]) if nz.any() else np.zeros(1, np.float32)
        total = np.abs(cur).sum(axis=1)
        step = np.abs(np.diff(cur, axis=0)).mean(axis=1) if len(cur) > 1 else np.zeros(1)

        # temporal autocorrelation of the injected vector, lag 1..4
        flat = cur.reshape(len(cur), -1)
        ac = []
        for lag in (1, 2, 4):
            if len(flat) > lag and flat.std() > 1e-12:
                a, b = flat[:-lag].ravel(), flat[lag:].ravel()
                ac.append(round(float(np.corrcoef(a, b)[0, 1]), 4))
            else:
                ac.append(float("nan"))

        # spatial spread over the retinotopic map, weighted by current
        uv = np.concatenate([vision.on_uv, vision.off_uv])
        order = np.argsort(np.concatenate([vision.on_idx, vision.off_idx]))
        uv = uv[order]
        w = np.abs(cur).mean(axis=0)
        centroid = (uv * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)

        out[name] = {
            "cells_available": int(len(inj)),
            "cells_driven_mean": round(float(nz.sum(axis=1).mean()), 1),
            "cells_driven_max": int(nz.any(axis=0).sum()),
            "sparsity": round(float(1.0 - nz.mean()), 4),
            "amplitude": {
                "mean": round(float(amp.mean()), 4),
                "p50": round(float(np.percentile(amp, 50)), 4),
                "p99": round(float(np.percentile(amp, 99)), 4),
                "max": round(float(amp.max()), 4),
            },
            "total_current": {"mean": round(float(total.mean()), 1),
                              "cv": round(float(total.std() / max(total.mean(), 1e-12)), 4)},
            "component_share": {
                "static": round(float(np.abs(static).sum() / max(np.abs(cur).sum(), 1e-12)), 4),
                "temporal": round(float(np.abs(temporal).sum() / max(np.abs(cur).sum(), 1e-12)), 4),
            },
            "step_to_step_change": round(float(step.mean()), 5),
            "autocorrelation_lag_1_2_4": ac,
            "retinotopic_centroid": [round(float(centroid[0]), 4),
                                     round(float(centroid[1]), 4)],
        }

    tot = np.array([v["total_current"]["mean"] for v in out.values()])
    drv = np.array([v["cells_driven_mean"] for v in out.values()])
    return {
        "bank": bank.which,
        "per_sequence": out,
        "across_sequences": {
            "total_current_cv": round(float(tot.std() / max(tot.mean(), 1e-12)), 4),
            "cells_driven_cv": round(float(drv.std() / max(drv.mean(), 1e-12)), 4),
            "total_current_range": [round(float(tot.min()), 1), round(float(tot.max()), 1)],
            "cells_driven_range": [round(float(drv.min()), 1), round(float(drv.max()), 1)],
        },
    }
