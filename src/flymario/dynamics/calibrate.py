"""Find an operating point, and check it region by region.

The activity target is revalidated per graph and the per-region distribution is reported with the
mean, since equal global rates can hide different regimes. "Calibrated" means three things at once:

  alive        the readout population is not silent
  unsaturated  no region is firing on most ticks
  balanced     regions are within a declared ratio of each other
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flymario import config
from flymario.connectome import datasets
from flymario.connectome.schema import ConnectomeGraph
from flymario.models.spiking_reservoir import LIFBrain, LIFParams

#: Region groups reported separately. Keys are dataset-agnostic; the values
#: are resolved through the registry so both connectomes work unchanged.
REGION_KEYS = ("sensory", "optic", "projection", "central", "motor")


@dataclass
class Operating:
    source: str = field(default="interface", kw_only=True)
    gain: float
    bias: float
    global_rate: float
    region_rates: dict[str, float]
    alive: bool
    saturated: bool
    spread: float          # max/min across non-empty regions

    @property
    def usable(self) -> bool:
        return self.alive and not self.saturated


def regions(g: ConnectomeGraph, dataset: str) -> dict[str, np.ndarray]:
    cls = np.asarray(g.node_class)
    spec = datasets.spec(dataset)
    if dataset == "flywire":
        groups = {"sensory": ("sensory",), "optic": ("optic",),
                  "projection": ("visual_projection",), "central": ("central",),
                  "motor": spec["motor_classes"]}
    else:
        groups = {"sensory": ("ol_sensory", "cb_sensory", "vnc_sensory"),
                  "optic": ("ol_intrinsic",), "projection": ("visual_projection",),
                  "central": ("cb_intrinsic",), "motor": spec["motor_classes"]}
    return {k: np.flatnonzero(np.isin(cls, v)).astype(np.int32)
            for k, v in groups.items()}


def probe_current(g: ConnectomeGraph, dataset: str, amplitude: float = 2.0) -> np.ndarray:
    """Legacy synthetic probe, kept to reproduce older measurements; not the calibration reference.

    Drives both eyes' lamina at a uniform 2.0 (about 4x the interface's total current).
    Use `interface_drive` for anything that must reflect what the model runs on.
    """
    cur = np.zeros(g.n_nodes, np.float32)
    cur[datasets.by_types(g, dataset, "lamina_types")] = amplitude
    return cur


def interface_drive(g: ConnectomeGraph, dataset: str, bank=None,
                    scenario: str | None = None, hold_still: bool = False):
    """The calibration reference: real frames through the real interface (`frames -> vision.encode`).

    Returns a callable `tick -> current`, since frames change every decision; motion widens the
    regional spread, so `hold_still` is available but not the default.
    """
    from flymario.env import stimulus_bank
    from flymario.env.vision import RetinotopicVision

    vision = RetinotopicVision(g)
    bank = bank if bank is not None else stimulus_bank.build(which="calibration")
    name = scenario or sorted(bank.sequences)[0]
    frames = bank.sequences[name]

    if hold_still:
        vision.reset()
        vision.encode(frames[0])
        cur = vision.encode(frames[0])
        return lambda t: cur

    # Advance one frame per policy decision, not per tick: the emulator does not
    # produce a new frame for every neural step.
    tpd = config.get("timing.ticks_per_decision")
    vision.reset()
    encoded = []
    for f in frames:
        encoded.append(vision.encode(f))
    return lambda t: encoded[(t // max(1, tpd)) % len(encoded)]


#: Ticks discarded before measuring. The network ignites between ticks 50 and 100 (measured
#: convergence: tick 94), so a warm-up of 100 keeps the transient out of the measurement.
WARMUP = config.get("dynamics.warmup_ticks")
MEASURE_TICKS = config.get("dynamics.measure_ticks")


def measure(g: ConnectomeGraph, dataset: str, gain: float, bias: float,
            ticks: int = WARMUP + MEASURE_TICKS, warmup: int = WARMUP, seed: int = 0,
            drive: bool = True, source: str = "interface",
            bank=None) -> Operating:
    """Measure the operating point; `source` selects the drive and is recorded with the result.

    "interface" (default): real frames through the visual interface; "legacy": the old synthetic probe.
    """
    brain = LIFBrain(g.to_sparse(), LIFParams(gain=gain, bias=bias,
                                              noise_std=0.02, seed=seed))
    if not drive:
        at = lambda t: None            # noqa: E731
    elif source == "legacy":
        legacy = probe_current(g, dataset)
        at = lambda t: legacy          # noqa: E731
    else:
        at = interface_drive(g, dataset, bank=bank)
    reg = regions(g, dataset)
    for t in range(warmup):
        brain.step(at(t))
    acc = {k: [] for k in reg}
    glob = []
    for t in range(warmup, ticks):
        s = brain.step(at(t))
        glob.append(float(s.mean()))
        for k, idx in reg.items():
            acc[k].append(float(s[idx].mean()) if idx.size else np.nan)
    rates = {k: float(np.nanmean(v)) for k, v in acc.items()}
    live = [v for v in rates.values() if np.isfinite(v) and v > 0]
    return Operating(
        source=source if drive else "none",
        gain=gain, bias=bias, global_rate=float(np.mean(glob)), region_rates=rates,
        alive=rates.get("motor", 0.0) > 0.002,
        saturated=any(v > 0.30 for v in rates.values() if np.isfinite(v)),
        spread=float(max(live) / min(live)) if len(live) > 1 else float("inf"),
    )


def sweep(dataset: str = "flywire", gains=(1, 2, 4, 8, 12, 18, 26),
          bias: float = 0.0, verbose: bool = True) -> list[Operating]:
    g = datasets.load(dataset)
    out = []
    if verbose:
        head = f"{'gain':>6}{'global':>9}" + "".join(f"{k:>11}" for k in REGION_KEYS)
        print(head + "   spread  verdict")
    for gain in gains:
        op = measure(g, dataset, float(gain), bias)
        out.append(op)
        if verbose:
            row = f"{gain:6.1f}{op.global_rate * 100:8.2f}%"
            row += "".join(f"{op.region_rates[k] * 100:10.2f}%" for k in REGION_KEYS)
            verdict = ("saturated" if op.saturated else
                       "motor silent" if not op.alive else "usable")
            print(f"{row}{op.spread:9.0f}x  {verdict}")
    return out


#: Population firing target, from configs/base.yaml (frozen at 7%). An engineering target,
#: not a measured biological firing rate.
TARGET_RATE = config.get("dynamics.target_rate")


def best(ops: list[Operating], target: float = TARGET_RATE) -> Operating | None:
    """Closest to the activity target among usable points, spread as tiebreak.

    Not by spread alone: that picks the hottest gain, since saturation flattens the regional ratio.
    """
    usable = [o for o in ops if o.usable]
    if not usable:
        return None
    return min(usable, key=lambda o: (abs(o.global_rate - target), o.spread))
