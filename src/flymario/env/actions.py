"""Discrete action vocabulary.

The spec's rule: derive macros from what the expert actually uses, rather than
exposing all 2^8 controller states. `DEFAULT_MACROS` is the starting set for
World 1-1; `macros_from_expert` narrows it to the combinations a given TAS
really presses, so the decoder spends no capacity on dead classes.
"""
from __future__ import annotations

import numpy as np

#: (name, buttons). Order is the class order the readout learns.
DEFAULT_MACROS: list[tuple[str, tuple[str, ...]]] = [
    ("NOOP", ()),
    ("RIGHT", ("RIGHT",)),
    ("RIGHT_B", ("RIGHT", "B")),
    ("RIGHT_A", ("RIGHT", "A")),
    ("RIGHT_A_B", ("RIGHT", "A", "B")),
    ("A", ("A",)),
    ("LEFT", ("LEFT",)),
    ("DOWN", ("DOWN",)),
]

NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")


def to_button_array(buttons: list[str], combo: tuple[str, ...]) -> np.ndarray:
    a = np.zeros(len(buttons), dtype=np.uint8)
    for name in combo:
        if name in buttons:
            a[buttons.index(name)] = 1
    return a


def macros_from_expert(expert_combos: list[tuple[str, ...]],
                       min_count: int = 5) -> list[tuple[str, tuple[str, ...]]]:
    """Keep only the combinations the expert presses at least `min_count` times."""
    from collections import Counter

    counts = Counter(tuple(sorted(c)) for c in expert_combos)
    keep = {c for c, n in counts.items() if n >= min_count}
    out = [(name, combo) for name, combo in DEFAULT_MACROS
           if tuple(sorted(combo)) in keep]
    if not any(combo == () for _, combo in out):
        out.insert(0, ("NOOP", ()))
    return out
