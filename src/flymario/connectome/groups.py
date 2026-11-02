"""Neuron groups for display: one table for both datasets.

FlyWire 783 and MaleCNS v1.0 name their super-classes differently (`optic` vs `ol_intrinsic`, ...);
every viewer that colours, counts or selects neurons by class goes through this table. These groups
are for display only: the analysis definition is `dynamics.calibrate.regions`.
"""
from __future__ import annotations

import numpy as np

#: Raw class label -> display group. Unmapped labels go to "other".
CLASS_GROUP: dict[str, str] = {
    # FlyWire 783
    "optic": "optic", "sensory": "sensory", "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal", "central": "central",
    "ascending": "ascending", "descending": "descending", "motor": "motor",
    "endocrine": "other",
    # MaleCNS v1.0
    "ol_intrinsic": "optic",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "cb_sensory_tbc": "sensory",
    "vnc_sensory": "sensory", "vnc_sensory_tbc": "sensory",
    "sensory_ascending": "sensory", "sensory_ascending_tbc": "sensory",
    "sensory_descending": "sensory",
    "visual_projection_tbc": "visual_projection",
    "cb_intrinsic": "central",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending", "descending_neuron_tbc": "descending",
    "cb_motor": "motor", "vnc_motor": "motor",
    "vnc_intrinsic": "vnc", "vnc_tbc": "vnc", "vnc_efferent": "vnc",
    "vnc_endocrine": "vnc",
}

#: Display colour per group (RGB 0-255), the same in the 3D view and on the curves. Optic,
#: sensory, projection and motor use the reference categorical palette (validated on #0f1116);
#: the central brain is a neutral, peripheral groups are muted.
GROUP_COLOUR: dict[str, tuple[int, int, int]] = {
    "optic": (57, 135, 229), "sensory": (25, 158, 112),
    "visual_projection": (144, 133, 233), "visual_centrifugal": (213, 81, 129),
    "central": (195, 194, 183), "ascending": (201, 133, 0),
    "descending": (230, 103, 103), "motor": (230, 103, 103),
    "vnc": (111, 122, 140), "other": (100, 100, 100),
}

#: Colours for the analysis regions of `dynamics.calibrate.regions`, which the
#: rate plots use -- mapped onto the same display colours.
REGION_COLOUR: dict[str, tuple[int, int, int]] = {
    "optic": GROUP_COLOUR["optic"], "sensory": GROUP_COLOUR["sensory"],
    "projection": GROUP_COLOUR["visual_projection"],
    "central": GROUP_COLOUR["central"], "motor": GROUP_COLOUR["motor"],
}

#: Plain-words names, for people who have never seen a connectome.
GROUP_LABEL: dict[str, str] = {
    "optic": "optic lobe (vision)", "sensory": "sensory neurons",
    "visual_projection": "visual projection (eye → brain)",
    "visual_centrifugal": "visual feedback (brain → eye)",
    "central": "central brain", "ascending": "ascending (body → brain)",
    "descending": "descending (brain → body)", "motor": "motor neurons",
    "vnc": "nerve cord", "other": "other",
}


def groups_of(classes) -> np.ndarray:
    """Display group for every neuron, as an array of strings."""
    return np.array([CLASS_GROUP.get(str(c), "other") for c in classes])


def colours_of(groups: np.ndarray) -> np.ndarray:
    """(n, 3) uint8 colours for an array of group names."""
    out = np.full((len(groups), 3), 140, dtype=np.uint8)
    for g, c in GROUP_COLOUR.items():
        out[groups == g] = c
    return out
