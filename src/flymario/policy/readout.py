"""Motor readouts. The primary one is fixed and untrainable by declaration (docs/PROTOCOL.md §11.2).

Each readout carries its category in `CATEGORY`, recorded with every number:
`FixedRandomProjection` (fixed, the primary controller), `RandomReadout` (floor_control, ignores the
network), `MotorPopulationReadout` (demonstration, closed-loop demos only), the Gate D linear decoder
(diagnostic). A plastic readout would be `part_of_the_learning_intervention`, never reported as the primary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Readout:
    """Interface. Every readout maps motor activity to one discrete action."""

    CATEGORY = "undeclared"

    def predict(self, z: np.ndarray) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def provenance(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------
@dataclass
class ProjectionSpec:
    """Declared before any comparison (docs/PROTOCOL.md §11.2); fixed here, never chosen per run."""

    #: Entries are i.i.d. standard normal, scaled by 1/sqrt(n_features) so the
    #: score variance does not depend on how many motor units a dataset has.
    distribution: str = "gaussian"
    scale_mode: str = "inv_sqrt_features"

    #: Each row is centred so it sums to zero: otherwise the common-mode motor
    #: activity alone decides the argmax through the row with the largest sum.
    centre_rows: bool = True


class FixedRandomProjection(Readout):
    """**Primary readout.** Motor activity -> fixed random projection -> action.

    Drawn once from its declared seed and exposed read-only. One realisation is
    shared by every topology in a block; a fresh seed is drawn between blocks.
    """

    CATEGORY = "fixed"

    def __init__(self, n_features: int, n_actions: int, seed: int,
                 spec: ProjectionSpec | None = None):
        self.spec = spec or ProjectionSpec()
        self.n_features, self.n_actions, self.seed = n_features, n_actions, int(seed)
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((n_actions, n_features)).astype(np.float32)
        if self.spec.centre_rows:
            W -= W.mean(axis=1, keepdims=True)
        if self.spec.scale_mode == "inv_sqrt_features":
            W /= np.sqrt(n_features)
        self._W = W
        self._W.flags.writeable = False          # untrainable, enforced

    @property
    def W(self) -> np.ndarray:
        return self._W

    def scores(self, z: np.ndarray) -> np.ndarray:
        return self._W @ np.asarray(z, dtype=np.float32)

    def predict(self, z: np.ndarray) -> int:
        return int(np.argmax(self.scores(z)))

    def provenance(self) -> dict:
        return {"readout": "FixedRandomProjection", "category": self.CATEGORY,
                "seed": self.seed, "n_features": self.n_features,
                "n_actions": self.n_actions,
                "distribution": self.spec.distribution,
                "scale_mode": self.spec.scale_mode,
                "centre_rows": self.spec.centre_rows,
                "trainable": False,
                "checksum": int(np.abs(self._W).sum() * 1e6) % 10**9}


# --------------------------------------------------------------------------
def action_balance(readout: FixedRandomProjection, states: np.ndarray) -> dict:
    """Does the projection favour an action regardless of what it is shown?

    `structural` feeds pure common-mode states (a centred projection must be
    near-indifferent); `empirical` feeds the real states (one action taking
    nearly everything means the readout is not reading).
    """
    n = readout.n_actions
    common = np.ones((16, readout.n_features), np.float32) * np.linspace(0.1, 4.0, 16)[:, None]
    struct = np.bincount([readout.predict(c) for c in common], minlength=n)
    emp = np.bincount([readout.predict(s) for s in np.atleast_2d(states)], minlength=n)

    def gini(counts):
        c = np.sort(np.asarray(counts, float))
        if c.sum() == 0:
            return 0.0
        idx = np.arange(1, len(c) + 1)
        return float((2 * idx - len(c) - 1).dot(c) / (len(c) * c.sum()))

    return {"n_actions": n,
            "structural_counts": struct.tolist(),
            "structural_distinct": int((struct > 0).sum()),
            "empirical_counts": emp.tolist(),
            "empirical_gini": round(gini(emp), 4),
            "empirical_top_share": round(float(emp.max() / max(1, emp.sum())), 4),
            # A centred projection should not resolve pure common-mode input
            # into a preference; anything above 1 distinct action there is a
            # scale effect and is reported rather than hidden.
            "structurally_unbiased": bool((struct > 0).sum() >= min(2, n))
            if not readout.spec.centre_rows else bool(struct.max() < len(common))}


def block_seeds(n_blocks: int, master_seed: int = 0) -> list[int]:
    """Independent readout seeds, one per block, derived reproducibly.

    Blocks must not share a projection.
    """
    ss = np.random.SeedSequence(master_seed)
    return [int(s.generate_state(1)[0] % (2**31 - 1)) for s in ss.spawn(n_blocks)]
