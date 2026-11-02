"""`configs/base.yaml`, loaded once: the single source of frozen values (defaults come from here).

    from flymario import config
    config.get("dynamics.gain")          # 12.0
    config.get("dynamics.gain", 14.0)    # 12.0 -- the file wins when present
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from flymario.paths import CONFIGS

BASE = CONFIGS / "base.yaml"

_MISSING = object()


@lru_cache(maxsize=1)
def load() -> dict:
    """Parse `configs/base.yaml`. Cached; call `load.cache_clear()` in tests."""
    if not BASE.exists():
        return {}
    with BASE.open() as fh:
        return yaml.safe_load(fh) or {}


def get(path: str, default: Any = _MISSING) -> Any:
    """Dotted lookup: `get("dynamics.gain")`.

    Raises when the key is absent and no default is given (a vanished frozen value is a
    provenance failure, not a reason to fall back silently).
    """
    node: Any = load()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"{path!r} not in {BASE}")
            return default
        node = node[part]
    return node
