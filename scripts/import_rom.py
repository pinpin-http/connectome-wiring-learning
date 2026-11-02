#!/usr/bin/env python
"""Import a user-supplied SMB ROM and make it visible to our integration.

`python -m retro.import roms/` copies the ROM into stable-retro's *packaged*
integration directory (site-packages/.../SuperMarioBros-Nes-v0/rom.nes). Our
extended integration lives in `integrations/SuperMarioBros-Nes/`, which is a
separate directory and therefore does not see it. This links the two.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from flymario.paths import INTEGRATIONS, ROMS

EXPECTED_SHA1 = "facee9c577a5262dbe33ac4930bb0b58c8c037f7"


def rom_sha1(path: Path) -> str:
    """stable-retro hashes the ROM without its 16-byte iNES header."""
    data = path.read_bytes()
    return hashlib.sha1(data[16:] if data[:4] == b"NES\x1a" else data).hexdigest()


def main() -> int:
    nes = sorted(ROMS.glob("*.nes"))
    if not nes:
        for z in ROMS.glob("*.zip"):
            subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(ROMS)], check=True)
        nes = sorted(ROMS.glob("*.nes"))
    if not nes:
        print(f"no .nes or .zip in {ROMS} — see docs/ROM.md")
        return 1

    good = [p for p in nes if rom_sha1(p) == EXPECTED_SHA1]
    for p in nes:
        sha = rom_sha1(p)
        print(f"  {'OK  ' if sha == EXPECTED_SHA1 else 'skip'} {p.name}  sha1 {sha[:16]}…")
    if not good:
        print(f"\nno ROM matches {EXPECTED_SHA1} (US/World). See docs/ROM.md.")
        return 1

    subprocess.run([sys.executable, "-m", "retro.import", str(ROMS)], check=True)

    import retro

    packaged = Path(retro.data.path()) / "stable" / "SuperMarioBros-Nes-v0" / "rom.nes"
    if not packaged.exists():
        print(f"expected import target missing: {packaged}")
        return 1
    link = INTEGRATIONS / "SuperMarioBros-Nes" / "rom.nes"
    link.unlink(missing_ok=True)
    link.symlink_to(packaged)
    print(f"\nlinked {link} -> {packaged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
