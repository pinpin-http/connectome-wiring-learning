#!/usr/bin/env python
"""Verify every piece of the setup and say exactly what is missing.

Exit code 0 means the environment is ready for work that does not need the ROM.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

from flymario.paths import CONNECTOME, INTEGRATIONS, RAW, ROMS

SMB_SHA1 = "facee9c577a5262dbe33ac4930bb0b58c8c037f7"
PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "torch", "retro",
            "gymnasium", "sklearn", "cv2", "stable_baselines3")

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'ok ' if passed else 'MISS'}] {label}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print(f"python {sys.version.split()[0]}")

    print("\ndependencies")
    for m in PACKAGES:
        try:
            mod = importlib.import_module(m)
            check(m, True, getattr(mod, "__version__", ""))
        except ImportError:
            check(m, False, "not installed")

    print("\ngpu (optional — CPU is sufficient)")
    try:
        import torch

        avail = torch.cuda.is_available()
        print(f"  [{'ok ' if avail else '--'}] torch backend"
              f"  — {torch.cuda.get_device_name(0) if avail else 'CPU only'}")
    except Exception as e:  # noqa: BLE001
        print(f"  [--] torch backend  — {e}")

    print("\nraw connectome tables")
    for f in ("flywire_proofread_connections_783.feather",
              "flywire_neuron_annotations_783.tsv",
              "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"):
        p = RAW / f
        check(f[:52], p.exists(), f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else "run `make download`")

    print("\nbuilt graphs")
    graphs = sorted(CONNECTOME.glob("*.npz"))
    check("cached ConnectomeGraph files", bool(graphs), f"{len(graphs)} found" or "run `make build-graphs`")

    print("\nemulator")
    try:
        import retro

        retro.data.Integrations.add_custom_path(str(INTEGRATIONS))
        check("stable-retro import", True, retro.__version__)
        check("custom SMB integration", (INTEGRATIONS / "SuperMarioBros-Nes" / "data.json").exists())
        # render_mode=None is mandatory here: see retro_env.HEADLESS_NOTE.
        env = retro.make("Airstriker-Genesis-v0", render_mode=None)
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        check("emulator core runs", True, "verified on the bundled free ROM")
    except Exception as e:  # noqa: BLE001
        check("emulator", False, str(e)[:70])

    print("\nrom (user-supplied — see docs/ROM.md)")
    roms = list(ROMS.glob("*.nes"))
    have_rom = False
    if not roms:
        # Not counted against readiness: the ROM is only needed to actually
        # play, and everything else in the pipeline runs without it.
        print("  [--] SuperMarioBros.nes  — absent: place your own dump in roms/")
    else:
        have_rom = True
        for r in roms:
            # stable-retro hashes the ROM *without* its 16-byte iNES header,
            # so the sha1 of the whole file will never match rom.sha.
            data = Path(r).read_bytes()
            body = data[16:] if data[:4] == b"NES\x1a" else data
            sha = hashlib.sha1(body).hexdigest()
            check(r.name, sha == SMB_SHA1,
                  "sha1 matches" if sha == SMB_SHA1 else f"sha1 {sha[:12]}… != expected")

    if ok:
        print("\nSETUP READY" + ("" if have_rom else "  (ROM absent — needed only to run the game)"))
    else:
        print("\nSETUP INCOMPLETE — see MISS lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
