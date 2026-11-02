#!/usr/bin/env python
"""Check result files and frozen protocol files against the SHA-256 manifests in results/.

    python tools/verify_results.py            # every results/SHA256SUMS.* manifest
    python tools/verify_results.py --strict   # also fail when a listed file is missing

Standard library only. Manifest lines are `<sha256>  <path from the repository root>` (`sha256sum`
format, # for comments). The manifests are records: a mismatch means the file changed after it.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(manifest: Path) -> tuple[int, int, int]:
    ok = missing = bad = 0
    for n, line in enumerate(manifest.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        target = ROOT / rel
        if not target.is_file():
            missing += 1
            print(f"  MISSING   {rel}")
        elif sha256(target) != digest:
            bad += 1
            print(f"  MISMATCH  {rel}")
        else:
            ok += 1
    return ok, missing, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="a missing file is a failure")
    args = ap.parse_args()
    manifests = sorted(RESULTS.glob("SHA256SUMS.*"))
    if not manifests:
        print("no manifest in results/")
        return 1
    failed = False
    for m in manifests:
        print(m.relative_to(ROOT))
        ok, missing, bad = check(m)
        print(f"  -> {ok} ok, {missing} missing, {bad} mismatched")
        failed |= bad > 0 or (args.strict and missing > 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
