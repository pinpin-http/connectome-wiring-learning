"""Fetch the raw connectome tables. Both datasets are CC-BY and need no account.

  flywire  Zenodo 10676866 + flyconnectome/flywire_annotations   ~845 MB
  malecns  Janelia public GCS bucket gs://flyem-male-cns          ~540 MB
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request

from flymario.connectome import datasets
from flymario.paths import (
    FLYWIRE_SOURCES, MALECNS_BASE, MALECNS_FILES, RAW, ensure_dirs,
)


def _progress(block: int, size: int, total: int) -> None:
    if total > 0:
        sys.stderr.write(f"\r    {min(100.0, block * size * 100.0 / total):5.1f}%")
        sys.stderr.flush()


# SHA-256 of the MaleCNS v1.0 tables every reported run was built from (recorded in
# deployment/manifest/data_manifest.json on 2026-10-09, mirrored in results/SHA256SUMS.malecns-v1.0).
# A mismatch means the bucket now serves different bytes: the graph would not be the one analysed.
# FlyWire files have no recorded hash (FlyWire is not used by the reported studies).
EXPECTED_SHA256 = {
    "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather":
        "9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather":
        "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "body-neurotransmitters-male-cns-v1.0.feather":
        "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(filename: str) -> bool:
    """True when the file on disk has its recorded hash (or none is recorded)."""
    expected = EXPECTED_SHA256.get(filename)
    if expected is None:
        return True
    actual = _sha256(RAW / filename)
    if actual != expected:
        print(f"[FAIL] {filename}: sha256 {actual}, expected {expected}", file=sys.stderr)
        return False
    print(f"[ ok ] {filename}: sha256 matches the record")
    return True


def _fetch(url: str, filename: str, approx_mb: int, force: bool) -> None:
    dest = RAW / filename
    if dest.exists() and not force:
        print(f"[skip] {filename} ({dest.stat().st_size / 1e6:.0f} MB on disk)")
    else:
        print(f"[get ] {filename} (~{approx_mb} MB)")
        urllib.request.urlretrieve(url, dest, _progress)
        sys.stderr.write("\n")
    if not verify(filename):
        raise SystemExit(f"{filename} does not match its recorded SHA-256 (re-fetch with force=True)")


def flywire(force: bool = False) -> None:
    ensure_dirs()
    for url, filename, mb in FLYWIRE_SOURCES.values():
        _fetch(url, filename, mb, force)


def malecns(force: bool = False) -> None:
    ensure_dirs()
    for filename, mb in MALECNS_FILES.values():
        _fetch(f"{MALECNS_BASE}/{filename}", filename, mb, force)


def primary(force: bool = False) -> None:
    """Fetch whichever dataset the registry calls primary.

    This used to fetch FlyWire unconditionally, with a docstring explaining
    that FlyWire was primary. It stopped being primary on 2026-09-21 and the
    function did not notice, so `make download` followed by `make build-graphs`
    downloaded one dataset and tried to build the other.
    """
    {"flywire": flywire, "malecns": malecns}[datasets.DEFAULT](force)


def all_datasets(force: bool = False) -> None:
    """Both datasets: the primary and the replication axis."""
    flywire(force)
    malecns(force)


if __name__ == "__main__":
    primary()
