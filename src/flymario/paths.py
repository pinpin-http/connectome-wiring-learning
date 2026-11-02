"""Canonical project paths and dataset locations."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"          # user-supplied: TAS movies, etc.
CONNECTOME = DATA / "connectome"      # built ConnectomeGraph .npz caches
PROCESSED = DATA / "processed"        # state/action datasets

ROMS = ROOT / "roms"
CONFIGS = ROOT / "configs"
INTEGRATIONS = ROOT / "integrations"

ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
CHECKPOINTS = ARTIFACTS / "checkpoints"
LOGS = ARTIFACTS / "logs"

# --- FlyWire 783 (Zenodo 10676866 + flyconnectome/flywire_annotations, CC-BY) ---
FLYWIRE_SOURCES = {
    "connections": (
        "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
        "flywire_proofread_connections_783.feather", 813,
    ),
    "annotations": (
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "flywire_neuron_annotations_783.tsv", 32,
    ),
}

# --- MaleCNS v1.0 (public GCS bucket, CC-BY). Second adapter, brain + VNC. ---
MALECNS_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)
MALECNS_FILES = {
    "weights": ("connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", 485),
    "annotations": ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14),
    "neurotransmitters": ("body-neurotransmitters-male-cns-v1.0.feather", 42),
}

# --- derived artifacts ---
GRAPH_FLYWIRE = CONNECTOME / "flywire_783.npz"
GRAPH_MALECNS = CONNECTOME / "malecns_v1.npz"


def ensure_dirs() -> None:
    for p in (RAW, EXTERNAL, CONNECTOME, PROCESSED, ROMS,
              RUNS, CHECKPOINTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)
