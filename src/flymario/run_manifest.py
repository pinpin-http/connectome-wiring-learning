"""Machine-readable provenance for experimental runs (protocol §24.1)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flymario.paths import CONFIGS, ROOT, RUNS


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _hash_paths(paths: dict[str, str | Path]) -> dict:
    out = {}
    for label, value in paths.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            out[label] = {"path": str(path), "exists": False}
            continue
        stat = path.stat()
        out[label] = {"path": str(path), "exists": True, "size_bytes": stat.st_size,
                      "sha256": sha256_file(path)}
    return out


def _runtime() -> dict:
    packages = ("numpy", "scipy", "torch", "pyarrow", "scikit-learn", "PyYAML",
                "polars", "gymnasium", "stable-retro")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    accelerator = {"configured_backend": None, "torch_device": None,
                   "torch_hip": None, "torch_cuda": None, "device_name": None}
    try:
        import torch
        accelerator["torch_hip"] = torch.version.hip
        accelerator["torch_cuda"] = torch.version.cuda
        accelerator["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            accelerator["device_name"] = torch.cuda.get_device_name(0)
        if torch.version.hip:
            try:
                info = subprocess.run(["rocm-smi", "--showproductname"], check=True,
                                      capture_output=True, text=True, timeout=10).stdout
                accelerator["rocm_smi_product"] = next(
                    (line.split(":", 1)[1].strip() for line in info.splitlines()
                     if "Card Series:" in line), None)
            except (OSError, subprocess.SubprocessError):
                accelerator["rocm_smi_product"] = None
    except Exception as exc:  # environment details should not prevent recording a run
        accelerator["error"] = f"{type(exc).__name__}: {exc}"
    config_path = CONFIGS / "base.yaml"
    try:
        import yaml
        import flymario.config as project_config
        accelerator["configured_backend"] = project_config.get("dynamics.backend")
        config_payload = yaml.safe_load(config_path.read_text())
    except Exception:
        config_payload = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_release": Path("/etc/os-release").read_text(errors="replace")
        if Path("/etc/os-release").is_file() else None,
        "packages": versions,
        "accelerator": accelerator,
        "base_config": config_payload,
        "base_config_file": {"path": str(config_path),
                              "sha256": sha256_file(config_path) if config_path.is_file() else None},
    }


def dataset_files(dataset: str | None = None, min_syn: int | None = None) -> dict[str, Path]:
    """Raw source and derived cache files for the selected connectome release."""
    from flymario import config
    from flymario.paths import (CONNECTOME, FLYWIRE_SOURCES, MALECNS_FILES, RAW)
    from flymario.connectome import datasets

    dataset = dataset or str(config.get("connectome.dataset"))
    min_syn = int(config.get("connectome.min_syn") if min_syn is None else min_syn)
    out: dict[str, Path] = {}
    if dataset == "malecns":
        for key, (filename, _) in MALECNS_FILES.items():
            out[f"raw_{key}"] = RAW / filename
    elif dataset == "flywire":
        for key, (_, filename, _) in FLYWIRE_SOURCES.items():
            out[f"raw_{key}"] = RAW / filename
    else:
        raise ValueError(f"no manifest source registry for dataset {dataset!r}")
    out["derived_graph_cache"] = datasets.cache_path(dataset, min_syn)
    return out


class RunManifest:
    """Write a start record, then finalize it with status, duration and outputs."""

    def __init__(self, run_id: str, study: str, phase: str, configuration: dict,
                 seeds: dict | list | None = None, dataset: str | None = None,
                 input_files: dict[str, str | Path] | None = None,
                 output_files: dict[str, str | Path] | None = None,
                 root: Path | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id
        self.study, self.phase = study, phase
        self.configuration = configuration
        self.seeds, self.dataset = seeds or {}, dataset
        self.input_files = dict(input_files or {})
        self.input_files.setdefault("run_manifest_implementation", Path(__file__))
        self.output_files = output_files or {}
        self.started_at, self.started_clock = _utc(), time.monotonic()
        self.path = (root or RUNS / "manifests") / f"{run_id}_{stamp}_{uuid.uuid4().hex[:8]}.json"
        self.record = {
            "manifest_version": 1,
            "run_id": run_id,
            "study": study,
            "phase": phase,
            "status": "running",
            "started_at_utc": self.started_at,
            "host": {"hostname": socket.gethostname(), "machine": platform.machine()},
            "git": {"commit": _git("rev-parse", "HEAD"),
                    "dirty": bool(_git("status", "--porcelain")),
                    "worktree_status": _git("status", "--short")},
            "runtime": _runtime(),
            "dataset": dataset,
            "configuration": configuration,
            "seed_registry": seeds or {},
            "inputs": _hash_paths(self.input_files),
            "expected_outputs": {k: str(Path(v).resolve()) for k, v in self.output_files.items()},
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.record, indent=2, default=str))
        tmp.replace(self.path)

    def finish(self, status: str = "complete", outputs: dict[str, str | Path] | None = None,
               error: str | None = None) -> Path:
        if status not in ("complete", "stopped", "failed"):
            raise ValueError(f"invalid run status {status!r}")
        self.record["status"] = status
        self.record["finished_at_utc"] = _utc()
        self.record["runtime_seconds"] = round(time.monotonic() - self.started_clock, 3)
        if error:
            self.record["error"] = error
        self.record["outputs"] = _hash_paths(outputs or self.output_files)
        self._write()
        return self.path
