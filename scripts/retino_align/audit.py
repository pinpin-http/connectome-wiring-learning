#!/usr/bin/env python
"""RETINO_ALIGN_V1 initial audit (mission §3): repository state, environment, hardware, data and their hashes,
neuron indexing, retinal columns, readout, simulator facts. Facts only; docs/retino_align/AUDIT.md interprets them.

    .venv/bin/python scripts/retino_align/audit.py [--out artifacts/retino_align/audit/audit.json]

Exit code 0 when every hard check passes, 1 otherwise (the failing checks are listed in the JSON).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p: str) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/retino_align/audit/audit.json")
    a = ap.parse_args()
    import cv2
    import igraph
    import scipy
    import torch
    import yaml

    from flymario import config
    from flymario.connectome import datasets, nullcache
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.retino_align import retina

    checks: dict[str, bool] = {}
    rec: dict = {"utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rec["repo"] = {"branch": sh("git rev-parse --abbrev-ref HEAD"), "commit": sh("git rev-parse HEAD"),
                   "status_short": sh("git status --short"), "last_commits": sh("git log --oneline -8").splitlines()}
    rec["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "igraph": igraph.__version__, "opencv": cv2.__version__, "pyyaml": yaml.__version__,
        "platform": platform.platform()}
    gpu = {}
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        gpu = {"name": pr.name, "memory_gb": round(pr.total_memory / 1e9, 2), "capability": f"{pr.major}.{pr.minor}",
               "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
               "memory_used_mb_now": sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")}
    quota, period = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    mem = read("/sys/fs/cgroup/memory/memory.limit_in_bytes") or read("/sys/fs/cgroup/memory.max")
    du = shutil.disk_usage(ROOT)
    rec["hardware"] = {"gpu": gpu, "cpus_visible": os.cpu_count(),
                       "cpu_quota_cores": (int(quota) / int(period)) if quota and period and int(quota) > 0 else None,
                       "ram_limit_gb": round(int(mem) / 1e9, 1) if mem and mem.isdigit() else mem,
                       "disk_total_gb": round(du.total / 1e9, 1), "disk_free_gb": round(du.free / 1e9, 1)}
    checks["gpu_visible"] = bool(gpu)

    # --- connectome -------------------------------------------------------------------------------------------
    gpath = datasets.cache_path(datasets.DEFAULT)
    g = datasets.load(datasets.DEFAULT)
    pairs = g.src.astype(np.int64) * g.n_nodes + g.dst
    W = g.to_sparse()
    stats = json.loads((gpath.parent / "malecns_stats.json").read_text()) if (gpath.parent / "malecns_stats.json").exists() else {}
    rec["connectome"] = {
        "file": str(gpath.relative_to(ROOT)), "sha256": sha256(gpath), "bytes": gpath.stat().st_size,
        "source": g.source, "n_nodes": int(g.n_nodes), "n_edges": int(g.n_edges),
        "duplicate_pairs": int(len(pairs) - len(np.unique(pairs))), "autapses": int((g.src == g.dst).sum()),
        "sign_counts": {str(k): int(v) for k, v in zip(*np.unique(g.sign, return_counts=True))},
        "weights_integer": bool(np.array_equal(g.weight, np.round(g.weight))),
        "weight_min_max": [float(g.weight.min()), float(g.weight.max())],
        "synapses_total": float(g.weight.astype(np.float64).sum()),
        "operator_nnz": int(W.nnz), "operator_max_abs": float(np.abs(W.data).max()),
        "operator_max_abs_row_sum": float(np.abs(W).sum(1).max()),
        "node_ids_sha256": hashlib.sha256(g.node_ids.astype(np.int64).tobytes()).hexdigest(),
        "stats_file": {k: stats.get(k) for k in ("n_nodes", "n_edges", "synapses") if k in stats}}
    checks["connectome_sha256_matches_prep"] = (rec["connectome"]["sha256"] ==
                                               "6befb67a0278568316c26d32c3f2cadf76fbb8cc35d93137819c86f804a1df9f")
    checks["connectome_counts_match_report"] = (g.n_nodes == 169_984 and g.n_edges == 25_563_197)
    checks["operator_integer"] = bool(np.array_equal(W.data, np.round(W.data)))

    # --- readout --------------------------------------------------------------------------------------------------
    motor = datasets.motor_index(g, datasets.DEFAULT)
    rec["readout"] = {"n": int(len(motor)),
                      "by_class": {str(k): int(v) for k, v in zip(*np.unique(g.node_class[motor], return_counts=True))},
                      "node_index_sha256": hashlib.sha256(motor.astype(np.int64).tobytes()).hexdigest(),
                      "body_ids_sha256": hashlib.sha256(g.node_ids[motor].astype(np.int64).tobytes()).hexdigest(),
                      "ordered_ascending": bool(np.all(np.diff(motor) > 0))}
    checks["readout_2227"] = len(motor) == 2227

    # --- retina -----------------------------------------------------------------------------------------------------
    vis = RetinotopicVision(g)
    ret = retina.build(g, vis)
    nb = ret.neighbours()
    deg = np.bincount(nb.ravel(), minlength=ret.n_cols)
    lam_types = np.asarray(g.node_type)
    lam_all = np.isin(lam_types, retina.LAMINA)
    rec["retina"] = {
        "eye": ret.side, "coverage": vis.coverage(), "n_injected": ret.n_cells, "n_columns": ret.n_cols,
        "types": {t: int((ret.types == t).sum()) for t in retina.LAMINA},
        "signatures": {str(s): int((ret.col_sig == s).sum()) for s in np.unique(ret.col_sig)},
        "lattice_neighbour_degree_hist": {int(k): int(v) for k, v in enumerate(np.bincount(deg)) if v},
        "lamina_cells_in_graph_by_side": {str(s): int((lam_all & (np.asarray(g.node_side) == s)).sum())
                                          for s in np.unique(np.asarray(g.node_side)[lam_all])},
        "lamina_cells_mapped_by_side": {str(s): int((lam_all & (np.asarray(g.node_side) == s) &
                                                     np.isfinite(g.node_retino).all(1)).sum())
                                        for s in np.unique(np.asarray(g.node_side)[lam_all])},
        "injected_in_degree_mean": float(np.diff(W.tocsr().indptr)[ret.inj].mean()),
        "retina_digest": ret.digest(),
        "encoder": {"hud_rows": vis.p.hud_rows, "on_gain": vis.p.on_gain, "off_gain": vis.p.off_gain,
                    "static_gain": vis.p.static_gain, "blur_sigma": vis.p.blur_sigma}}
    # static convention: two identical frames after reset give only the static term
    frame = np.full((240, 256, 3), 0.4 * 255, np.float32)
    vis.reset()
    c1 = vis.encode(frame)
    c2 = vis.encode(frame)
    on, off = vis.on_idx, vis.off_idx
    checks["static_two_identical_frames"] = bool(np.allclose(c2[on], 0.4 * vis.p.static_gain, atol=1e-5) and
                                                 np.allclose(c2[off], 0.6 * vis.p.static_gain, atol=1e-5) and
                                                 np.array_equal(c1, c2))
    checks["retina_columns_892"] = ret.n_cols == 892
    rp = retina.report(ret, retina.column_permutation(ret, 1), retina.cell_permutation(ret, retina.column_permutation(ret, 1)))
    rec["retina"]["permutable_fraction"] = rp["frac_cells_permutable"]
    checks["permutable_fraction_ge_0.90"] = rp["frac_cells_permutable"] >= 0.90

    # --- simulator facts ------------------------------------------------------------------------------------------
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    rec["simulator"] = {
        "tau_ms": p.tau_ms, "dt_ms": p.dt_ms, "decay": p.decay, "threshold": p.threshold, "v_reset": p.v_reset,
        "refractory_ms": p.refractory_ms, "refractory_steps": p.refractory_steps, "gain": p.gain,
        "noise_std": p.noise_std, "bias": p.bias, "ticks_per_decision": int(config.get("timing.ticks_per_decision")),
        "weight_transform": config.get("dynamics.weight_transform"), "backend_config": config.get("dynamics.backend"),
        "max_count_per_window": int(np.ceil(int(config.get("timing.ticks_per_decision")) / (p.refractory_steps + 1)))}

    # --- nulls -------------------------------------------------------------------------------------------------------
    local = json.loads((ROOT / "artifacts/retino_align_prep/nullin_hashes_local.json").read_text())["graphs"]
    server_p = ROOT / "artifacts/retino_align_prep/nullin_hashes_server.json"
    server = json.loads(server_p.read_text())["graphs"] if server_p.exists() else {}
    nulls = {}
    for s in list(range(40)) + [100, 101]:
        spec = f"nullin:{s}"
        path = nullcache.null_path("depth_stratified_lateral_instrength", datasets.DEFAULT, s)
        nulls[spec] = {"present": path.exists(), "sha256_local_pc": local.get(spec, {}).get("sha256"),
                       "sha256_server": server.get(spec, {}).get("sha256"),
                       "build_s_server": server.get(spec, {}).get("seconds")}
        nulls[spec]["identical"] = (nulls[spec]["sha256_server"] is not None and
                                    nulls[spec]["sha256_server"] == nulls[spec]["sha256_local_pc"])
    rec["nulls"] = {"kind": "depth_stratified_lateral_instrength", "graphs": nulls,
                    "n_present": sum(v["present"] for v in nulls.values()),
                    "n_identical_to_pc": sum(v["identical"] for v in nulls.values())}

    rec["checks"] = checks
    rec["all_pass"] = all(checks.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, a.out)
    print(json.dumps(checks, indent=1))
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
