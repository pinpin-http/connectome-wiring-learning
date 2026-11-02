#!/usr/bin/env python
"""The declared reading of docs/RL_STUDY.md §17 (N1-N4), applied exactly as written.

Reads `artifacts/runs/platform/{operator-N1N2,samenoise-*,cs_noise_spread-nvidia}.json`, the ROCm
reference curve (`artifacts/rocm_reference/common_stream-connectome-20261009.json`) and the NVIDIA
production curves (`artifacts/runs/common_stream-connectome-20261009.json`, and
`common_stream-perm-connectome-20261009.json` if it exists); writes `platform/report-17.json`.

    .venv/bin/python scripts/diagnostics/platform_report.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PL = ROOT / "artifacts/runs/platform"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def curve_arr(curve):
    """(points, readouts) array and the frames of each point."""
    return np.array([c["auc"] for c in curve]), [c["frames"] for c in curve]


def D(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def main():
    out = {}
    # ---------------- N1 / N2
    op = load(PL / "operator-N1N2.json")
    if op:
        out["N1"] = {"pass": op["N1_pass"], "max_rel_err_segment": op["max_rel_err_segment"],
                     "max_rel_err_spmm": op["max_rel_err_spmm"], "tolerance": 1e-5}
        out["N2"] = {"a_pass": op["N2a_pass"], "segment_identical_all": op["segment_all_identical"],
                     "spmm_identical_all": op["spmm_all_identical"]}
    # ---------------- N3
    sn = {lab: load(PL / f"samenoise-{lab}.json") for lab in ("cpu", "cuda-seg", "cuda-seg-rerun", "cuda-spmm",
                                                               "cuda-seg-perm")}
    if all(sn.values()):
        A = {k: curve_arr(v["curve"])[0] for k, v in sn.items()}
        ms = {}
        for k in sn:
            mp = PL / f"samenoise-{k}-m.pt"
            if mp.exists():
                import torch
                ms[k] = torch.load(mp, map_location="cpu", weights_only=False)["m"].double()
        ref_pass = bool(np.array_equal(A["cuda-seg"], A["cuda-seg-rerun"]) and
                        ("cuda-seg" in ms and bool((ms["cuda-seg"] == ms["cuda-seg-rerun"]).all())))
        neg_fail = bool(np.abs(A["cuda-seg-perm"][1:] - A["cuda-seg"][1:]).max() > 0.02)
        d_seg, d_spmm = D(A["cuda-seg"], A["cpu"]), D(A["cuda-spmm"], A["cpu"])
        wd = {}
        if {"cpu", "cuda-seg", "cuda-spmm"} <= set(ms):
            import sys

            import scipy.sparse as scs
            import torch
            sys.path.insert(0, str(ROOT / "src"))
            from flymario.connectome import graphsel
            _g, W, motor = graphsel.load("connectome")
            Wm = scs.csr_matrix(W, dtype=np.float32)[np.asarray(motor, dtype=np.int64)]
            Wm.sort_indices()
            m0 = torch.as_tensor(np.abs(Wm.data), dtype=torch.float64)[None, :]
            scale = (ms["cpu"] - m0).norm()                  # the learnt change of the CPU run
            for k in ("cuda-seg", "cuda-spmm", "cuda-seg-perm"):
                wd[k] = float((ms[k] - ms["cpu"]).norm() / scale)
        out["N3"] = {"auc": {k: v.tolist() for k, v in A.items()},
                     "frames": curve_arr(sn["cpu"]["curve"])[1],
                     "first_sweep_motor_spikes": {k: v["first_sweep_motor_spikes"] for k, v in sn.items()},
                     "reference_rerun_identical": ref_pass, "negative_permuted_flagged": neg_fail,
                     "max_abs_dAUC_seg_vs_cpu": d_seg, "max_abs_dAUC_spmm_vs_cpu": d_spmm,
                     "weights_rel_diff_vs_cpu": wd,
                     "pass": bool(ref_pass and neg_fail and d_seg <= 0.02 and d_spmm <= 0.02),
                     "references_ok": bool(ref_pass and neg_fail)}
    # ---------------- N4
    ns = load(PL / "cs_noise_spread-nvidia.json")
    rocm = load(ROOT / "artifacts/rocm_reference/common_stream-connectome-20261009.json")
    nv = load(ROOT / "artifacts/runs/common_stream-connectome-20261009.json")
    nvp = load(ROOT / "artifacts/runs/common_stream-perm-connectome-20261009.json")
    if ns and ns.get("complete"):
        names = [b["name"] for b in ns["brains"]]
        M = np.array([c["auc"] for c in ns["curve"]])                       # (points, brains)

        def run(prefix):
            return M[:, [names.index(f"{prefix}_ro{i + 1}") for i in range(3)]]
        sets = {s: run(f"s{s}") for s in range(ns["run_args"]["n_sets"])}
        perm, eta = run("perm"), run("eta_half")
        r_rocm = curve_arr(rocm["results"]["connectome"])[0]
        r_nv = curve_arr(nv["results"]["connectome"])[0]
        batch_ok = bool(np.array_equal(sets[0], r_nv))
        perm_batch = None
        if nvp:
            perm_batch = bool(np.array_equal(perm, curve_arr(nvp["results"]["connectome"])[0]))
        cal = [D(sets[a], sets[b]) for a, b in combinations(range(1, 5), 2)]
        T = max(cal)
        val = [D(sets[a], sets[b]) for a, b in combinations(range(5, 9), 2)]
        perm_vs = [D(perm, sets[s]) for s in range(5, 9)]
        eta_vs = [D(eta, sets[s]) for s in range(5, 9)]
        valid = bool(sum(v > T for v in val) <= 1 and all(p > T for p in perm_vs))
        d_rocm = D(r_rocm, sets[0])
        out["N4"] = {"batch_reproduces_cs_scripted": batch_ok, "perm_batch_reproduces_cs_permuted": perm_batch,
                     "calibration_D": cal, "T": T, "validation_D": val, "perm_vs_validation_D": perm_vs,
                     "eta_half_vs_validation_D": eta_vs, "eta_half_flagged": int(sum(e > T for e in eta_vs)),
                     "criterion_valid": valid, "D_rocm_vs_nvidia_set0": d_rocm,
                     "D_rocm_vs_each_set": [D(r_rocm, sets[s]) for s in sets],
                     "coherent": bool(valid and batch_ok and d_rocm <= T),
                     "all_pairs_D_sets": [D(sets[a], sets[b]) for a, b in combinations(sets, 2)]}
    (PL / "report-17.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("auc",)}, default=str))


if __name__ == "__main__":
    main()
