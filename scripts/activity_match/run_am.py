#!/usr/bin/env python
"""ACTIVITY_MATCH orchestrator (docs/activity_match/PROTOCOL_FROZEN.md): resumable, atomic outputs, one status file.

    run_am.py controls                       checks (gamma = 1 reproduces V1, gain effect, reset, ...)
    run_am.py [--config C] <command>          C: configs/activity_match/protocol.yaml (V1, default), protocol_v2.yaml or protocol_v2b.yaml
    run_am.py scan [--graphs ...] [--gammas ...]   development: label-free activity over a wide gamma range
    run_am.py calibrate [--pilot]            label-free grid, common range, bounded search, CAL_CHECK
    run_am.py prepare  --run R [--pilot]     banks, inputs, subsets, lot plan with each lot's gamma
    run_am.py validate --run R               data checks: balance, uniqueness, disjointness, leakage
    run_am.py extract  --run R               simulation lots (graph x condition x noise seed), resumable
    run_am.py decode   --run R [--workers N] [--shuffled-control]
    run_am.py analyse  --run R [--replicates N]
    run_am.py status   --run R               lots expected / done, decoders, measured lot times, ETA
    run_am.py estimate --run R               planned volume and time from measured lot times; launches nothing

Runs: artifacts/activity_match/<run>/; calibration: artifacts/activity_match/calibration[_pilot]/. A main run refuses
to prepare or extract until the protocol is frozen and the calibration check has passed. The only intervention on the
RETINO_ALIGN_V1 modules is `gain = gamma x dynamics.gain`. Exit codes: 0 done, 1 failure, 2 refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/retino_align"))
from run_study import lot_order, oriented_energy_predict, save_npz, sha256_arrays, sha256_file, write_json  # noqa: E402

CONFIG = ROOT / "configs/activity_match/protocol.yaml"
ARTS = ROOT / "artifacts/activity_match"
PARTS = ("train", "val", "test")
TASKS = ("orient", "lum")
# Bank namespaces (uid = namespace x 1e6 + position), disjoint from RETINO_ALIGN_V1's 1-7.
AM_BANKS = {"am_dev": 20, "am_pilot_train": 21, "am_pilot_val": 22, "am_pilot_test": 23, "am_devcal_fit": 24,
            "am_devcal_check": 25, "am_cal_fit": 26, "am_cal_check": 27, "am_train": 31, "am_val": 32,
            "am_test": 33,
            # ACTIVITY_MATCH_V2
            "am2_cal_fit": 41, "am2_cal_check": 42, "am2_devcal_fit": 43, "am2_devcal_check": 44,
            "am2_pilot_train": 45, "am2_pilot_val": 46, "am2_pilot_test": 47, "am2_train": 51, "am2_val": 52,
            "am2_test": 53,
            # ACTIVITY_MATCH_V2B (family S nulls)
            "am3_cal_fit": 61, "am3_cal_check": 62, "am3_devcal_fit": 63, "am3_devcal_check": 64,
            "am3_pilot_train": 65, "am3_pilot_val": 66, "am3_pilot_test": 67, "am3_train": 71, "am3_val": 72,
            "am3_test": 73}
CODE = ("src/flymario/retino_align/simulate.py", "src/flymario/retino_align/stimuli.py",
        "src/flymario/retino_align/decode.py", "src/flymario/activity_match/calibrate.py",
        "src/flymario/activity_match/analysis.py", "scripts/activity_match/run_am.py")


def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def register_banks() -> None:
    from flymario.retino_align import stimuli
    for k, v in AM_BANKS.items():
        if stimuli.BANKS.get(k, v) != v or (k not in stimuli.BANKS and v in stimuli.BANKS.values()):
            raise RuntimeError(f"bank namespace collision for {k}")
    stimuli.BANKS.update(AM_BANKS)


def code_record() -> dict:
    import subprocess

    import scipy
    import torch
    run = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    return {"utc": utc(), "commit": run("git rev-parse HEAD"), "branch": run("git rev-parse --abbrev-ref HEAD"),
            "dirty": bool(run("git status --porcelain -- src scripts configs")),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "code_sha256": {p: sha256_file(ROOT / p) for p in CODE}, "config_sha256": sha256_file(CONFIG)}


def check_config(cfg: dict, main: bool) -> list[str]:
    need = ["study.id", "study.final_reserve_utc", "graphs.null_seeds", "simulation.main_noise_seeds",
            "calibration.gamma_grid", "calibration.noise_seeds", "dataset.generator.amp_range", "dataset.banks.test",
            "decoder.subset_draw_seeds", "decoder.normalized_loss_lambda_grid", "analysis.practical_margin",
            "analysis.bootstrap_seed", "conditions"]
    problems = []
    for k in need:
        node = cfg
        for part in k.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node in (None, "", []):
            problems.append(f"undefined: {k}")
    if main:
        if cfg["study"].get("protocol_status") != "frozen":
            problems.append("protocol_status is not 'frozen'")
        if not cfg["calibration"].get("verification_passed"):
            problems.append("calibration.verification_passed is not true")
        if not cfg["calibration"].get("selected_gains_manifest"):
            problems.append("calibration.selected_gains_manifest is not set")
    return problems


def past_final_reserve(cfg: dict) -> bool:
    return datetime.now(UTC) >= datetime.fromisoformat(cfg["study"]["final_reserve_utc"].replace("Z", "+00:00"))


def tag(spec: str) -> str:
    return spec.replace(":", "")


NULL_KIND = {"nullin": "depth_stratified_lateral_instrength", "nulllat": "depth_stratified_lateral"}


def graph_specs(cfg: dict, pilot: bool) -> list[str]:
    """`graphs.null_prefix`: nullin (family E, default) or nulllat (family S, ACTIVITY_MATCH_V2B)."""
    pre = cfg["graphs"].get("null_prefix", "nullin")
    if pilot:
        return [f"{pre}:{s}" for s in cfg["graphs"]["pilot_null_seeds"]]
    return [cfg["graphs"]["connectome"]] + [f"{pre}:{s}" for s in cfg["graphs"]["null_seeds"]]


def graph_file(spec: str) -> Path:
    from flymario.connectome import datasets, nullcache
    if spec == "connectome":
        return datasets.cache_path(datasets.DEFAULT)
    pre, seed = spec.split(":")
    return nullcache.null_path(NULL_KIND[pre], datasets.DEFAULT, int(seed))


def gen_params(cfg: dict):
    from flymario.retino_align import stimuli
    return stimuli.GenParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["dataset"]["generator"].items()})


class Eye:
    """The connectome's eye and retina (the same in every graph: nulls rewire edges, not nodes)."""

    def __init__(self):
        from flymario.connectome import datasets
        from flymario.env.vision import RetinotopicVision
        from flymario.retino_align import retina
        self.g = datasets.load(datasets.DEFAULT)
        self.vis = RetinotopicVision(self.g)
        self.ret = retina.build(self.g, self.vis)
        self.readout = datasets.motor_index(self.g, datasets.DEFAULT)

    def encode(self, bank, gp, keep: int = 0):
        from flymario.retino_align import stimuli
        return stimuli.encode_bank(bank, gp, self.vis, self.ret.inj, self.ret.col_uv, keep_images=keep)


def base_params():
    from flymario import config
    from flymario.models.spiking_reservoir import LIFParams
    return LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))


def set_gamma(net, base, gamma: float) -> None:
    """The intervention: a fresh parameter object from the read-only base each time (never cumulative)."""
    net.p = dataclasses.replace(base, gain=float(base.gain) * float(gamma))


def make_net(spec: str, eye: Eye, cfg: dict):
    from flymario.connectome import graphsel
    from flymario.retino_align import simulate
    _g, W, motor = graphsel.load(spec)
    if not np.array_equal(motor, eye.readout):
        raise RuntimeError("readout differs between graphs")
    base = base_params()
    net = simulate.FrozenNet(W, eye.readout, eye.ret.inj, base, int(cfg["simulation"]["steps"]),
                             batch=int(cfg["simulation"]["batch"]))
    return net, base


def window_s(cfg: dict) -> float:
    return int(cfg["simulation"]["steps"]) * float(cfg["simulation"]["dt_ms"]) / 1000.0


def cal_dir(cfg: dict, pilot: bool) -> Path:
    """Calibration folder: `calibration.dir` (V2) or the V1 names."""
    base = cfg["calibration"].get("dir", "calibration")
    return ARTS / (base + "_pilot" if pilot else base)


def max_spikes(cfg: dict) -> int:
    from flymario.activity_match import calibrate
    return calibrate.max_count(int(cfg["simulation"]["steps"]), int(base_params().refractory_steps))


# --- calibration -------------------------------------------------------------------------------------------------
def cal_inputs(cfg: dict, eye: Eye, which: str, pilot: bool, out: Path) -> dict:
    """Encoded calibration bank WITHOUT its labels (they are dropped here, never saved, never passed on)."""
    from flymario.retino_align import stimuli
    p = out / f"inputs_{which}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    name, per, seed = cfg["calibration"][("pilot_" if pilot else "") + f"{which}_bank"]
    bank = stimuli.draw_bank(name, int(per), int(seed), gen_params(cfg))
    X, side = eye.encode(bank, gen_params(cfg))
    rec = {"X": X, "uid": bank.uid, "digest": side["digests"]}
    save_npz(p, **rec)
    return rec


def cal_evaluate(net, base, gamma: float, inp: dict, seeds: list[int], cfg: dict) -> dict:
    from flymario.activity_match import calibrate
    t = time.time()
    set_gamma(net, base, gamma)
    d0 = net.device_operator_digest()
    per_seed, finite, rate_all = [], True, []
    rules = cfg["calibration"]
    for s in seeds:
        o = net.run(inp["X"], inp["uid"], int(s))
        finite &= bool(np.isfinite(o.v_readout_mean).all() and np.isfinite(o.v_readout_sd).all()
                       and np.isfinite(o.rate_all).all())
        per_seed.append(calibrate.activity(o.counts, window_s(cfg), max_spikes(cfg),
                                           float(rules["saturation_threshold_fraction_of_max_spikes"])))
        rate_all.append(float(o.rate_all.mean()))
    if net.device_operator_digest() != d0 or d0 != net.digest:
        raise RuntimeError("operator changed during calibration")
    pt = calibrate.combine(per_seed, finite, rules)
    pt.update({"gamma": float(gamma), "gain": float(net.p.gain), "per_seed": per_seed,
               "rate_all_network_frac": float(np.mean(rate_all)), "seconds": round(time.time() - t, 2)})
    return pt


def cmd_calibrate(a) -> int:
    from flymario.activity_match import calibrate
    register_banks()
    cfg = load_config()
    c = cfg["calibration"]
    if c.get("use_labels"):
        print("refused: calibration.use_labels must be false")
        return 2
    out = cal_dir(cfg, a.pilot)
    out.mkdir(parents=True, exist_ok=True)
    graphs = graph_specs(cfg, a.pilot)
    seeds = c["pilot_noise_seeds"] if a.pilot else c["noise_seeds"]
    if c.get("mode", "common_range") == "match_reference":
        return calibrate_reference(a, cfg, out, graphs, seeds)
    eye = Eye()
    fit = cal_inputs(cfg, eye, "fit", a.pilot, out)
    chk = cal_inputs(cfg, eye, "check", a.pilot, out)
    if set(fit["uid"]) & set(chk["uid"]) or set(fit["digest"]) & set(chk["digest"]):
        raise RuntimeError("CAL_FIT and CAL_CHECK overlap")
    path = out / "calibration.json"
    rec = json.loads(path.read_text()) if path.exists() else {
        "study": cfg["study"]["id"], "pilot": bool(a.pilot), "started_utc": utc(), "env": code_record(),
        "graphs": graphs, "noise_seeds": seeds, "window_s": window_s(cfg), "max_spikes": max_spikes(cfg),
        "fit_uid_sha256": sha256_arrays(fit["uid"]), "check_uid_sha256": sha256_arrays(chk["uid"]),
        "labels_used": False, "grid": {}, "search": {}, "check": {}}

    def save():
        rec["updated_utc"] = utc()
        write_json(path, rec)

    grid = [float(x) for x in c["gamma_grid"]]
    # 1. grid on CAL_FIT, every graph
    for spec in graphs:
        if spec in rec["grid"] and all(str(gm) in rec["grid"][spec] for gm in grid):
            continue
        net, base = make_net(spec, eye, cfg)
        rec["grid"].setdefault(spec, {})
        for gm in grid:
            rec["grid"][spec][str(gm)] = cal_evaluate(net, base, gm, fit, seeds, cfg)
        save()
        print(utc(), "grid", spec, {g: round(p["rate_hz"], 2) for g, p in rec["grid"][spec].items()}, flush=True)
        del net
    curves = {s: {float(g): p for g, p in rec["grid"][s].items()} for s in graphs}
    cr = calibrate.common_range(curves)
    if not cr["exists"] and not rec.get("extension_done"):
        ext = [float(x) for x in c["grid_extension_if_no_common_range"]]
        rec["extension_reason"] = cr["reason"]
        for spec in graphs:
            net, base = make_net(spec, eye, cfg)
            for gm in ext:
                rec["grid"][spec][str(gm)] = cal_evaluate(net, base, gm, fit, seeds, cfg)
            del net
            save()
        rec["extension_done"] = True
        curves = {s: {float(g): p for g, p in rec["grid"][s].items()} for s in graphs}
        cr = calibrate.common_range(curves)
    rec["common_range"] = {k: v for k, v in cr.items() if k != "envelopes"}
    rec["envelopes"] = cr["envelopes"]
    if not cr["exists"]:
        save()
        print("no common range on the grid:", cr["reason"])
        return 1
    tg = calibrate.targets(cr["r_min"], cr["r_max"], c["target_positions_in_common_range"])
    rec["targets_hz"] = tg
    order = {k: tg[k] for k in c["search_order"]}
    save()
    # 2. bounded search per graph on CAL_FIT
    for spec in graphs:
        if spec in rec["search"]:
            continue
        net, base = make_net(spec, eye, cfg)
        pts = dict(curves[spec])
        log = calibrate.search(pts, order, lambda gm: cal_evaluate(net, base, gm, fit, seeds, cfg),
                               int(c["max_refinement_evaluations_per_graph"]),
                               int(c["refinement_evaluations_per_target"]), float(c["fit_tolerance"]))
        sel = {}
        for name, t in tg.items():
            gm, rule = calibrate.select(pts, t, float(c["fit_tolerance"]))
            sel[name] = {"gamma": gm, "rule": rule, "fit_rate_hz": pts[gm]["rate_hz"],
                         "fit_rel_err": calibrate.rel_err(pts[gm]["rate_hz"], t)}
        rec["search"][spec] = {"points": {str(g): p for g, p in sorted(pts.items())},
                               "evaluations": log.evaluations, "notes": log.notes, "selected": sel}
        save()
        print(utc(), "search", spec, {k: (v["gamma"], round(v["fit_rel_err"], 4)) for k, v in sel.items()},
              log.notes[-1], flush=True)
        del net
    # 3. CAL_CHECK with the selected gains, unchanged (and gamma = 1 for C0, descriptive)
    tol = float(c["relative_rate_tolerance"])
    for spec in graphs:
        if spec in rec["check"]:
            continue
        net, base = make_net(spec, eye, cfg)
        ck = {"C0": cal_evaluate(net, base, 1.0, chk, seeds, cfg)}
        for name, t in tg.items():
            gm = rec["search"][spec]["selected"][name]["gamma"]
            p = cal_evaluate(net, base, gm, chk, seeds, cfg)
            p["target_hz"] = t
            p["rel_err"] = calibrate.rel_err(p["rate_hz"], t)
            p["rel_err_by_seed"] = [calibrate.rel_err(r, t) for r in p["rate_hz_by_seed"]]
            p["pass"] = bool(p["valid"] and p["rel_err"] <= tol)
            ck[name] = p
        rec["check"][spec] = ck
        save()
        print(utc(), "check", spec, {k: round(v.get("rel_err", float("nan")), 4) for k, v in ck.items() if k != "C0"},
              flush=True)
        del net
    passed = {name: all(rec["check"][s][name]["pass"] for s in graphs) for name in tg}
    rec["check_passed"] = passed
    rec["complete_utc"] = utc()
    save()
    sel_path = out / "selection.json"
    write_json(sel_path, {"study": cfg["study"]["id"], "pilot": bool(a.pilot), "utc": utc(),
                          "calibration_sha256": sha256_file(path), "targets_hz": tg, "check_passed": passed,
                          "gammas": {s: {"C0": 1.0, **{k: rec["search"][s]["selected"][k]["gamma"] for k in tg}}
                                     for s in graphs}})
    print("targets", {k: round(v, 3) for k, v in tg.items()}, "check passed", passed)
    return 0


def calibrate_reference(a, cfg: dict, out: Path, graphs: list[str], seeds: list[int]) -> int:
    """ACTIVITY_MATCH_V2: every graph is calibrated to the reference graph's rate on CAL_FIT (reference at its fixed
    gamma); CAL_CHECK compares rates on the same independent images. No label is read."""
    from flymario.activity_match import calibrate
    c = cfg["calibration"]
    ref = c["pilot_reference" if a.pilot else "reference"]
    ref_spec, ref_gamma = ref["graph"], float(ref["gamma"])
    if ref_spec not in graphs:
        raise RuntimeError(f"reference {ref_spec} is not among the graphs")
    others = [g for g in graphs if g != ref_spec]
    eye = Eye()
    fit = cal_inputs(cfg, eye, "fit", a.pilot, out)
    chk = cal_inputs(cfg, eye, "check", a.pilot, out)
    if set(fit["uid"]) & set(chk["uid"]) or set(fit["digest"]) & set(chk["digest"]):
        raise RuntimeError("CAL_FIT and CAL_CHECK overlap")
    path = out / "calibration.json"
    rec = json.loads(path.read_text()) if path.exists() else {
        "study": cfg["study"]["id"], "mode": "match_reference", "pilot": bool(a.pilot), "started_utc": utc(),
        "env": code_record(), "graphs": graphs, "reference": {"graph": ref_spec, "gamma": ref_gamma},
        "noise_seeds": seeds, "window_s": window_s(cfg), "max_spikes": max_spikes(cfg),
        "fit_uid_sha256": sha256_arrays(fit["uid"]), "check_uid_sha256": sha256_arrays(chk["uid"]),
        "labels_used": False, "grid": {}, "search": {}, "check": {}}

    def save():
        rec["updated_utc"] = utc()
        write_json(path, rec)

    tol, fit_tol = float(c["relative_rate_tolerance"]), float(c["fit_tolerance"])
    # 1. the reference's rate on CAL_FIT and CAL_CHECK, at its fixed gamma
    if "reference_fit" not in rec:
        net, base = make_net(ref_spec, eye, cfg)
        rec["reference_fit"] = cal_evaluate(net, base, ref_gamma, fit, seeds, cfg)
        rec["reference_check"] = cal_evaluate(net, base, ref_gamma, chk, seeds, cfg)
        del net
        save()
    if not (rec["reference_fit"]["valid"] and rec["reference_check"]["valid"]):
        print("refused: the reference point is not valid", rec["reference_fit"]["invalid_reasons"])
        return 1
    target = float(rec["reference_fit"]["rate_hz"])
    rec["targets_hz"] = {"CN": target}
    save()
    print(utc(), "reference", ref_spec, "gamma", ref_gamma, "CAL_FIT", round(target, 3), "Hz; CAL_CHECK",
          round(rec["reference_check"]["rate_hz"], 3), "Hz", flush=True)
    grid = [float(x) for x in c["gamma_grid"]]
    for spec in others:
        if spec in rec["check"]:
            continue
        net, base = make_net(spec, eye, cfg)
        if spec not in rec["grid"]:
            rec["grid"][spec] = {str(gm): cal_evaluate(net, base, gm, fit, seeds, cfg) for gm in grid}
            save()
        pts = {float(g): p for g, p in rec["grid"][spec].items()}
        log = calibrate.search(pts, {"CN": target}, lambda gm: cal_evaluate(net, base, gm, fit, seeds, cfg),
                               int(c["max_refinement_evaluations_per_graph"]),
                               int(c["refinement_evaluations_per_target"]), fit_tol)
        gm, rule = calibrate.select(pts, target, fit_tol)
        rec["search"][spec] = {"points": {str(g): p for g, p in sorted(pts.items())}, "evaluations": log.evaluations,
                               "notes": log.notes, "selected": {"CN": {
                                   "gamma": gm, "rule": rule, "fit_rate_hz": pts[gm]["rate_hz"],
                                   "fit_rel_err": calibrate.rel_err(pts[gm]["rate_hz"], target)}}}
        p = cal_evaluate(net, base, gm, chk, seeds, cfg)
        r_ref = float(rec["reference_check"]["rate_hz"])
        p["reference_rate_hz"] = r_ref
        p["rel_err"] = calibrate.rel_err(p["rate_hz"], r_ref)
        p["rel_err_by_seed"] = [calibrate.rel_err(r, rr) for r, rr in
                                zip(p["rate_hz_by_seed"], rec["reference_check"]["rate_hz_by_seed"])]
        p["rel_err_vs_fit_target"] = calibrate.rel_err(p["rate_hz"], target)
        p["pass"] = bool(p["valid"] and p["rel_err"] <= tol)
        rec["check"][spec] = {"CN": p}
        save()
        print(utc(), spec, "gamma", gm, "fit err %.4f" % rec["search"][spec]["selected"]["CN"]["fit_rel_err"],
              "check err %.4f" % p["rel_err"], log.notes[-1], flush=True)
        del net
    passed = all(rec["check"][s]["CN"]["pass"] for s in others)
    rec["check_passed"] = {"CN": passed}
    rec["complete_utc"] = utc()
    save()
    gammas = {s: {"C0": 1.0, "CN": rec["search"][s]["selected"]["CN"]["gamma"]} for s in others}
    gammas[ref_spec] = {"C0": 1.0, "CN": ref_gamma}
    write_json(out / "selection.json", {"study": cfg["study"]["id"], "pilot": bool(a.pilot), "utc": utc(),
                                        "calibration_sha256": sha256_file(path), "targets_hz": {"CN": target},
                                        "reference": rec["reference"], "check_passed": {"CN": passed},
                                        "gammas": {s: gammas[s] for s in graphs}})
    print("target", round(target, 3), "Hz; check passed", passed)
    return 0


def cmd_scan(a) -> int:
    """Development, label-free: activity over a wide gamma range on a few graphs, with the DEV calibration bank
    (am_devcal_fit) and the pilot noise seeds, to decide a grid extension before the freeze."""
    register_banks()
    cfg = load_config()
    out = ARTS / "calibration_pilot"
    out.mkdir(parents=True, exist_ok=True)
    eye = Eye()
    fit = cal_inputs(cfg, eye, "fit", True, out)
    gammas = [float(x) for x in a.gammas.split(",")]
    path = out / "scan.json"
    rec = json.loads(path.read_text()) if path.exists() else {"utc": utc(), "bank": "am_devcal_fit",
                                                               "labels_used": False, "graphs": {}}
    for spec in a.graphs.split(","):
        net, base = make_net(spec, eye, cfg)
        rec["graphs"].setdefault(spec, {})
        for gm in gammas:
            if str(gm) in rec["graphs"][spec]:
                continue
            p = cal_evaluate(net, base, gm, fit, cfg["calibration"]["pilot_noise_seeds"], cfg)
            rec["graphs"][spec][str(gm)] = {k: v for k, v in p.items() if k != "per_seed"} | {
                "ge7_frac_mean": float(np.mean([d["ge7_frac_mean"] for d in p["per_seed"]])),
                "n_constant_features": int(np.mean([d["n_constant_features"] for d in p["per_seed"]]))}
        del net
        rec["updated_utc"] = utc()
        write_json(path, rec)
        print(spec, {g: round(v["rate_hz"], 1) for g, v in rec["graphs"][spec].items()}, flush=True)
    return 0


# --- controls ----------------------------------------------------------------------------------------------------
def cmd_controls(a) -> int:
    """Control checks 1-5 (6-8 are structural, tested in tests/unit/test_activity_match.py)."""
    import scipy.sparse as sp
    import torch

    from flymario.retino_align import simulate, stimuli
    register_banks()
    cfg = load_config()
    eye = Eye()
    rec = {"utc": utc(), "env": code_record()}
    # 1. gamma = 1 reproduces RETINO_ALIGN_V1 exactly: its saved features, connectome and nullin:0, ordered, seed 1
    v1 = ROOT / "artifacts/retino_align/main_v1"
    inp = np.load(v1 / "inputs/inputs.npz")
    idx = np.arange(256)
    rep = {}
    for spec in ("connectome", "nullin:0"):
        net, base = make_net(spec, eye, cfg)
        set_gamma(net, base, 2.0)
        _ = net.run(inp["X"][idx], inp["uid"][idx], 1)
        set_gamma(net, base, 1.0)
        gain1 = float(net.p.gain)
        o = net.run(inp["X"][idx], inp["uid"][idx], 1)
        ref = np.load(v1 / "features" / tag(spec) / "ordered_s1.npz")["counts"][idx]
        set_gamma(net, base, 2.0)
        o2 = net.run(inp["X"][idx], inp["uid"][idx], 1)
        rep[spec] = {"gamma1_equals_v1_features": bool(np.array_equal(o.counts, ref)),
                     "gain_at_gamma1": gain1, "v1_gain": float(base.gain),
                     "gamma2_differs": bool(not np.array_equal(o2.counts, ref)),
                     "gamma2_mean_count": float(o2.counts.mean()), "gamma1_mean_count": float(o.counts.mean()),
                     "operator_unchanged": net.device_operator_digest() == net.digest}
        del net
    rec["gamma1_reproduces_v1"] = rep
    # 2. the factor multiplies W @ s exactly once (analytic two-neuron case, no noise)
    base = dataclasses.replace(base_params(), noise_std=0.0)
    w = 1.0                                                                  # 0.369 at gamma 2: below threshold
    W = sp.csr_matrix(np.array([[0.0, 0.0], [w, 0.0]], np.float32))          # neuron 0 -> neuron 1
    tiny = simulate.FrozenNet(W, np.array([1]), np.array([0]), base, 2, batch=4)
    ana = {}
    for gm in (0.5, 1.0, 2.0, 1.0):
        set_gamma(tiny, base, gm)
        o = tiny.run(np.array([[5.0]], np.float32), np.array([0]), 0)
        ana[str(gm) if str(gm) not in ana else f"{gm}_again"] = {"v1_after_tick1": float(o.v_readout_mean[0]),
                                                                  "expected": w * float(base.gain) * gm}
    rec["gain_multiplies_recurrent_input_once"] = {
        "cases": ana, "pass": all(abs(v["v1_after_tick1"] - v["expected"]) < 1e-6 for v in ana.values())}
    # max spikes permitted by the coded refractoriness: drive one isolated neuron far above threshold
    lone = simulate.FrozenNet(sp.csr_matrix((1, 1), dtype=np.float32), np.array([0]), np.array([0]), base,
                              int(cfg["simulation"]["steps"]), batch=1)
    o = lone.run(np.array([[50.0]], np.float32), np.array([0]), 0)
    rec["max_spikes"] = {"simulated": int(o.counts.max()), "formula": max_spikes(cfg),
                         "refractory_steps": int(base.refractory_steps), "ticks": int(cfg["simulation"]["steps"])}
    # 3. weights intact, no plasticity: operator digests vs the file's operator, before and after runs (item 1)
    rec["no_plasticity"] = {"simulator_has_no_weight_update": True,
                            "note": "FrozenNet holds W as a read-only CSR tensor; the device digest equals the "
                                    "file operator's after every run (item 1, operator_unchanged)"}
    # 4. state reset: an image simulated alone equals the same image simulated after others (batch of 1)
    net, base = make_net("nullin:100", eye, cfg)
    set_gamma(net, base, 1.5)
    alone = net.run(inp["X"][[5]], inp["uid"][[5]], 1).counts
    in_batch = net.run(inp["X"][[7, 9, 5]], inp["uid"][[7, 9, 5]], 1).counts[2:]
    net.B = 1
    after = net.run(inp["X"][[7, 9, 5]], inp["uid"][[7, 9, 5]], 1).counts[2:]
    rec["reset_between_images"] = {"alone_equals_after_two_others_sequential": bool(np.array_equal(alone, after)),
                                   "alone_equals_same_batch": bool(np.array_equal(alone, in_batch))}
    rec["reset_between_images"]["identical"] = all(rec["reset_between_images"].values())
    del net
    # 5. the generator carries orientation (pixel-only oriented energy on the dev bank)
    gp = gen_params(cfg)
    name, per, seed = cfg["dataset"]["dev_bank"]
    bank = stimuli.draw_bank(name, int(per), int(seed), gp)
    pts = stimuli.sample_points(eye.ret.col_uv)
    pred = np.array([oriented_energy_predict(stimuli.render(bank, i, gp, pts, eye.vis.p.blur_sigma))
                     for i in range(len(bank))])
    acc = float((pred == bank.orient).mean())
    rec["positive_control_oriented_energy"] = {"bank": name, "n": len(bank), "accuracy": acc, "chance": 0.25,
                                               "pass": acc >= 0.9}
    torch.cuda.empty_cache()
    rec["pass"] = bool(all(v["gamma1_equals_v1_features"] and v["gamma2_differs"] and v["operator_unchanged"]
                           for v in rep.values()) and rec["gain_multiplies_recurrent_input_once"]["pass"]
                       and rec["max_spikes"]["simulated"] == rec["max_spikes"]["formula"]
                       and rec["reset_between_images"]["identical"] and rec["positive_control_oriented_energy"]["pass"])
    write_json(ARTS / "controls/controls.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "env"}, indent=1))
    return 0 if rec["pass"] else 1


# --- runs --------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, run: str):
        self.run = run
        self.dir = ARTS / run
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.cfg = self.manifest["config"]
        self.pilot = self.manifest["pilot"]
        self._inputs = None

    def inputs(self) -> dict:
        if self._inputs is None:
            z = np.load(self.dir / "inputs/inputs.npz")
            self._inputs = {k: z[k] for k in z.files}
        return self._inputs

    def feature_path(self, spec: str, cond: str, seed: int) -> Path:
        return self.dir / "features" / tag(spec) / f"{cond}_s{seed}.npz"

    def lots(self) -> list[tuple]:
        return [tuple(x) for x in self.manifest["lots"]]

    def gamma(self, spec: str, cond: str) -> float:
        return float(self.manifest["gammas"][spec][cond])


def cmd_prepare(a) -> int:
    from flymario.retino_align import decode, stimuli
    register_banks()
    cfg = load_config()
    problems = check_config(cfg, main=not a.pilot)
    if problems:
        print("refused:", problems)
        return 2
    d = ARTS / a.run
    if (d / "manifest.json").exists() and not a.force:
        print(f"{d}/manifest.json exists")
        return 0
    sel_path = (cal_dir(cfg, True) / "selection.json") if a.pilot else ROOT / cfg["calibration"]["selected_gains_manifest"]
    sel = json.loads(Path(sel_path).read_text())
    conds = cfg["conditions"]
    graphs = graph_specs(cfg, a.pilot)
    missing = [(s, c) for s in graphs for c in conds if c not in sel["gammas"].get(s, {})]
    if missing:
        print("refused: no gamma for", missing[:5])
        return 2
    t0 = time.time()
    eye = Eye()
    gp = gen_params(cfg)
    banks_cfg = cfg["dataset"]["pilot_banks" if a.pilot else "banks"]
    Xs, uids, part, orient, lum, digests, lat = [], [], [], [], [], [], {}
    for pi_, p in enumerate(PARTS):
        name, per, seed = banks_cfg[p]
        b = stimuli.draw_bank(name, int(per), int(seed), gp)
        X, side = eye.encode(b, gp)
        Xs.append(X), uids.append(b.uid), part.append(np.full(len(b), pi_, np.int8))
        orient.append(b.orient), lum.append(b.lum), digests.append(side["digests"])
        for k, v in b.latent.items():
            lat.setdefault(k, []).append(v)
    inputs = dict(X=np.concatenate(Xs), uid=np.concatenate(uids), part=np.concatenate(part),
                  orient=np.concatenate(orient), lum=np.concatenate(lum), digest=np.concatenate(digests),
                  inj=eye.ret.inj, readout=eye.readout, **{f"lat_{k}": np.concatenate(v) for k, v in lat.items()})
    h_inputs = save_npz(d / "inputs/inputs.npz", **inputs)
    tr = inputs["part"] == 0
    combo = inputs["orient"][tr].astype(int) * 4 + inputs["lum"][tr]
    subs = decode.subsets(combo, cfg["decoder"]["subset_draw_seeds"], tuple(cfg["decoder"]["per_combination"]))
    save_npz(d / "inputs/subsets.npz", subsets=subs)
    seeds = cfg["simulation"]["pilot_noise_seeds" if a.pilot else "main_noise_seeds"]
    lots = lot_order(graphs, conds, seeds)
    manifest = {"run": a.run, "pilot": bool(a.pilot), "created_utc": utc(), "config": cfg, "env": code_record(),
                "graphs": graphs, "graph_sha256": {s: sha256_file(graph_file(s)) for s in graphs},
                "conditions": conds, "noise_seeds": seeds, "lots": lots,
                "gammas": {s: {c: float(sel["gammas"][s][c]) for c in conds} for s in graphs},
                "targets_hz": sel["targets_hz"], "reference": sel.get("reference"),
                "selection_file": str(Path(sel_path).relative_to(ROOT)),
                "selection_sha256": sha256_file(Path(sel_path)), "base_gain": float(base_params().gain),
                "n_images": {p: int((inputs["part"] == i).sum()) for i, p in enumerate(PARTS)},
                "inputs_sha256": h_inputs, "readout_n": int(len(eye.readout)),
                "readout_node_index_sha256": sha256_arrays(eye.readout.astype(np.int64)),
                "readout_body_ids_sha256": sha256_arrays(eye.g.node_ids[eye.readout].astype(np.int64)),
                "retina_digest": eye.ret.digest(), "generator": gp.as_dict(), "banks": banks_cfg,
                "prepare_s": round(time.time() - t0, 1)}
    write_json(d / "manifest.json", manifest)
    vol = len(inputs["X"]) * len(eye.readout) * len(lots)
    print(f"prepared {a.run}: {len(graphs)} graphs x {len(conds)} conditions x {len(seeds)} seeds = {len(lots)} lots; "
          f"features {vol / 1e9:.2f} GB as uint8")
    return 0


def cmd_validate(a) -> int:
    """Data checks on a prepared run, plus disjointness from RETINO_ALIGN_V1 and the calibration banks."""
    ctx = Ctx(a.run)
    inp = ctx.inputs()
    ck = {}
    part = inp["part"]
    for pi_, p in enumerate(PARTS):
        m = part == pi_
        combo = inp["orient"][m].astype(int) * 4 + inp["lum"][m]
        cnt = np.bincount(combo, minlength=16)
        ck[f"{p}_balanced_16_cells"] = bool((cnt == cnt[0]).all())
    ck["uids_unique"] = bool(len(np.unique(inp["uid"])) == len(inp["uid"]))
    ck["image_digests_unique"] = bool(len(np.unique(inp["digest"])) == len(inp["digest"]))
    ck["latent_noise_seeds_unique"] = bool(len(np.unique(inp["lat_noise_seed"])) == len(inp["lat_noise_seed"]))
    others = {"retino_v1_main": ROOT / "artifacts/retino_align/main_v1/inputs/inputs.npz",
              "retino_v1_pilot": ROOT / "artifacts/retino_align/pilot_e2e/inputs/inputs.npz"}
    for cal in sorted(p.name for p in ARTS.glob("calibration*") if p.is_dir()):
        for w in ("fit", "check"):
            others[f"{cal}_{w}"] = ARTS / cal / f"inputs_{w}.npz"
    for run in sorted(p.parent.parent.name for p in ARTS.glob("*/inputs/inputs.npz")):
        if run != a.run:
            others[f"activity_match_{run}"] = ARTS / run / "inputs/inputs.npz"
    for name, p in others.items():
        if p.exists():
            z = np.load(p)
            ck[f"disjoint_from_{name}"] = bool(not (set(z["uid"]) & set(inp["uid"]))
                                               and not (set(z["digest"]) & set(inp["digest"])))
    # labels: independence by construction (balanced cells); order carries no label (uid position vs label)
    pos = inp["uid"] % 1_000_000
    for t in TASKS:
        r = np.corrcoef(pos, inp[t])[0, 1]
        ck[f"uid_position_uncorrelated_with_{t}"] = bool(abs(r) < 0.1)
    ck["orient_lum_independent_within_each_partition"] = all(ck[f"{p}_balanced_16_cells"] for p in PARTS)
    ck["pass"] = all(v for k, v in ck.items())
    write_json(ctx.dir / "validation/data_checks.json", {"utc": utc(), **ck})
    print(json.dumps(ck, indent=1))
    return 0 if ck["pass"] else 1


def lot_done(ctx: Ctx, lot) -> bool:
    j = ctx.feature_path(*lot).with_suffix(".json")
    if not j.exists() or not ctx.feature_path(*lot).exists():
        return False
    r = json.loads(j.read_text())
    return bool(r.get("complete")) and abs(r.get("gamma", -1) - ctx.gamma(lot[0], lot[1])) < 1e-12


def write_status(ctx: Ctx, extra: dict | None = None) -> dict:
    lots = ctx.lots()
    done = [l for l in lots if lot_done(ctx, l)]
    secs = [json.loads(ctx.feature_path(*l).with_suffix(".json").read_text())["seconds"] for l in done]
    dec = sorted({(g, c) for g, c, _ in lots})
    n_dec = sum((ctx.dir / "predictions" / tag(g) / f"{c}.npz").exists() for g, c in dec)
    st = {"utc": utc(), "lots_expected": len(lots), "lots_done": len(done),
          "lot_seconds_median": float(np.median(secs)) if secs else None,
          "eta_extract_s": round(float(np.median(secs)) * (len(lots) - len(done))) if secs else None,
          "decoders_expected": len(dec), "decoders_done": int(n_dec), **(extra or {})}
    write_json(ctx.dir / "status.json", st)
    return st


def lot_activity(C: np.ndarray, part: np.ndarray, cfg: dict) -> dict:
    from flymario.activity_match import calibrate
    ws, cm = window_s(cfg), max_spikes(cfg)
    sf = float(cfg["calibration"]["saturation_threshold_fraction_of_max_spikes"])
    out = {"all": {k: v for k, v in calibrate.activity(C, ws, cm, sf).items()}}
    for pi_, p in enumerate(PARTS):
        a_ = calibrate.activity(C[part == pi_], ws, cm, sf)
        out[p] = {k: a_[k] for k in ("rate_hz", "active_frac_mean", "saturated_frac_mean", "n_constant_features")}
    return out


def cmd_extract(a) -> int:
    ctx = Ctx(a.run)
    cfg = ctx.cfg
    live = load_config()
    if not ctx.pilot:
        problems = check_config(live, main=True)
        if problems:
            print("refused:", problems)
            return 2
        if sha256_file(CONFIG) != ctx.manifest["env"]["config_sha256"]:
            print("refused: configs/activity_match/protocol.yaml changed since prepare (a new protocol version?)")
            return 2
    eye = Eye()
    inp = ctx.inputs()
    if not np.array_equal(inp["readout"], eye.readout) or not np.array_equal(inp["inj"], eye.ret.inj):
        raise RuntimeError("readout or injection differs from the manifest's")
    lots = [l for l in ctx.lots() if not lot_done(ctx, l)]
    if a.limit:
        lots = lots[: a.limit]
    order = [l[0] for l in ctx.lots()]
    by_graph: dict[str, list] = {}
    for l in lots:
        by_graph.setdefault(l[0], []).append(l)
    fpath = ctx.dir / "logs/failures.json"
    failures = json.loads(fpath.read_text()) if fpath.exists() else []
    X = inp["X"]
    for spec in sorted(by_graph, key=order.index):
        if not ctx.pilot and past_final_reserve(cfg):
            print("final reserve reached: no new lot started")
            write_status(ctx, {"stopped": "final_reserve"})
            return 2
        t_load = time.time()
        g_sha = sha256_file(graph_file(spec))
        if g_sha != ctx.manifest["graph_sha256"][spec]:
            raise RuntimeError(f"{spec}: graph file differs from the manifest's")
        net, base = make_net(spec, eye, cfg)
        t_load = time.time() - t_load
        for lot in by_graph[spec]:
            _, cond, seed = lot
            gm = ctx.gamma(spec, cond)
            for attempt in range(1, int(cfg["execution"]["technical_retry"]["max_attempts"]) + 1):
                try:
                    import torch
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    set_gamma(net, base, gm)
                    d0 = net.device_operator_digest()
                    o = net.run(X, inp["uid"], int(seed))
                    if net.device_operator_digest() != d0 or d0 != net.digest:
                        raise RuntimeError("operator changed during the lot")
                    finite = bool(np.isfinite(o.v_readout_mean).all() and np.isfinite(o.rate_all).all())
                    secs = time.time() - t0
                    path = ctx.feature_path(*lot)
                    h = save_npz(path, counts=o.counts, uid=inp["uid"], part=inp["part"], rate_all=o.rate_all,
                                 rate_input=o.rate_input, v_readout_mean=o.v_readout_mean, v_readout_sd=o.v_readout_sd)
                    rec = {"lot": list(lot), "complete": True, "utc": utc(), "seconds": round(secs, 2),
                           "graph_load_s": round(t_load, 2), "attempt": attempt, "sha256": h, "gamma": gm,
                           "gain": float(net.p.gain), "noise_seed": int(seed), "finite": finite,
                           "graph_file": graph_file(spec).name, "graph_sha256": g_sha, "operator_digest": net.digest,
                           "inputs_sha256": ctx.manifest["inputs_sha256"],
                           "readout_node_index_sha256": ctx.manifest["readout_node_index_sha256"],
                           "config_sha256": ctx.manifest["env"]["config_sha256"],
                           "code_sha256": ctx.manifest["env"]["code_sha256"],
                           "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                           "activity": lot_activity(o.counts, inp["part"], cfg)}
                    write_json(path.with_suffix(".json"), rec)
                    break
                except (RuntimeError, OSError) as e:
                    failures.append({"lot": list(lot), "utc": utc(), "attempt": attempt, "error": repr(e),
                                     "trace": traceback.format_exc()[-2000:]})
                    write_json(fpath, failures)
                    if "operator changed" in repr(e) or "differs" in repr(e):
                        raise
                    import torch
                    torch.cuda.empty_cache()
            st = write_status(ctx)
            print(f"{utc()} {lot} gamma={gm:.4f} done={st['lots_done']}/{st['lots_expected']} "
                  f"eta={st['eta_extract_s']}s", flush=True)
        del net
    write_status(ctx)
    return 0


# --- decode ------------------------------------------------------------------------------------------------------
def fit_predict(F: np.ndarray, y: dict, part: np.ndarray, subs: np.ndarray, lams: list[float],
                shuffle: bool = False) -> dict:
    """The study's decoder for one feature matrix: per (draw, budget) one standardisation + SVD of the training subset
    (both tasks), lambda chosen on validation, predictions on test. Reads test labels nowhere."""
    from flymario.retino_align import decode
    tr, va, te = part == 0, part == 1, part == 2
    R, K = subs.shape[:2]
    T, d, n_te = len(TASKS), F.shape[1], int(te.sum())
    out = {"W": np.zeros((R, K, T, d, 4), np.float32), "b": np.zeros((R, K, T, 4), np.float32),
           "val_acc": np.zeros((R, K, T, len(lams)), np.float32), "lam_idx": np.zeros((R, K, T), np.int8),
           "pred": np.zeros((R, K, T, n_te), np.uint8), "margin": np.zeros((R, K, T, n_te), np.float16)}
    Ftr, Fva, Fte = F[tr], F[va], F[te]
    for r in range(R):
        for k in range(K):
            sel = subs[r, k][subs[r, k] >= 0]
            ys = []
            for ti, t in enumerate(TASKS):
                ytr = y[t][tr][sel]
                if shuffle:
                    ytr = ytr[np.random.default_rng(777 + 100 * r + k + 10_000 * ti).permutation(len(ytr))]
                ys.append(ytr)
            fits = decode.fit_tasks(Ftr[sel], ys, lams)
            Zva = (Fva - fits[0].mu) / fits[0].sd
            Zte = (Fte - fits[0].mu) / fits[0].sd
            for ti, t in enumerate(TASKS):
                f = fits[ti]
                va_a = ((np.einsum("md,ldk->lmk", Zva, f.W) + f.b).argmax(2) == y[t][va]).mean(1)
                j = decode.select(va_a, lams)
                ste = Zte @ f.W[j] + f.b
                srt = np.sort(ste, 1)
                out["W"][r, k, ti], out["b"][r, k, ti] = f.W[j], f.b
                out["val_acc"][r, k, ti], out["lam_idx"][r, k, ti] = va_a, j
                out["pred"][r, k, ti], out["margin"][r, k, ti] = ste.argmax(1), srt[:, -1] - srt[:, -2]
    return out


def decode_one(run: str, spec: str, cond: str, shuffle: bool = False) -> str:
    ctx = Ctx(run)
    cfg = ctx.cfg
    inp = ctx.inputs()
    seeds = ctx.manifest["noise_seeds"]
    sfx = "_shuffled" if shuffle else ""
    out_dec = ctx.dir / "decoders" / tag(spec) / f"{cond}{sfx}.npz"
    out_pred = ctx.dir / "predictions" / tag(spec) / f"{cond}{sfx}.npz"
    if out_dec.exists() and out_pred.exists():
        return "exists"
    subs = np.load(ctx.dir / "inputs/subsets.npz")["subsets"]
    part = inp["part"]
    te = part == 2
    y = {"orient": inp["orient"].astype(int), "lum": inp["lum"].astype(int)}
    lams = cfg["decoder"]["normalized_loss_lambda_grid"]
    t0 = time.time()
    res, lot_sha = [], {}
    for seed in seeds:
        fp = ctx.feature_path(spec, cond, seed)
        lot_sha[str(seed)] = json.loads(fp.with_suffix(".json").read_text())["sha256"]
        res.append(fit_predict(np.load(fp)["counts"].astype(np.float64), y, part, subs, lams, shuffle))
    st = {k: np.stack([r_[k] for r_ in res]) for k in res[0]}                 # (S, R, K, T, ...)
    conf = np.zeros(st["pred"].shape[:4] + (4, 4), np.int32)
    for idx in np.ndindex(*st["pred"].shape[:4]):
        np.add.at(conf[idx], (y[TASKS[idx[3]]][te], st["pred"][idx]), 1)
    h1 = save_npz(out_dec, W=st["W"], b=st["b"], val_acc=st["val_acc"], lam_idx=st["lam_idx"], lambdas=np.array(lams),
                  seeds=np.array(seeds), budgets_per_class=4 * np.array(cfg["decoder"]["per_combination"]),
                  tasks=np.array(TASKS))
    h2 = save_npz(out_pred, pred=st["pred"], margin=st["margin"], conf=conf, uid_test=inp["uid"][te],
                  y_orient=y["orient"][te], y_lum=y["lum"][te])
    acc = (st["pred"] == np.stack([y["orient"][te], y["lum"][te]])[None, None, None, :, :]).mean(-1)
    write_json(ctx.dir / "metrics" / tag(spec) / f"{cond}{sfx}.json",
               {"graph": spec, "condition": cond, "gamma": ctx.gamma(spec, cond), "shuffled_labels": shuffle,
                "utc": utc(), "seconds": round(time.time() - t0, 1), "decoders_sha256": h1, "predictions_sha256": h2,
                "lots_sha256": lot_sha, "acc_by_seed_draw_budget_task": acc.tolist(),
                "M": {t: float(acc[..., ti].mean()) for ti, t in enumerate(TASKS)},
                "lambda_chosen_counts": {t: np.bincount(st["lam_idx"][..., ti].ravel(), minlength=len(lams)).tolist()
                                         for ti, t in enumerate(TASKS)}})
    return "done"


def cmd_decode(a) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = Ctx(a.run)
    pairs = sorted({(g, c) for g, c, _ in ctx.lots()},
                   key=lambda x: (ctx.manifest["graphs"].index(x[0]), ctx.manifest["conditions"].index(x[1])))
    jobs = [(g, c, False) for g, c in pairs
            if all(lot_done(ctx, (g, c, s)) for s in ctx.manifest["noise_seeds"])
            and not (ctx.dir / "predictions" / tag(g) / f"{c}.npz").exists()]
    if a.shuffled_control:
        jobs.append((ctx.manifest["graphs"][0], "C0", True))
    print(f"{len(jobs)} decoder jobs", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(decode_one, a.run, g, c, s): (g, c, s) for g, c, s in jobs}
        for f in as_completed(futs):
            print(utc(), futs[f], f.result(), flush=True)
    write_status(ctx)
    return 0


# --- analyse -----------------------------------------------------------------------------------------------------
def cmd_analyse(a) -> int:
    from flymario.activity_match import analysis
    from flymario.activity_match import calibrate as cal
    ctx = Ctx(a.run)
    cfg = ctx.cfg
    graphs, conds = ctx.manifest["graphs"], ctx.manifest["conditions"]
    seeds = ctx.manifest["noise_seeds"]
    inp = ctx.inputs()
    te = inp["part"] == 2
    strata = (inp["orient"].astype(int) * 4 + inp["lum"])[te]
    y = np.stack([inp["orient"][te], inp["lum"][te]]).astype(np.uint8)
    n = int(te.sum())
    G, C, S = len(graphs), len(conds), len(seeds)
    R_, K_ = len(cfg["decoder"]["subset_draw_seeds"]), len(cfg["decoder"]["per_combination"])
    corr = np.full((G, C, 2, S, R_, K_, n), np.nan, np.float32)
    lam = np.full((G, C, S, R_, K_, 2), -1, np.int8)
    missing = []
    for gi, g in enumerate(graphs):
        for ci, c in enumerate(conds):
            p = ctx.dir / "predictions" / tag(g) / f"{c}.npz"
            if not p.exists():
                missing.append([g, c])
                continue
            pr = np.load(p)["pred"]                                          # (S, R, K, T, n)
            corr[gi, ci] = np.moveaxis(pr == y[None, None, None, :, :], 3, 0)
            lam[gi, ci] = np.load(ctx.dir / "decoders" / tag(g) / f"{c}.npz")["lam_idx"]
    res = {"run": ctx.run, "utc": utc(), "pilot": ctx.pilot, "missing_graph_conditions": missing,
           "conditions": conds, "graphs": graphs}
    if missing:
        write_json(ctx.dir / "metrics/analysis.json", res)
        print("incomplete:", missing[:5])
        return 1
    cbar = corr.mean(axis=(3, 4, 5))                                         # (G, C, T, n)
    M = cbar.mean(-1)
    res["M"] = {g: {c: {"orient": float(M[gi, ci, 0]), "lum": float(M[gi, ci, 1])} for ci, c in enumerate(conds)}
                for gi, g in enumerate(graphs)}
    m = cfg["analysis"].get("matched_condition", "CM")
    res["point"] = analysis.contrasts(M, conds, matched=m)
    t0 = time.time()
    bs = analysis.bootstrap(cbar, strata, conds, int(a.replicates or cfg["analysis"]["bootstrap_replicates"]),
                            int(cfg["analysis"]["bootstrap_seed"]), matched=m)
    iv = analysis.intervals(bs, float(cfg["analysis"]["confidence_level"]))
    delta = float(cfg["analysis"]["practical_margin"])
    res["bootstrap"] = {"replicates": len(next(iter(bs.values()))), "seed": int(cfg["analysis"]["bootstrap_seed"]),
                        "seconds": round(time.time() - t0, 1), "method": "percentile", "intervals_95": iv,
                        "scope": "conditional on the train/validation banks, the calibrated gains, the 41 graphs, the "
                                 "fitted decoders and the two noise seeds; test images resampled within the 16 label "
                                 "cells (paired across graphs, conditions, seeds, budgets, draws and tasks), nulls "
                                 "resampled as whole graphs with all their conditions"}
    if m in conds:
        lo, hi = iv[f"G_{m}_orient"]
        res[f"reading_G_{m}_orient"] = analysis.reading_G(lo, hi, delta)
    # activity on the test images of every lot: matching transport, per graph and condition
    act = {}
    for g in graphs:
        act[g] = {}
        for c in conds:
            recs = [json.loads(ctx.feature_path(g, c, s).with_suffix(".json").read_text())["activity"] for s in seeds]
            act[g][c] = {k: float(np.mean([r_["test"][k] for r_ in recs])) for k in
                         ("rate_hz", "active_frac_mean", "saturated_frac_mean", "n_constant_features")}
            act[g][c]["rate_hz_by_seed"] = [r_["test"]["rate_hz"] for r_ in recs]
            if c in ctx.manifest["targets_hz"]:
                t = ctx.manifest["targets_hz"][c]
                act[g][c]["target_hz"] = t
                act[g][c]["rel_err_test"] = cal.rel_err(act[g][c]["rate_hz"], t)
    ref = ctx.manifest.get("reference")
    if ref:                      # V2: matching is judged against the reference graph's rate on the same test images
        for c in conds:
            if c in ctx.manifest["targets_hz"]:
                r_ref = act[ref["graph"]][c]["rate_hz"]
                for g in graphs:
                    act[g][c]["fit_target_hz"] = act[g][c]["target_hz"]
                    act[g][c]["target_hz"] = r_ref
                    act[g][c]["rel_err_test"] = cal.rel_err(act[g][c]["rate_hz"], r_ref)
    tol = float(cfg["calibration"]["relative_rate_tolerance"])
    res["activity_test"] = act
    res["transport"] = {c: {"max_rel_err": max(act[g][c]["rel_err_test"] for g in graphs),
                            "all_within_tolerance": all(act[g][c]["rel_err_test"] <= tol for g in graphs)}
                        for c in conds if c in ctx.manifest["targets_hz"]}
    if m == "CM" and "CM" in conds and "C0" in conds:
        matching_valid = bool(res["transport"]["CM"]["all_within_tolerance"])
        res["strong_interpretation_criteria"] = analysis.strong_criteria(
            iv, delta, float(cfg["analysis"]["informative_accuracy_threshold"]), matching_valid)
    if cfg["analysis"].get("interpretation") == "v2" and m in conds and "C0" in conds:
        matching_valid = bool(res["transport"][m]["all_within_tolerance"])
        gi0 = graphs.index(ref["graph"])
        t_zero = bool(np.array_equal(corr[gi0, conds.index("C0")], corr[gi0, conds.index(m)]))
        res["interpretation_criteria_v2"] = analysis.criteria_v2(
            iv, delta, float(cfg["analysis"]["informative_accuracy_threshold"]), matching_valid, t_zero)
    # descriptive rank of the connectome per condition (no p-value)
    res["rank_orient_descriptive"] = {c: int((M[1:, ci, 0] > M[0, ci, 0]).sum()) + 1 for ci, c in enumerate(conds)}
    # secondary sensitivity: per seed, per budget, per draw (descriptive)
    sens = {"per_seed": {}, "per_budget": {}, f"per_draw_G_{m}_orient": []}
    for si, s in enumerate(seeds):
        sens["per_seed"][str(s)] = analysis.contrasts(corr[:, :, :, si].mean(axis=(3, 4, 5)), conds, matched=m)
    bpc = 4 * np.array(cfg["decoder"]["per_combination"])
    for k in range(K_):
        sens["per_budget"][str(int(bpc[k]))] = analysis.contrasts(corr[:, :, :, :, :, k].mean(axis=(3, 4, 5)), conds,
                                                                  matched=m)
    if m in conds:
        for r in range(R_):
            sens[f"per_draw_G_{m}_orient"].append(analysis.contrasts(corr[:, :, :, :, r].mean(axis=(3, 4, 5)),
                                                                     conds, matched=m)[f"G_{m}_orient"])
    res["sensitivity_secondary"] = sens
    res["curves"] = corr.mean(axis=(3, 4, 6)).tolist()                       # (G, C, T, K)
    res["lambda_counts"] = {c: {t: np.bincount(lam[:, ci, ..., ti].ravel(), minlength=6).tolist()
                                for ti, t in enumerate(TASKS)} for ci, c in enumerate(conds)}
    np.savez(ctx.dir / "metrics/bootstrap.npz", **bs)
    np.savez(ctx.dir / "metrics/cbar.npz", cbar=cbar, graphs=np.array(graphs), conds=np.array(conds), strata=strata)
    write_json(ctx.dir / "metrics/analysis.json", res)
    # score table
    lams = cfg["decoder"]["normalized_loss_lambda_grid"]
    acc = corr.mean(-1)                                                      # (G, C, T, S, R, K)
    with open(ctx.dir / "metrics/scores.csv.part", "w") as f:
        f.write("experiment_id,graph_id,graph_role,condition,gamma,target_rate,noise_seed,task,train_budget,"
                "subset_id,lambda,test_n,accuracy,output_rate_hz,active_fraction,saturation_fraction\n")
        for gi, g in enumerate(graphs):
            role = "connectome" if g == "connectome" else "null"
            for ci, c in enumerate(conds):
                tgt = ctx.manifest["targets_hz"].get(c, "")
                for si, s in enumerate(seeds):
                    ar = json.loads(ctx.feature_path(g, c, s).with_suffix(".json").read_text())["activity"]["test"]
                    for ti, t in enumerate(TASKS):
                        for r in range(R_):
                            for k in range(K_):
                                f.write(f"{cfg['study']['id']},{g},{role},{c},{ctx.gamma(g, c):.6f},{tgt},{s},{t},"
                                        f"{int(bpc[k])},{r},{lams[lam[gi, ci, si, r, k, ti]]},{n},"
                                        f"{acc[gi, ci, ti, si, r, k]:.6f},{ar['rate_hz']:.4f},"
                                        f"{ar['active_frac_mean']:.5f},{ar['saturated_frac_mean']:.6f}\n")
    os.replace(ctx.dir / "metrics/scores.csv.part", ctx.dir / "metrics/scores.csv")
    p = res["point"]
    print({k: round(p[k], 4) for k in p if k.startswith(("G_", "R_", "T_"))})
    if m in conds:
        print(f"G_{m}_orient", iv[f"G_{m}_orient"], res[f"reading_G_{m}_orient"])
    return 0


def cmd_status(a) -> int:
    print(json.dumps(write_status(Ctx(a.run)), indent=1))
    return 0


def cmd_estimate(a) -> int:
    """Volume and time of a prepared run from its own measured lots (or the RETINO_ALIGN_V1 lots if none yet)."""
    ctx = Ctx(a.run)
    lots = ctx.lots()
    n_img = sum(ctx.manifest["n_images"].values())
    done = [l for l in lots if lot_done(ctx, l)]
    if done:
        secs = [json.loads(ctx.feature_path(*l).with_suffix(".json").read_text())["seconds"] for l in done]
        src = f"{len(done)} measured lots of this run"
    else:
        import glob
        secs = [json.loads(Path(p).read_text())["seconds"] for p in
                glob.glob(str(ROOT / "artifacts/retino_align/main_v1/features/*/*.json"))]
        src = "RETINO_ALIGN_V1 main lots (same simulator, 4,096 images)"
    per = float(np.percentile(secs, 90))
    n_graphs = len(ctx.manifest["graphs"])
    est = {"lots": len(lots), "lots_left": len(lots) - len(done), "images_per_lot": n_img,
           "features_gb_uint8": n_img * ctx.manifest["readout_n"] * len(lots) / 1e9,
           "lot_seconds_p90": per, "source": src,
           "extract_hours_p90": (per * (len(lots) - len(done)) + 6.0 * n_graphs) / 3600}
    print(json.dumps(est, indent=1))
    return 0


def main() -> int:
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG.relative_to(ROOT)),
                    help="protocol configuration (V1 default; V2: configs/activity_match/protocol_v2.yaml; V2B: protocol_v2b.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("controls")
    sc = sub.add_parser("scan")
    sc.add_argument("--graphs", default="connectome,nullin:100,nullin:101")
    sc.add_argument("--gammas", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,6,8,12,16")
    c = sub.add_parser("calibrate")
    c.add_argument("--pilot", action="store_true")
    for name in ("prepare", "validate", "extract", "decode", "analyse", "status", "estimate"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name == "prepare":
            s.add_argument("--pilot", action="store_true")
            s.add_argument("--force", action="store_true")
        if name == "extract":
            s.add_argument("--limit", type=int, default=0)
        if name == "decode":
            s.add_argument("--workers", type=int, default=3)
            s.add_argument("--shuffled-control", action="store_true")
        if name == "analyse":
            s.add_argument("--replicates", type=int, default=0)
    a = ap.parse_args()
    CONFIG = (ROOT / a.config).resolve()
    return {"controls": cmd_controls, "scan": cmd_scan, "calibrate": cmd_calibrate, "prepare": cmd_prepare, "validate": cmd_validate,
            "extract": cmd_extract, "decode": cmd_decode, "analyse": cmd_analyse, "status": cmd_status,
            "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
