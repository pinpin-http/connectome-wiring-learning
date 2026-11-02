#!/usr/bin/env python
"""RETINO_ALIGN_V1 orchestrator (docs/retino_align/PROTOCOL_FROZEN.md): resumable, one status file, atomic outputs.

    run_study.py prepare  --run R [--pilot]   datasets, permutations, manifests, lot plan; prints the planned volume
    run_study.py validate --run R             graph invariants, permutation tests, dataset checks, input controls
    run_study.py extract  --run R             simulation lots (graph x condition x dynamic seed), resumable
    run_study.py decode   --run R [--workers N] ridge decoders per graph and condition, resumable
    run_study.py analyse  --run R             M, L, D, G, S, bootstrap, sensitivity tables
    run_study.py status   --run R             lots expected / done / failed / missing, decoders, ETA
    run_study.py resume   --run R             extract + decode + analyse, continuing where it stopped

A run is artifacts/retino_align/<run>/. `--pilot` runs use the reserved graphs (nullin:100, 101) and the pilot
banks; a main run refuses to extract while configs/retino_align/protocol.yaml is not frozen. Exit codes: 0 done,
1 failure, 2 refused (protocol not frozen, final reserve reached, or a precondition missing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "configs/retino_align/protocol.yaml"
ARTS = ROOT / "artifacts/retino_align"
PARTS = ("train", "val", "test")
TASKS = ("orient", "lum")


# --- small utilities ---------------------------------------------------------------------------------------------
def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, p)


def save_npz(p: Path, **arrays) -> str:
    """Atomic: written as <name>.part, then renamed. Returns the file's sha256."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, p)
    return sha256_file(p)


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def check_config(cfg: dict, main: bool) -> list[str]:
    """Refuse the main launch while an essential field is undefined or the protocol is a draft."""
    need = ["study.id", "study.final_reserve_utc", "graphs.null_seeds", "retina.permutation_seeds",
            "dataset.generator.amp_range", "dataset.banks.train", "dataset.banks.val", "dataset.banks.test",
            "simulation.steps", "simulation.dynamic_seeds", "decoder.subset_draw_seeds",
            "decoder.normalized_loss_lambda_grid", "analysis.practical_margin", "analysis.bootstrap_seed"]
    problems = []
    for k in need:
        node = cfg
        for part in k.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node in (None, "", [], "discover_and_verify"):
            problems.append(f"undefined: {k}")
    if main and cfg["study"].get("protocol_status") != "frozen":
        problems.append("protocol_status is not 'frozen'")
    return problems


def past_final_reserve(cfg: dict) -> bool:
    return datetime.now(UTC) >= datetime.fromisoformat(cfg["study"]["final_reserve_utc"].replace("Z", "+00:00"))


def run_dir(run: str) -> Path:
    return ARTS / run


def graph_specs(cfg: dict, pilot: bool) -> list[str]:
    if pilot:
        return [f"nullin:{s}" for s in cfg["graphs"]["pilot_null_seeds"]]
    return [cfg["graphs"]["connectome"]] + [f"nullin:{s}" for s in cfg["graphs"]["null_seeds"]]


def tag(spec: str) -> str:
    return spec.replace(":", "")


def lot_order(graphs: list[str], conds: list[str], seeds: list[int], order_seed: int = 20261019) -> list[tuple]:
    """Balanced, performance-independent order: graphs in a fixed random order; within a graph, the condition order is
    rotated by the graph's position and the two dynamic seeds alternate, so no condition is tied to a period."""
    rng = np.random.default_rng(order_seed)
    out = []
    for gi, gidx in enumerate(rng.permutation(len(graphs))):
        rot = conds[gi % len(conds):] + conds[:gi % len(conds)]
        for si, s in enumerate(seeds if gi % 2 == 0 else seeds[::-1]):
            for c in (rot if si == 0 else rot[::-1]):
                out.append((graphs[gidx], c, s))
    return out


def env_record() -> dict:
    import scipy
    import torch
    return {"utc": utc(), "commit": subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True,
                                                    cwd=ROOT).stdout.strip(),
            "dirty": bool(subprocess.run("git status --porcelain -- src scripts configs", shell=True,
                                         capture_output=True, text=True, cwd=ROOT).stdout.strip()),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "simulate_py_sha256": sha256_file(ROOT / "src/flymario/retino_align/simulate.py"),
            "decode_py_sha256": sha256_file(ROOT / "src/flymario/retino_align/decode.py"),
            "stimuli_py_sha256": sha256_file(ROOT / "src/flymario/retino_align/stimuli.py"),
            "config_sha256": sha256_file(CONFIG)}


# --- shared objects ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, run: str):
        self.run = run
        self.dir = run_dir(run)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.cfg = self.manifest["config"]
        self.pilot = self.manifest["pilot"]
        self._inputs = None

    def inputs(self) -> dict:
        if self._inputs is None:
            z = np.load(self.dir / "inputs/inputs.npz")
            self._inputs = {k: z[k] for k in z.files}
        return self._inputs

    def sigma(self, cond: str) -> np.ndarray | None:
        if cond == "ordered":
            return None
        return np.load(self.dir / "permutations/permutations.npz")[f"{cond}_sigma"]

    def X(self, cond: str) -> np.ndarray:
        X = self.inputs()["X"]
        s = self.sigma(cond)
        return X if s is None else X[:, s]

    def feature_path(self, spec: str, cond: str, seed: int) -> Path:
        return self.dir / "features" / tag(spec) / f"{cond}_s{seed}.npz"

    def lots(self) -> list[tuple]:
        return [tuple(x) for x in self.manifest["lots"]]


# --- prepare -----------------------------------------------------------------------------------------------------
def cmd_prepare(a) -> int:
    from flymario.connectome import datasets
    from flymario.env.vision import RetinotopicVision
    from flymario.retino_align import decode, retina, stimuli
    cfg = load_config()
    problems = check_config(cfg, main=False)
    if problems:
        print("config problems:", problems)
        return 2
    d = run_dir(a.run)
    if (d / "manifest.json").exists() and not a.force:
        print(f"{d}/manifest.json exists; prepare is done (use --force to rebuild inputs; features are kept)")
        return 0
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = datasets.load(datasets.DEFAULT)
    vis = RetinotopicVision(g)
    ret = retina.build(g, vis)
    motor = datasets.motor_index(g, datasets.DEFAULT)
    gp = stimuli.GenParams(**{k: tuple(v) if isinstance(v, list) else v
                              for k, v in cfg["dataset"]["generator"].items()})
    banks_cfg = ({p: (v[0], v[1], v[2]) for p, v in cfg["dataset"]["pilot_banks"].items()} if a.pilot else
                 {p: (p, v[0], v[1]) for p, v in cfg["dataset"]["banks"].items()})
    Xs, uids, part, orient, lum, lat, digests, col_lum = [], [], [], [], [], {}, [], []
    keep = {}
    for pi_, p in enumerate(PARTS):
        name, per, seed = banks_cfg[p]
        b = stimuli.draw_bank(name, per, seed, gp)
        X, side = stimuli.encode_bank(b, gp, vis, ret.inj, ret.col_uv, keep_images=16)
        Xs.append(X), uids.append(b.uid), part.append(np.full(len(b), pi_, np.int8))
        orient.append(b.orient), lum.append(b.lum), digests.append(side["digests"]), col_lum.append(side["col_lum"])
        for k, v in b.latent.items():
            lat.setdefault(k, []).append(v)
        keep[p] = side["images"]
    X = np.concatenate(Xs)
    inputs = dict(X=X, uid=np.concatenate(uids), part=np.concatenate(part), orient=np.concatenate(orient),
                  lum=np.concatenate(lum), digest=np.concatenate(digests), col_lum=np.concatenate(col_lum),
                  inj=ret.inj, readout=motor, **{f"lat_{k}": np.concatenate(v) for k, v in lat.items()})
    h_inputs = save_npz(d / "inputs/inputs.npz", **inputs)
    save_npz(d / "inputs/example_images.npz", **keep)
    # permutations
    perms, prep = {}, {}
    for cond, seed in cfg["retina"]["permutation_seeds"].items():
        pi = retina.column_permutation(ret, seed, cfg["retina"]["min_group_columns"])
        sigma = retina.cell_permutation(ret, pi)
        perms[f"{cond}_pi"], perms[f"{cond}_sigma"] = pi, sigma
        prep[cond] = {"seed": seed, "sha256_sigma": sha256_arrays(sigma.astype(np.int64)),
                      **retina.report(ret, pi, sigma, cfg["retina"]["min_group_columns"])}
    h_perm = save_npz(d / "permutations/permutations.npz", col=ret.col, col_uv=ret.col_uv, col_xy=ret.col_xy,
                      col_sig=ret.col_sig, types=ret.types, **perms)
    # training subsets (pool = the train partition)
    tr = inputs["part"] == 0
    combo = inputs["orient"][tr].astype(int) * 4 + inputs["lum"][tr]
    subs = decode.subsets(combo, cfg["decoder"]["subset_draw_seeds"], tuple(cfg["decoder"]["per_combination"]))
    save_npz(d / "inputs/subsets.npz", subsets=subs)
    graphs = graph_specs(cfg, a.pilot)
    conds = cfg["retina"]["conditions"]
    seeds = cfg["simulation"]["dynamic_seeds"]
    lots = lot_order(graphs, conds, seeds)
    n_img = len(X)
    manifest = {"run": a.run, "pilot": bool(a.pilot), "created_utc": utc(), "config": cfg, "env": env_record(),
                "graphs": graphs, "conditions": conds, "dynamic_seeds": seeds, "lots": lots,
                "n_images": {p: int((inputs["part"] == i).sum()) for i, p in enumerate(PARTS)},
                "inputs_sha256": h_inputs, "permutations_sha256": h_perm, "permutations": prep,
                "retina_digest": ret.digest(), "readout_n": int(len(motor)),
                "readout_node_index_sha256": sha256_arrays(motor.astype(np.int64)),
                "readout_body_ids_sha256": sha256_arrays(g.node_ids[motor].astype(np.int64)),
                "generator": gp.as_dict(), "banks": banks_cfg, "prepare_s": round(time.time() - t0, 1)}
    write_json(d / "manifest.json", manifest)
    vol = n_img * len(motor) * len(lots)
    print(f"prepared {a.run}: {len(graphs)} graphs x {len(conds)} conditions x {len(seeds)} seeds = {len(lots)} lots; "
          f"{n_img} images per lot; features {vol / 1e9:.2f} GB as uint8 ({4 * vol / 1e9:.1f} GB as float32); "
          f"decoder fits {len(graphs) * len(conds) * len(seeds) * len(cfg['decoder']['subset_draw_seeds']) * 5 * 2}")
    return 0


# --- validate ----------------------------------------------------------------------------------------------------
def edge_class_counts(gg, st, side) -> dict:
    c = np.asarray(st.edge_class(gg.src, gg.dst, gg.sign), np.int64)
    k = int(side.max()) + 1
    c = (c * k + side[gg.src]) * k + side[gg.dst]
    u, cnt = np.unique(c, return_counts=True)
    return dict(zip(u.tolist(), cnt.tolist()))


def validate_graph(spec: str, g0, st, side, cls0: dict) -> dict:
    """Graph invariants of one graph against the connectome (identities, edges, degrees, signs, in-strength, strata,
    sides, autapses, duplicates, edge overlap)."""
    from flymario.connectome import datasets, nullcache, randomize
    if spec == "connectome":
        return {"spec": spec, "is_connectome": True}
    s = int(spec.split(":")[1])
    path = nullcache.null_path("depth_stratified_lateral_instrength", datasets.DEFAULT, s)
    g, status = nullcache.load_or_build("depth_stratified_lateral_instrength", g0, s, datasets.DEFAULT, build=False)
    if g is None:
        return {"spec": spec, "present": False}
    n = g0.n_nodes
    rec = {"spec": spec, "file": path.name, "sha256": sha256_file(path), "present": True}
    rec["node_ids_identical"] = bool(np.array_equal(g.node_ids, g0.node_ids))
    rec["n_edges_equal"] = bool(g.n_edges == g0.n_edges)
    pairs = g.src.astype(np.int64) * n + g.dst
    rec["duplicate_pairs"] = int(len(pairs) - len(np.unique(pairs)))
    rec["autapses"] = int((g.src == g.dst).sum())
    rec["autapses_connectome"] = int((g0.src == g0.dst).sum())
    mix = randomize.mixing_report(g0, g, sample_size=10**9)
    rec.update({k: mix[k] for k in ("out_degree_exact", "in_degree_exact", "sign_multiset_equal",
                                    "weight_multiset_equal")})
    rec["edge_overlap_exact"] = mix["edge_overlap_sampled"]
    rec["frac_edges_changed"] = 1.0 - mix["edge_overlap_sampled"]
    sw0 = np.bincount(g0.dst, weights=g0.weight * g0.sign, minlength=n)
    sw = np.bincount(g.dst, weights=g.weight * g.sign, minlength=n)
    rec["signed_instrength_exact"] = bool(np.array_equal(sw0, sw))
    rec["instrength_unsigned_exact"] = bool(np.array_equal(np.bincount(g0.dst, weights=g0.weight, minlength=n),
                                                           np.bincount(g.dst, weights=g.weight, minlength=n)))
    rec["outstrength_max_abs_change"] = float(np.abs(np.bincount(g0.src, weights=g0.weight * g0.sign, minlength=n) -
                                                     np.bincount(g.src, weights=g.weight * g.sign, minlength=n)).max())
    # per-node incoming sign multiset (Dale on the source side: every source keeps its outgoing sign)
    out_sign0 = np.bincount(g0.src, weights=(g0.sign > 0), minlength=n), np.bincount(g0.src, weights=(g0.sign < 0), minlength=n)
    out_sign = np.bincount(g.src, weights=(g.sign > 0), minlength=n), np.bincount(g.src, weights=(g.sign < 0), minlength=n)
    rec["per_source_sign_counts_exact"] = bool(np.array_equal(out_sign0[0], out_sign[0]) and
                                               np.array_equal(out_sign0[1], out_sign[1]))
    rec["strata_side_edge_classes_equal"] = edge_class_counts(g, st, side) == cls0
    rec["meta"] = {k: g.meta.get(k) for k in ("null", "seed", "strata_key", "swap_multiplier", "n_strata")}
    rec["swaps"] = ("attempted per class = swap_multiplier x edges in the class (igraph Graph.rewire); the number "
                    "accepted is not exposed by igraph; the exact edge turnover is reported instead")
    rec["pass"] = all(rec[k] for k in ("node_ids_identical", "n_edges_equal", "out_degree_exact", "in_degree_exact",
                                       "sign_multiset_equal", "weight_multiset_equal", "signed_instrength_exact",
                                       "per_source_sign_counts_exact", "strata_side_edge_classes_equal")) \
        and rec["duplicate_pairs"] == 0
    return rec


def oriented_energy_predict(img: np.ndarray, centre=(135.38, 124.16), radius=100.0) -> int:
    """Pixel-only orientation estimate: FFT power of the disc around the eye's centre, summed in four
    ±22.5° sectors of the stripes' normal direction; argmax. Uses no label and no latent."""
    from flymario.retino_align.stimuli import ORIENT_DEG
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2
    z = np.where(m, img - img[m].mean(), 0.0)
    P = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(img.shape[0])), np.fft.fftshift(np.fft.fftfreq(img.shape[1])),
                         indexing="ij")
    ang = np.degrees(np.arctan2(-fy, fx)) % 180.0                    # direction of the wave vector, y up
    rad = np.hypot(fx, fy)
    band = (rad > 1 / 60) & (rad < 1 / 16)
    e = []
    for th in ORIENT_DEG:
        normal = (th + 90.0) % 180.0
        dd = np.abs((ang - normal + 90.0) % 180.0 - 90.0)
        e.append(P[band & (dd <= 22.5)].sum())
    return int(np.argmax(e))


def global_descriptors(X: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Mean, variance and quantiles of the input, overall and per cell type (no spatial layout)."""
    feats = []
    groups = [np.ones(X.shape[1], bool)] + [types == t for t in ("L1", "L2", "L3", "L5")]
    for m in groups:
        x = X[:, m].astype(np.float64)
        feats += [x.mean(1), x.var(1)] + [np.quantile(x, q, axis=1) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    return np.stack(feats, 1)


def decode_curve(F_tr, y_tr, F_va, y_va, F_te, y_te, subs, per_combo, shuffle_seed=None) -> dict:
    """Test accuracy per (draw, budget) with the study's decoder; optionally shuffled training labels."""
    from flymario.retino_align import decode
    acc = np.zeros(subs.shape[:2])
    for r in range(subs.shape[0]):
        for k in range(subs.shape[1]):
            sel = subs[r, k][subs[r, k] >= 0]
            y = y_tr[sel]
            if shuffle_seed is not None:
                y = y[np.random.default_rng(shuffle_seed + 100 * r + k).permutation(len(y))]
            f = decode.fit_all_lambdas(F_tr[sel], y)
            j = decode.select((decode.scores(f, F_va).argmax(2) == y_va).mean(1))
            acc[r, k] = (decode.scores(f, F_te)[j].argmax(1) == y_te).mean()
    return {"acc": acc.tolist(), "mean_by_budget": acc.mean(0).tolist(), "mean": float(acc.mean()),
            "budgets_per_class": [4 * m for m in per_combo]}


def validate_graphs(ctx: Ctx, a) -> bool:
    from flymario.connectome import datasets, strata
    g0 = datasets.load(datasets.DEFAULT)
    sensory = datasets.by_types(g0, datasets.DEFAULT, "lamina_types")
    readout = datasets.motor_index(g0, datasets.DEFAULT)
    st = strata.compute(g0, sensory, readout, "both")
    side = np.unique(np.asarray(g0.node_side), return_inverse=True)[1]
    cls0 = edge_class_counts(g0, st, side)
    specs = ctx.manifest["graphs"] if not a.graph_list else a.graph_list.split(",")
    gpath = ARTS / "graph_validation/graphs.json"
    gv = json.loads(gpath.read_text()) if gpath.exists() else {}
    for spec in specs:
        if spec in gv and gv[spec].get("pass") is not None:
            continue
        t1 = time.time()
        gv[spec] = validate_graph(spec, g0, st, side, cls0)
        gv[spec]["seconds"] = round(time.time() - t1, 1)
        print(spec, gv[spec].get("pass", "connectome"), flush=True)
        write_json(gpath, gv)
    return all(v.get("pass", True) for v in gv.values())


def cmd_validate(a) -> int:
    if a.graphs_only:
        return 0 if validate_graphs(Ctx(a.run), a) else 1
    from flymario.retino_align import decode, retina
    ctx = Ctx(a.run)
    cfg = ctx.cfg
    inp = ctx.inputs()
    out = {"run": a.run, "utc": utc(), "checks": {}}
    ck = out["checks"]
    # permutations
    pz = np.load(ctx.dir / "permutations/permutations.npz", allow_pickle=False)
    types = pz["types"]
    X = inp["X"]
    sig = {}
    for cond in cfg["retina"]["conditions"][1:]:
        s = pz[f"{cond}_sigma"]
        sig[cond] = sha256_arrays(s.astype(np.int64))
        Px = X[:, s]
        ck[f"{cond}_bijection"] = bool(np.array_equal(np.sort(s), np.arange(len(s))))
        ck[f"{cond}_inverse_exact"] = bool(np.array_equal(Px[:, retina.inverse(s)], X))
        ck[f"{cond}_types_preserved"] = bool(np.array_equal(types[s], types))
        ck[f"{cond}_multiset_global"] = bool(np.array_equal(np.sort(Px, 1), np.sort(X, 1)))
        ck[f"{cond}_multiset_per_type"] = all(bool(np.array_equal(np.sort(Px[:, types == t], 1),
                                                                  np.sort(X[:, types == t], 1))) for t in retina.LAMINA)
        col = pz["col"]
        ck[f"{cond}_on_off_together"] = all(len(np.unique(col[s][col == c])) == 1 for c in range(len(pz["col_uv"])))
    ck["permutations_distinct"] = len(set(sig.values())) == len(sig)
    out["permutation_sha256"] = sig
    # datasets
    part, uid = inp["part"], inp["uid"]
    combo = inp["orient"].astype(int) * 4 + inp["lum"]
    for i, p in enumerate(PARTS):
        cnt = np.bincount(combo[part == i], minlength=16)
        ck[f"{p}_balanced"] = bool(np.all(cnt == cnt[0]))
    ck["uids_unique"] = len(np.unique(uid)) == len(uid)
    ck["image_digests_unique"] = len(np.unique(inp["digest"])) == len(inp["digest"])
    lat = np.stack([inp[f"lat_{k}"] for k in ("phase", "period", "amp", "lum_offset", "dx", "dy")], 1)
    ck["latent_draws_unique"] = len(np.unique(lat, axis=0)) == len(lat)
    ck["noise_seeds_unique"] = len(np.unique(inp["lat_noise_seed"])) == len(inp["lat_noise_seed"])
    subs = np.load(ctx.dir / "inputs/subsets.npz")["subsets"]
    trc = combo[part == 0]
    per = cfg["decoder"]["per_combination"]
    ck["subsets_balanced_and_nested"] = all(
        np.all(np.bincount(trc[subs[r, k][subs[r, k] >= 0]], minlength=16) == per[k]) and
        (k == 0 or set(subs[r, k - 1][subs[r, k - 1] >= 0]) <= set(subs[r, k][subs[r, k] >= 0]))
        for r in range(subs.shape[0]) for k in range(subs.shape[1]))
    # luminance as constructed: eye-mean luminance equals the generator's base
    base = inp["lat_base"]
    eye_mean = inp["col_lum"].astype(np.float64).mean(1)
    out["luminance"] = {"max_abs_eye_mean_minus_base": float(np.abs(eye_mean - base).max()),
                        "bayes_rule_accuracy_from_eye_mean": float(
                            (np.digitize(eye_mean, [0.40, 0.50, 0.60]) == inp["lum"]).mean()),
                        "nominal_overlap": "adjacent classes overlap by 0.03 (centres 0.10 apart, jitter ±0.065)"}
    ck["luminance_as_constructed"] = out["luminance"]["max_abs_eye_mean_minus_base"] < 1e-4
    # summary stats per orientation (input)
    gd = global_descriptors(X, types)
    out["global_descriptors_by_orientation"] = {
        str(o): {"mean": float(gd[inp["orient"] == o, 0].mean()), "var": float(gd[inp["orient"] == o, 1].mean()),
                 "q05": float(gd[inp["orient"] == o, 2].mean()), "q95": float(gd[inp["orient"] == o, 6].mean())}
        for o in range(4)}
    # input-level decoders: direct (ordered vs permuted), global descriptors
    y = {t: inp["orient" if t == "orient" else "lum"].astype(int) for t in TASKS}
    tr, va, te = (part == 0), (part == 1), (part == 2)
    trX, vaX, teX = X[tr].astype(np.float64), X[va].astype(np.float64), X[te].astype(np.float64)
    inv = {}
    t0 = time.time()
    for cond in cfg["retina"]["conditions"][1:]:
        s = pz[f"{cond}_sigma"]
        worst, n_pred_diff, near_tie = 0.0, 0, 0
        for r in range(subs.shape[0]):
            for k in range(subs.shape[1]):
                sel = subs[r, k][subs[r, k] >= 0]
                for t in TASKS:
                    f1 = decode.fit_all_lambdas(trX[sel], y[t][tr][sel])
                    f2 = decode.fit_all_lambdas(trX[sel][:, s], y[t][tr][sel])
                    s1, s2 = decode.scores(f1, teX), decode.scores(f2, teX[:, s])
                    worst = max(worst, float(np.abs(s1 - s2).max()))
                    d = s1.argmax(2) != s2.argmax(2)
                    n_pred_diff += int(d.sum())
                    srt = np.sort(s1, 2)
                    near_tie += int(((srt[..., -1] - srt[..., -2]) < 1e-9).sum())
        inv[cond] = {"max_abs_score_diff": worst, "n_prediction_differences": n_pred_diff, "n_near_ties": near_tie}
    tol = 1e-8
    out["direct_decoder_invariance"] = {"tolerance_abs_score": tol, "by_condition": inv, "seconds": round(time.time() - t0, 1)}
    ck["direct_decoder_invariant"] = all(v["max_abs_score_diff"] <= tol and v["n_prediction_differences"] <= v["n_near_ties"]
                                         for v in inv.values())
    out["direct_decoder_accuracy"] = {t: decode_curve(trX, y[t][tr], vaX, y[t][va], teX, y[t][te], subs, per)
                                      for t in TASKS}
    out["global_descriptor_decoder"] = {t: decode_curve(gd[tr], y[t][tr], gd[va], y[t][va], gd[te], y[t][te], subs, per)
                                        for t in TASKS}
    # permutation-invariant reference (declared secondary): the multiset of input values per cell type,
    # identical in every condition by construction; what an arrangement-blind linear reader can get from the input
    srt = np.concatenate([np.sort(X[:, types == t_], 1) for t_ in retina.LAMINA], 1).astype(np.float64)
    out["multiset_reference_decoder"] = {t: decode_curve(srt[tr], y[t][tr], srt[va], y[t][va], srt[te], y[t][te], subs,
                                                         per) for t in TASKS}
    # pixel positive control on the test images (re-rendered, no label used by the estimator)
    from flymario.retino_align.stimuli import Bank, GenParams, render, sample_points
    gp = GenParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in ctx.manifest["generator"].items()})
    idx = np.flatnonzero(te)[:: max(1, int(te.sum()) // 512)]
    bk = Bank(name="x", uid=uid[idx], orient=inp["orient"][idx], lum=inp["lum"][idx],
              latent={k[4:]: inp[k][idx] for k in inp if k.startswith("lat_")})
    pts = sample_points(pz["col_uv"])
    pred = np.array([oriented_energy_predict(render(bk, i, gp, pts, 1.0), gp.centre) for i in range(len(idx))])
    out["pixel_orientation_control"] = {"n": int(len(idx)), "accuracy": float((pred == bk.orient).mean()),
                                        "method": "FFT power in ±22.5° sectors of the normal, disc r=100 px"}
    ck["pixel_control_detects_orientation"] = out["pixel_orientation_control"]["accuracy"] >= 0.90
    # graphs
    if a.graphs:
        ck["all_graphs_pass"] = validate_graphs(ctx, a)
    out["all_pass"] = all(ck.values())
    write_json(ctx.dir / "validation/validation.json", out)
    print(json.dumps({k: v for k, v in ck.items() if not v} or "all checks pass"))
    return 0 if out["all_pass"] else 1


# --- extract -----------------------------------------------------------------------------------------------------
def lot_done(ctx: Ctx, lot) -> bool:
    p = ctx.feature_path(*lot)
    j = p.with_suffix(".json")
    if not (p.exists() and j.exists()):
        return False
    rec = json.loads(j.read_text())
    return rec.get("complete") and rec.get("sha256") == sha256_file(p)


def write_status(ctx: Ctx, extra: dict | None = None) -> dict:
    lots = ctx.lots()
    done = [l for l in lots if lot_done(ctx, l)]
    failed = []
    fpath = ctx.dir / "logs/failures.json"
    if fpath.exists():
        failed = [tuple(x["lot"]) for x in json.loads(fpath.read_text()) if tuple(x["lot"]) not in done]
    secs = [json.loads(ctx.feature_path(*l).with_suffix(".json").read_text())["seconds"] for l in done]
    dec = sorted({(g, c) for g, c, _ in lots})
    dec_done = [x for x in dec if (ctx.dir / "decoders" / tag(x[0]) / f"{x[1]}.npz").exists()]
    st = {"run": ctx.run, "utc": utc(), "lots_expected": len(lots), "lots_done": len(done),
          "lots_failed": len(set(failed)), "lots_missing": len(lots) - len(done),
          "missing": [list(l) for l in lots if l not in done][:400],
          "sim_seconds_total": round(float(np.sum(secs)), 1) if secs else 0.0,
          "eta_extract_s": round(float(np.mean(secs)) * (len(lots) - len(done)), 0) if secs else None,
          "decoders_expected": len(dec), "decoders_done": len(dec_done), **(extra or {})}
    write_json(ctx.dir / "status.json", st)
    return st


def cmd_extract(a) -> int:
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.retino_align import simulate
    ctx = Ctx(a.run)
    cfg = ctx.cfg
    live = load_config()
    if not ctx.pilot:
        problems = check_config(live, main=True)
        if problems:
            print("refused:", problems)
            return 2
        if sha256_file(CONFIG) != ctx.manifest["env"]["config_sha256"] and not a.allow_config_change:
            print("refused: configs/retino_align/protocol.yaml changed since prepare (a new protocol version?)")
            return 2
    inp = ctx.inputs()
    p = LIFParams(gain=float(_cfg_get("dynamics.gain")), noise_std=float(_cfg_get("dynamics.noise_std")))
    if int(_cfg_get("timing.ticks_per_decision")) != int(cfg["simulation"]["steps"]):
        print("refused: simulation.steps differs from timing.ticks_per_decision")
        return 2
    lots = [l for l in ctx.lots() if not lot_done(ctx, l)]
    if a.limit:
        lots = lots[: a.limit]
    by_graph: dict[str, list] = {}
    for l in lots:
        by_graph.setdefault(l[0], []).append(l)
    failures = json.loads((ctx.dir / "logs/failures.json").read_text()) if (ctx.dir / "logs/failures.json").exists() else []
    readout = inp["readout"]
    for spec in sorted(by_graph, key=lambda s: [l[0] for l in ctx.lots()].index(s)):
        if not ctx.pilot and past_final_reserve(cfg):
            print("final reserve reached: no new lot started")
            write_status(ctx, {"stopped": "final_reserve"})
            return 2
        t_load = time.time()
        _g, W, motor = graphsel.load(spec)
        if not np.array_equal(motor, readout):
            raise RuntimeError("readout differs from the manifest's")
        net = simulate.FrozenNet(W, readout, inp["inj"], p, int(cfg["simulation"]["steps"]),
                                 batch=int(cfg["simulation"]["batch"]))
        t_load = time.time() - t_load
        gpath = _graph_file(spec)
        g_sha = sha256_file(gpath)
        for lot in by_graph[spec]:
            _, cond, seed = lot
            for attempt in range(1, int(cfg["execution"]["technical_retry"]["max_attempts"]) + 1):
                try:
                    t0 = time.time()
                    d_before = net.device_operator_digest()
                    o = net.run(ctx.X(cond), inp["uid"], int(seed))
                    d_after = net.device_operator_digest()
                    if d_before != d_after or d_after != net.digest:
                        raise RuntimeError("operator changed during the lot")
                    secs = time.time() - t0
                    path = ctx.feature_path(*lot)
                    h = save_npz(path, counts=o.counts, uid=inp["uid"], part=inp["part"], rate_all=o.rate_all,
                                 rate_input=o.rate_input, v_readout_mean=o.v_readout_mean, v_readout_sd=o.v_readout_sd)
                    C = o.counts
                    rec = {"lot": list(lot), "complete": True, "utc": utc(), "seconds": round(secs, 2),
                           "graph_load_s": round(t_load, 2), "attempt": attempt, "sha256": h,
                           "graph_file": gpath.name, "graph_sha256": g_sha, "operator_digest": net.digest,
                           "inputs_sha256": ctx.manifest["inputs_sha256"],
                           "permutation_sha256": (ctx.manifest["permutations"][cond]["sha256_sigma"]
                                                  if cond != "ordered" else None),
                           "config_sha256": ctx.manifest["env"]["config_sha256"],
                           "simulator": {"commit": env_record()["commit"],
                                         "simulate_py_sha256": sha256_file(ROOT / "src/flymario/retino_align/simulate.py")},
                           "activity": activity_summary(C, o, inp)}
                    write_json(path.with_suffix(".json"), rec)
                    break
                except (RuntimeError, OSError) as e:
                    failures.append({"lot": list(lot), "utc": utc(), "attempt": attempt, "error": repr(e),
                                     "trace": traceback.format_exc()[-2000:]})
                    write_json(ctx.dir / "logs/failures.json", failures)
                    if "operator changed" in repr(e):
                        raise
                    import torch
                    torch.cuda.empty_cache()
            st = write_status(ctx)
            print(f"{utc()} {lot} done={st['lots_done']}/{st['lots_expected']} eta={st['eta_extract_s']}s", flush=True)
        del net
    write_status(ctx)
    return 0


def activity_summary(C: np.ndarray, o, inp) -> dict:
    """Activity summary per lot, computed on all images of the lot."""
    Cf = C.astype(np.float64)
    return {"mean_count": float(Cf.mean()), "sd_count": float(Cf.std()),
            "frac_silent_entries": float((C == 0).mean()),
            "frac_neurons_silent_on_all_images": float((C.max(0) == 0).mean()),
            "frac_near_ceiling_entries_ge7": float((C >= 7).mean()), "max_count": int(C.max()),
            "mean_count_by_orient": [float(Cf[inp["orient"] == k].mean()) for k in range(4)],
            "mean_count_by_lum": [float(Cf[inp["lum"] == k].mean()) for k in range(4)],
            "rate_all": float(o.rate_all.mean()), "rate_input": float(o.rate_input.mean()),
            "v_readout_mean": float(o.v_readout_mean.mean()), "v_readout_sd": float(o.v_readout_sd.mean())}


def _cfg_get(key: str):
    from flymario import config
    return config.get(key)


def _graph_file(spec: str) -> Path:
    from flymario.connectome import datasets, nullcache
    if spec == "connectome":
        return datasets.cache_path(datasets.DEFAULT)
    return nullcache.null_path("depth_stratified_lateral_instrength", datasets.DEFAULT, int(spec.split(":")[1]))


# --- decode ------------------------------------------------------------------------------------------------------
def decode_one(run: str, spec: str, cond: str, shuffle: bool = False) -> str:
    from flymario.retino_align import decode
    ctx = Ctx(run)
    cfg = ctx.cfg
    inp = ctx.inputs()
    seeds = ctx.manifest["dynamic_seeds"]
    out_dec = ctx.dir / "decoders" / tag(spec) / f"{cond}{'_shuffled' if shuffle else ''}.npz"
    out_pred = ctx.dir / "predictions" / tag(spec) / f"{cond}{'_shuffled' if shuffle else ''}.npz"
    if out_dec.exists() and out_pred.exists():
        return "exists"
    subs = np.load(ctx.dir / "inputs/subsets.npz")["subsets"]
    part = inp["part"]
    tr, va, te = part == 0, part == 1, part == 2
    y = {"orient": inp["orient"].astype(int), "lum": inp["lum"].astype(int)}
    lams = cfg["decoder"]["normalized_loss_lambda_grid"]
    R, K = subs.shape[:2]
    S, T = len(seeds), len(TASKS)
    d = len(inp["readout"])
    n_te = int(te.sum())
    W = np.zeros((S, R, K, T, d, 4), np.float32)
    b = np.zeros((S, R, K, T, 4), np.float32)
    mu = np.zeros((S, R, K, d), np.float32)
    sd = np.zeros((S, R, K, d), np.float32)
    val_acc = np.zeros((S, R, K, T, len(lams)), np.float32)
    lam_idx = np.zeros((S, R, K, T), np.int8)
    pred = np.zeros((S, R, K, T, n_te), np.uint8)
    margin = np.zeros((S, R, K, T, n_te), np.float16)
    conf = np.zeros((S, R, K, T, 4, 4), np.int32)
    t0 = time.time()
    lot_sha = {}
    for si, seed in enumerate(seeds):
        fp = ctx.feature_path(spec, cond, seed)
        lot_sha[str(seed)] = json.loads(fp.with_suffix(".json").read_text())["sha256"]
        F = np.load(fp)["counts"].astype(np.float64)
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
                mu[si, r, k], sd[si, r, k] = fits[0].mu, fits[0].sd
                for ti, t in enumerate(TASKS):
                    f = fits[ti]
                    sva = np.einsum("md,ldk->lmk", Zva, f.W) + f.b
                    va_a = (sva.argmax(2) == y[t][va]).mean(1)
                    j = decode.select(va_a, lams)
                    ste = Zte @ f.W[j] + f.b
                    p_ = ste.argmax(1)
                    srt = np.sort(ste, 1)
                    W[si, r, k, ti], b[si, r, k, ti] = f.W[j], f.b
                    val_acc[si, r, k, ti], lam_idx[si, r, k, ti] = va_a, j
                    pred[si, r, k, ti], margin[si, r, k, ti] = p_, srt[:, -1] - srt[:, -2]
                    np.add.at(conf[si, r, k, ti], (y[t][te], p_), 1)
    h1 = save_npz(out_dec, W=W, b=b, mu=mu, sd=sd, val_acc=val_acc, lam_idx=lam_idx, lambdas=np.array(lams),
                  seeds=np.array(seeds), budgets_per_class=4 * np.array(cfg["decoder"]["per_combination"]),
                  tasks=np.array(TASKS))
    h2 = save_npz(out_pred, pred=pred, margin=margin, conf=conf, uid_test=inp["uid"][te], y_orient=y["orient"][te],
                  y_lum=y["lum"][te])
    acc = (pred == np.stack([y["orient"][te], y["lum"][te]])[None, None, None, :, :]).mean(-1)
    write_json(ctx.dir / "metrics" / tag(spec) / f"{cond}{'_shuffled' if shuffle else ''}.json",
               {"graph": spec, "condition": cond, "shuffled_labels": shuffle, "utc": utc(),
                "seconds": round(time.time() - t0, 1), "decoders_sha256": h1, "predictions_sha256": h2,
                "lots_sha256": lot_sha, "acc_by_seed_draw_budget_task": acc.tolist(),
                "M": {t: float(acc[..., ti].mean()) for ti, t in enumerate(TASKS)},
                "lambda_chosen_counts": {t: np.bincount(lam_idx[..., ti].ravel(), minlength=len(lams)).tolist()
                                         for ti, t in enumerate(TASKS)}})
    return "done"


def cmd_decode(a) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = Ctx(a.run)
    pairs = sorted({(g, c) for g, c, _ in ctx.lots()}, key=lambda x: (ctx.manifest["graphs"].index(x[0]), x[1]))
    ready = [(g, c) for g, c in pairs
             if all(lot_done(ctx, (g, c, s)) for s in ctx.manifest["dynamic_seeds"])
             and not (ctx.dir / "decoders" / tag(g) / f"{c}.npz").exists()]
    jobs = [(g, c, False) for g, c in ready]
    if a.shuffled_control:
        g0 = ctx.manifest["graphs"][0]
        jobs.append((g0, "ordered", True))
    print(f"{len(jobs)} decoder jobs", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(decode_one, a.run, g, c, s): (g, c, s) for g, c, s in jobs}
        for f in as_completed(futs):
            print(utc(), futs[f], f.result(), flush=True)
    write_status(ctx)
    return 0


# --- analyse -----------------------------------------------------------------------------------------------------
def cmd_analyse(a) -> int:
    from flymario.retino_align import analysis
    ctx = Ctx(a.run)
    cfg = ctx.cfg
    graphs, conds = ctx.manifest["graphs"], ctx.manifest["conditions"]
    inp = ctx.inputs()
    te = inp["part"] == 2
    strata = (inp["orient"].astype(int) * 4 + inp["lum"])[te]
    y = np.stack([inp["orient"][te], inp["lum"][te]]).astype(np.uint8)
    n = int(te.sum())
    G, C = len(graphs), len(conds)
    S = len(ctx.manifest["dynamic_seeds"])
    missing = []
    R_, K_ = len(cfg['decoder']['subset_draw_seeds']), len(cfg['decoder']['per_combination'])
    corr = np.full((G, C, 2, S, R_, K_, n), np.nan, np.float32)
    for gi, g in enumerate(graphs):
        for ci, c in enumerate(conds):
            p = ctx.dir / "predictions" / tag(g) / f"{c}.npz"
            if not p.exists():
                missing.append([g, c])
                continue
            pr = np.load(p)["pred"]                                  # (S, R, K, T, n)
            corr[gi, ci] = np.moveaxis(pr == y[None, None, None, :, :], 3, 0)
    complete_graphs = [gi for gi in range(G) if not np.isnan(corr[gi]).any()]
    res = {"run": ctx.run, "utc": utc(), "missing_graph_conditions": missing,
           "graphs_complete": [graphs[i] for i in complete_graphs], "pilot": ctx.pilot}
    if len(complete_graphs) < 2 or (not ctx.pilot and 0 not in complete_graphs):
        write_json(ctx.dir / "metrics/analysis.json", res)
        print("not enough complete graphs for the contrasts", missing[:5])
        return 1
    corr = corr[complete_graphs]
    cbar = corr.mean(axis=(3, 4, 5))                                 # (G, C, T, n): over seeds, draws, budgets
    M = analysis.scores(cbar)
    c = analysis.contrasts(M)
    res["M"] = {graphs[i]: {cond: {"orient": float(M[k, ci, 0]), "lum": float(M[k, ci, 1])}
                            for ci, cond in enumerate(conds)} for k, i in enumerate(complete_graphs)}
    res["L"] = {graphs[i]: {"orient": float(c["L"][k, 0]), "lum": float(c["L"][k, 1])}
                for k, i in enumerate(complete_graphs)}
    res["point"] = {k: c[k] for k in ("D_orientation", "D_brightness", "G", "S")}
    delta = float(cfg["analysis"]["practical_margin"])
    t0 = time.time()
    bs = analysis.bootstrap(cbar, strata, int(a.replicates or cfg["analysis"]["bootstrap_replicates"]),
                            int(cfg["analysis"]["bootstrap_seed"]))
    res["bootstrap"] = {"replicates": len(bs["D_orientation"]), "seed": int(cfg["analysis"]["bootstrap_seed"]),
                        "seconds": round(time.time() - t0, 1),
                        "intervals_95": {k: analysis.interval(v) for k, v in bs.items()},
                        "scope": "conditional on the connectome, the train/val banks, the three permutations, the two "
                                 "dynamic seeds and the subsets used; test images resampled within the 16 cells "
                                 "(paired across graphs and conditions), nulls resampled as whole graphs"}
    lo, hi = res["bootstrap"]["intervals_95"]["D_orientation"]
    res["reading_D_orientation"] = analysis.reading(lo, hi, delta)
    res["rank_L_orientation"] = analysis.rank_of_connectome(c["L"][:, 0])
    res["rank_L_brightness"] = analysis.rank_of_connectome(c["L"][:, 1])
    # sensitivity (secondary): per permutation, per seed, per budget
    sens = {"per_permutation": {}, "per_seed": {}, "per_budget": {}}
    for pi_, cond in enumerate(conds[1:], start=1):
        Mp = M[:, [0, pi_], :]
        sens["per_permutation"][cond] = {k: v for k, v in analysis.contrasts(Mp).items() if k != "L"}
    for si in range(S):
        Ms = analysis.scores(corr[:, :, :, si].mean(axis=(3, 4)))
        sens["per_seed"][str(ctx.manifest["dynamic_seeds"][si])] = {k: v for k, v in analysis.contrasts(Ms).items() if k != "L"}
    for k, bpc in enumerate(4 * np.array(cfg["decoder"]["per_combination"])):
        Mk = analysis.scores(corr[:, :, :, :, :, k].mean(axis=(3, 4)))
        sens["per_budget"][str(int(bpc))] = {kk: v for kk, v in analysis.contrasts(Mk).items() if kk != "L"}
    res["sensitivity_secondary"] = sens
    curves = corr.mean(axis=(3, 4, 6))                                # (G, C, T, K) accuracy by budget
    res["curves"] = curves.tolist()
    np.savez(ctx.dir / "metrics/bootstrap.npz", **bs)
    np.savez(ctx.dir / "metrics/cbar.npz", cbar=cbar, graphs=np.array([graphs[i] for i in complete_graphs]),
             conds=np.array(conds), strata=strata)
    write_json(ctx.dir / "metrics/analysis.json", res)
    print(json.dumps(res["point"]), res["bootstrap"]["intervals_95"]["D_orientation"], res["reading_D_orientation"])
    return 0


def cmd_status(a) -> int:
    st = write_status(Ctx(a.run))
    print(json.dumps({k: v for k, v in st.items() if k != "missing"}, indent=1))
    return 0


def cmd_resume(a) -> int:
    rc = cmd_extract(a)
    if rc not in (0,):
        return rc
    rc = cmd_decode(a)
    return cmd_analyse(a) if rc == 0 else rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "validate", "extract", "decode", "analyse", "status", "resume"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name == "prepare":
            s.add_argument("--pilot", action="store_true")
            s.add_argument("--force", action="store_true")
        if name == "validate":
            s.add_argument("--graphs", action="store_true", help="also validate every graph's invariants (§5.1)")
            s.add_argument("--graph-list", default="", help="comma-separated specs (default: the run's graphs)")
            s.add_argument("--graphs-only", action="store_true")
        if name in ("extract", "resume"):
            s.add_argument("--limit", type=int, default=0)
            s.add_argument("--allow-config-change", action="store_true")
        if name in ("decode", "resume"):
            s.add_argument("--workers", type=int, default=3)
            s.add_argument("--shuffled-control", action="store_true")
        if name in ("analyse", "resume"):
            s.add_argument("--replicates", type=int, default=0)
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "validate": cmd_validate, "extract": cmd_extract, "decode": cmd_decode,
            "analyse": cmd_analyse, "status": cmd_status, "resume": cmd_resume}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
