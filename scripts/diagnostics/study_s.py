#!/usr/bin/env python
"""Study S (docs/STUDY_S.md): controlled visual tasks learnt from reward by the three-factor rule, per graph.

    .venv/bin/python scripts/diagnostics/study_s.py run --graph nulllat:100 [--trials 3000] [--eta 0.001] [--tag pilot]
    .venv/bin/python scripts/diagnostics/study_s.py refs --graph nulllat:100 --trials T --eta E     # §7 references

One GPU process per graph: 12 brains (2 tasks x 3 readouts x 2 noise seed sets), common random numbers for every
graph (stimulus streams cached once in artifacts/runs/study_s/stimuli-*.npz). Output: artifacts/runs/study_s/
<graph>[-tag].json with E1-E5 per brain and per task, and the training curves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts/runs/study_s"
TASKS = ("where", "loom")
H, W_ = 240, 256
BG, DARK = 1.0, 0.0                      # A1 (2026-10-11 11:47 UTC): large dark stimuli on white
FIELD_C = [(32 + 64 * i, 84 + 104 * j) for j in range(2) for i in range(4)]
LOOM_C = [(64, 85), (192, 85), (64, 185), (192, 185)]
KAPPA, EBAR, RD, TAU, EPS, BRATE = 0.2, 0.05, 0.95, 2.0, 0.1, 0.1
N_TEST = 25
_YY, _XX = np.mgrid[0:H, 0:W_]


def _rgb(img):
    return np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, 2)


def stimulus(task, cls, rng, size=(64, 104), lum=DARK, loom_radii=(30, 90), jit=8):
    """(previous frame, stimulus frame) for one trial of class `cls` (A1 tasks)."""
    if task == "where":                              # T-field
        cx, cy = FIELD_C[cls]
        j = rng.integers(-jit, jit + 1, 2)
        w, h = size
        img = np.full((H, W_), BG)
        x0, y0 = cx - w // 2 + j[0], cy - h // 2 + j[1]
        img[max(32, y0):max(32, y0 + h), max(0, x0):max(0, x0 + w)] = lum
        return _rgb(np.full((H, W_), BG)), _rgb(img)
    q, kind = cls % 4, cls // 4                      # T-loom: classes 0-3 looming, 4-7 receding
    x, y = LOOM_C[q]
    j = rng.uniform(-jit, jit, 2)
    r0, r1 = loom_radii if kind == 0 else loom_radii[::-1]
    a = np.full((H, W_), BG)
    a[(_XX - x - j[0]) ** 2 + (_YY - y - j[1]) ** 2 <= r0 ** 2] = lum
    bimg = np.full((H, W_), BG)
    bimg[(_XX - x - j[0]) ** 2 + (_YY - y - j[1]) ** 2 <= r1 ** 2] = lum
    return _rgb(a), _rgb(bimg)


def class_stream(n, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(8) for _ in range(-(-n // 8))])[:n]


def build_stimuli(g, n_train, sb=0):
    """Training streams (per task and seed set), test and generalisation sets, encoded once (compact)."""
    from flymario.env.vision import RetinotopicVision
    path = OUT / f"stimuli-A1-T{n_train}{f'-sb{sb}' if sb else ''}.npz"   # F1: sb > 0 redraws the training streams only
    vis = RetinotopicVision(g)
    idx = np.unique(np.r_[vis.on_idx, vis.off_idx]).astype(np.int64)
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, idx

    def enc(prev, cur):
        vis.reset()
        vis.encode(prev)
        return vis.encode(cur)[idx].astype(np.float32)
    out = {}
    for ti, task in enumerate(TASKS):
        for ss in range(2):
            cls = class_stream(n_train, 20262000 + 1000 * sb + 10 * ti + ss)
            rng = np.random.default_rng(20262100 + 1000 * sb + 10 * ti + ss)
            out[f"train_{task}_{ss}_cls"] = cls
            out[f"train_{task}_{ss}_x"] = np.stack([enc(*stimulus(task, int(c), rng)) for c in cls], 1)
        rng = np.random.default_rng(20262200 + ti)
        tc = np.repeat(np.arange(8), N_TEST)
        out[f"test_{task}_cls"] = tc
        out[f"test_{task}_x"] = np.stack([enc(*stimulus(task, int(c), rng)) for c in tc], 1)
        gens = ([dict(size=(48, 80)), dict(lum=0.5), dict(jit=16)] if task == "where" else
                [dict(loom_radii=(24, 72)), dict(lum=0.5), dict(loom_radii=(40, 110))])
        for gi, kw in enumerate(gens):
            rng = np.random.default_rng(20262300 + 10 * ti + gi)
            out[f"gen{gi}_{task}_x"] = np.stack([enc(*stimulus(task, int(c), rng, **kw)) for c in tc], 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out, idx


def run(a):
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    from flymario.models.torch_lif import upload_operator
    torch.set_num_threads(4)
    t0 = time.time()
    g0, _w, _m = graphsel.load("connectome")
    del _w
    S, idx = build_stimuli(g0, a.trials, a.seed_block)
    g, W, motor = graphsel.load(a.graph)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    ro = block_seeds(3, master_seed=a.seed_block)              # F1: seed block 1 = fresh readouts, noise, streams
    # learners: (task, readout, seed set, kind); kinds: true | supervised | shuffled | frozen
    kinds = (["true"] if a.mode == "run" else ["true", "supervised", "shuffled", "frozen"]) if not a.kinds else a.kinds.split(",")
    L_ = [(t, r, s, k) for k in kinds for t in TASKS for r in range(3) for s in range(2)]
    B = len(L_)
    seeds = [8300 + 1000 * a.seed_block + 10 * s + r for (_t, r, s, _k) in L_]
    b = PlasticLIF(W, motor, params, seeds=seeds, cache_key=None)
    R3 = np.stack([np.asarray(FixedRandomProjection(b.M, 8, r_).W) for r_ in ro])
    R = torch.as_tensor(R3[[r for (_t, r, _s, _k) in L_]], device=b.dev)
    n = b.n
    eta = torch.tensor([0.0 if k == "frozen" else a.eta for (*_x, k) in L_], device=b.dev)

    def cur_for(cols):
        out = torch.zeros((n, B), device=b.dev)
        out[torch.as_tensor(idx, device=b.dev)] = torch.as_tensor(np.stack(cols, 1), device=b.dev)
        return out
    rng_u = [np.random.default_rng(20262400 + 1000 * a.seed_block + 10 * TASKS.index(t) + s) for (t, _r, s, _k) in L_]   # shared per task/seedset
    base = np.zeros((B, 8))
    rbar = np.full(B, 0.125)
    ebar, ms = None, torch.zeros(B, device=b.dev)
    correct = np.zeros((a.trials, B), np.int8)
    for k in range(a.trials):
        cols, cls = [], []
        for (t, _r, s, _k) in L_:
            cls.append(int(S[f"train_{t}_{s}_cls"][k]))
            cols.append(S[f"train_{t}_{s}_x"][:, k])
        cls = np.array(cls)
        c, e = sweep_once(b, params, cur_for(cols), K, 0.5, True)
        sc = torch.einsum("bam,mb->ba", R, c)
        pi = torch.softmax(sc / TAU, 1)
        pin = pi.cpu().numpy().astype(np.float64)
        act = np.empty(B, int)
        for j in range(B):
            u, v = rng_u[j].random(2)
            act[j] = int(v * 8) if u < EPS else int(np.searchsorted(np.cumsum(pin[j] / pin[j].sum()), rng_u[j].random()))
        act = np.minimum(act, 7)
        r = (act == cls).astype(np.float64)
        correct[k] = r
        g_ = np.zeros((B, 8))
        for j, (*_x, kind) in enumerate(L_):
            if kind == "supervised":
                g_[j] = np.eye(8)[cls[j]] - pin[j]
            elif kind in ("true", "shuffled", "frozen"):
                rr = r[j]
                if kind == "shuffled":                 # A3: the true learners' pilot reward rate (per task)
                    rate = {"where": a.shuffle_rate_field, "loom": a.shuffle_rate_loom}[L_[j][0]]
                    rr = float(rng_u[j].random() < (rate if rate is not None else rbar[j]))
                g_[j] = (np.eye(8)[act[j]] - pin[j]) * (rr - base[j, cls[j]])
                base[j, cls[j]] += BRATE * (rr - base[j, cls[j]])
            rbar[j] += 0.01 * (r[j] - rbar[j])
        if ebar is None:
            ebar = e.clone()
        ec = (e - ebar) + KAPPA * ebar
        ebar.mul_(1 - EBAR).add_(e, alpha=EBAR)
        Lsig = torch.einsum("ba,bam->mb", torch.as_tensor(g_, device=b.dev, dtype=torch.float32), R) / TAU
        U = ec * Lsig.index_select(0, b.post) * b.sign[:, None]
        ms = RD * ms + (1 - RD) * (U * U).mean(0)
        U = U / torch.sqrt(ms / (1 - RD ** (k + 1)) + 1e-30)[None, :]
        b.update((U * eta[None, :]).t() * b.m0[None, :])

    def evaluate(key_fn):
        accs = []
        tc = S["test_where_cls"]
        hits = np.zeros(B)
        for q in range(len(tc)):
            cols = [S[key_fn(t)][:, q] for (t, _r, _s, _k) in L_]
            c, _ = sweep_once(b, params, cur_for(cols), K, 0.5, False)
            a_ = torch.einsum("bam,mb->ba", R, c).argmax(1).cpu().numpy()
            hits += (a_ == tc[q])
        accs = hits / len(tc)
        return accs
    A_test = evaluate(lambda t: f"test_{t}_x")
    A_gen = np.mean([evaluate(lambda t, gi=gi: f"gen{gi}_{t}_x") for gi in range(3)], 0)
    p_noise = LIFParams(gain=params.gain, noise_std=2 * params.noise_std)
    b.p = p_noise
    A_noise = evaluate(lambda t: f"test_{t}_x")
    b.p = params
    # 10% of all edges deleted (frozen operator and plastic edges), shared seeds by index
    import scipy.sparse as sp
    Wc = sp.csr_matrix(W, dtype=np.float32)
    keep = np.ones(Wc.shape[0], np.float32)
    keep[np.asarray(motor)] = 0.0
    frozen = sp.csr_matrix(sp.diags(keep) @ Wc)
    frozen.eliminate_zeros()
    mask = np.random.default_rng(20262500).random(frozen.nnz) >= 0.10
    frozen.data = frozen.data * mask
    frozen.eliminate_zeros()
    b.W, _nnz = upload_operator(frozen, False, str(b.dev), key=None)
    pmask = torch.as_tensor(np.random.default_rng(20262501).random(b.P) >= 0.10, device=b.dev, dtype=torch.float32)
    b.set_weights(b.m * pmask[None, :])
    A_rob = evaluate(lambda t: f"test_{t}_x")
    res = {"graph": a.graph, "mode": a.mode, "trials": a.trials, "eta": a.eta, "learners": [], "seconds": 0}
    for j, (t, r_, s, kind) in enumerate(L_):
        res["learners"].append({"task": t, "readout": r_, "seed_set": s, "kind": kind,
                                "E1": float(correct[:, j].mean()), "A_test": float(A_test[j]), "A_gen": float(A_gen[j]),
                                "A_rob": float(A_rob[j]), "A_noise": float(A_noise[j]),
                                "curve": [round(float(x), 3) for x in correct[:, j].reshape(-1, 100).mean(1)]
                                if a.trials % 100 == 0 else None})
    summ = {}
    for kind in kinds:
        for t in TASKS:
            ls = [x for x in res["learners"] if x["task"] == t and x["kind"] == kind]
            summ[f"{kind}/{t}"] = {k: round(float(np.mean([x[k] for x in ls])), 4) for k in ("E1", "A_test", "A_gen", "A_rob", "A_noise")}
            summ[f"{kind}/{t}"]["E4"] = round(summ[f"{kind}/{t}"]["A_test"] - summ[f"{kind}/{t}"]["A_rob"], 4)
            summ[f"{kind}/{t}"]["E5"] = round(summ[f"{kind}/{t}"]["A_test"] - summ[f"{kind}/{t}"]["A_noise"], 4)
    res["summary"] = summ
    res["seconds"] = round(time.time() - t0)
    res["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.graph.replace(':', '')}{'-' + a.tag if a.tag else ''}.json"
    tmp = OUT / (name + ".part")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, OUT / name)
    print(json.dumps(summ), res["seconds"], "s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "refs"))
    ap.add_argument("--graph", required=True)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shuffle-rate-field", type=float, default=None)
    ap.add_argument("--shuffle-rate-loom", type=float, default=None)
    ap.add_argument("--seed-block", type=int, default=0)
    ap.add_argument("--kinds", default="", help="comma list overriding the learner kinds (references, debugging)")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
