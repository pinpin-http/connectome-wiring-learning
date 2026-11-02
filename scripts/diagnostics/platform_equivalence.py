#!/usr/bin/env python
"""What the platform computes (docs/RL_STUDY.md §17 N1-N3). Outputs under artifacts/runs/platform/.

    operator   N1/N2: motor input of ROCm-trained weights by CUDA segment reduction, CUDA SpMM and CPU float64.
    samenoise  N3: the cs-scripted common stream with every tick's noise drawn from one CPU generator.
    lockstep   a CPU brain and a CUDA brain trained in lock-step with identical noise.

    .venv/bin/python scripts/diagnostics/platform_equivalence.py operator
    .venv/bin/python scripts/diagnostics/platform_equivalence.py samenoise --device cuda --product segment --label cuda-seg
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
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

OUT = ROOT / "artifacts/runs/platform"
NOISE_SEED = 20261201


def _load():
    from flymario import config
    from flymario.connectome import graphsel
    from flymario.models.spiking_reservoir import LIFParams
    g, W, motor = graphsel.load("connectome")
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    return g, W, motor, params, int(config.get("timing.ticks_per_decision"))


def _streams(g):
    from common_stream import stream
    tr_cur, tr_lab, _ = stream(g, ("Level1-1", "Level2-1", "Level4-1", "Level7-1"), 2500, 20261073)
    order = np.random.default_rng(20261074).permutation(tr_cur.shape[1])[:4000]
    te_cur, te_lab, _ = stream(g, ("Level5-1", "Level8-1"), 1500, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])
    return tr_cur, tr_lab, order, te_cur[:, keep], te_lab[keep]


def _write(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, p)
    print("wrote", p, flush=True)


# --------------------------------------------------------------------------- N1 / N2
def operator(_a):
    import scipy.sparse as sp
    import torch

    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    g, W, motor, params, K = _load()
    st = torch.load(ROOT / "artifacts/rocm_reference/checkpoints/common_stream_connectome.pt",
                    map_location="cpu", weights_only=False)
    m = st["brain"]["m"].float()
    frac = float((m != m.round()).float().mean())
    os.environ["FLYMARIO_PLASTIC_PRODUCT"] = "segment"
    b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], cache_key=("stream", "connectome"))
    b.set_weights(m.to(b.dev))
    tr_cur, _lab, order, _tc, _tl = _streams(g)
    # record presynaptic spike vectors (n, 3) of real sweeps: ticks 1..16 of 2 frames, keep 20 non-empty ones
    rec = []
    for f in order[:2]:
        cur = torch.as_tensor(np.repeat(tr_cur[:, f:f + 1], 3, 1), device=b.dev)
        from flymario.training.sweep import reset_network
        reset_network(b)
        for _ in range(K):
            b.step(cur)
            if bool(b.spikes.any()):
                rec.append(b.spikes_f.clone())
    rec = rec[:20]
    # float64 reference: signed plastic rows, per brain
    pre, post, sign, dev = b.pre.cpu().numpy(), b.post.cpu().numpy(), b.sign.cpu().numpy(), b.dev
    M, n = b.M, b.n
    mats = [sp.csr_matrix(((m[i].numpy().astype(np.float64) * sign), (post, pre)), shape=(M, n)) for i in range(3)]

    def seg(s):
        return torch.segment_reduce(b._wsig * s.index_select(0, b.pre), "sum", lengths=b._indeg, axis=0)

    vals = (b.m * b.sign[None, :]).reshape(-1)
    A = torch.sparse_csr_tensor(b._crow, b._col, vals, size=(M * 3, n * 3))

    def spmm(s):
        return (A @ s.t().reshape(-1, 1)).view(3, M).t()

    rows = []
    for t, s in enumerate(rec):
        s64 = s.cpu().numpy().astype(np.float64)
        x64 = np.stack([mats[i] @ s64[:, i] for i in range(3)], 1)
        scale = float(np.abs(x64).max())
        xs, xp = [seg(s) for _ in range(50)], [spmm(s) for _ in range(50)]
        same_s = sum(bool(torch.equal(xs[0], x)) for x in xs)
        same_p = sum(bool(torch.equal(xp[0], x)) for x in xp)
        es = float(np.abs(xs[0].cpu().numpy().astype(np.float64) - x64).max()) / scale
        ep = max(float(np.abs(x.cpu().numpy().astype(np.float64) - x64).max()) for x in xp) / scale
        rows.append({"tick": t, "active_pre": int(s.sum()), "max_abs_ref": scale, "rel_err_segment": es,
                     "rel_err_spmm_worst_of_50": ep, "segment_identical_of_50": same_s, "spmm_identical_of_50": same_p})
        print(rows[-1], flush=True)
    rep = {"what": "RL_STUDY §17 N1/N2: motor input of ROCm-trained fractional weights", "fractional_share": frac,
           "rows": rows, "max_rel_err_segment": max(r["rel_err_segment"] for r in rows),
           "max_rel_err_spmm": max(r["rel_err_spmm_worst_of_50"] for r in rows),
           "segment_all_identical": all(r["segment_identical_of_50"] == 50 for r in rows),
           "spmm_all_identical": all(r["spmm_identical_of_50"] == 50 for r in rows),
           "N1_pass": max(max(r["rel_err_segment"], r["rel_err_spmm_worst_of_50"]) for r in rows) <= 1e-5,
           "N2a_pass": all(r["segment_identical_of_50"] == 50 for r in rows),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    print({k: v for k, v in rep.items() if k != "rows"}, flush=True)
    _write("operator-N1N2.json", rep)


# --------------------------------------------------------------------------- N3
def samenoise(a):
    os.environ["FLYMARIO_PLASTIC_PRODUCT"] = a.product
    import torch
    from sklearn.metrics import roc_auc_score

    from common_stream import EBAR, ETA, JUMPS, KAPPA, PRESS, RD, RUN, TAU
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(a.threads)          # the cgroup quota is ~12 cores; default = 64 visible
    g, W, motor, params, K = _load()
    tr_cur, tr_lab, order, te_cur, te_lab = _streams(g)
    if a.permuted:
        tr_lab = tr_lab[np.random.default_rng(20261077).permutation(len(tr_lab))]
    b = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], device=a.device,
                   cache_key=None if a.device == "cpu" else ("stream", "connectome"))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(NOISE_SEED)
    orig_step = b.step

    def step(current=None, observe=False, noise=None):        # identical noise on every implementation
        z = torch.randn((b.n, b.B), generator=gen) * params.noise_std
        return orig_step(current, observe=observe, noise=z.to(b.dev))
    b.step = step
    ro = block_seeds(3)
    R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro]),
                        device=b.dev)
    first = {}

    def score():
        P = []
        for s0 in range(te_cur.shape[1]):
            c, _ = sweep_once(b, params, torch.as_tensor(np.repeat(te_cur[:, s0:s0 + 1], 3, 1), device=b.dev),
                              K, 0.5, False)
            if "spikes" not in first:
                first["spikes"] = c.sum(0).cpu().tolist()
            z = torch.einsum("bam,mb->ba", R, c) / TAU
            P.append(torch.softmax(z, 1)[:, JUMPS].sum(1).cpu().numpy())
        P = np.array(P)
        return [round(float(roc_auc_score(te_lab, P[:, i])), 4) for i in range(3)], P

    t0 = time.time()
    curve, probs = [], {}
    auc, P = score()
    curve.append({"frames": 0, "auc": auc})
    probs[0] = P.tolist()
    print(a.label, curve[-1], round(time.time() - t0), "s", flush=True)
    ebar, ms = None, torch.zeros(3, device=b.dev)
    for k, f in enumerate(order[:a.frames]):
        cur = torch.as_tensor(np.repeat(tr_cur[:, f:f + 1], 3, 1), device=b.dev)
        c, e = sweep_once(b, params, cur, K, 0.5, True)
        if ebar is None:
            ebar = e.clone()
        ec = (e - ebar) + KAPPA * ebar
        ebar.mul_(1 - EBAR).add_(e, alpha=EBAR)
        pi = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)
        tgt = torch.full((3,), PRESS if tr_lab[f] else RUN, device=b.dev, dtype=torch.long)
        oh = torch.nn.functional.one_hot(tgt, pi.shape[1]).float()
        L = torch.einsum("ba,bam->mb", oh - pi, R) / TAU
        U = ec * L.index_select(0, b.post) * b.sign[:, None]
        ms = RD * ms + (1 - RD) * (U * U).mean(0)
        U = U / torch.sqrt(ms / (1 - RD ** (k + 1)) + 1e-30)[None, :]
        b.update((U * ETA).t() * b.m0[None, :])
        if (k + 1) % a.every == 0:
            auc, P = score()
            curve.append({"frames": k + 1, "auc": auc})
            probs[k + 1] = P.tolist()
            print(a.label, curve[-1], round(time.time() - t0), "s", flush=True)
    dm = (b.m - b.m0[None, :]).cpu()
    rep = {"what": "RL_STUDY §17 N3: common stream with identical CPU-drawn noise", "label": a.label,
           "device": a.device, "product": a.product, "segment_path": bool(b.segment), "permuted": a.permuted,
           "frames": a.frames, "every": a.every, "noise_seed": NOISE_SEED, "curve": curve,
           "first_sweep_motor_spikes": first["spikes"], "pjump": probs,
           "dm_l2": [float(x) for x in dm.norm(dim=1)], "torch": torch.__version__,
           "seconds": round(time.time() - t0), "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    torch.save({"m": b.m.cpu()}, OUT / f"samenoise-{a.label}-m.pt")
    _write(f"samenoise-{a.label}.json", rep)


def lockstep(a):
    """A CPU brain and a CUDA brain trained in lock-step with the same noise every tick (docs/RL_STUDY.md §17).
    Per step: differing motor counts and relative differences of eligibility, update and weights,
    to tell a wrong learning computation from amplified rounding."""
    import torch

    from common_stream import EBAR, ETA, KAPPA, PRESS, RD, RUN, TAU
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    torch.set_num_threads(a.threads)
    g, W, motor, params, K = _load()
    tr_cur, tr_lab, order, _tc, _tl = _streams(g)
    os.environ["FLYMARIO_PLASTIC_PRODUCT"] = "spmm"
    bc = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], device="cpu")
    os.environ["FLYMARIO_PLASTIC_PRODUCT"] = "segment"
    bg = PlasticLIF(W, motor, params, seeds=[8300, 8301, 8302], device="cuda", cache_key=("stream", "connectome"))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(NOISE_SEED)
    shared = {}
    steps = {"cpu": bc.step, "cuda": bg.step}

    def mk(dev, brain):
        def step(current=None, observe=False, noise=None):
            if dev == "cpu":                     # the CPU brain draws; the CUDA brain reuses the same tensor
                shared["z"] = torch.randn((brain.n, brain.B), generator=gen) * params.noise_std
            return steps[dev](current, observe=observe, noise=shared["z"].to(brain.dev))
        return step
    ro = block_seeds(3)
    R = np.stack([np.asarray(FixedRandomProjection(bc.M, len(ACTION_NAMES), r).W) for r in ro])
    st = {d: {"b": b, "R": torch.as_tensor(R, device=b.dev), "ebar": None, "ms": torch.zeros(3, device=b.dev)}
          for d, b in (("cpu", bc), ("cuda", bg))}
    rows = []
    m0 = bc.m0[None, :].double()
    for k, f in enumerate(order[:a.frames]):
        out = {}
        # the CPU sweep draws the noise tick by tick; record it, then replay it for the CUDA sweep
        zs = []
        bc.step = lambda current=None, observe=False, noise=None: (
            zs.append(torch.randn((bc.n, bc.B), generator=gen) * params.noise_std),
            steps["cpu"](current, observe=observe, noise=zs[-1]))[1]
        it = iter(range(10 ** 9))
        bg.step = lambda current=None, observe=False, noise=None: steps["cuda"](
            current, observe=observe, noise=zs[next(it)].to(bg.dev))
        for d in ("cpu", "cuda"):
            s = st[d]
            b = s["b"]
            cur = torch.as_tensor(np.repeat(tr_cur[:, f:f + 1], 3, 1), device=b.dev)
            c, e = sweep_once(b, params, cur, K, 0.5, True)
            if s["ebar"] is None:
                s["ebar"] = e.clone()
            ec = (e - s["ebar"]) + KAPPA * s["ebar"]
            s["ebar"].mul_(1 - EBAR).add_(e, alpha=EBAR)
            pi = torch.softmax(torch.einsum("bam,mb->ba", s["R"], c) / TAU, 1)
            tgt = torch.full((3,), PRESS if tr_lab[f] else RUN, device=b.dev, dtype=torch.long)
            oh = torch.nn.functional.one_hot(tgt, pi.shape[1]).float()
            L = torch.einsum("ba,bam->mb", oh - pi, s["R"]) / TAU
            U = ec * L.index_select(0, b.post) * b.sign[:, None]
            s["ms"] = RD * s["ms"] + (1 - RD) * (U * U).mean(0)
            U = U / torch.sqrt(s["ms"] / (1 - RD ** (k + 1)) + 1e-30)[None, :]
            b.update((U * ETA).t() * b.m0[None, :])
            out[d] = (c.cpu().double(), e.cpu().double(), U.cpu().double())
        (cc, ec_, uc), (cg, eg, ug) = out["cpu"], out["cuda"]
        mc, mg = bc.m.double(), bg.m.cpu().double()
        rows.append({"step": k + 1, "motor_count_diff": float((cc - cg).abs().sum()),
                     "motor_cells_differing": int(((cc - cg).abs() > 0).sum()),
                     "elig_rel": float((ec_ - eg).norm() / ec_.norm()),
                     "update_rel": float((uc - ug).norm() / uc.norm()),
                     "weights_rel": float((mc - mg).norm() / max(1e-30, float((mc - m0).norm())))})
        print(rows[-1], flush=True)
    _write("lockstep-N3b.json", {"what": "§17 bounded investigation 1: CPU vs CUDA in lock-step, identical noise",
                                 "rows": rows, "utc": datetime.now(UTC).isoformat(timespec="seconds")})
    del mk


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("operator")
    ls = sub.add_parser("lockstep")
    ls.add_argument("--frames", type=int, default=40)
    ls.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("samenoise")
    s.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    s.add_argument("--product", default="segment", choices=["segment", "spmm"])
    s.add_argument("--permuted", action="store_true")
    s.add_argument("--frames", type=int, default=500)
    s.add_argument("--every", type=int, default=250)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--label", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"operator": operator, "samenoise": samenoise, "lockstep": lockstep}[a.cmd](a)


if __name__ == "__main__":
    main()
