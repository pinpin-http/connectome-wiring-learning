#!/usr/bin/env python
"""Noise-realisation spread of the common-stream curve on one platform (docs/RL_STUDY.md §17 N4).

The `cs-scripted` procedure of `common_stream.py`, batched: noise sets s = 0..8 (readout i gets
seed 8300 + 100 s + i; s = 0 is production), plus, on set 0, three brains with eta halved and
three with permuted labels. Writes `artifacts/runs/platform/cs_noise_spread-<tag>.json`; state is
saved at every scoring point and on SIGTERM (exit 75), and resumed with the same options.

    .venv/bin/python scripts/diagnostics/cs_noise_spread.py [--sets 9] [--tag nvidia]
"""
from __future__ import annotations

import json
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common_stream import EBAR, ETA, JUMPS, KAPPA, PRESS, RD, RUN, TAU, stream  # noqa: E402

from flymario import config  # noqa: E402
from flymario.env.retro_env import ACTION_NAMES  # noqa: E402
from flymario.models.spiking_reservoir import LIFParams  # noqa: E402
from flymario.paths import RUNS  # noqa: E402
from flymario.policy.readout import FixedRandomProjection, block_seeds  # noqa: E402
from flymario.training.plastic import PlasticLIF  # noqa: E402
from flymario.training.sweep import sweep_once  # noqa: E402


def brains(n_sets):
    out = [{"name": f"s{s}_ro{i + 1}", "ro": i, "noise": 8300 + 100 * s + i, "eta": ETA, "perm": False, "set": s}
           for s in range(n_sets) for i in range(3)]
    out += [{"name": f"eta_half_ro{i + 1}", "ro": i, "noise": 8300 + i, "eta": ETA / 2, "perm": False, "set": "eta_half"}
            for i in range(3)]
    out += [{"name": f"perm_ro{i + 1}", "ro": i, "noise": 8300 + i, "eta": ETA, "perm": True, "set": "perm"}
            for i in range(3)]
    return out


def main():
    import torch
    from sklearn.metrics import roc_auc_score

    from flymario.connectome import graphsel
    a = sys.argv
    n_sets = int(a[a.index("--sets") + 1]) if "--sets" in a else 9
    tag = a[a.index("--tag") + 1] if "--tag" in a else "nvidia"
    n_train, every = 4000, 500
    spec = "connectome"
    out_dir = RUNS / "platform"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cs_noise_spread-{tag}.json"
    state_path = out_dir / f"cs_noise_spread_state-{tag}.pt"
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def atomic(path, write):
        tmp = path.with_name(path.name + ".part")
        write(tmp)
        os.replace(tmp, path)

    g, W, motor = graphsel.load(spec)
    tr_cur, tr_lab, _ = stream(g, ("Level1-1", "Level2-1", "Level4-1", "Level7-1"), 2500, 20261073)
    order = np.random.default_rng(20261074).permutation(tr_cur.shape[1])[:n_train]
    perm_lab = tr_lab[np.random.default_rng(20261077).permutation(len(tr_lab))]
    te_cur, te_lab, _ = stream(g, ("Level5-1", "Level8-1"), 1500, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])
    te_cur, te_lab = te_cur[:, keep], te_lab[keep]
    print("stream", tr_cur.shape[1], "train frames, label rate", round(float(tr_lab.mean()), 3),
          "| test", te_cur.shape[1], round(float(te_lab.mean()), 3), flush=True)
    bl = brains(n_sets)
    B = len(bl)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    ro = block_seeds(3)
    b = PlasticLIF(W, motor, params, seeds=[x["noise"] for x in bl], cache_key=("stream", spec))
    R3 = np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro])
    R = torch.as_tensor(R3[[x["ro"] for x in bl]], device=b.dev)                      # (B, A, M)
    eta = torch.tensor([x["eta"] for x in bl], device=b.dev, dtype=torch.float32)
    is_perm = np.array([x["perm"] for x in bl])
    ebar, ms, curve, k0 = None, torch.zeros(B, device=b.dev), [], 0
    run_args = {"n_sets": n_sets, "n_train": n_train, "every": every, "brains": [x["name"] for x in bl]}

    def score():
        P = []
        for s0 in range(te_cur.shape[1]):
            c, _ = sweep_once(b, params, torch.as_tensor(np.repeat(te_cur[:, s0:s0 + 1], B, 1), device=b.dev),
                              K, 0.5, False)
            z = torch.einsum("bam,mb->ba", R, c) / TAU
            P.append(torch.softmax(z, 1)[:, JUMPS].sum(1).cpu().numpy())
        P = np.array(P)
        return [round(float(roc_auc_score(te_lab, P[:, i])), 4) for i in range(B)]

    def write(complete):
        rep = {"what": "noise-realisation spread of the common-stream curve (RL_STUDY §17 N4)",
               "platform": {"torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
               "brains": bl, "curve": curve, "complete": complete, "run_args": run_args,
               "test_label_rate": float(te_lab.mean()), "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        atomic(out, lambda p: p.write_text(json.dumps(rep, indent=1)))

    def save_state(k_done):
        atomic(state_path, lambda p: torch.save({"args": run_args, "k": k_done, "brain": b.state_dict(),
                                                 "ebar": None if ebar is None else ebar.cpu(), "ms": ms.cpu(),
                                                 "curve": curve}, p))

    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["args"] != run_args:
            sys.exit(f"{state_path} belongs to another configuration")
        b.load_state_dict({k: (v.to(b.dev) if hasattr(v, "to") and k != "gens" else v) for k, v in st["brain"].items()})
        ebar = None if st["ebar"] is None else st["ebar"].to(b.dev)
        ms, curve, k0 = st["ms"].to(b.dev), st["curve"], int(st["k"])
        print(f"resumed at frame {k0}", flush=True)
    else:
        curve.append({"frames": 0, "auc": score()})
        print(0, curve[-1]["auc"][:3], flush=True)
        write(False)
    for k, f in enumerate(order):
        if k < k0:
            continue
        cur = torch.as_tensor(np.repeat(tr_cur[:, f:f + 1], B, 1), device=b.dev)
        c, e = sweep_once(b, params, cur, K, 0.5, True)
        if ebar is None:
            ebar = e.clone()
        ec = (e - ebar) + KAPPA * ebar
        ebar.mul_(1 - EBAR).add_(e, alpha=EBAR)
        pi = torch.softmax(torch.einsum("bam,mb->ba", R, c) / TAU, 1)
        lab = np.where(is_perm, perm_lab[f], tr_lab[f])
        tgt = torch.as_tensor(np.where(lab, PRESS, RUN), device=b.dev, dtype=torch.long)
        oh = torch.nn.functional.one_hot(tgt, pi.shape[1]).float()
        L = torch.einsum("ba,bam->mb", oh - pi, R) / TAU
        U = ec * L.index_select(0, b.post) * b.sign[:, None]
        ms = RD * ms + (1 - RD) * (U * U).mean(0)
        U = U / torch.sqrt(ms / (1 - RD ** (k + 1)) + 1e-30)[None, :]
        b.update((U * eta[None, :]).t() * b.m0[None, :])
        if (k + 1) % every == 0:
            curve.append({"frames": k + 1, "auc": score()})
            print(k + 1, curve[-1]["auc"][:3], flush=True)
            save_state(k + 1)
            write(False)
        if stop["flag"]:
            save_state(k + 1)
            print(f"state saved at frame {k + 1}", flush=True)
            sys.exit(75)
    write(True)
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
