#!/usr/bin/env python
"""Common-stream learning diagnostic (docs/RL_STUDY.md §6): every brain learns from the same fixed frame stream.

Scripted-play frames on the training levels, take-off label v2 as target, Stage D supervised rule; every
`--every` frames, AUC of P(jump) on 300 held-out frames of levels 5-1 and 8-1.

    .venv/bin/python scripts/diagnostics/common_stream.py [--graphs connectome nulllat:0 nulllat:1]
        [--n 4000] [--every 500] [--save] [--permuted] [--stream expert]
        [--tag T] [--no-resume] [--steps S --test-steps S] [--stop-after K]

Per-graph state is checkpointed to `checkpoints/common_stream_state<suffix>_<graph>.pt` and on SIGTERM (exit 75);
reruns resume. --tag isolates outputs (required by --steps/--test-steps); --save writes trainer-format checkpoints;
--permuted shuffles the training labels (seed 20261077). Output: `common_stream-<tag>-20261009.json`.
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

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.training.plastic import PlasticLIF
from flymario.training.sweep import sweep_once

RUN, PRESS = ACTION_NAMES.index("RIGHT+B"), ACTION_NAMES.index("RIGHT+A+B")
JUMPS = [ACTION_NAMES.index(a) for a in ("RIGHT+A", "RIGHT+A+B", "A")]
TAU, ETA, KAPPA, EBAR, RD = 2.0, 0.001, 0.2, 0.05, 0.95


def stream(g, levels, steps, seed, every=2, kind="scripted"):
    """Frames of scripted play (run, jump with p .25), each with its take-off label v2.

    kind="expert" (RL_STUDY §14): a random run/NOOP prefix of 5-40 decisions, then a
    press-release expert that jumps while the label is on."""
    if kind == "expert":
        return expert_stream(g, levels, steps, seed, every)
    from flymario.env.retro_env import EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    from flymario.env.vision import RetinotopicVision
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    cur, lab, lv = [], [], []
    for L in levels:
        env = MarioEnv(EnvConfig(state=L))
        prev, _ = env.reset()
        for n in range(steps):
            obs, _r, term, trunc, _i = env.step(PRESS if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram()
            if n % every == 0 and ram[0xB5] == 1:
                vis.reset()
                vis.encode(prev)
                cur.append(vis.encode(obs).astype(np.float32))
                lab.append(bool(takeoff_label_v2(ram)))
                lv.append(L)
            prev = obs
            if term or trunc:
                prev, _ = env.reset()
        env.close()
    return np.stack(cur, 1), np.array(lab), np.array(lv)


def expert_stream(g, levels, steps, seed, every=2):
    from flymario.env.retro_env import EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    from flymario.env.vision import RetinotopicVision
    noop = ACTION_NAMES.index("NOOP")
    rng = np.random.default_rng(seed)
    vis = RetinotopicVision(g)
    cur, lab, lv, blocked = [], [], [], []
    for L in levels:
        env = MarioEnv(EnvConfig(state=L))
        n = 0
        while n < steps:
            prev, _ = env.reset()
            for _ in range(int(rng.integers(5, 41))):
                prev, _r, term, trunc, _i = env.step(RUN if rng.random() < 0.6 else noop)
                if term or trunc:
                    break
            ram, held, xs = env.env.unwrapped.get_ram(), 0, []
            while n < steps:
                a = PRESS if takeoff_label_v2(ram) and held < 6 else RUN
                held = held + 1 if a == PRESS else 0
                obs, _r, term, trunc, info = env.step(a)
                ram = env.env.unwrapped.get_ram()
                xs.append(info["mario_x"])
                if n % every == 0 and ram[0xB5] == 1:
                    vis.reset()
                    vis.encode(prev)
                    cur.append(vis.encode(obs).astype(np.float32))
                    lab.append(bool(takeoff_label_v2(ram)))
                    lv.append(L)
                    blocked.append(len(xs) >= 5 and xs[-1] - xs[-5] <= 2)
                prev = obs
                n += 1
                if term or trunc or info.get("died"):
                    break
        env.close()
    lab, blocked = np.array(lab), np.array(blocked)
    print(f"expert stream: {len(lab)} frames, label on {lab.mean():.3f}, blocked among label-on "
          f"{blocked[lab].mean():.3f}", flush=True)
    return np.stack(cur, 1), lab, np.array(lv)


def main():
    import torch
    from sklearn.metrics import roc_auc_score

    from flymario.connectome import graphsel
    a = sys.argv
    graphs = [x for x in a[a.index("--graphs") + 1:] if ":" in x or x == "connectome"] if "--graphs" in a \
        else ["connectome", "nulllat:0", "nulllat:1"]
    n_train = int(a[a.index("--n") + 1]) if "--n" in a else 4000
    every = int(a[a.index("--every") + 1]) if "--every" in a else 500
    save, permuted = "--save" in a, "--permuted" in a
    kind = a[a.index("--stream") + 1] if "--stream" in a else "scripted"        # scripted | expert (§14)
    # --block B (RL_STUDY §23): an independent block = new readouts block_seeds(3(B+1))[3B:]
    # and new noise seeds 8300 + 1000 B + i
    blk = int(a[a.index("--block") + 1]) if "--block" in a else 0
    # --gain-file F (RL_STUDY §25): each graph runs at the gain F[spec]["gain"] (motor activity
    # matched to the connectome's); outputs carry "rm"
    gain_file = json.loads(Path(a[a.index("--gain-file") + 1]).read_text()) if "--gain-file" in a else None
    tag = a[a.index("--tag") + 1] if "--tag" in a else ""
    steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 2500
    test_steps = int(a[a.index("--test-steps") + 1]) if "--test-steps" in a else 1500
    stop_after = int(a[a.index("--stop-after") + 1]) if "--stop-after" in a else None
    resume = "--no-resume" not in a
    if (steps, test_steps) != (2500, 1500) and not tag:
        sys.exit("--steps / --test-steps change the data: only allowed with --tag (test runs)")
    suffix = ("_expert" if kind == "expert" else "") + (f"b{blk}" if blk else "") + ("rm" if gain_file else "") + ("_perm" if permuted else "") + (f"_{tag}" if tag else "")
    run_args = {"n_train": n_train, "every": every, "kind": kind, "permuted": permuted,
                "steps": steps, "test_steps": test_steps}
    if blk:
        run_args["block"] = blk
    if gain_file:
        run_args["gain_file"] = {sp: gain_file[sp]["gain"] for sp in graphs}
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
        print(f"signal {signum}: saving the state after the current frame", flush=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    def atomic_save(obj, path):
        tmp = path.with_name(path.name + ".part")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_state(path, spec, k_done, b, ebar, ms, curve):
        atomic_save({"spec": spec, "args": run_args, "k": k_done, "brain": b.state_dict(),
                     "ebar": None if ebar is None else ebar.detach().cpu(), "ms": ms.detach().cpu(),
                     "curve": list(curve), "utc": datetime.now(UTC).isoformat(timespec="seconds")}, path)
    g, _W, motor = graphsel.load("connectome")
    del _W
    tr_cur, tr_lab, _ = stream(g, ("Level1-1", "Level2-1", "Level4-1", "Level7-1"), steps, 20261073, kind=kind)
    order = np.random.default_rng(20261074).permutation(tr_cur.shape[1])[:n_train]
    if permuted:                                   # same frames and order, labels shuffled
        tr_lab = tr_lab[np.random.default_rng(20261077).permutation(len(tr_lab))]
    te_cur, te_lab, te_lv = stream(g, ("Level5-1", "Level8-1"), test_steps, 20261075, every=3)
    keep = np.sort(np.random.default_rng(20261076).permutation(te_cur.shape[1])[:300])   # 300 held-out frames
    CHECKPOINTS = RUNS / "checkpoints"             # where the trainer keeps its checkpoints
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    te_cur, te_lab, te_lv = te_cur[:, keep], te_lab[keep], te_lv[keep]
    print("stream", tr_cur.shape[1], "train frames, label rate", round(float(tr_lab.mean()), 3),
          "| test", te_cur.shape[1], round(float(te_lab.mean()), 3), flush=True)
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    res = {}
    for spec in graphs:
        _, W, _ = graphsel.load(spec)
        if gain_file:
            params = LIFParams(gain=float(gain_file[spec]["gain"]), noise_std=float(config.get("dynamics.noise_std")))
            print(spec, "matched-regime gain", params.gain, flush=True)
        ro = block_seeds(3 * (blk + 1))[3 * blk:]
        nseeds = [8300 + 1000 * blk + i for i in range(3)]
        b = PlasticLIF(W, motor, params, seeds=nseeds, cache_key=("stream", spec))
        R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro]),
                            device=b.dev)
        ebar, ms = None, torch.zeros(3, device=b.dev)
        curve = []

        def score(b, R):
            P = []
            for s0 in range(0, te_cur.shape[1], 1):
                c, _ = sweep_once(b, params, torch.as_tensor(np.repeat(te_cur[:, s0:s0 + 1], 3, 1), device=b.dev),
                                  K, 0.5, False)
                z = torch.einsum("bam,mb->ba", R, c) / TAU
                P.append(torch.softmax(z, 1)[:, JUMPS].sum(1).cpu().numpy())
            P = np.array(P)
            return [round(float(roc_auc_score(te_lab, P[:, i])), 4) for i in range(3)]

        state_path = CHECKPOINTS / f"common_stream_state{suffix}_{spec.replace(':', '')}.pt"
        k0 = 0
        if resume and state_path.exists():
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            if st["args"] != run_args or st["spec"] != spec:
                sys.exit(f"{state_path} belongs to another configuration ({st['args']}); "
                         "remove it or pass --no-resume")
            b.load_state_dict({k: (v.to(b.dev) if hasattr(v, "to") and k != "gens" else v)
                               for k, v in st["brain"].items()})
            ebar = None if st["ebar"] is None else st["ebar"].to(b.dev)
            ms, curve, k0 = st["ms"].to(b.dev), st["curve"], int(st["k"])
            print(f"{spec}: resumed at frame {k0} from {state_path.name}", flush=True)
        else:
            curve.append({"frames": 0, "auc": score(b, R)})
            print(spec, curve[-1], flush=True)
        for k, f in enumerate(order):
            if k < k0:
                continue
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
            if (k + 1) % every == 0:
                curve.append({"frames": k + 1, "auc": score(b, R)})
                print(spec, curve[-1], flush=True)
                save_state(state_path, spec, k + 1, b, ebar, ms, curve)
            if stop["flag"] or (stop_after is not None and k + 1 == stop_after):
                save_state(state_path, spec, k + 1, b, ebar, ms, curve)
                print(f"{spec}: state saved at frame {k + 1}; relaunch with the same options to resume", flush=True)
                sys.exit(75)
        res[spec] = curve
        if save:
            pre = ("csperm" if permuted else "cs") + ("x" if kind == "expert" else "") + ("b" * (blk > 0))
            st = {"learners": [{"name": f"{pre}_ro{i + 1}", "readout_seed": int(ro[i]), "noise_seed": nseeds[i],
                                "view": "pixels"} for i in range(3)],
                  "brain": {"m": b.m.detach().cpu().clone()}, "graph": spec, "frames": n_train,
                  "what": "common-stream brains (RL_STUDY §6.1 / §12), " + ("permuted labels" if permuted else "take-off label v2")}
            st["stream"] = kind
            atomic_save(st, CHECKPOINTS / f"common_stream{suffix}_{spec.replace(':', '')}.pt")
        state_path.unlink(missing_ok=True)
        del b, W
        torch.cuda.empty_cache()
    rep = {"what": "supervised learning on one common stream of frames and take-off labels (RL_STUDY §6 diagnostic)",
           "n_train": n_train, "test_frames": int(te_cur.shape[1]), "test_label_rate": float(te_lab.mean()),
           "results": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    rep["run_args"] = run_args
    if gain_file:
        rep["gains"] = run_args["gain_file"]
    if save or permuted or kind == "expert" or tag:
        name = (("expert" + (f"b{blk}" if blk else "") + ("rm" if gain_file else "") + "-" if kind == "expert" else "") + ("perm-" if permuted else "")
                + "-".join(x.replace(":", "") for x in graphs) + (f"-{tag}" if tag else ""))
        rep["permuted"] = permuted
        out = RUNS / f"common_stream-{name}-20261009.json"
    else:
        out = RUNS / "common_stream-20261007.json"
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
