#!/usr/bin/env python
"""Gates C and D, re-certified for lin_raw at the Gate E operating point (fraction 0.0158, gain 0.18452).

    python scripts/gates/gate_cd_recert.py          # GPU, live: make follow

C1 settling: cross-seed SD of the 16-tick rolling global rate < 0.1 pt by the warm-up (150 ticks).
C2 distinct: mean pairwise cosine between 10 scenes' seed-averaged motor states < 0.9.
C3 input-driven: with no drive the motor readout is exactly silent.
D  coherence: at 16 ticks, on two seed sets (1-10, 11-20), the scene decoder beats shuffled-label
   and no-input controls by > 2 SE (SE = max(binomial SE, half the gap between sets)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from flymario import config
from flymario.connectome import datasets
from flymario.dynamics.calibrate import interface_drive
from flymario.env import stimulus_bank
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.models.torch_lif import TorchLIF
from flymario.paths import ROOT, RUNS
from flymario.runctl import RunControl, StopRequested

NOISE = 0.02
TICKS_D = (8, 16, 32, 64)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def decode(X, y, groups, seed: int = 0) -> float:
    """scripts/diagnostics/decode.py's decoder, unchanged."""
    n_splits = min(len(np.unique(groups)), 5)
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(float(clf.score(sc.transform(X[te]), y[te])))
    return float(np.mean(accs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--amp", type=float, default=2.5, help="as decode.py")
    p.add_argument("--classes", type=int, default=4, help="as decode.py")
    args = p.parse_args()

    t0 = time.perf_counter()
    import torch
    g = datasets.load(args.dataset)
    W = g.to_sparse()
    wt = config.get("dynamics.weight_transform")
    norm = bool(wt["normalize_in_degree"])
    gain = float(config.get("dynamics.gain"))
    warm = int(config.get("dynamics.warmup_ticks"))
    target = float(config.get("dynamics.target_rate"))
    read = datasets.motor_index(g, args.dataset)
    bank = stimulus_bank.build(which="calibration")
    vision = RetinotopicVision(g)
    names = sorted(bank.sequences)
    n = g.n_nodes
    rep = {"gain": gain, "transform": wt, "warmup": warm, "C": {}, "D": {}}
    out = RUNS / "gate_cd_recert.json"

    def flush():
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=2))
        tmp.replace(out)

    tasks = ["C1 settling", "C2 distinct states", "C3 silent without input",
             "D seeds 1-10", "D seeds 11-20"]
    ctl = RunControl("gate_cd_recert", total=len(tasks), checkpoint=flush, unit="task", meta={
        "question": "Is the brain still well-behaved, and still fast enough, with the new conversion?",
        "why": ("Gates C and D were certified with the old conversion. C asks whether the network "
                "settles, responds to its input and nothing else; D asks whether a change on the "
                "screen reaches the motor neurons within one decision (16 ticks)."),
        "method": ("The same examinations as the first time, at Gate E's new operating point, on "
                   "the GPU. Gate D on two independent sets of seeds, after Gate E's peak "
                   "vanished on fresh ones.")})
    for t in tasks:
        ctl.task(t, "todo")
    ctl.chart("settle", "Does the brain settle? (8 seeds)", "tick", "global firing rate",
              ["seed"], yfmt="pct", refs=[{"axis": "x", "value": warm, "label": "warm-up ends"},
                                          {"axis": "y", "value": target, "label": "target"}],
              note="Each line is one noise seed under real game frames. They must merge before 150.")
    ctl.chart("gated", "Does a change reach the motor neurons in time?", "ticks after the change",
              "decoder accuracy", ["seeds 1-10", "seeds 11-20"],
              refs=[{"axis": "y", "value": 1 / args.classes, "label": "chance"},
                    {"axis": "x", "value": 16, "label": "one decision"}],
              note="One decision is 16 ticks. Two independent sets of seeds, same stimuli.")

    try:
        # ---- C1 -------------------------------------------------------------
        ctl.begin("C1 settling"); ctl.task("C1 settling", "running")
        seeds8 = list(range(1, 9))
        at = interface_drive(g, args.dataset, bank=bank)
        b = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE), batch=8, normalize=norm,
                     seeds=seeds8, noise="numpy", cache_key=("cd", 0))
        traj = []
        for t in range(300):
            traj.append(b.step(at(t)).float().mean(0).cpu().numpy())
            if t % 30 == 0:
                ctl.sub(t, 300, "tick")
        traj = np.array(traj)                                    # (300, 8)
        k = 16
        roll = np.stack([np.convolve(traj[:, j], np.ones(k) / k, mode="valid") for j in range(8)], 1)
        sd = roll.std(1, ddof=1)                                 # per rolling window end
        ends = np.arange(k - 1, 300)
        conv = None
        for i in range(len(sd)):
            if (sd[i:] < 0.001).all():
                conv = int(ends[i]); break
        win = traj[warm:warm + 60].mean(0)
        c1 = {"convergence_tick": conv, "cross_seed_sd_150_210_pts": float(win.std(ddof=1) * 100),
              "mean_rate_150_210": float(win.mean())}
        c1["pass"] = bool(conv is not None and conv <= warm and c1["cross_seed_sd_150_210_pts"] < 0.1)
        rep["C"]["C1"] = c1
        for j in range(8):
            for t in range(0, 300, 3):
                ctl.point("settle", "seed", t, float(traj[t, j]), line=f"seed {j + 1}")
        flush(); ctl.task("C1 settling", "done")
        ctl.done({"converged_at": conv, "sd_pts": round(c1["cross_seed_sd_150_210_pts"], 3)},
                 label="C1 settling", kind="C")
        print(f"C1 settling: converged at tick {conv}; cross-seed SD over 150-210 "
              f"{c1['cross_seed_sd_150_210_pts']:.3f} pts; rate {win.mean() * 100:.2f}% "
              f"-> {'PASS' if c1['pass'] else 'FAIL'}", flush=True)

        # ---- C2 -------------------------------------------------------------
        ctl.begin("C2 distinct states"); ctl.task("C2 distinct states", "running")
        C, S = len(names), 8
        enc = []
        for nm in names:
            vision.reset()
            enc.append(vision.encode(bank.sequences[nm][0]))
        bs = [sd for _ in range(C) for sd in seeds8]
        b = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE), batch=C * S, normalize=norm,
                     seeds=bs, noise="numpy", cache_key=("cd", 0))
        drive = torch.from_numpy(np.stack(enc, 1)).to(b.device).repeat_interleave(S, dim=1)
        read_t = torch.as_tensor(read, device=b.device, dtype=torch.long)
        for t in range(warm):
            b.step(drive)
        cnt = torch.zeros((len(read), C * S), device=b.device)
        for _ in range(16):
            cnt += b.step(drive).index_select(0, read_t).float()
        M = cnt.t().cpu().numpy().reshape(C, S, -1).mean(1)       # (scenes, motor)
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        cos = Mn @ Mn.T
        iu = np.triu_indices(C, 1)
        c2 = {"mean_pairwise_cosine": float(cos[iu].mean()), "max_pairwise_cosine": float(cos[iu].max())}
        c2["pass"] = bool(c2["mean_pairwise_cosine"] < 0.9)
        rep["C"]["C2"] = c2
        flush(); ctl.task("C2 distinct states", "done")
        ctl.done({"mean_cosine": round(c2["mean_pairwise_cosine"], 3)}, label="C2 distinct states", kind="C")
        print(f"C2 distinct: mean pairwise cosine {c2['mean_pairwise_cosine']:.3f} "
              f"(max {c2['max_pairwise_cosine']:.3f}) -> {'PASS' if c2['pass'] else 'FAIL'}", flush=True)

        # ---- C3 -------------------------------------------------------------
        ctl.begin("C3 silent without input"); ctl.task("C3 silent without input", "running")
        b = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE), batch=8, normalize=norm,
                     seeds=seeds8, noise="numpy", cache_key=("cd", 0))
        motor_spikes, any_spikes = 0, 0
        for _ in range(warm + 60):
            s = b.step(None)
            motor_spikes += int(s.index_select(0, read_t).sum())
            any_spikes += int(s.sum())
        c3 = {"motor_spikes": motor_spikes, "all_spikes": any_spikes, "pass": motor_spikes == 0}
        rep["C"]["C3"] = c3
        flush(); ctl.task("C3 silent without input", "done")
        ctl.done({"motor_spikes": motor_spikes}, label="C3 silent", kind="C")
        print(f"C3 input-driven: {motor_spikes} motor spikes, {any_spikes} spikes anywhere, "
              f"no drive, 8 seeds x {warm + 60} ticks -> {'PASS' if c3['pass'] else 'FAIL'}", flush=True)

        # ---- D --------------------------------------------------------------
        lam = datasets.by_types(g, args.dataset, "lamina_types")
        rng = np.random.default_rng(0)                          # decode.py --stim-seed 0
        stimuli = [rng.choice(lam, len(lam) // 3, replace=False) for _ in range(args.classes)]
        cur = np.zeros((n, args.classes), np.float32)
        for ci, st in enumerate(stimuli):
            cur[st, ci] = args.amp
        for name, seeds in (("D seeds 1-10", list(range(1, 11))),
                            ("D seeds 11-20", list(range(11, 21)))):
            ctl.begin(name); ctl.task(name, "running")
            K, S = args.classes, len(seeds)
            bs = [sd for _ in range(K) for sd in seeds]
            y = np.repeat(np.arange(K), S)
            grp = np.tile(np.asarray(seeds), K)
            res = {}
            for on in (True, False):
                b = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE), batch=K * S,
                             normalize=norm, seeds=bs, noise="numpy", cache_key=("cd", 0))
                dr = torch.from_numpy(cur).to(b.device).repeat_interleave(S, dim=1)
                for _ in range(warm):
                    b.step(None)                                 # silent warm-up, as decode.py
                cnt = torch.zeros((len(read), K * S), device=b.device)
                snaps = {}
                for t in range(1, max(TICKS_D) + 1):
                    cnt += b.step(dr if on else None).index_select(0, read_t).float()
                    if t in TICKS_D:
                        snaps[t] = cnt.t().cpu().numpy().copy()
                    if t % 16 == 0:
                        ctl.sub(t, max(TICKS_D), "tick")
                for t, X in snaps.items():
                    r = res.setdefault(t, {})
                    if on:
                        r["decode"] = decode(X, y, grp)
                        r["shuffled"] = decode(X, np.random.default_rng(1).permutation(y), grp)
                    else:
                        r["no_input"] = decode(X, y, grp)
            rep["D"][name] = {str(t): v for t, v in res.items()}
            for t, v in res.items():
                ctl.point("gated", name.replace("D ", ""), t, v["decode"])
            flush(); ctl.task(name, "done")
            ctl.done({"decode_16": round(res[16]["decode"], 3)}, label=name, kind="D")
            print(f"{name}: " + "  ".join(f"{t} ticks {v['decode']:.3f} (sh {v['shuffled']:.2f}, "
                                            f"no {v['no_input']:.2f})" for t, v in sorted(res.items())),
                  flush=True)
    except StopRequested:
        flush(); ctl.close("stopped"); print("stopped; relaunch to rerun"); return

    # ---- D verdict, with the honest uncertainty -----------------------------
    a, bb = rep["D"]["D seeds 1-10"]["16"], rep["D"]["D seeds 11-20"]["16"]
    nsamp = args.classes * 10
    chance = 1 / args.classes
    binom = max(np.sqrt(a["decode"] * (1 - a["decode"]) / nsamp),
                np.sqrt(bb["decode"] * (1 - bb["decode"]) / nsamp))
    gap = abs(a["decode"] - bb["decode"]) / 2
    se = max(binom, gap)
    ok = all(v["decode"] - max(v["shuffled"], v["no_input"]) > 2 * se
             and abs(v["shuffled"] - chance) <= 2 * max(se, np.sqrt(chance * (1 - chance) / nsamp))
             and abs(v["no_input"] - chance) <= 2 * max(se, np.sqrt(chance * (1 - chance) / nsamp))
             for v in (a, bb))
    rep["D"]["verdict"] = {"se_used": float(se), "binomial_se": float(binom),
                           "half_gap_between_seed_sets": float(gap), "pass": bool(ok)}
    rep["C"]["pass"] = bool(all(rep["C"][k]["pass"] for k in ("C1", "C2", "C3")))
    rep["provenance"] = {"utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
                         "commit": git_sha(), "backend": "gpu",
                         "graph_sha256": hashlib.sha256(datasets.cache_path(args.dataset).read_bytes()).hexdigest(),
                         "wall_seconds": round(time.perf_counter() - t0, 1)}
    flush(); ctl.close("complete")
    print("=" * 70)
    print(f"GATE C {'PASS' if rep['C']['pass'] else 'FAIL'}   ·   GATE D {'PASS' if ok else 'FAIL'} "
          f"(16 ticks: {a['decode']:.3f} and {bb['decode']:.3f}; SE used {se:.3f} = "
          f"max(binomial {binom:.3f}, half-gap {gap:.3f}))")
    print("=" * 70, f"\n-> {out}")


if __name__ == "__main__":
    main()
