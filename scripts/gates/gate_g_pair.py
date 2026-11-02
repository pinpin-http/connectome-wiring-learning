#!/usr/bin/env python
"""Gate G task: which two game scenes are the cues (docs/GATE_G_PLAN.md §2).

Picks, among the 45 pairs of the 10 calibration scenes, the pair with the largest decodability under the
Gate E instrument (gate_e_recert.py, frozen operating point), on seeds 1-12 only. Ties: higher mean
held-out probability of the true scene, then alphabetical order.

Run:  .venv/bin/python scripts/gates/gate_g_pair.py
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from flymario import config
from flymario.connectome import datasets
from flymario.env import stimulus_bank
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams
from flymario.models.torch_lif import TorchLIF
from flymario.paths import RUNS

SEEDS = list(range(1, 13))


def decode(X, y, groups, folds: int = 4) -> tuple[float, float]:
    """gate_e_recert.decode, plus the held-out probability of the true class."""
    n = min(folds, len(np.unique(groups)))
    accs, probs = [], []
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
        p = clf.predict_proba(sc.transform(X[te]))
        probs.append(float(np.mean(p[np.arange(len(te)), np.searchsorted(clf.classes_, y[te])])))
    return float(np.mean(accs)), float(np.mean(probs))


def main() -> None:
    import torch
    t0 = time.time()
    ds = datasets.DEFAULT
    g = datasets.load(ds)
    W = g.to_sparse()
    norm = bool(config.get("dynamics.weight_transform")["normalize_in_degree"])
    warm = int(config.get("dynamics.warmup_ticks"))
    tpd = int(config.get("timing.ticks_per_decision"))
    gain = float(config.get("dynamics.gain"))
    noise = float(config.get("dynamics.noise_std"))
    read = datasets.motor_index(g, ds)
    bank = stimulus_bank.build(which="calibration")
    vision = RetinotopicVision(g)
    names = sorted(bank.sequences)
    enc = []
    for nm in names:
        vision.reset()
        enc.append(vision.encode(bank.sequences[nm][0]))
    C, S = len(names), len(SEEDS)
    brain_seeds = [sd for _ in range(C) for sd in SEEDS]
    b = TorchLIF(W, LIFParams(gain=gain, noise_std=noise), batch=C * S, normalize=norm,
                 seeds=brain_seeds, noise="numpy")
    drive = torch.from_numpy(np.stack(enc, 1)).to(b.device).repeat_interleave(S, dim=1)
    read_t = torch.as_tensor(read, device=b.device, dtype=torch.long)
    for _ in range(warm):
        b.step(drive)
    X = torch.zeros((len(read), C * S), device=b.device)
    for _ in range(tpd):
        X += b.step(drive).index_select(0, read_t).float()
    X = X.t().cpu().numpy()
    y_all = np.repeat(np.arange(C), S)
    grp = np.tile(np.asarray(SEEDS), C)
    pairs = []
    for a, c in itertools.combinations(range(C), 2):
        m = (y_all == a) | (y_all == c)
        acc, prob = decode(X[m], (y_all[m] == c).astype(int), grp[m])
        pairs.append({"pair": [names[a], names[c]], "decode": acc, "p_true": prob})
        print(f"{names[a]:>22s} vs {names[c]:<22s} decode {acc:.3f}  p(true) {prob:.3f}", flush=True)
    ranked = sorted(pairs, key=lambda r: (-r["decode"], -r["p_true"], r["pair"]))
    all_dec, _ = decode(X, y_all, grp)
    rep = {"what": "Gate G cue pair (GATE_G_PLAN §2)", "seeds": SEEDS, "gain": gain,
           "noise": noise, "warmup": warm, "tpd": tpd, "scenes": names,
           "ten_scene_decode_same_run": all_dec,
           "rule": "max decode; ties by mean held-out p(true scene); then alphabetical",
           "chosen": ranked[0], "runner_up": ranked[1], "pairs": ranked,
           "n_perfect": int(sum(r["decode"] == 1.0 for r in pairs)),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "seconds": round(time.time() - t0, 1)}
    (RUNS / "gate_g_pair.json").write_text(json.dumps(rep, indent=2))
    print(f"\n10-scene decode (same run): {all_dec:.3f}  perfect pairs: {rep['n_perfect']}/45")
    print("chosen:", ranked[0], "\nrunner-up:", ranked[1])


if __name__ == "__main__":
    main()
