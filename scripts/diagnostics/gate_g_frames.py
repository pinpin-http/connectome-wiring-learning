#!/usr/bin/env python
"""Is "which scene" readable from one sweep on frames never seen in training?

Exploratory diagnostic (2026-10-02), no learning. The Gate G pair's two gameplay
sequences (48 decisions each, encoded in order so ON/OFF transients are
present); every frame is swept from rest on `--seeds` noise seeds. Motor spike
counts over ticks [0, K) are decoded by the project's float64 StandardScaler +
logistic C=.05 decoder under three splits, each with a shuffled-label reference:

    block        train frames 0-23, test frames 24-47 (the stage-1 learning test)
    interleaved  train even frames, test odd frames (neighbouring frames)
    noise        all frames, train on half the noise seeds, test the other half

`block` near `noise` means the scene identity generalises across frames and the
learner, not the representation, limited stage-1 test 3. `block` near chance
while `interleaved` is high means the late frames do not share the early
frames' distinguishing features: the test asked for something the input may
not contain. Also reported for comparison: the same splits on the injected
lamina current (the input itself), which bounds what any network could keep.

    .venv/bin/python scripts/diagnostics/gate_g_frames.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config  # noqa: E402
from flymario.connectome import datasets  # noqa: E402
from flymario.env import stimulus_bank  # noqa: E402
from flymario.env.vision import RetinotopicVision  # noqa: E402
from flymario.models.spiking_reservoir import LIFParams  # noqa: E402
from flymario.models.torch_lif import TorchLIF  # noqa: E402
from flymario.paths import RUNS  # noqa: E402
from flymario.run_manifest import RunManifest, dataset_files  # noqa: E402


def score(Xtr, ytr, Xte, yte, shuffled=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64)
    keep = Xtr.std(0) > 0
    if shuffled:
        ytr = np.random.default_rng(20261002).permutation(ytr)
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Xtr[:, keep]), ytr)
    return float(clf.score(sc.transform(Xte[:, keep]), yte))


def splits(X, scene, frame, seed, S):
    out = {}
    masks = {"block": (frame < 24, frame >= 24),
             "interleaved": (frame % 2 == 0, frame % 2 == 1),
             "noise": (seed < S // 2, seed >= S // 2)}
    for name, (tr, te) in masks.items():
        out[name] = {"decode": round(score(X[tr], scene[tr], X[te], scene[te]), 4),
                     "shuffled": round(score(X[tr], scene[tr], X[te], scene[te], True), 4),
                     "n_train": int(tr.sum()), "n_test": int(te.sum())}
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ticks", default="16,32")
    ap.add_argument("--tag", default="20261002")
    a = ap.parse_args()
    run_id = f"gate_g_frames-{a.tag}"
    out = RUNS / f"{run_id}.json"
    if out.exists():
        raise FileExistsError(out)
    g = datasets.load(datasets.DEFAULT)
    W, motor = g.to_sparse(), datasets.motor_index(g, datasets.DEFAULT)
    pair = json.loads((RUNS / "gate_g_pair.json").read_text())["chosen"]["pair"]
    bank = stimulus_bank.build(which="calibration")
    vision = RetinotopicVision(g)
    enc = []
    for nm in pair:
        vision.reset()
        enc.append(np.stack([vision.encode(f) for f in bank.sequences[nm]]))
    enc = np.stack(enc).astype(np.float32)                       # (2, F, n)
    F, S = enc.shape[1], a.seeds
    Ks = [int(k) for k in a.ticks.split(",")]
    manifest = RunManifest(run_id, "Gate G", "exploratory frame-generalisation observability",
                           vars(a), dataset=str(config.get("connectome.dataset")),
                           input_files={**dataset_files(), "script": Path(__file__)},
                           output_files={"report": out})
    # every (scene, frame, seed) is one brain; chunks of one scene-frame block
    scene = np.repeat(np.arange(2), F * S)
    frame = np.tile(np.repeat(np.arange(F), S), 2)
    seed = np.tile(np.arange(S), 2 * F)
    N = len(scene)
    counts = {k: np.zeros((N, len(motor)), np.float32) for k in Ks}
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    mot = torch.as_tensor(motor, device="cuda", dtype=torch.long)
    chunk = 192
    for s0 in range(0, N, chunk):
        idx = np.arange(s0, min(N, s0 + chunk))
        b = TorchLIF(W, p, batch=len(idx), normalize=False, seeds=[880000 + int(i) for i in idx],
                     noise="torch", cache_key="malecns")
        cur = torch.from_numpy(enc[scene[idx], frame[idx]].T.copy()).cuda()
        c = torch.zeros((len(motor), len(idx)), device="cuda")
        for t in range(max(Ks)):
            c += b.step(cur).index_select(0, mot).float()
            if t + 1 in Ks:
                counts[t + 1][idx] = c.t().cpu().numpy()
        del b
        torch.cuda.empty_cache()
        print(f"brains {idx[-1] + 1}/{N}", flush=True)
    res = {f"motor_K{k}": splits(counts[k], scene, frame, seed, S) for k in Ks}
    lam = np.concatenate([vision.on_idx, vision.off_idx])
    X_in = enc[scene, frame][:, lam]                              # the injected current itself
    res["input_lamina_current"] = splits(X_in, scene, frame, seed, S)
    rep = {"what": "frame generalisation of scene identity from one sweep", "pair": pair,
           "frames": F, "seeds": S, "noise_seeds": f"880000..{880000 + N - 1}",
           "results": res, "complete": True, "manifest": str(manifest.path),
           "scope": "exploratory diagnostic; no learning",
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    out.write_text(json.dumps(rep, indent=2))
    manifest.finish("complete")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
