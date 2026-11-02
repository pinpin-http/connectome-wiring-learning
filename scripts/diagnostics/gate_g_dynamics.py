#!/usr/bin/env python
"""Why the continuous Gate G task fails: does the network represent the CURRENT cue?

Exploratory diagnostics (2026-09-30), no weight update anywhere. Three modes:

`switch`  Cue-switch test. Reference brains start from rest and hold cue A (or
          B) for `warm + nwin*win` ticks. Switched brains hold A for `warm`
          ticks, then B. For each depth stratum and each window after the
          switch, the index

              f = <(x - mu_from) . (mu_to - mu_from)> / |mu_to - mu_from|^2

          is 0 if the switched brains still look like the OLD cue and 1 if they
          look like fresh brains on the NEW cue (mu: reference means at the same
          window). `reset_t4` counts neurons whose cue contrast between the two
          reference groups has |t| > 4; `switched_t4` counts neurons separating
          the two switched groups; `agree` counts those whose switched contrast
          has the sign of the NEW cue's reference contrast. A network that
          follows its input has f -> 1 and agree ~ switched_t4; a locked one has
          f ~ 0 and agree ~ 0.
          `--variants` runs several model variants in one invocation.

`decode`  Sweep regime. Fresh brains (reset) see one of the ten calibration
          scenes; motor spike counts over ticks [0, K) are decoded (10 classes)
          by the project's C=.05 standardised logistic decoder, trained on half
          the noise seeds and scored on the other half. Matched comparator on
          the SAME brains: the Gate E instrument's window, counts over ticks
          [150, 166). A shuffled-label decoder is the negative reference.

`smooth`  Along a real gameplay sequence (48 consecutive decisions, encoded in
          order so ON/OFF transients are present), how similar are the sweep
          responses of neighbouring frames, compared with the same frame under
          different noise? Exact-frame identification across seed halves.

Model variants (`switch`) are exploratory probes of the frozen LIF, not
candidate operating points: gain, weight transform, short-term depression
(Tsodyks-Markram style resources x_j, recovery 1/tau_rec per tick, use U per
spike), spike-frequency adaptation (threshold + beta * a, a decays by
exp(-1/tau) per tick), and a graded-rate stand-in (x <- lam x + (1-lam)(g W
phi(x) + u), phi = clip(x, 0, 1)).

    .venv/bin/python scripts/diagnostics/gate_g_dynamics.py switch --variants all
    .venv/bin/python scripts/diagnostics/gate_g_dynamics.py decode
    .venv/bin/python scripts/diagnostics/gate_g_dynamics.py smooth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.connectome import datasets, strata
from flymario.env import stimulus_bank
from flymario.env.vision import RetinotopicVision
from flymario.models.spiking_reservoir import LIFParams, normalize_in_degree
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl

PAIR = ("1-1_jumping", "1-1_open")        # the Gate G pair (artifacts/runs/gate_g_pair.json)

#: name -> keyword overrides of the frozen operating point. Declared before the
#: runs; every one is a probe, none is proposed as a new operating point.
VARIANTS = {
    "frozen":        {},
    "gain0.12":      {"gain": 0.12},
    "gain0.08":      {"gain": 0.08},
    "gain0.05":      {"gain": 0.05},
    "gain0.03":      {"gain": 0.03},
    "lin_norm10":    {"transform": "lin_norm", "gain": 10.0},
    "lin_norm20":    {"transform": "lin_norm", "gain": 20.0},
    "lin_norm40":    {"transform": "lin_norm", "gain": 40.0},
    "log_norm12":    {"transform": "log_norm", "gain": 12.0},
    "log_norm20":    {"transform": "log_norm", "gain": 20.0},
    "log_norm40":    {"transform": "log_norm", "gain": 40.0},
    "std_U.2_t30":   {"U": 0.2, "tau_rec": 30.0},
    "std_U.5_t30":   {"U": 0.5, "tau_rec": 30.0},
    "std_U.2_t100":  {"U": 0.2, "tau_rec": 100.0},
    "sfa_b.5_t30":   {"sfa_beta": 0.5, "sfa_tau": 30.0},
    "sfa_b2_t30":    {"sfa_beta": 2.0, "sfa_tau": 30.0},
    "sfa_b1_t100":   {"sfa_beta": 1.0, "sfa_tau": 100.0},
    "rate_lin_norm": {"model": "rate", "transform": "lin_norm", "gain": 0.9,
                      "noise": 0.05, "inscale": 0.5},
}

_OPS: dict = {}


def operator(transform: str):
    """The recurrent operator as a GPU CSR tensor, cached per transform."""
    import torch
    if transform in _OPS:
        return _OPS[transform]
    g = datasets.load(datasets.DEFAULT)
    W = g.to_sparse().tocsr().astype(np.float32)
    if transform.startswith("log"):
        W.data = np.sign(W.data) * np.log1p(np.abs(W.data))
    if transform.endswith("norm"):
        W = normalize_in_degree(W).tocsr()
    W.sort_indices()
    T = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int32)),
                                torch.from_numpy(W.indices.astype(np.int32)),
                                torch.from_numpy(W.data.astype(np.float32)),
                                size=W.shape).to("cuda")
    _OPS[transform] = (T, W.shape[0])
    return _OPS[transform]


class Sim:
    """B brains, same equation as TorchLIF/LIFBrain, plus optional STD and SFA,
    or a graded-rate stand-in (model='rate')."""

    def __init__(self, B, seeds, gain=None, noise=None, U=0.0, tau_rec=60.0,
                 sfa_beta=0.0, sfa_tau=60.0, transform="lin_raw", model="lif",
                 inscale=1.0):
        import torch
        self.torch = torch
        p = LIFParams()
        self.W, self.n = operator(transform)
        self.B, self.model, self.inscale = B, model, float(inscale)
        self.gain = float(config.get("dynamics.gain") if gain is None else gain)
        self.noise = float(config.get("dynamics.noise_std") if noise is None else noise)
        self.U, self.rec = float(U), 1.0 / float(tau_rec)
        self.sfa_beta, self.rho = float(sfa_beta), float(np.exp(-1.0 / sfa_tau))
        self.decay, self.refr, self.thr = float(p.decay), int(p.refractory_steps), float(p.threshold)
        self.gen = []
        for s in seeds:
            gg = torch.Generator(device="cuda")
            gg.manual_seed(int(s))
            self.gen.append(gg)
        n = self.n
        self.v = torch.zeros((n, B), device="cuda")
        self.ref = torch.zeros((n, B), device="cuda", dtype=torch.int32)
        self.s = torch.zeros((B, n), device="cuda").t()
        self.x = torch.ones((n, B), device="cuda")
        self.a = torch.zeros((n, B), device="cuda")

    def _z(self):
        return self.torch.stack([self.torch.randn(self.n, generator=g, device="cuda")
                                 for g in self.gen], 1)

    def step(self, cur):
        cur = cur * self.inscale
        if self.model == "rate":
            drive = (self.W @ self.s) * self.gain + cur
            self.v.mul_(self.decay).add_(drive * (1 - self.decay))
            if self.noise > 0:
                self.v.add_(self._z() * self.noise)
            self.s.copy_(self.v.clamp(0, 1))
            return self.s
        self.v.mul_(self.decay)
        pre = self.s * self.x if self.U > 0 else self.s
        self.v.add_((self.W @ pre) * self.gain)
        self.v.add_(cur)
        if self.noise > 0:
            self.v.add_(self._z() * self.noise)
        thr = self.thr + self.sfa_beta * self.a if self.sfa_beta > 0 else self.thr
        sp = (self.ref <= 0) & (self.v >= thr)
        self.v.masked_fill_(sp, 0.0)
        self.ref.sub_(1).clamp_(min=0)
        self.ref.masked_fill_(sp, self.refr)
        sf = sp.float()
        if self.U > 0:
            self.x.add_((1 - self.x) * self.rec).sub_(self.U * self.x * sf)
        if self.sfa_beta > 0:
            self.a.mul_(self.rho).add_(sf)
        self.s.copy_(sf)
        return self.s


def inputs(which: str = "first"):
    """Encoded calibration scenes. 'first': each scene's first frame from a reset
    eye (the Gate E / Gate G cue); 'seq': whole 48-decision sequences in order."""
    g = datasets.load(datasets.DEFAULT)
    vision = RetinotopicVision(g)
    bank = stimulus_bank.build(which="calibration")
    names = sorted(bank.sequences)
    out = []
    for nm in names:
        vision.reset()
        if which == "first":
            out.append(vision.encode(bank.sequences[nm][0]))
        else:
            out.append(np.stack([vision.encode(f) for f in bank.sequences[nm]]))
    return names, np.stack(out).astype(np.float32), g


def depth_groups(g) -> dict:
    ds = datasets.DEFAULT
    motor = datasets.motor_index(g, ds)
    vision = RetinotopicVision(g)
    lam = np.concatenate([vision.on_idx, vision.off_idx])
    st = strata.compute(g, lam, motor, key="both")
    groups = {f"depth{k}": np.flatnonzero(st.from_input == k) for k in range(5)}
    groups["motor"] = motor
    return groups


def switch_test(kw: dict, cur, groups, B=64, warm=150, nwin=8, win=16, seed0=5000):
    import torch
    H = B // 2
    half = torch.as_tensor(np.repeat([0, 1], H), device="cuda")

    def run(seed, first, second):
        b = Sim(B, [seed + i for i in range(B)], **kw)
        c1, c2 = cur.index_select(1, first), cur.index_select(1, second)
        for _ in range(warm):
            b.step(c1)
        out = torch.zeros((nwin, b.n, B), device="cuda")
        for w in range(nwin):
            for _ in range(win):
                out[w] += b.step(c2).float()
        return out

    ref = run(seed0, half, half)            # 0..H-1 hold cue 0, H..B-1 hold cue 1
    sw = run(seed0 + 2000, half, 1 - half)  # 0..H-1: 0 -> 1 ; H..B-1: 1 -> 0
    res = {}
    for name, idx in groups.items():
        it = torch.as_tensor(idx, device="cuda")
        x0, x1 = ref[-1][it][:, :H], ref[-1][it][:, H:]
        tt = (x1.mean(1) - x0.mean(1)) / torch.sqrt(x0.var(1) / H + x1.var(1) / H + 1e-9)
        y1, y0 = sw[-1][it][:, :H], sw[-1][it][:, H:]      # now seeing cue 1 / cue 0
        ts = (y1.mean(1) - y0.mean(1)) / torch.sqrt(y1.var(1) / H + y0.var(1) / H + 1e-9)
        fs = []
        for w in range(nwin):
            m0, m1 = ref[w][it][:, :H].mean(1), ref[w][it][:, H:].mean(1)
            dv = m1 - m0
            den = float((dv * dv).sum()) + 1e-12
            f01 = float(((sw[w][it][:, :H].mean(1) - m0) * dv).sum()) / den
            f10 = float(((sw[w][it][:, H:].mean(1) - m1) * (-dv)).sum()) / den
            fs.append(round((f01 + f10) / 2, 3))
        sig_r, sig_s = tt.abs() > 4, ts.abs() > 4
        res[name] = {"n": len(idx),
                     "reset_t4": int(sig_r.sum()), "switched_t4": int(sig_s.sum()),
                     "agree": int((sig_r & sig_s & (torch.sign(tt) == torch.sign(ts))).sum()),
                     "disagree": int((sig_r & sig_s & (torch.sign(tt) != torch.sign(ts))).sum()),
                     "rate": round(float(torch.cat([x0, x1], 1).mean()) / win, 4),
                     "f_by_window": fs}
    return {"global_rate": round(float(ref[-1].mean()) / win, 4), "groups": res}


def decoder_score(Xtr, ytr, Xte, yte, shuffled=False, seed=20260930):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64)
    keep = Xtr.std(0) > 0
    if keep.sum() == 0:
        return 1.0 / len(np.unique(ytr))
    if shuffled:
        ytr = np.random.default_rng(seed).permutation(ytr)
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Xtr[:, keep]), ytr)
    return float(clf.score(sc.transform(Xte[:, keep]), yte))


def run_switch(args, ctl):
    import torch
    names, first, g = inputs("first")
    cur = torch.from_numpy(first[[names.index(p) for p in PAIR]].T.copy()).cuda()
    groups = depth_groups(g)
    chosen = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    out = {"what": "cue-switch hysteresis test per model variant", "pair": PAIR,
           "design": {"B": 64, "warm": 150, "windows": 8, "window_ticks": 16,
                      "reference_seeds": "5000..5063", "switched_seeds": "7000..7063",
                      "variants": {k: VARIANTS[k] for k in chosen}},
           "complete": False, "results": {}}
    part = args.output.with_suffix(".json.part")
    for i, name in enumerate(chosen):
        ctl.begin(name)
        t0 = time.time()
        out["results"][name] = {**switch_test(VARIANTS[name], cur, groups),
                                "seconds": round(time.time() - t0, 1)}
        part.write_text(json.dumps(out, indent=2))
        part.replace(args.output)
        m = out["results"][name]["groups"]["motor"]
        ctl.done({"variant": name, "motor_f_last": m["f_by_window"][-1],
                  "motor_reset_t4": m["reset_t4"], "motor_agree": m["agree"]})
        print(name, json.dumps({k: (v["f_by_window"][-1], v["reset_t4"], v["agree"])
                                for k, v in out["results"][name]["groups"].items()}), flush=True)
    return out


def run_decode(args, ctl):
    import torch
    names, first, g = inputs("first")
    motor = torch.as_tensor(datasets.motor_index(g, datasets.DEFAULT), device="cuda")
    S, per, T = len(names), args.per, 166
    lab = np.repeat(np.arange(S), per)
    cur = torch.from_numpy(first.T.copy()).cuda().index_select(1, torch.as_tensor(lab, device="cuda"))
    ctl.begin("simulate")
    b = Sim(S * per, [70000 + i for i in range(S * per)])
    C = torch.zeros((T, len(motor), S * per), device="cuda", dtype=torch.float16)
    for t in range(T):
        C[t] = b.step(cur).index_select(0, motor).half()
        if t % 20 == 0:
            ctl.sub(t, T, "tick")
    C = C.float().cpu().numpy()
    ctl.done({"brains": S * per})
    tr = np.tile(np.arange(per) < per // 2, S)
    windows = {f"sweep_0_{K}": (0, K) for K in (8, 16, 24, 32)}
    windows["gate_e_150_166"] = (150, 166)
    res = {}
    for name, (a0, a1) in windows.items():
        X = C[a0:a1].sum(0).T
        res[name] = {"decode": decoder_score(X[tr], lab[tr], X[~tr], lab[~tr]),
                     "shuffled": decoder_score(X[tr], lab[tr], X[~tr], lab[~tr], shuffled=True),
                     "features_nonconstant": int((X[tr].std(0) > 0).sum())}
    # the pair used by Gate G, same brains
    pa, pb = names.index(PAIR[0]), names.index(PAIR[1])
    m = (lab == pa) | (lab == pb)
    for name, (a0, a1) in windows.items():
        X = C[a0:a1].sum(0).T[m]
        y = (lab[m] == pb).astype(int)
        t2 = tr[m]
        res[name]["pair_decode"] = decoder_score(X[t2], y[t2], X[~t2], y[~t2])
    return {"what": "ten-scene decoding from motor counts, sweep windows vs the Gate E window",
            "scenes": names, "seeds_per_scene": per, "train_seeds": "first half of each scene",
            "test_seeds": "second half", "chance": 1 / S, "decoder": "float64 StandardScaler + "
            "logistic C=.05 (Gate E / Gate G decoder), constant training columns dropped",
            "noise_seeds": f"70000..{70000 + S * per - 1}", "results": res, "complete": True}


def run_smooth(args, ctl):
    import torch
    names, seqs, g = inputs("seq")
    motor = torch.as_tensor(datasets.motor_index(g, datasets.DEFAULT), device="cuda")
    out = {"what": "smoothness of the sweep response along gameplay sequences",
           "K": 16, "seeds_per_frame": args.seeds, "results": {}, "complete": False}
    for scene in args.scenes.split(","):
        ctl.begin(scene)
        seq = seqs[names.index(scene)]
        F, S = len(seq), args.seeds
        lab = np.repeat(np.arange(F), S)
        cur = torch.from_numpy(seq.T.copy()).cuda().index_select(1, torch.as_tensor(lab, device="cuda"))
        b = Sim(F * S, [123000 + i for i in range(F * S)])
        C = torch.zeros((len(motor), F * S), device="cuda")
        for _ in range(16):
            C += b.step(cur).index_select(0, motor).float()
        C = C.cpu().numpy().T.reshape(F, S, -1)
        mA, mB = C[:, :S // 2].mean(1), C[:, S // 2:].mean(1)

        def corr(x, y):
            x, y = x - x.mean(), y - y.mean()
            return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
        Z = np.array([[corr(mA[i], mB[j]) for j in range(F)] for i in range(F)])
        out["results"][scene] = {
            "same_frame_splithalf_r": round(float(np.mean(np.diag(Z))), 4),
            "r_by_frame_distance": {d: round(float(np.mean([Z[f, f + d] for f in range(F - d)])), 4)
                                    for d in (1, 2, 4, 8, 16, 32)},
            "exact_frame_identification": round(float(np.mean(Z.argmax(1) == np.arange(F))), 4),
            "within_one_frame": round(float(np.mean(np.abs(Z.argmax(1) - np.arange(F)) <= 1)), 4),
            "chance": round(1 / F, 4)}
        ctl.done({"scene": scene, **{k: v for k, v in out["results"][scene].items()
                                     if k != "r_by_frame_distance"}})
    out["complete"] = True
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["switch", "decode", "smooth"])
    ap.add_argument("--variants", default="all")
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--scenes", default="1-1_open,1-1_obstacles")
    ap.add_argument("--tag", default="20260930")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("declared GPU unavailable")
    run_id = f"gate_g_dynamics_{args.mode}-{args.tag}"
    args.output = RUNS / f"{run_id}.json"
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = RunManifest(run_id, "Gate G", f"exploratory dynamics diagnostic ({args.mode})",
                           {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                           dataset=str(config.get("connectome.dataset")),
                           input_files={**dataset_files(), "script": Path(__file__),
                                        "config": ROOT / "configs/base.yaml"},
                           output_files={"report": args.output})
    total = {"switch": len(VARIANTS) if args.variants == "all" else len(args.variants.split(",")),
             "decode": 1, "smooth": len(args.scenes.split(","))}[args.mode]
    ctl = RunControl(run_id, total=total, checkpoint=lambda: None, unit=args.mode,
                     meta={"question": "Does the network represent the current cue?"})
    try:
        fn = {"switch": run_switch, "decode": run_decode, "smooth": run_smooth}[args.mode]
        rep = fn(args, ctl)
        rep.update({"complete": True, "scope": "exploratory diagnostic; no learning",
                    "manifest": str(manifest.path),
                    "utc": datetime.now(UTC).isoformat(timespec="seconds")})
        part = args.output.with_suffix(".json.part")
        part.write_text(json.dumps(rep, indent=2, allow_nan=False))
        part.replace(args.output)
        ctl.close("complete")
        manifest.finish("complete")
        if args.mode != "switch":
            print(json.dumps(rep["results"], indent=1))
    except BaseException as exc:
        ctl.close("failed")
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
