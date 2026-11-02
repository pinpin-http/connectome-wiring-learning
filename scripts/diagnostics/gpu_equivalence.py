#!/usr/bin/env python
"""Is the GPU brain the same brain? Three questions, each able to fail.

    python scripts/diagnostics/gpu_equivalence.py

Criteria (models/torch_lif.py vs `LIFBrain.step`, same noise), declared before running:
  A  identical spikes on the first tick; tick of first divergence reported.
  B  10 seeds, steady state: global and per-region rates within 3 SE of their difference.
  C  the campaign's decodability measurements within 2 binomial SE of the CPU values
     (100 samples each), both controls at chance. Also reports wall time per backend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gates"))
from transform_exposure import (NOISE_FRACTION, build_operator,  # noqa: E402
                                measure, measure_gpu)

from flymario import config  # noqa: E402
from flymario.agent import AgentConfig, FlyAgent  # noqa: E402
from flymario.connectome import datasets  # noqa: E402
from flymario.dynamics.calibrate import regions  # noqa: E402
from flymario.env import stimulus_bank  # noqa: E402
from flymario.models.spiking_reservoir import LIFBrain, LIFParams, normalize_in_degree  # noqa: E402
from flymario.models.torch_lif import TorchBrain, TorchLIF  # noqa: E402
from flymario.paths import RUNS  # noqa: E402

CAMPAIGN = RUNS / "transform_exposure_malecns.json"


def main() -> None:
    t0 = time.perf_counter()
    g = datasets.load("malecns")
    agent = FlyAgent(g)
    tpd = AgentConfig().ticks_per_decision
    warmup = int(config.get("dynamics.warmup_ticks"))
    bank = stimulus_bank.build(which="calibration", length=6)
    reg = regions(g, "malecns")
    camp = json.loads(CAMPAIGN.read_text())
    out = {"cases": {}}
    frame = bank.sequences[sorted(bank.sequences)[0]][2]
    agent.vision.reset()
    cur = agent.vision.encode(frame)

    # the two operators that matter: the chosen one and the old one
    cases = {"lin_raw": (False, False, 0.0158), "log_norm": (True, True, 0.05)}
    for name, (logw, norm, frac) in cases.items():
        W, gain, mass = build_operator(g, logw, norm, frac)
        W = normalize_in_degree(W) if norm else W.tocsc()
        params = LIFParams(gain=gain, noise_std=NOISE_FRACTION, seed=7)
        print(f"\n=== {name}  (gain {gain:.5g}) ===")

        # ---- A: trajectory -------------------------------------------------
        cpu = LIFBrain(W, params, normalize=False)
        gpu = TorchBrain(W, params, normalize=False)
        first_diff, same, T = None, [], 300
        for t in range(T):
            a, b = cpu.step(cur), gpu.step(cur)
            eq = bool(np.array_equal(a, b))
            same.append(float((a == b).mean()))
            if not eq and first_diff is None:
                first_diff = t + 1
        a_pass = same[0] == 1.0
        print(f"A  trajectory: tick 1 identical: {same[0] == 1.0}; first differing tick: "
              f"{first_diff or f'none in {T}'}; neuron-ticks identical over {T}: "
              f"{np.mean(same) * 100:.3f} %  -> {'PASS' if a_pass else 'FAIL'}")

        # ---- B: activity -----------------------------------------------------
        seeds = list(range(1, 11))
        def rates_cpu(sd):
            b = LIFBrain(W, LIFParams(gain=gain, noise_std=NOISE_FRACTION, seed=sd), normalize=False)
            for _ in range(warmup):
                b.step(cur)
            acc = []
            for _ in range(96):
                s = b.step(cur)
                acc.append([s.mean()] + [s[i].mean() for i in reg.values()])
            return np.mean(acc, 0)
        rc = np.array([rates_cpu(sd) for sd in seeds])
        gb = TorchLIF(W, LIFParams(gain=gain, noise_std=NOISE_FRACTION), batch=len(seeds),
                      normalize=False, seeds=seeds, noise="numpy")
        for _ in range(warmup):
            gb.step(cur)
        import torch
        idx = [torch.as_tensor(i, device=gb.device, dtype=torch.long) for i in reg.values()]
        acc = []
        for _ in range(96):
            s = gb.step(cur).float()
            acc.append(torch.stack([s.mean(0)] + [s[i].mean(0) for i in idx]).cpu().numpy())
        rg = np.mean(acc, 0).T                                   # (seeds, 1+regions)
        labels = ["global"] + list(reg)
        b_pass = True
        print("B  steady-state rates, mean over 10 seeds (CPU vs GPU, |diff| / SE of diff):")
        for j, lab in enumerate(labels):
            d = rg[:, j].mean() - rc[:, j].mean()
            se = np.sqrt(rg[:, j].var(ddof=1) / len(seeds) + rc[:, j].var(ddof=1) / len(seeds))
            z = abs(d) / se if se > 0 else 0.0
            ok = z < 3
            b_pass &= ok
            print(f"     {lab:<11} {rc[:, j].mean() * 100:6.3f} %  {rg[:, j].mean() * 100:6.3f} %"
                  f"   {z:4.2f} SE  {'ok' if ok else 'DIFFERENT'}")
        print(f"   -> {'PASS' if b_pass else 'FAIL'}")
        out["cases"][name] = {"first_diff_tick": first_diff, "identical_frac": float(np.mean(same)),
                              "A_pass": a_pass, "B_pass": bool(b_pass),
                              "rates_cpu": rc.mean(0).tolist(), "rates_gpu": rg.mean(0).tolist()}

    # ---- C: the campaign's own measurements, on the GPU --------------------
    print("\n=== C  the campaign's decodability measurements, re-run on the GPU ===")
    seeds = np.asarray(camp["params"]["seeds"])
    variants = ("log_norm", "lin_norm", "log_raw", "lin_raw")
    comps = ("shared_parameter", "matched_regime")
    c_pass = True
    out["C"] = []
    for comp, t in (("matched_regime", "lin_raw"), ("shared_parameter", "log_norm")):
        ref = next(r for r in camp["results"] if r["comparison"] == comp
                   and r["transform"] == t and r["arm"] == "connectome")
        logw, norm = ref["log_weights"], ref["normalised"]
        rng = np.random.default_rng([comps.index(comp), variants.index(t), 0])
        tg = time.perf_counter()
        r = measure_gpu(g, logw, norm, ref["fraction_used"], agent.vision, bank,
                        agent.motor_idx, reg, warmup, tpd, seeds, rng)
        dt_gpu = time.perf_counter() - tg
        se = np.sqrt(ref["decode"] * (1 - ref["decode"]) / (len(seeds) * 10))
        ok = (abs(r["decode"] - ref["decode"]) <= 2 * se and r["shuffled"] <= 0.25
              and r["no_input"] <= 0.25)
        c_pass &= ok
        print(f"   {t:9} {comp:17} CPU {ref['decode']:.3f}  GPU {r['decode']:.3f}  "
              f"(2 SE = {2 * se:.3f})  controls GPU {r['shuffled']:.2f}/{r['no_input']:.2f}  "
              f"rate CPU {ref['global_rate'] * 100:.2f}% GPU {r['global_rate'] * 100:.2f}%  "
              f"GPU {dt_gpu:.0f} s  -> {'ok' if ok else 'DIFFERENT'}")
        out["C"].append({"transform": t, "comparison": comp, "cpu": ref["decode"],
                         "gpu": r["decode"], "two_se": 2 * se, "gpu_seconds": dt_gpu,
                         "gpu_controls": [r["shuffled"], r["no_input"]], "ok": bool(ok)})
    print(f"   -> {'PASS' if c_pass else 'FAIL'}")

    # ---- speed: the same measurement on the CPU, for the record -------------
    ref = next(r for r in camp["results"] if r["comparison"] == "shared_parameter"
               and r["transform"] == "log_norm" and r["arm"] == "connectome")
    tc = time.perf_counter()
    measure(g, True, True, ref["fraction_used"], agent.vision, bank, agent.motor_idx,
            reg, warmup, tpd, seeds, np.random.default_rng([0, 0, 0]))
    dt_cpu = time.perf_counter() - tc
    print(f"\nspeed, one full measurement (100 brains x {warmup + 2 * tpd} ticks, x2 collects): "
          f"CPU {dt_cpu:.0f} s on one core   GPU {out['C'][1]['gpu_seconds']:.0f} s")
    out["speed"] = {"cpu_seconds": dt_cpu, "gpu_seconds": out["C"][1]["gpu_seconds"]}
    out["verdict"] = bool(all(c["A_pass"] and c["B_pass"] for c in out["cases"].values()) and c_pass)
    out["wall_seconds"] = time.perf_counter() - t0
    (RUNS / "gpu_equivalence.json").write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {'the GPU brain is the same brain' if out['verdict'] else 'NOT EQUIVALENT'}"
          f"  -> {RUNS / 'gpu_equivalence.json'}")


if __name__ == "__main__":
    main()
