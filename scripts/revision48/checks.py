#!/usr/bin/env python
"""REVISION48 technical checks on the server's GPU, on a development graph only (no analysed graph, no score).

    checks.py --graph nulllat:100 [--slow]       -> artifacts/revision48/checks/checks.json (exit 1 if a check fails)
    checks.py --graph nulllat:100 --bench-only   -> artifacts/revision48/checks/benchmark.json

Checks: 1 explicit eta/tau equal the defaults bit for bit; 2 repeatability (raw, log CSR and log segment products);
3 resume equals continuous training; 4 evaluation is read-only, frozen learners predict identically; 5 benchmark of
one 12-learner main batch, projected to campaign hours from configs/revision48/campaign.yaml.
Log-weight checks use --base-gain (default 20.0); the frozen gain is calibrated later (calibrate_gain.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/revision48"


def main() -> int:
    import torch

    from flymario.experience_commune import learn
    from flymario.experience_commune.learn import Learner
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="nulllat:100")
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--base-gain", type=float, default=20.0)
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--bench-decisions", type=int, default=200)
    ap.add_argument("--bench-tag", default="", help="concurrency test: benchmark_<tag>.json (run N at once)")
    a = ap.parse_args()
    if a.graph.split(":")[0] not in ("nulllat", "nullin", "nulllatL", "nullinL") or not a.graph.endswith(("100", "101")):
        raise SystemExit("checks run on development graphs only (…:100 / …:101)")
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    data, dev = ART / "data", c["blocks"]["dev"][0]
    out = ART / "checks"
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"graph": a.graph, "slow": a.slow, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                 else "cpu", "torch": torch.__version__}
    kw = {"slow": a.slow, "eval_cols": 64}

    def steps(gr, n):
        for _ in range(n):
            gr.step()
        return gr

    if not a.bench_only:
        ls = [Learner(t, "U", dev) for t in c["tasks"]]
        lx = [Learner(t, "U", dev, "main", 0.001, 2.0) for t in c["tasks"]]
        g1 = steps(learn.GraphRun(a.graph, ls, data, **kw), 40)
        g2 = steps(learn.GraphRun(a.graph, lx, data, **kw), 40)
        rec["1_default_equals_explicit"] = bool(torch.equal(g1.brain.m, g2.brain.m))
        del g1, g2
        rep = {}
        for name, opt in {"raw": {}, "lognorm_csr": {"weights": "lognorm", "base_gain": a.base_gain},
                          "lognorm_segment": {"weights": "lognorm", "base_gain": a.base_gain,
                                              "frozen_product": "segment"}}.items():
            t0 = time.time()
            x = steps(learn.GraphRun(a.graph, ls, data, **kw, **opt), 40)
            y = steps(learn.GraphRun(a.graph, ls, data, **kw, **opt), 40)
            rep[name] = {"identical": bool(torch.equal(x.brain.m, y.brain.m)),
                         "max_abs_diff": float((x.brain.m - y.brain.m).abs().max()),
                         "seconds_per_step": round((time.time() - t0) / 80, 4)}
            if name == "lognorm_csr":
                m_csr = x.brain.m.clone()
            if name == "lognorm_segment":
                rep["segment_vs_csr_max_rel_diff"] = float(((x.brain.m - m_csr).abs() / m_csr.abs().clamp_min(1e-12)).max())
            del x, y
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        rec["2_repeatability"] = rep
        res = {}
        for name, opt in {"raw": {}, "lognorm": {"weights": "lognorm", "base_gain": a.base_gain}}.items():
            base = out / f"resume_{name}"
            shutil.rmtree(base, ignore_errors=True)
            cont = learn.run(a.graph, ls, data, base / "continuous", stop_at=500, banks_final=(), **kw, **opt)
            learn.run(a.graph, ls, data, base / "resumed", stop_at=250, banks_final=(), **kw, **opt)
            r2 = learn.run(a.graph, ls, data, base / "resumed", stop_at=500, banks_final=(), **kw, **opt)
            res[name] = {"resumed_from": r2["resumed_from"],
                         "digests_equal_at_500": cont["weight_digests"]["500"] == r2["weight_digests"]["500"]}
            shutil.rmtree(base, ignore_errors=True)
        rec["3_resume"] = res
        gr = steps(learn.GraphRun(a.graph, ls + [Learner(t, "U", dev, "frozen") for t in c["tasks"]], data, **kw), 60)
        before = {"m": gr.brain.m.clone(), "gens": [g.get_state().clone() for g in gr.brain._gens],
                  "base": gr.base.copy(), "ms": gr.ms.clone(), "ebar": gr.ebar.clone()}
        p0 = gr.evaluate("progress")
        p1 = gr.evaluate("progress")
        frozen_idx = [j for j, l in enumerate(gr.learners) if l.kind == "frozen"]
        g0 = learn.GraphRun(a.graph, gr.learners, data, **kw)          # frozen learners at k = 0
        q0 = g0.evaluate("progress")
        rec["4_evaluation"] = {
            "read_only": bool(torch.equal(before["m"], gr.brain.m) and np.array_equal(before["base"], gr.base)
                              and torch.equal(before["ms"], gr.ms) and torch.equal(before["ebar"], gr.ebar)
                              and all(torch.equal(x, g.get_state()) for x, g in zip(before["gens"], gr.brain._gens))),
            "repeatable": bool(np.array_equal(p0, p1)),
            "frozen_identical_k0_k60": bool(all(np.array_equal(p0[j], q0[j]) for j in frozen_idx))}
        del gr, g0
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # 5 benchmark: one main batch on the development graph
    ls12 = [Learner(t, "U", b) for b in c["blocks"]["main"] for t in c["tasks"]]
    gr = learn.GraphRun(a.graph, ls12, data, slow=a.slow)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _ = gr.evaluate("progress")
    ev = time.time() - t0
    steps(gr, 10)
    t0 = time.time()
    steps(gr, a.bench_decisions)
    st = (time.time() - t0) / a.bench_decisions
    t0 = time.time()
    _ = gr.evaluate("final")
    evf = time.time() - t0
    t0 = time.time()
    _ = gr.evaluate("ood")
    evo = time.time() - t0
    main_s = 6000 * st + len(c["checkpoints"]) * ev + evf + evo + gr.load_s
    tune_s = c["tuning"]["horizon"] * st + len(c["tuning"]["checkpoints"]) * ev + gr.load_s
    n_an = len(json.loads((ROOT / c["graphs"]["manifest"]).read_text())["order"])
    n_aut = len(c["graphs"]["autapse"]["S"]) + len(c["graphs"]["autapse"]["E"])
    n_gamma = len({t["gamma"] for t in c["tuning"]["trials"]})
    hours = {"tuning": n_an * n_gamma * tune_s / 3600, "reference": n_an * main_s / 3600,
             "tuned_upper_bound": n_an * main_s / 3600, "supervised": n_an * main_s / 3600,
             "lognorm": n_an * main_s / 3600, "autapse": n_aut * main_s / 3600}
    rec["5_benchmark"] = {"graph": a.graph, "learners": len(ls12), "train_s_per_decision": round(st, 4),
                          "eval_progress_s": round(ev, 1), "eval_final_s": round(evf, 1), "eval_ood_s": round(evo, 1),
                          "graph_load_s": round(gr.load_s, 1), "projected_main_run_s": round(main_s),
                          "projected_tuning_batch_s": round(tune_s),
                          "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                          if torch.cuda.is_available() else None,
                          "projected_hours": {k: round(v, 2) for k, v in hours.items()},
                          "projected_hours_total": round(sum(hours.values()), 2),
                          "note": "development graph, raw weights; log weights may differ (see 2_repeatability "
                                  "seconds_per_step); analysed graphs may differ in activity"}
    ok = True
    if not a.bench_only:
        ok = (rec["1_default_equals_explicit"] and rec["2_repeatability"]["raw"]["identical"]
              and all(v["digests_equal_at_500"] for v in rec["3_resume"].values())
              and all(rec["4_evaluation"].values()))
        rec["lognorm_frozen_product_recommendation"] = (
            "csr" if rec["2_repeatability"]["lognorm_csr"]["identical"] else
            "segment" if rec["2_repeatability"]["lognorm_segment"]["identical"] else "none repeatable: report")
        rec["pass"] = bool(ok)
    name = (f"benchmark_{a.bench_tag}.json" if a.bench_tag else "benchmark.json") if a.bench_only else "checks.json"
    p = out / name
    tmp = p.with_name(name + ".part")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)
    print(json.dumps(rec, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
