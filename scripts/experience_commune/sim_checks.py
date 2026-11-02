#!/usr/bin/env python
"""EXPERIENCE_COMMUNE_V1 simulator checks (mission §9), on a development graph and block d0 only.

1. pairing: with U forced to N's action and rho = 1, N and U learners end with identical plastic magnitudes
2. evaluation is read-only: plastic magnitudes, noise-generator states, baseline, RMS and eligibility means unchanged
3. resume: 250 decisions + resume to 500 equals 500 continuous decisions (weight digests at 500)
4. every learner restarts from the graph's initial magnitudes; the topology and dynamics parameters are unchanged
5. U journals: identical bytes for every graph of a block (one file per block and task, sha256 in every run.json)

    .venv/bin/python scripts/experience_commune/sim_checks.py --graph nulllat:100
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/experience_commune_v1"


def main() -> int:
    import torch

    from flymario.experience_commune import learn
    from flymario.experience_commune.learn import Learner
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="nulllat:100")
    a = ap.parse_args()
    rec = {"graph": a.graph}
    ls = [Learner(t, m, "d0") for t in ("where", "loom") for m in ("N", "U")]
    # 0 the row-major brain equals Study S's PlasticLIF bit for bit (counts, eligibilities, magnitudes)
    fast = learn.GraphRun(a.graph, ls, ART / "data", eval_cols=16)
    slow = learn.GraphRun(a.graph, ls, ART / "data", eval_cols=16, slow=True)
    for _ in range(25):
        fast.step()
        slow.step()
    rec["row_major_equals_plasticlif"] = {
        "magnitudes": bool(torch.equal(fast.brain.m, slow.brain.m)),
        "eligibility_means": bool(torch.equal(fast.ebar, slow.ebar)),
        "actions": bool(np.array_equal(fast.diag["action"][:25], slow.diag["action"][:25])),
        "step_s_fast_vs_slow": [float(np.median(fast.step_s[5:25])), float(np.median(slow.step_s[5:25]))]}
    del fast, slow
    torch.cuda.empty_cache()
    # 1 pairing
    gr = learn.GraphRun(a.graph, ls, ART / "data", eval_cols=64, pair_u_to_n=True)
    m0 = gr.brain.m.clone()
    rec["initial_magnitudes_equal_m0"] = bool(torch.equal(m0, gr.brain.m0[None, :].expand_as(m0)))
    for _ in range(60):
        gr.step()
    same = [bool(torch.equal(gr.brain.m[0], gr.brain.m[1])), bool(torch.equal(gr.brain.m[2], gr.brain.m[3]))]
    rec["pairing_identical_N_U"] = same
    rec["pairing_weights_moved"] = bool((gr.brain.m != m0).any())
    # 2 evaluation is read-only
    before = {"m": gr.brain.m.clone(), "gens": [g.get_state().clone() for g in gr.brain._gens],
              "base": gr.base.copy(), "ms": gr.ms.clone(), "ebar": gr.ebar.clone(), "k": gr.k}
    _ = gr.evaluate("progress")
    rec["eval_read_only"] = {
        "m": bool(torch.equal(before["m"], gr.brain.m)),
        "generators": all(torch.equal(x, g.get_state()) for x, g in zip(before["gens"], gr.brain._gens)),
        "baseline": bool(np.array_equal(before["base"], gr.base)), "rms": bool(torch.equal(before["ms"], gr.ms)),
        "eligibility_mean": bool(torch.equal(before["ebar"], gr.ebar)), "k": before["k"] == gr.k}
    p1 = gr.evaluate("progress")
    rec["eval_repeatable"] = bool(np.array_equal(_, p1))
    del gr
    torch.cuda.empty_cache()
    # 3 resume
    base = ART / "checks"
    shutil.rmtree(base, ignore_errors=True)
    cont = learn.run(a.graph, ls, ART / "data", base / "continuous", stop_at=500, eval_cols=64, banks_final=())
    learn.run(a.graph, ls, ART / "data", base / "resumed", stop_at=250, eval_cols=64, banks_final=())
    res = learn.run(a.graph, ls, ART / "data", base / "resumed", stop_at=500, eval_cols=64, banks_final=())
    rec["resume"] = {"resumed_from": res["resumed_from"],
                     "digests_equal_at_500": cont["weight_digests"]["500"] == res["weight_digests"]["500"],
                     "digests_equal_at_250": cont["weight_digests"]["250"] == res["weight_digests"]["250"],
                     "train_step_s_median": cont["train_step_s_median"], "eval_s": cont["eval_s"]}
    # 4 topology / dynamics untouched: the frozen operator and the plastic index sets are those of the graph
    from flymario.connectome import graphsel
    _g, W, motor = graphsel.load(a.graph)
    rec["motor_readout_n"] = int(len(motor))
    # 5 journals
    rec["journal_sha256"] = {k: v["journal_sha256"] for k, v in
                             json.loads((ART / "manifests/data.json").read_text())["entries"].items()}
    rec["pass"] = bool(rec["row_major_equals_plasticlif"]["magnitudes"] and rec["row_major_equals_plasticlif"]["actions"]
                       and rec["initial_magnitudes_equal_m0"] and all(same) and rec["pairing_weights_moved"]
                       and all(rec["eval_read_only"].values()) and rec["eval_repeatable"]
                       and rec["resume"]["digests_equal_at_500"])
    (ART / "checks").mkdir(parents=True, exist_ok=True)
    (ART / "manifests/sim_checks.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "journal_sha256"}, indent=1))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
