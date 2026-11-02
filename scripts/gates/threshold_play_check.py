#!/usr/bin/env python
"""Calibrated-decision play test (docs/PERCEPTION_CREDIT_STUDY.md §20): threshold moving on P(jump).

1. Calibration on training levels (1-1, 2-1, 4-1, 7-1; seed 20261017): per brain, tau_b = the Youden threshold
   of P(jump) (softmax(scores / 2) summed over jump buttons) for the take-off label v2.
2. Play on unseen levels: RIGHT+A+B when P(jump) >= tau_b, else the best non-jump action; against K = 5
   rate-matched random jumpers (PASS: Wilcoxon p < .05).

    .venv/bin/python scripts/gates/threshold_play_check.py --ckpt artifacts/runs/checkpoints/mario_stageD.pt \
        --tag stageD [--graph connectome] [--k 5]
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
sys.path.insert(0, str(ROOT / "scripts/gates"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from criterion_e_relabel import frames as relabel_frames
from mario_generalisation_eval import JUMP as JUMPS
from mario_generalisation_eval import TAU, Evaluator
from oracle_criterion_check import JUMP, RUN, criterion, play

from flymario.env.retro_env import ACTION_NAMES
from flymario.paths import RUNS
from flymario.training.sweep import sweep_once

LEVELS = ("Level5-1", "Level8-1", "Level3-1", "Level6-1")
DELAYS = tuple(range(0, 22, 3))
TRAIN = ("Level1-1", "Level2-1", "Level4-1", "Level7-1")
PRESS = ACTION_NAMES.index("RIGHT+A+B")


def pjump(sc):
    z = sc / TAU
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p[..., JUMPS].sum(-1)


class ThresholdEvaluator(Evaluator):
    """Greedy evaluation of rewritten scores: argmax then equals the threshold policy."""

    tau: np.ndarray

    def scores(self, frames):
        sc = super().scores(frames)
        pj = pjump(sc)
        out = sc.copy()
        out[:, JUMPS] = -np.inf
        jump = pj >= self.tau
        out[jump] = -np.inf
        out[jump, PRESS] = 0.0
        self._trace.append(jump.copy())
        return out

    def episode(self, suite, level, delay):
        """Adds, per brain, the share of jump decisions that follow a jump decision
        (A held: no new jump in this game) -- a failure-analysis field (RL_STUDY §4.3)."""
        self._trace = []
        rec = super().episode(suite, level, delay)
        tr = np.array(self._trace)                      # (steps, B); brain b valid for its decisions
        for b, e in enumerate(rec):
            j = tr[:e["decisions"], b] if len(tr) else np.zeros(0, bool)
            n = int(j[1:].sum())
            e["jump_after_jump"] = round(float((j[1:] & j[:-1]).sum() / n), 3) if n else None
        return rec

    def calibrate(self, fpr=None):
        """fpr (RL_STUDY §22, exploratory): instead of Youden, the lowest threshold whose
        false-positive rate on the same calibration frames is <= fpr."""
        torch = self.torch
        F = relabel_frames(self.g, levels=TRAIN, steps=2000, cap=60, seed=20261017)
        tko = np.array([f[3] for f in F])
        P = np.zeros((len(F), self.B))
        for i, f in enumerate(F):
            cur = torch.as_tensor(np.repeat(f[2][:, None], self.B, 1).astype(np.float32), device=self.brain.dev)
            c, _ = sweep_once(self.brain, self.params, cur, self.K, 0.5, False)
            P[i] = pjump(torch.einsum("bam,mb->ba", self.R, c).cpu().numpy())
        taus, info = [], {}
        for b, name in enumerate(self.names):
            cand = np.unique(P[:, b])
            j = [(tko & (P[:, b] >= t)).sum() / max(1, tko.sum()) - (~tko & (P[:, b] >= t)).sum() / max(1, (~tko).sum())
                 for t in cand]
            k = int(np.argmax(j))
            if fpr is not None:
                fp = np.array([(~tko & (P[:, b] >= t)).sum() / max(1, (~tko).sum()) for t in cand])
                ok = np.flatnonzero(fp <= fpr)
                k = int(ok[0]) if len(ok) else len(cand) - 1
            taus.append(float(cand[k]))
            info[name] = {"tau": round(float(cand[k]), 4), "youden": round(float(j[k]), 4),
                          "train_jump_rate": round(float((P[:, b] >= cand[k]).mean()), 3)}
        self.tau = np.array(taus)
        return {"n_frames": len(F), "label_rate": round(float(tko.mean()), 3), "brains": info, "fpr_target": fpr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--graph", default="connectome")
    a = ap.parse_args()
    ev = ThresholdEvaluator(a.ckpt, 1, 600, graph=a.graph)
    cal = ev.calibrate()
    print("calibration", json.dumps(cal), flush=True)
    rows = []
    for level in LEVELS:
        for d in DELAYS:
            rec = ev.episode("threshold", level, d)
            for b, e in enumerate(rec):
                q = e["jumps"] / max(1, e["decisions"])
                if e["learner"].startswith("untrained"):
                    r = {"best_x": float("nan")}
                else:
                    gens = [np.random.default_rng(20261018 + 1000 * LEVELS.index(level) + 10 * d + b + 100000 * j)
                            for j in range(a.k)]
                    r = {"best_x": float(np.mean([play(level, d, lambda ram, g=g, q=q: JUMP if g.random() < q else RUN)
                                                  ["best_x"] for g in gens]))}
                rows.append({"learner": e["learner"], "level": level, "delay": d, "best_x": e["best_x"],
                             "jump_rate": round(q, 3), "jump_after_jump": e.get("jump_after_jump"),
                             "random_best_x": r["best_x"], "end": e["end"]})
            print(f"{level} delay {d:2d}: " + " ".join(
                f"{r['learner'][:5]}{r['learner'][-1]}={r['best_x']:.0f}/{r['random_best_x']:.0f}"
                for r in rows[-len(rec):]), flush=True)
    res = {}
    for name in ev.names:
        R = [r for r in rows if r["learner"] == name]
        if np.isnan(R[0]["random_best_x"]):
            continue
        res[name] = {**criterion([r["best_x"] for r in R], [r["random_best_x"] for r in R]),
                     "mean_jump_rate": round(float(np.mean([r["jump_rate"] for r in R])), 3)}
        print(name, res[name], flush=True)
    rep = {"what": "calibrated-decision play (threshold moving on training levels) vs rate-matched random jumping",
           "checkpoint": a.ckpt, "graph": a.graph, "calibration": cal, "levels": LEVELS, "delays": DELAYS,
           "results": res, "rows": rows, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (RUNS / f"threshold_play_check-{a.tag}-20261006.json").write_text(json.dumps(rep, indent=1, default=float))
    ev.pool.close()


if __name__ == "__main__":
    main()
