#!/usr/bin/env python
"""RL_STUDY §21 follow-up (exploratory, descriptive): per brain, on the decisions of its own frozen
play (threshold rule, harness-v2 training suite), the true-positive rate of P(jump) for the take-off
label at a 5% (and 10%) false-positive rate, and the AUC. Reads the play logs only.

    .venv/bin/python scripts/diagnostics/own_frame_roc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
FP = ROOT / "artifacts/runs/frozen_probe"


def main():
    out = {}
    for letter, stream in (("", "scripted"), ("x", "expert"), ("d", "dagger"), ("c", "daggerctl")):
        for g in ("connectome", "nulllat0", "nulllat1", "nulllat2"):
            p = FP / f"playv2{letter}-training-{g}-20261009.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for name in dict.fromkeys(r["learner"] for r in d["rows"]):
                if name.startswith("ref"):
                    continue
                rr = [r for r in d["rows"] if r["learner"] == name]
                lab = np.concatenate([np.asarray(r["log"]["label"], bool) for r in rr])
                pj = np.concatenate([np.asarray(r["log"]["p"], float) for r in rr])
                fpr, tpr, _ = roc_curve(lab, pj)
                row = {"auc": round(float(roc_auc_score(lab, pj)), 3),
                       "tpr_at_fpr05": round(float(np.interp(0.05, fpr, tpr)), 3),
                       "tpr_at_fpr10": round(float(np.interp(0.10, fpr, tpr)), 3),
                       "label_rate": round(float(lab.mean()), 3), "n": int(len(lab))}
                out[f"{stream}/{g}/{name}"] = row
                print(f"{stream:9s} {g:10s} {name:14s} {row}")
    (ROOT / "artifacts/runs/mission/own_frame_roc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
