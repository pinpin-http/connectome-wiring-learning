#!/usr/bin/env python
"""Study S: variance of the learning speed V (E1) within a graph and between graphs (descriptive, no decision rule).

    .venv/bin/python scripts/diagnostics/study_s_variance.py          # reads artifacts/runs/study_s/*.json

Per task, in points of % correct: SD between graphs (40 nulls per family); SD within a graph (six
learners, pooled) and the implied SD of a graph value; SD between draws from the -rep1 replicates;
and the between-graph share of variance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "artifacts/runs/study_s"
TASKS = ("where", "loom")
FAMILIES = {"S": "nulllat", "E": "nullin"}


def learners(graph: dict, task: str) -> np.ndarray:
    return np.array([lr["E1"] for lr in graph["learners"] if lr["task"] == task and lr["kind"] == "true"]) * 100


def describe(graphs: dict[str, dict], reps: dict[str, dict]) -> dict:
    out = {}
    for task in TASKS:
        row = {}
        within = [learners(g, task).std(ddof=1) for g in graphs.values()]
        pooled = float(np.sqrt(np.mean(np.square(within))))
        n = len(learners(next(iter(graphs.values())), task))
        row["within_graph_learner_sd"] = pooled
        row["within_graph_value_sd"] = pooled / np.sqrt(n)
        for fam, prefix in FAMILIES.items():
            vals = np.array([learners(g, task).mean() for k, g in graphs.items() if k.startswith(prefix)])
            if vals.size >= 2:
                sd = float(vals.std(ddof=1))
                row[f"between_graphs_sd_{fam}"] = sd
                row[f"n_{fam}"] = int(vals.size)
                row[f"between_share_{fam}"] = sd ** 2 / (sd ** 2 + row["within_graph_value_sd"] ** 2)
        diffs = [learners(graphs[k], task).mean() - learners(r, task).mean() for k, r in reps.items() if k in graphs]
        if diffs:
            row["between_draws_value_sd"] = float(np.sqrt(np.mean(np.square(diffs)) / 2))
            row["n_replicated_graphs"] = len(diffs)
        out[task] = row
    return out


def load() -> tuple[dict, dict]:
    graphs, reps = {}, {}
    for p in sorted(D.glob("*.json")):
        name = p.stem
        if name == "connectome" or any(name == f"{pre}{i}" for pre in FAMILIES.values() for i in range(40)):
            graphs[name] = json.loads(p.read_text())
        elif name.endswith("-rep1"):
            reps[name[: -len("-rep1")]] = json.loads(p.read_text())
    return graphs, reps


def main() -> int:
    graphs, reps = load()
    if not graphs:
        print(f"no Study S file in {D} (see results/README.md)", file=sys.stderr)
        return 1
    out = describe(graphs, reps)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
