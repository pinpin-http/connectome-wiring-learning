#!/usr/bin/env python
"""Revision campaign, sensitivity analysis: effect of equal-budget tuning on measures that do not saturate.

Declared in docs/REVISION_UNSATURATED.md; post hoc, so a sensitivity analysis, never confirmatory.

    .venv/bin/python scripts/diagnostics/revision_unsaturated.py exams.csv [--out FILE] [--reps 10000]
    .venv/bin/python scripts/diagnostics/revision_unsaturated.py --references      # synthetic checks only
    .venv/bin/python scripts/diagnostics/revision_unsaturated.py --r48 artifacts/revision48/analysis/tables/per_run_block.csv

Input CSV, one row per exam: condition (reference|tuned), family (connectome|S|E), graph, block, task
(where|loom), decisions (0 ... 6000), accuracy (%). Measures: gap250, gap1000 (accuracy gap, points) and
log2_t50, log2_t75 (log2 decisions to threshold, censored at 6,000; gap = family - connectome).
D = gap(tuned) - gap(reference), paired bootstrap over blocks and graphs, read against the declared margins.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXAMS = np.array([0, 250, 500, 1000, 2000, 4000, 6000], dtype=float)
THRESHOLDS = {"log2_t50": 50.0, "log2_t75": 75.0}
CHANCE = 12.5
MARGIN = {"gap250": 3.0, "gap1000": 3.0, "log2_t50": np.log2(1.25), "log2_t75": np.log2(1.25)}
MEASURES = ("gap250", "gap1000", "log2_t50", "log2_t75")
MAX_CENSORED = 0.20      # a time measure is not interpreted if more than 20 % of a cell's runs are censored


def time_to(acc: np.ndarray, thr: float) -> tuple[float, str]:
    """log2 of the decisions at which a curve sampled at EXAMS first reaches thr (linear interpolation).

    Flags: "ok"; "right" if never reached (counted as 6,000); "left" if reached within 250 decisions.
    """
    hit = np.nonzero(acc >= thr)[0]
    if hit.size == 0:
        return float(np.log2(EXAMS[-1])), "right"
    i = int(hit[0])
    if i == 0:                       # already above threshold before learning
        return float(np.log2(EXAMS[1])), "left"
    a0, a1 = acc[i - 1], acc[i]
    frac = 1.0 if a1 == a0 else (thr - a0) / (a1 - a0)
    t = EXAMS[i - 1] + frac * (EXAMS[i] - EXAMS[i - 1])
    return float(np.log2(max(t, 1.0))), ("left" if i == 1 else "ok")


def run_measures(acc: np.ndarray) -> dict[str, float]:
    """acc: accuracies at EXAMS for one learning run."""
    out = {"gap250": float(acc[1]), "gap1000": float(acc[3])}
    for name, thr in THRESHOLDS.items():
        out[name], out[name + "_flag"] = time_to(acc, thr)
    return out


def tensor(rows: list[dict]) -> tuple[dict, list[str], list[str], dict]:
    """-> values[condition][task][measure] = array (graphs, blocks), graph order, block order, censoring counts."""
    curves = defaultdict(dict)
    for r in rows:
        key = (r["condition"], r["task"], r["graph"], r["block"])
        curves[key][float(r["decisions"])] = float(r["accuracy"])
    graphs = sorted({k[2] for k in curves}, key=lambda g: (g != "connectome", g))
    blocks = sorted({k[3] for k in curves})
    vals, censored = {}, defaultdict(int)
    flags = {}
    for cond in ("reference", "tuned"):
        vals[cond] = {}
        for task in ("where", "loom"):
            vals[cond][task] = {m: np.full((len(graphs), len(blocks)), np.nan) for m in MEASURES}
            for k in THRESHOLDS:
                flags[(cond, task, k)] = np.zeros((len(graphs), len(blocks)), dtype=bool)
            for gi, g in enumerate(graphs):
                for bi, b in enumerate(blocks):
                    c = curves.get((cond, task, g, b))
                    if c is None:
                        continue
                    if sorted(c) != list(EXAMS):
                        raise ValueError(f"{cond}/{task}/{g}/{b}: exams {sorted(c)} != {list(EXAMS)}")
                    m = run_measures(np.array([c[t] for t in EXAMS]))
                    for k in MEASURES:
                        vals[cond][task][k][gi, bi] = m[k]
                    for k in THRESHOLDS:
                        if m[k + "_flag"] != "ok":
                            flags[(cond, task, k)][gi, bi] = True
                            censored[f"{cond}/{task}/{g}/{k}/{m[k + '_flag']}"] += 1
    vals["_flags"] = flags
    return vals, graphs, blocks, dict(censored)


def gap(v: np.ndarray, conn: int, fam: np.ndarray, measure: str) -> float:
    """v: (graphs, blocks) already restricted to the drawn blocks; fam: drawn family row indices."""
    g = np.nanmean(v, axis=1)
    d = g[conn] - g[fam].mean()
    return float(-d if measure.startswith("log2_t") else d)


def bootstrap(ref: np.ndarray, tun: np.ndarray, fam_rows: np.ndarray, measure: str,
              reps: int, rng: np.random.Generator) -> dict:
    nb = ref.shape[1]
    point = {"reference": gap(ref, 0, fam_rows, measure), "tuned": gap(tun, 0, fam_rows, measure)}
    point["D"] = point["tuned"] - point["reference"]
    draws = np.empty((reps, 3))
    for i in range(reps):
        b = rng.integers(0, nb, nb)
        f = fam_rows[rng.integers(0, fam_rows.size, fam_rows.size)]
        r, t = gap(ref[:, b], 0, f, measure), gap(tun[:, b], 0, f, measure)
        draws[i] = r, t, t - r
    ci = np.percentile(draws, [2.5, 97.5], axis=0)
    out = {}
    for j, k in enumerate(("reference", "tuned", "D")):
        lo, hi = float(ci[0, j]), float(ci[1, j])
        out[k] = {"point": point[k], "ci95": [lo, hi], "reading": reading(lo, hi, MARGIN[measure])}
    return out


def reading(lo: float, hi: float, margin: float) -> str:
    """Declared reading of an interval against a margin (docs/REVISION_UNSATURATED.md §4)."""
    if lo > margin:
        return "above margin"
    if hi < -margin:
        return "below margin"
    if -margin <= lo and hi <= margin:
        return "within margin"
    return "not decided"


def analyse(rows: list[dict], reps: int = 10000, seed: int = 20261102) -> dict:
    vals, graphs, blocks, censored = tensor(rows)
    fams = {"S": [], "E": []}
    fam_of = {r["graph"]: r["family"] for r in rows}
    for gi, g in enumerate(graphs):
        if fam_of[g] in fams:
            fams[fam_of[g]].append(gi)
    if graphs[0] != "connectome":
        raise ValueError("no connectome rows")
    rng = np.random.default_rng(seed)
    out = {"graphs": graphs, "blocks": blocks, "reps": reps, "seed": seed,
           "margins": {k: float(v) for k, v in MARGIN.items()}, "censored_runs": censored, "results": {}}
    for task in ("where", "loom"):
        for fam, rows_ in fams.items():
            if not rows_:
                continue
            for m in MEASURES:
                r = bootstrap(vals["reference"][task][m], vals["tuned"][task][m], np.array(rows_), m, reps, rng)
                if m in THRESHOLDS:
                    frac = {f"{cond}/{who}": float(vals["_flags"][(cond, task, m)][idx].mean())
                            for cond in ("reference", "tuned")
                            for who, idx in (("connectome", [0]), (fam, rows_))}
                    r["censored_fraction"] = frac
                    if max(frac.values()) > MAX_CENSORED:
                        for k in ("reference", "tuned", "D"):
                            r[k]["reading"] = "not interpretable (censoring)"
                out["results"][f"{task}/{fam}/{m}"] = r
            out.setdefault("verdict", {})[f"{task}/{fam}"] = verdict(out["results"], task, fam)
    return out


def verdict(results: dict, task: str, fam: str) -> dict:
    """Declared reading (docs/REVISION_UNSATURATED.md §4): log2_t75, else log2_t50, else undecidable."""
    words = {"below margin": "the tuning reduces the connectome's speed advantage beyond the ceiling",
             "within margin": "the reduction of L is compatible with the ceiling alone",
             "above margin": "the tuning increases the connectome's speed advantage",
             "not decided": "not decided at this precision"}
    for m in ("log2_t75", "log2_t50"):
        d = results[f"{task}/{fam}/{m}"]["D"]
        if d["reading"] in words:
            return {"measure": m, "D": d["point"], "ci95": d["ci95"], "reading": words[d["reading"]]}
    return {"measure": None, "reading": "these exams cannot separate the tuning from the ceiling"}


# ----------------------------------------------------------------------------------------- synthetic references
def curve(rate: float, final: float = 100.0) -> np.ndarray:
    """Exam accuracies of an exponential approach from chance to `final` with time constant 1/rate."""
    return CHANCE + (final - CHANCE) * (1.0 - np.exp(-rate * EXAMS))


def synthetic(kind: str, n_null: int = 10, n_block: int = 6, seed: int = 0) -> list[dict]:
    """Reference data sets with a known answer (docs/REVISION_UNSATURATED.md §5).

    ceiling: connectome 2x faster, tuning x4 for all (D on log2 times within margin); shrink: tuning x8 for
    nulls, x2 for the connectome (D below margin); identical: every gap and D within margin.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = {"ceiling": (2.0, 4.0, 4.0), "shrink": (2.0, 2.0, 8.0), "identical": (1.0, 4.0, 4.0)}[kind]
    ratio, tune_c, tune_n = base
    k0 = 1.0 / 2000.0
    graphs = [("connectome", "connectome")] + [(f"nulllat{i}", "S") for i in range(n_null)] \
        + [(f"nullin{i}", "E") for i in range(n_null)]
    for g, fam in graphs:
        jitter = 1.0 if fam == "connectome" else float(np.exp(rng.normal(0, 0.05)))
        k = k0 * (ratio if fam == "connectome" else 1.0) * jitter
        for cond, mult in (("reference", 1.0), ("tuned", tune_c if fam == "connectome" else tune_n)):
            for task in ("where", "loom"):
                for b in range(n_block):
                    acc = curve(k * mult * float(np.exp(rng.normal(0, 0.02))))
                    for t, a in zip(EXAMS, acc):
                        rows.append(dict(condition=cond, family=fam, graph=g, block=f"m{b}", task=task,
                                         decisions=t, accuracy=a))
    return rows


def learning_area(acc: np.ndarray) -> float:
    """Area under the exam curve over 0-6,000 decisions, normalised (the campaign's L)."""
    return float(np.trapezoid(acc, EXAMS) / EXAMS[-1])


def check_references(reps: int = 2000) -> dict:
    """Run the three references; raise AssertionError if any declared expectation fails (declaration §5)."""
    res = {k: analyse(synthetic(k), reps=reps)["results"] for k in ("ceiling", "shrink", "identical")}
    for key, r in res["ceiling"].items():
        m = key.split("/")[2]
        if m == "log2_t75":
            assert r["D"]["reading"] == "within margin", ("ceiling", key, r["D"])
        elif m == "log2_t50":
            assert r["D"]["reading"] in ("within margin", "not interpretable (censoring)"), ("ceiling", key, r["D"])
        else:   # fixed-time gaps move under a pure common speed-up: they cannot separate tuning from ceiling
            assert r["D"]["reading"] != "within margin", ("ceiling", key, r["D"])
    for key, r in res["shrink"].items():
        if key.split("/")[2] == "log2_t75":
            assert r["D"]["reading"] == "below margin", ("shrink", key, r["D"])
    for key, r in res["identical"].items():
        for k in ("reference", "tuned", "D"):
            assert r[k]["reading"] in ("within margin", "not interpretable (censoring)"), ("identical", key, k, r[k])
        if key.split("/")[2] != "log2_t50":
            assert r["D"]["reading"] == "within margin", ("identical", key, r["D"])
    # the learning area itself shows the ceiling artefact on the ceiling reference
    k0 = 1.0 / 2000.0
    d_area = (learning_area(curve(8 * k0)) - learning_area(curve(4 * k0))) \
        - (learning_area(curve(2 * k0)) - learning_area(curve(k0)))
    assert d_area < -3.0, d_area
    summary = {kind: {k: [round(r["D"]["point"], 3), r["D"]["reading"]] for k, r in v.items() if k.startswith("where/S")}
               for kind, v in res.items()}
    return {"references": "pass", "ceiling_reference_D_learning_area": round(d_area, 2), "where/S": summary}


def rows_from_r48(path: str) -> list[dict]:
    """The campaign's wide table (scripts/revision48/figures.py, columns A_0 ... A_6000 as fractions)
    -> the long rows above, reference and tuned conditions only."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] not in ("reference", "tuned"):
                continue
            for t in EXAMS:
                out.append(dict(condition=r["condition"], family=r["family"], graph=r["graph"], block=r["block"],
                                task=r["task"], decisions=t, accuracy=100.0 * float(r[f"A_{int(t)}"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--out", default="artifacts/runs/revision/unsaturated.json")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--references", action="store_true", help="synthetic checks only")
    ap.add_argument("--r48", help="the campaign's per_run_block.csv instead of a long CSV")
    a = ap.parse_args()
    refs = check_references()                 # references always run first; a failure stops before any data
    print(json.dumps(refs))
    if a.references or not (a.csv or a.r48):
        return 0
    if a.r48:
        rows = rows_from_r48(a.r48)
    else:
        with open(a.csv, newline="") as f:
            rows = list(csv.DictReader(f))
    out = analyse(rows, reps=a.reps)
    out["references"] = refs
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    for k, r in out["results"].items():
        d = r["D"]
        print(f"{k:24s} ref {r['reference']['point']:+7.2f}  tuned {r['tuned']['point']:+7.2f}  "
              f"D {d['point']:+7.2f} [{d['ci95'][0]:+.2f}; {d['ci95'][1]:+.2f}]  {d['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
