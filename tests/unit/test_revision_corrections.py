"""Checks of the two analyses written for the 2026-11-02 manuscript corrections (docs/REVISION_UNSATURATED.md).

They run on synthetic data with a known answer only; no campaign or study file is read.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/diagnostics/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _module("revision_unsaturated")
V = _module("study_s_variance")


def test_time_to_threshold_interpolates_and_flags():
    acc = np.array([12.5, 30.0, 50.0, 70.0, 80.0, 90.0, 95.0])     # exams 0..6000
    t, flag = R.time_to(acc, 60.0)                                  # between 500 (50) and 1000 (70)
    assert flag == "ok" and t == pytest.approx(np.log2(750.0))
    assert R.time_to(acc, 99.0) == (pytest.approx(np.log2(6000.0)), "right")
    assert R.time_to(acc, 20.0)[1] == "left"


def test_reading_against_margin():
    assert R.reading(3.5, 9.0, 3.0) == "above margin"
    assert R.reading(-9.0, -3.5, 3.0) == "below margin"
    assert R.reading(-1.0, 2.0, 3.0) == "within margin"
    assert R.reading(1.0, 5.0, 3.0) == "not decided"


def test_synthetic_references_pass():
    out = R.check_references(reps=500)
    assert out["references"] == "pass"
    assert out["ceiling_reference_D_learning_area"] < -3.0     # the learning area shows the ceiling artefact


def test_gap_sign_convention_for_times():
    v = np.array([[8.0, 8.0], [10.0, 10.0], [12.0, 12.0]])       # connectome row 0 reaches the criterion sooner
    assert R.gap(v, 0, np.array([1, 2]), "log2_t75") == pytest.approx(3.0)
    assert R.gap(v, 0, np.array([1, 2]), "gap250") == pytest.approx(-3.0)


def _graph(e1s, task_pair=("where", "loom")):
    return {"learners": [{"task": t, "kind": "true", "E1": e} for t in task_pair for e in e1s]}


def test_variance_decomposition_on_known_values():
    rng = np.random.default_rng(0)
    graphs = {"connectome": _graph([0.40] * 6)}
    for i in range(40):
        base = 0.25 + 0.016 * rng.standard_normal()
        graphs[f"nulllat{i}"] = _graph(list(base + 0.01 * rng.standard_normal(6)))
        graphs[f"nullin{i}"] = _graph(list(base + 0.01 * rng.standard_normal(6)))
    out = V.describe(graphs, {"nulllat0": graphs["nulllat0"]})
    row = out["where"]
    assert row["between_graphs_sd_S"] == pytest.approx(1.6, abs=0.5)        # points
    assert row["within_graph_learner_sd"] == pytest.approx(1.0, abs=0.2)
    assert row["within_graph_value_sd"] == pytest.approx(row["within_graph_learner_sd"] / np.sqrt(6))
    assert row["between_draws_value_sd"] == 0.0                              # identical replicate
    assert 0.5 < row["between_share_S"] < 1.0
